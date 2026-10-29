//! Rendezvous, address records, hole punching and the policing check, end to end over
//! loopback. Spec: "Rendezvous trait" and "NAT traversal".

use std::net::SocketAddr;
use std::time::Duration;

use letify_link::Identity;
use letify_link::policing::{self, Verdict};
use letify_link::punch::{self, Plan, now_ms};
use letify_link::record::{AddressRecord, unix_now};
use letify_link::rendezvous::{MemoryHub, Message, Proto, Punch, recv_message, send_message};
use letify_link::stun::NatType;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream, UdpSocket};

/// Reserve `count` loopback TCP ports. The sockets are closed again, so the punch can bind them.
async fn free_tcp_addrs(count: usize) -> Vec<SocketAddr> {
    let mut held = Vec::new();
    for _ in 0..count {
        held.push(punch::reusable_socket("127.0.0.1:0".parse().unwrap()).unwrap());
    }
    held.iter().map(|s| s.local_addr().unwrap()).collect()
}

#[tokio::test]
async fn two_nodes_exchange_records_and_punch_tcp_over_the_in_memory_rendezvous() {
    let hub = MemoryHub::new();
    let (ida, idb) = (Identity::generate(), Identity::generate());
    let (a, b) = (ida.node_id(), idb.node_id());
    let (ra, rb) = (hub.endpoint(a), hub.endpoint(b));
    let addrs_a = free_tcp_addrs(punch::PUNCH_SOCKETS).await;
    let addrs_b = free_tcp_addrs(punch::PUNCH_SOCKETS).await;

    // First contact: A says hello and asks for B's record, B answers with it, A sends its own.
    send_message(
        &ra,
        b,
        &Message::Hello {
            key: *a.as_bytes(),
            want_record: true,
        },
    )
    .await
    .unwrap();
    let (from, hello) = recv_message(&rb).await.unwrap();
    assert_eq!(from, a);
    assert!(matches!(
        hello,
        Message::Hello {
            want_record: true,
            ..
        }
    ));
    let mut rec_b = AddressRecord::new(b, 1);
    rec_b.nat = NatType::EndpointIndependent;
    rec_b.tcp = addrs_b.clone();
    send_message(&rb, a, &Message::Record(rec_b.encode(&idb)))
        .await
        .unwrap();
    let mut rec_a = AddressRecord::new(a, 1);
    rec_a.nat = NatType::EndpointIndependent;
    rec_a.tcp = addrs_a.clone();
    send_message(&ra, b, &Message::Record(rec_a.encode(&ida)))
        .await
        .unwrap();

    let (_, Message::Record(bytes)) = recv_message(&ra).await.unwrap() else {
        panic!()
    };
    let seen_b = AddressRecord::decode(&bytes, unix_now()).unwrap();
    let (_, Message::Record(bytes)) = recv_message(&rb).await.unwrap() else {
        panic!()
    };
    let seen_a = AddressRecord::decode(&bytes, unix_now()).unwrap();
    assert_eq!((seen_a.node, seen_b.node), (a, b));
    assert_eq!(seen_b.tcp, addrs_b);

    // Punch offer and answer, then both sides dial at the agreed start.
    let offer = Punch {
        proto: Proto::Tcp,
        token: [7; 16],
        start_at_ms: now_ms() + 300,
        until_ms: now_ms() + 10_000,
        mappings: addrs_a.clone(),
    };
    send_message(&ra, b, &Message::PunchOffer(offer.clone()))
        .await
        .unwrap();
    let (_, Message::PunchOffer(got)) = recv_message(&rb).await.unwrap() else {
        panic!()
    };
    let answer = Punch {
        mappings: addrs_b.clone(),
        until_ms: got.until_ms - 1000,
        ..got.clone()
    };
    send_message(&rb, a, &Message::PunchAnswer(answer))
        .await
        .unwrap();
    let (_, Message::PunchAnswer(answer)) = recv_message(&ra).await.unwrap() else {
        panic!()
    };

    let plan_a = Plan::of(&offer).with_peer_until(answer.until_ms);
    let plan_b = Plan::of(&got).with_peer_until(answer.until_ms);
    assert_eq!(plan_a.until_ms, offer.until_ms - 1000);
    let (sa, sb) = tokio::join!(
        punch::punch_tcp_many(&addrs_a, &answer.mappings, &offer.token, true, plan_a),
        punch::punch_tcp_many(&addrs_b, &seen_a.tcp, &got.token, false, plan_b),
    );
    let (mut sa, mut sb) = (sa.unwrap(), sb.unwrap());
    assert_eq!(sa.len(), punch::PUNCH_SOCKETS);
    assert_eq!(sb.len(), punch::PUNCH_SOCKETS);
    // Pair i on one side is pair i on the other: bytes written on one arrive on the other.
    for (i, (x, y)) in sa.iter_mut().zip(sb.iter_mut()).enumerate() {
        x.write_all(&[i as u8; 3]).await.unwrap();
        let mut got = [0u8; 3];
        y.read_exact(&mut got).await.unwrap();
        assert_eq!(got, [i as u8; 3]);
    }
}

#[tokio::test]
async fn a_tcp_punch_to_a_peer_that_never_dials_times_out_and_names_the_dials() {
    let addrs = free_tcp_addrs(2).await;
    let now = now_ms();
    let err = punch::punch_tcp(
        addrs[0],
        addrs[1],
        b"t",
        true,
        Plan {
            start_at_ms: now,
            until_ms: now + 800,
        },
    )
    .await
    .unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::TimedOut);
    assert!(err.to_string().contains("ConnectionRefused"), "{err}");
}

#[tokio::test]
async fn udp_punch_over_loopback() {
    let (sa, sb) = (
        UdpSocket::bind("127.0.0.1:0").await.unwrap(),
        UdpSocket::bind("127.0.0.1:0").await.unwrap(),
    );
    let (aa, ab) = (sa.local_addr().unwrap(), sb.local_addr().unwrap());
    let now = now_ms();
    let plan = Plan {
        start_at_ms: now + 100,
        until_ms: now + 5000,
    };
    let token = [4u8; 16];
    let (to_b, to_a) = ([ab], [aa]);
    let (x, y) = tokio::join!(
        punch::punch_udp(vec![sa], &to_b, &token, plan, punch::SPRAY_PER_SECOND),
        punch::punch_udp(vec![sb], &to_a, &token, plan, punch::SPRAY_PER_SECOND),
    );
    let ((sa, from_a), (sb, from_b)) = (x.unwrap(), y.unwrap());
    assert_eq!((from_a, from_b), (ab, aa));
    // The returned sockets are the punched ones and carry data.
    sa.send_to(b"ping", ab).await.unwrap();
    let mut buf = [0u8; 16];
    loop {
        let (n, _) = sb.recv_from(&mut buf).await.unwrap();
        if &buf[..n] == b"ping" {
            break;
        }
    }
}

#[tokio::test]
async fn a_udp_spray_finds_the_one_right_pair_among_wrong_ports() {
    // Side A has four sockets and aims each at a window of ports, only one of which is B's.
    let b = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let ab = b.local_addr().unwrap();
    let mut a_socks = Vec::new();
    for _ in 0..4 {
        a_socks.push(UdpSocket::bind("127.0.0.1:0").await.unwrap());
    }
    let a_addr = a_socks[2].local_addr().unwrap();
    let wrong = punch::predicted_destinations(ab, NatType::Symmetric { step: 1 }, 5);
    let now = now_ms();
    let plan = Plan {
        start_at_ms: now,
        until_ms: now + 5000,
    };
    let token = [5u8; 16];
    let to_a = [a_addr];
    let (x, y) = tokio::join!(
        punch::punch_udp(a_socks, &wrong, &token, plan, punch::SPRAY_PER_SECOND),
        punch::punch_udp(vec![b], &to_a, &token, plan, punch::SPRAY_PER_SECOND),
    );
    // ab itself is not in the predicted window (it starts at ab + 1), so A's probes all miss.
    // B probes A's third socket, which answers, and B's extra acks let A succeed on that socket.
    let ((chosen, from), _) = (x.unwrap(), y.unwrap());
    assert_eq!(chosen.local_addr().unwrap(), a_addr);
    assert_eq!(from, ab);
}

/// A loopback proxy that limits both directions to `rate` bytes per second after the first
/// 4 KiB, so round trips stay fast and bulk data crawls.
async fn policing_proxy(target: SocketAddr, rate: f64) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        while let Ok((front, _)) = listener.accept().await {
            tokio::spawn(async move {
                let Ok(back) = TcpStream::connect(target).await else {
                    return;
                };
                let (fr, fw) = front.into_split();
                let (br, bw) = back.into_split();
                tokio::join!(pump(fr, bw, rate), pump(br, fw, rate));
            });
        }
    });
    addr
}

async fn pump(
    mut from: tokio::net::tcp::OwnedReadHalf,
    mut to: tokio::net::tcp::OwnedWriteHalf,
    rate: f64,
) {
    let mut buf = vec![0u8; 1024];
    let mut passed = 0usize;
    let began = tokio::time::Instant::now();
    while let Ok(n) = from.read(&mut buf).await {
        if n == 0 || to.write_all(&buf[..n]).await.is_err() {
            break;
        }
        passed += n;
        if passed > 4096 {
            let due = began + Duration::from_secs_f64((passed - 4096) as f64 / rate);
            tokio::time::sleep_until(due).await;
        }
    }
    let _ = to.shutdown().await;
}

async fn responder() -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        while let Ok((s, _)) = listener.accept().await {
            tokio::spawn(async move {
                let _ = policing::serve(s).await;
            });
        }
    });
    addr
}

const FLOOR: u64 = 1024 * 1024;

#[tokio::test]
async fn a_path_that_answers_pings_but_throttles_bulk_data_is_reported_policed() {
    let server = responder().await;
    // 20 KiB/s is under a twentieth of the 1 MiB/s floor (52 KiB/s).
    let proxy = policing_proxy(server, 20.0 * 1024.0).await;
    let stream = TcpStream::connect(proxy).await.unwrap();
    let verdict = policing::check(stream, FLOOR, Duration::from_millis(500))
        .await
        .unwrap();
    let Verdict::Policed(rates) = verdict else {
        panic!("{verdict:?}")
    };
    assert!(rates.up_bps < FLOOR as f64 / 20.0, "{rates:?}");
    assert!(rates.down_bps < FLOOR as f64 / 20.0, "{rates:?}");
    assert!(rates.rtt < Duration::from_millis(200), "{rates:?}");
}

#[tokio::test]
async fn an_unthrottled_path_is_usable() {
    let server = responder().await;
    let stream = TcpStream::connect(server).await.unwrap();
    let verdict = policing::check(stream, FLOOR, Duration::from_millis(500))
        .await
        .unwrap();
    let Verdict::Usable(rates) = verdict else {
        panic!("{verdict:?}")
    };
    assert!(
        rates.up_bps > FLOOR as f64 && rates.down_bps > FLOOR as f64,
        "{rates:?}"
    );
}
