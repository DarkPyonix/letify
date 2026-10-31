//! Pins SPEC.md letify-link, NAT traversal item 6 (Address observers), TCP dial-back: a QAD
//! responder tests one TCP port of the requester, at the observed source IP only, and the
//! signed reply says whether it was reachable.

mod common;

use std::net::SocketAddr;
use std::time::Duration;

use common::HeldPort;
use letify_link::observe::{AddressObserver, QadLimits, QadObserver, QadResponder, Reachability};
use letify_link::{Identity, NodeId};
use tokio::net::{TcpListener, UdpSocket};
use tokio::time::timeout;

async fn responder(limits: QadLimits) -> (SocketAddr, NodeId) {
    let r = QadResponder::bind("127.0.0.1:0", Identity::generate())
        .await
        .unwrap()
        .with_limits(limits);
    let out = (r.local_addr(), r.node_id());
    tokio::spawn(r.run());
    out
}

async fn client() -> UdpSocket {
    UdpSocket::bind("127.0.0.1:0").await.unwrap()
}

/// A raw request: `LQAD`, version 1, a nonce of 7s, flags, port, then `tail` at offset 24.
fn raw_request(flags: u8, port: u16, tail: &[u8]) -> [u8; 128] {
    let mut r = [0u8; 128];
    r[..4].copy_from_slice(b"LQAD");
    r[4] = 1;
    r[5..21].fill(7);
    r[21] = flags;
    r[22..24].copy_from_slice(&port.to_be_bytes());
    r[24..24 + tail.len()].copy_from_slice(tail);
    r
}

async fn raw_reply(socket: &UdpSocket, to: SocketAddr, request: &[u8]) -> Option<Vec<u8>> {
    socket.send_to(request, to).await.unwrap();
    let mut buf = [0u8; 256];
    match timeout(Duration::from_secs(2), socket.recv_from(&mut buf)).await {
        Ok(Ok((n, _))) => Some(buf[..n].to_vec()),
        _ => None,
    }
}

#[tokio::test]
async fn a_dial_back_to_a_listening_port_is_reachable() {
    let (addr, id) = responder(QadLimits::default()).await;
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let socket = client().await;
    let (m, reach) = QadObserver::new(addr, Some(id))
        .observe_tcp(&socket, port)
        .await
        .unwrap();
    assert_eq!(m.mapped, socket.local_addr().unwrap());
    assert_eq!(reach, Reachability::Reachable);
}

#[tokio::test]
async fn a_dial_back_to_a_closed_port_is_not_reachable() {
    let (addr, id) = responder(QadLimits::default()).await;
    let held = HeldPort::new();
    let socket = client().await;
    let (_, reach) = QadObserver::new(addr, Some(id))
        .observe_tcp(&socket, held.addr.port())
        .await
        .unwrap();
    assert_eq!(reach, Reachability::Unreachable);
}

#[tokio::test]
async fn the_responder_dials_only_the_observed_source_address() {
    let (addr, _) = responder(QadLimits::default()).await;
    // A listener on another loopback address, on a port the observed address does not have.
    // macOS has no 127.0.0.2 unless an alias is added, so the test runs where it exists.
    let other = match TcpListener::bind("127.0.0.2:0").await {
        Ok(listener) => listener,
        Err(e) if e.kind() == std::io::ErrorKind::AddrNotAvailable => {
            eprintln!("skipped: 127.0.0.2 is not assigned on this machine");
            return;
        }
        Err(e) => panic!("{e}"),
    };
    let port = other.local_addr().unwrap().port();
    // The request names that address and port in every spare byte, in the formats a
    // careless parser might read, and asks to test the same port number at the source.
    let mut tail = vec![4u8];
    tail.extend_from_slice(&port.to_be_bytes());
    tail.extend_from_slice(&[127, 0, 0, 2]);
    tail.extend_from_slice(&[127, 0, 0, 2]);
    tail.extend_from_slice(&port.to_be_bytes());
    let socket = client().await;
    let reply = raw_reply(&socket, addr, &raw_request(1, port, &tail))
        .await
        .expect("a reply");
    // The byte before the signature is the verdict: 0 not reachable.
    assert_eq!(reply[reply.len() - 65], 0, "{reply:?}");
    assert!(
        timeout(Duration::from_millis(300), other.accept())
            .await
            .is_err(),
        "the responder connected to an address it was handed"
    );
}

#[tokio::test]
async fn a_dial_back_is_charged_more_than_a_plain_request() {
    // A burst of 7 tokens: one dial-back costs 1 + 4, so the second is answered but untested.
    let (addr, id) = responder(QadLimits {
        source_burst: 7,
        source_per_s: 0.001,
        dialback_cost: 4,
        ..QadLimits::default()
    })
    .await;
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let socket = client().await;
    let obs = QadObserver::new(addr, Some(id));
    let (_, first) = obs.observe_tcp(&socket, port).await.unwrap();
    assert_eq!(first, Reachability::Reachable);
    let (_, second) = obs.observe_tcp(&socket, port).await.unwrap();
    assert_eq!(second, Reachability::NotTested);
    // A plain request still costs one token and is still answered.
    obs.observe(&socket).await.unwrap();
}

#[tokio::test]
async fn a_dial_back_over_the_concurrency_cap_is_not_tested() {
    let (addr, id) = responder(QadLimits {
        dialback_concurrent: 0,
        ..QadLimits::default()
    })
    .await;
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let socket = client().await;
    let (m, reach) = QadObserver::new(addr, Some(id))
        .observe_tcp(&socket, listener.local_addr().unwrap().port())
        .await
        .unwrap();
    assert_eq!(m.mapped, socket.local_addr().unwrap());
    assert_eq!(reach, Reachability::NotTested);
}

#[tokio::test]
async fn port_zero_is_not_tested() {
    let (addr, id) = responder(QadLimits::default()).await;
    let socket = client().await;
    let (_, reach) = QadObserver::new(addr, Some(id))
        .observe_tcp(&socket, 0)
        .await
        .unwrap();
    assert_eq!(reach, Reachability::NotTested);
}

#[tokio::test]
async fn an_old_format_request_still_gets_the_old_format_reply() {
    let (addr, id) = responder(QadLimits::default()).await;
    let socket = client().await;
    let reply = raw_reply(&socket, addr, &raw_request(0, 0, &[]))
        .await
        .expect("a reply");
    // 4 magic, 1 version, 16 nonce, 1 family, 2 port, 4 IP, 64 signature.
    assert_eq!(reply.len(), 92);
    // The unchanged observe call reads it.
    let m = QadObserver::new(addr, Some(id))
        .observe(&socket)
        .await
        .unwrap();
    assert_eq!(m.mapped, socket.local_addr().unwrap());
    // A flagged request gets one byte more.
    let flagged = raw_reply(&socket, addr, &raw_request(1, 9, &[]))
        .await
        .expect("a reply");
    assert_eq!(flagged.len(), 93);
}

#[tokio::test]
async fn a_reply_without_the_reachable_byte_reads_as_not_tested() {
    // A responder from before the dial-back ignores the flag bytes and answers in the old
    // layout, which this stand-in does by not setting the bit it would honor: it echoes a
    // reply produced by a real responder to an unflagged request.
    let (real, id) = responder(QadLimits::default()).await;
    let old = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let old_addr = old.local_addr().unwrap();
    tokio::spawn(async move {
        let up = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let mut buf = [0u8; 256];
        loop {
            let Ok((n, from)) = old.recv_from(&mut buf).await else {
                return;
            };
            let mut request = buf[..n].to_vec();
            request[21..24].fill(0);
            up.send_to(&request, real).await.unwrap();
            let mut reply = [0u8; 256];
            let (m, _) = up.recv_from(&mut reply).await.unwrap();
            old.send_to(&reply[..m], from).await.unwrap();
        }
    });
    let socket = client().await;
    let (_, reach) = QadObserver::new(old_addr, None)
        .observe_tcp(&socket, 8080)
        .await
        .unwrap();
    assert_eq!(reach, Reachability::NotTested);
    let _ = id;
}
