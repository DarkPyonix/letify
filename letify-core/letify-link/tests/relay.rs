//! Pins SPEC.md letify-link, Relay items 7 to 13: the relay handshake and policy, the pairing
//! request, the splice, the limits, the dialer and promotion to a direct path.
#![cfg(feature = "relay")]

mod common;

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use common::*;
use letify_link::relay::{Relay, RelayConfig, RelayDialer, new_token};
use letify_link::{
    Accepted, Acceptor, Candidate, Config, Identity, ManagerConfig, PathEvent, PathManager,
    PeerPolicy,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::time::timeout;

async fn start_relay(policy: PeerPolicy, cfg: RelayConfig) -> (Identity, std::net::SocketAddr) {
    let id = Identity::generate();
    let relay = Relay::bind("127.0.0.1:0", id.clone(), policy, cfg)
        .await
        .unwrap();
    let addr = relay.local_addr();
    tokio::spawn(relay.run());
    (id, addr)
}

fn open() -> PeerPolicy {
    PeerPolicy::from_fn(|_| true)
}

/// A TCP forwarder that records every byte that passes toward the relay and back.
async fn recorder(target: std::net::SocketAddr) -> (std::net::SocketAddr, Arc<Mutex<Vec<u8>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let seen = Arc::new(Mutex::new(Vec::new()));
    tokio::spawn({
        let seen = seen.clone();
        async move {
            while let Ok((front, _)) = listener.accept().await {
                let seen = seen.clone();
                tokio::spawn(async move {
                    let Ok(back) = TcpStream::connect(target).await else {
                        return;
                    };
                    let (mut fr, mut fw) = front.into_split();
                    let (mut br, mut bw) = back.into_split();
                    let up = {
                        let seen = seen.clone();
                        async move {
                            let mut buf = vec![0u8; 16 << 10];
                            while let Ok(n) = fr.read(&mut buf).await {
                                if n == 0 || bw.write_all(&buf[..n]).await.is_err() {
                                    break;
                                }
                                seen.lock().unwrap().extend_from_slice(&buf[..n]);
                            }
                        }
                    };
                    let down = async move {
                        let mut buf = vec![0u8; 16 << 10];
                        while let Ok(n) = br.read(&mut buf).await {
                            if n == 0 || fw.write_all(&buf[..n]).await.is_err() {
                                break;
                            }
                            seen.lock().unwrap().extend_from_slice(&buf[..n]);
                        }
                    };
                    tokio::join!(up, down);
                });
            }
        }
    });
    (addr, seen)
}

#[tokio::test(flavor = "multi_thread")]
async fn a_session_runs_through_a_relay_and_the_relay_sees_only_ciphertext() {
    let (relay_id, relay_addr) = start_relay(open(), RelayConfig::default()).await;
    let (tap, seen) = recorder(relay_addr).await;
    let a = Identity::generate();
    let b = Identity::generate();
    let token = new_token();
    let da = RelayDialer::new(tap.to_string(), relay_id.node_id(), a.clone());
    let db = RelayDialer::new(tap.to_string(), relay_id.node_id(), b.clone());

    let server = tokio::spawn({
        let (b, a_id) = (b.clone(), a.node_id());
        async move {
            let t = db.connect(a_id, token).await.unwrap();
            let acceptor = Acceptor::new(b, PeerPolicy::allow(vec![a_id]), Config::default());
            let Accepted::Session(s) = acceptor.accept(t).await.unwrap() else {
                panic!("expected a new session");
            };
            let mut stream = s.accept_stream().await.unwrap();
            let mut got = Vec::new();
            let mut buf = vec![0u8; 64 << 10];
            while got.len() < (1 << 20) {
                let n = stream.read(&mut buf).await.unwrap();
                assert!(n > 0);
                got.extend_from_slice(&buf[..n]);
            }
            (got, s)
        }
    });
    let t = da.connect(b.node_id(), token).await.unwrap();
    let session = letify_link::Session::connect(t, &a, b.node_id(), Config::default())
        .await
        .unwrap();
    let mut stream = session.open_stream().await.unwrap();
    // The payload repeats a known marker, so any plaintext leak would be found.
    let marker = b"PLAINTEXT-MARKER-0123456789-abcdef";
    let mut payload = Vec::new();
    while payload.len() < (1 << 20) {
        payload.extend_from_slice(marker);
    }
    payload.truncate(1 << 20);
    stream.write_all(&payload).await.unwrap();
    let (got, _s) = timeout(Duration::from_secs(20), server)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(got, payload);

    let seen = seen.lock().unwrap();
    assert!(seen.len() > payload.len(), "the relay carried the payload");
    assert!(
        !seen.windows(8).any(|w| w == &marker[..8]),
        "the relay saw a plaintext pattern"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn an_unauthenticated_client_is_refused_by_the_relay() {
    let (relay_id, relay_addr) = start_relay(
        open(),
        RelayConfig {
            handshake_timeout: Duration::from_millis(500),
            ..RelayConfig::default()
        },
    )
    .await;
    // Bytes that are not a handshake are closed without a reply.
    let mut raw = TcpStream::connect(relay_addr).await.unwrap();
    raw.write_all(&[7u8; 200]).await.unwrap();
    let mut buf = [0u8; 16];
    let n = timeout(Duration::from_secs(5), raw.read(&mut buf))
        .await
        .unwrap()
        .unwrap_or(0);
    assert_eq!(n, 0, "the relay answers garbage with a closed connection");

    // A client that holds a different key for the relay cannot complete the handshake.
    let wrong = Identity::generate().node_id();
    let client = Identity::generate();
    let dialer = RelayDialer::new(relay_addr.to_string(), wrong, client);
    let err = dialer
        .connect(Identity::generate().node_id(), new_token())
        .await
        .expect_err("the handshake fails");
    assert!(!err.to_string().is_empty());
    let _ = relay_id;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_relay_with_an_allow_list_refuses_other_nodes() {
    let allowed = Identity::generate();
    let stranger = Identity::generate();
    let peer = Identity::generate();
    let (relay_id, relay_addr) = start_relay(
        PeerPolicy::allow(vec![allowed.node_id(), peer.node_id()]),
        RelayConfig::default(),
    )
    .await;
    let d = RelayDialer::new(relay_addr.to_string(), relay_id.node_id(), stranger);
    assert!(d.connect(peer.node_id(), new_token()).await.is_err());

    // An allowed pair still pairs on the same relay.
    let token = new_token();
    let da = RelayDialer::new(relay_addr.to_string(), relay_id.node_id(), allowed.clone());
    let dp = RelayDialer::new(relay_addr.to_string(), relay_id.node_id(), peer.clone());
    let (ra, rp) = tokio::join!(
        da.connect(peer.node_id(), token),
        dp.connect(allowed.node_id(), token)
    );
    let (mut ta, mut tp) = (ra.unwrap(), rp.unwrap());
    ta.write_all(b"hello").await.unwrap();
    let mut buf = [0u8; 5];
    tp.read_exact(&mut buf).await.unwrap();
    assert_eq!(&buf, b"hello");
}

#[tokio::test(flavor = "multi_thread")]
async fn requests_with_different_tokens_do_not_pair() {
    let (relay_id, relay_addr) = start_relay(
        open(),
        RelayConfig {
            pair_timeout: Duration::from_millis(600),
            ..RelayConfig::default()
        },
    )
    .await;
    let a = Identity::generate();
    let b = Identity::generate();
    let da = RelayDialer::new(relay_addr.to_string(), relay_id.node_id(), a.clone());
    let db = RelayDialer::new(relay_addr.to_string(), relay_id.node_id(), b.clone());
    let (ra, rb) = tokio::join!(
        da.connect(b.node_id(), new_token()),
        db.connect(a.node_id(), new_token())
    );
    assert!(ra.is_err() && rb.is_err(), "no pair without the same token");
}

#[tokio::test(flavor = "multi_thread")]
async fn per_client_rate_limit_is_enforced() {
    let (relay_id, relay_addr) = start_relay(
        open(),
        RelayConfig {
            rate_limit: Some(64 * 1024),
            ..RelayConfig::default()
        },
    )
    .await;
    let a = Identity::generate();
    let b = Identity::generate();
    let token = new_token();
    let da = RelayDialer::new(relay_addr.to_string(), relay_id.node_id(), a.clone());
    let db = RelayDialer::new(relay_addr.to_string(), relay_id.node_id(), b.clone());
    let (ra, rb) = tokio::join!(
        da.connect(b.node_id(), token),
        db.connect(a.node_id(), token)
    );
    let (mut ta, mut tb) = (ra.unwrap(), rb.unwrap());
    let data = pattern(128 * 1024, 5);
    let start = Instant::now();
    let writer = tokio::spawn(async move {
        ta.write_all(&data).await.unwrap();
        ta
    });
    let mut got = vec![0u8; 128 * 1024];
    timeout(Duration::from_secs(20), tb.read_exact(&mut got))
        .await
        .unwrap()
        .unwrap();
    let took = start.elapsed();
    let _ = writer.await;
    assert_eq!(got, pattern(128 * 1024, 5));
    // 128 KiB at 64 KiB/s takes about 2 s. The limiter allows a first chunk without delay.
    assert!(took >= Duration::from_millis(1400), "took {took:?}");
    assert!(took < Duration::from_secs(8), "took {took:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_relayed_path_is_promoted_to_a_direct_path_when_one_appears() {
    let (relay_id, relay_addr) = start_relay(open(), RelayConfig::default()).await;
    // The client reaches the relay through a slow link, so the relayed path is slow.
    let slow = Throttle::to(relay_addr, 150_000.0).await;
    let a = Identity::generate();
    let b = Identity::generate();
    let token = new_token();
    let acceptor = Arc::new(Acceptor::new(
        b.clone(),
        PeerPolicy::allow(vec![a.node_id()]),
        Config::default(),
    ));

    // The direct port is reserved now and listens only after the relayed path is running.
    let reserved = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let direct = reserved.local_addr().unwrap();
    drop(reserved);

    let (tx, mut sessions) = tokio::sync::mpsc::channel(4);
    {
        let (acceptor, tx) = (acceptor.clone(), tx.clone());
        let db = RelayDialer::new(relay_addr.to_string(), relay_id.node_id(), b.clone());
        let a_id = a.node_id();
        tokio::spawn(async move {
            let t = db.connect(a_id, token).await.unwrap();
            if let Ok(Accepted::Session(s)) = acceptor.accept(t).await {
                let _ = tx.send(s).await;
            }
        });
    }
    let da = RelayDialer::new(slow.addr.to_string(), relay_id.node_id(), a.clone());
    let candidates = vec![
        da.candidate("relay", b.node_id(), token),
        Candidate::tcp("direct", direct),
    ];
    let cfg = ManagerConfig {
        measure_interval: Duration::from_millis(100),
        redial_initial: Duration::from_millis(50),
        redial_max: Duration::from_millis(200),
        policy: letify_link::hysteresis::Policy {
            dwell: Duration::ZERO,
            confirmations: 2,
            spacing: Duration::from_millis(100),
            ..Default::default()
        },
        ..ManagerConfig::default()
    };
    let mut pm = PathManager::start(a.clone(), b.node_id(), candidates, Config::default(), cfg)
        .await
        .unwrap();
    let server_session = sessions.recv().await.unwrap();
    let mut stream = pm.session().open_stream().await.unwrap();
    tokio::spawn(async move {
        let chunk = pattern(1 << 20, 3);
        while stream.write_all(&chunk).await.is_ok() {}
    });
    tokio::spawn(async move {
        let mut sb = server_session.accept_stream().await.unwrap();
        let mut buf = vec![0u8; 1 << 20];
        while let Ok(n) = sb.read(&mut buf).await {
            if n == 0 {
                break;
            }
        }
    });
    tokio::time::sleep(Duration::from_millis(500)).await;
    let listener = TcpListener::bind(direct).await.unwrap();
    let _direct_sessions = spawn_server_into(listener, acceptor);

    let moved = timeout(Duration::from_secs(40), async {
        loop {
            if let Some(PathEvent::Preferred { from, to }) = pm.next_event().await
                && to == "direct"
            {
                return from;
            }
        }
    })
    .await
    .expect("the direct path becomes preferred");
    assert!(moved.is_none() || moved.as_deref() == Some("relay"));
}

/// Route connections that arrive on `listener` through an acceptor that already holds the
/// session, so they join it as subflows.
fn spawn_server_into(
    listener: TcpListener,
    acceptor: Arc<Acceptor>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        while let Ok((tcp, _)) = listener.accept().await {
            let acceptor = acceptor.clone();
            tokio::spawn(async move {
                let _ = acceptor.accept_tcp(tcp).await;
            });
        }
    })
}

#[tokio::test(flavor = "multi_thread")]
async fn a_relay_path_is_dropped_as_soon_as_a_direct_path_carries_traffic() {
    let (relay_id, relay_addr) = start_relay(open(), RelayConfig::default()).await;
    let a = Identity::generate();
    let b = Identity::generate();
    let token = new_token();
    let acceptor = Arc::new(Acceptor::new(
        b.clone(),
        PeerPolicy::allow(vec![a.node_id()]),
        Config::default(),
    ));
    let reserved = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let direct = reserved.local_addr().unwrap();
    drop(reserved);

    let (tx, mut sessions) = tokio::sync::mpsc::channel(4);
    {
        let (acceptor, tx) = (acceptor.clone(), tx.clone());
        let db = RelayDialer::new(relay_addr.to_string(), relay_id.node_id(), b.clone());
        let a_id = a.node_id();
        tokio::spawn(async move {
            let t = db.connect(a_id, token).await.unwrap();
            if let Ok(Accepted::Session(s)) = acceptor.accept(t).await {
                let _ = tx.send(s).await;
            }
        });
    }
    let da = RelayDialer::new(relay_addr.to_string(), relay_id.node_id(), a.clone());
    let candidates = vec![
        da.candidate("relay", b.node_id(), token),
        Candidate::tcp("direct", direct),
    ];
    // The switching rules would need a 60 s dwell and 100 confirmations, so only the relay
    // rule can drop the relayed path within the test.
    let cfg = ManagerConfig {
        measure_interval: Duration::from_millis(100),
        redial_initial: Duration::from_millis(50),
        redial_max: Duration::from_millis(200),
        policy: letify_link::hysteresis::Policy {
            dwell: Duration::from_secs(60),
            confirmations: 100,
            spacing: Duration::from_secs(60),
            ..Default::default()
        },
        ..ManagerConfig::default()
    };
    let mut pm = PathManager::start(a.clone(), b.node_id(), candidates, Config::default(), cfg)
        .await
        .unwrap();
    let server_session = sessions.recv().await.unwrap();
    let mut stream = pm.session().open_stream().await.unwrap();
    tokio::spawn(async move {
        let chunk = pattern(1 << 16, 3);
        while stream.write_all(&chunk).await.is_ok() {}
    });
    tokio::spawn(async move {
        let mut sb = server_session.accept_stream().await.unwrap();
        let mut buf = vec![0u8; 1 << 16];
        while let Ok(n) = sb.read(&mut buf).await {
            if n == 0 {
                break;
            }
        }
    });
    // With no direct path, the relayed path stays for as long as we look.
    let early = timeout(Duration::from_millis(1200), async {
        loop {
            if let Some(PathEvent::Lost { name, .. }) = pm.next_event().await
                && name == "relay"
            {
                return;
            }
        }
    })
    .await;
    assert!(early.is_err(), "the relay was dropped with no direct path");
    let listener = TcpListener::bind(direct).await.unwrap();
    let _direct_sessions = spawn_server_into(listener, acceptor);
    let started = Instant::now();
    let (mut direct_added, mut reason) = (false, None);
    timeout(Duration::from_secs(10), async {
        loop {
            match pm.next_event().await {
                Some(PathEvent::Added { name }) if name == "direct" => direct_added = true,
                Some(PathEvent::Lost { name, reason: r }) if name == "relay" => {
                    reason = Some(r);
                    return;
                }
                _ => {}
            }
        }
    })
    .await
    .expect("the relayed path is dropped once the direct path carries traffic");
    assert!(direct_added, "the direct path joined before the relay left");
    assert_eq!(reason.as_deref(), Some("moved"));
    // The first direct dial waits out the redial back-off at most, then one measurement tick.
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "{:?}",
        started.elapsed()
    );
}

/// Pair `a` with `b` through the relay and return their ends.
async fn pair(
    relay_id: &Identity,
    addr: std::net::SocketAddr,
    a: &Identity,
    b: &Identity,
) -> (TcpStream, TcpStream) {
    let token = new_token();
    let da = RelayDialer::new(addr.to_string(), relay_id.node_id(), a.clone());
    let db = RelayDialer::new(addr.to_string(), relay_id.node_id(), b.clone());
    let (ra, rb) = tokio::join!(
        da.connect(b.node_id(), token),
        db.connect(a.node_id(), token)
    );
    (ra.unwrap(), rb.unwrap())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_revoked_node_is_disconnected_from_the_relay_at_once() {
    let a = Identity::generate();
    let b = Identity::generate();
    let id = Identity::generate();
    let relay = Relay::bind("127.0.0.1:0", id.clone(), open(), RelayConfig::default())
        .await
        .unwrap();
    let addr = relay.local_addr();
    let handle = relay.handle();
    tokio::spawn(relay.run());
    let (mut ta, mut tb) = pair(&id, addr, &a, &b).await;
    ta.write_all(b"ping").await.unwrap();
    let mut buf = [0u8; 4];
    tb.read_exact(&mut buf).await.unwrap();

    handle.revoke(a.node_id());
    // Both ends of the splice see the close well inside a second.
    let mut rest = Vec::new();
    let closed = timeout(Duration::from_millis(500), tb.read_to_end(&mut rest)).await;
    assert!(closed.is_ok(), "the other end of the splice was not closed");
    let mut sink = [0u8; 8];
    let end = timeout(Duration::from_millis(500), ta.read(&mut sink))
        .await
        .expect("the revoked end was not closed");
    assert!(matches!(end, Ok(0) | Err(_)));

    // New pairings of the revoked node are refused, and allow admits it again.
    let d = RelayDialer::new(addr.to_string(), id.node_id(), a.clone());
    assert!(d.connect(b.node_id(), new_token()).await.is_err());
    handle.allow(a.node_id());
    let (mut ta, mut tb) = pair(&id, addr, &a, &b).await;
    ta.write_all(b"back").await.unwrap();
    tb.read_exact(&mut buf).await.unwrap();
    assert_eq!(&buf, b"back");
}

#[tokio::test(flavor = "multi_thread")]
async fn per_pair_rate_shares_are_fair() {
    // Client a holds two pairs, with b and with c, under one 128 KiB/s budget. Each pair gets
    // 64 KiB/s, so 96 KiB takes about 1.5 s on both at once, not 3 s on one of them.
    let (relay_id, relay_addr) = start_relay(
        open(),
        RelayConfig {
            rate_limit: Some(128 * 1024),
            ..RelayConfig::default()
        },
    )
    .await;
    let (a, b, c) = (
        Identity::generate(),
        Identity::generate(),
        Identity::generate(),
    );
    let (mut tab, tba) = pair(&relay_id, relay_addr, &a, &b).await;
    let (mut tac, tca) = pair(&relay_id, relay_addr, &a, &c).await;
    let size = 96 * 1024;
    let start = Instant::now();
    let w1 = tokio::spawn(async move {
        tab.write_all(&pattern(size, 1)).await.unwrap();
        tab
    });
    let w2 = tokio::spawn(async move {
        tac.write_all(&pattern(size, 2)).await.unwrap();
        tac
    });
    let read = |mut t: TcpStream| async move {
        let mut got = vec![0u8; size];
        timeout(Duration::from_secs(20), t.read_exact(&mut got))
            .await
            .unwrap()
            .unwrap();
        (start.elapsed(), t)
    };
    let ((t1, _k1), (t2, _k2)) = tokio::join!(read(tba), read(tca));
    let _ = (w1.await, w2.await);
    // Both finish near 1.5 s. Neither is much ahead of the other.
    for t in [t1, t2] {
        assert!(t >= Duration::from_millis(1000), "took {t:?}");
        assert!(t < Duration::from_millis(2600), "took {t:?}");
    }
    let gap = t1.abs_diff(t2);
    assert!(gap < Duration::from_millis(700), "gap {gap:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_pair_rate_cap_holds_one_pair_below_the_client_budget() {
    let (relay_id, relay_addr) = start_relay(
        open(),
        RelayConfig {
            rate_limit: Some(1 << 20),
            pair_rate_limit: Some(64 * 1024),
            ..RelayConfig::default()
        },
    )
    .await;
    let (a, b) = (Identity::generate(), Identity::generate());
    let (mut ta, mut tb) = pair(&relay_id, relay_addr, &a, &b).await;
    let start = Instant::now();
    let w = tokio::spawn(async move {
        ta.write_all(&pattern(128 * 1024, 3)).await.unwrap();
        ta
    });
    let mut got = vec![0u8; 128 * 1024];
    timeout(Duration::from_secs(20), tb.read_exact(&mut got))
        .await
        .unwrap()
        .unwrap();
    let _ = w.await;
    assert!(start.elapsed() >= Duration::from_millis(1400));
}

#[tokio::test(flavor = "multi_thread")]
async fn many_pending_pairings_on_a_hub() {
    // The 40 devices dial from one address, as devices behind one NAT would.
    let hub_cfg = RelayConfig {
        max_preauth_per_ip: 64,
        ..RelayConfig::default()
    };
    let (relay_id, relay_addr) = start_relay(open(), hub_cfg).await;
    assert_eq!(RelayConfig::default().max_pending, 64);
    let hub_client = Identity::generate();
    // One device waits on 40 peers at once, more than the old limit of 8.
    let mut waits = Vec::new();
    let mut peers = Vec::new();
    for _ in 0..40 {
        let peer = Identity::generate();
        let token = new_token();
        let d = RelayDialer::new(
            relay_addr.to_string(),
            relay_id.node_id(),
            hub_client.clone(),
        );
        let peer_id = peer.node_id();
        waits.push(tokio::spawn(async move { d.connect(peer_id, token).await }));
        peers.push((peer, token));
    }
    tokio::time::sleep(Duration::from_millis(300)).await;
    for (peer, token) in peers {
        let d = RelayDialer::new(relay_addr.to_string(), relay_id.node_id(), peer);
        let hub = hub_client.node_id();
        tokio::spawn(async move { d.connect(hub, token).await });
    }
    for w in waits {
        let r = timeout(Duration::from_secs(20), w).await.unwrap().unwrap();
        assert!(r.is_ok(), "{:?}", r.err());
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn revoking_an_alternate_encoding_of_a_node_refuses_the_node_at_the_relay() {
    // A node whose key has the sign bit set, so the other encoding differs from its id.
    let a = loop {
        let i = Identity::generate();
        if i.node_id().as_bytes()[31] & 0x80 != 0 {
            break i;
        }
    };
    let b = Identity::generate();
    let id = Identity::generate();
    let relay = Relay::bind("127.0.0.1:0", id.clone(), open(), RelayConfig::default())
        .await
        .unwrap();
    let addr = relay.local_addr();
    let handle = relay.handle();
    tokio::spawn(relay.run());
    let mut other = *a.node_id().as_bytes();
    other[31] ^= 0x80;
    handle.revoke(letify_link::NodeId::from_bytes(other));
    let d = RelayDialer::new(addr.to_string(), id.node_id(), a.clone());
    assert!(d.connect(b.node_id(), new_token()).await.is_err());
}

/// Hold `n` connections to the relay that send nothing.
async fn silent(addr: std::net::SocketAddr, n: usize) -> Vec<TcpStream> {
    let mut held = Vec::new();
    for _ in 0..n {
        held.push(TcpStream::connect(addr).await.unwrap());
    }
    // Let the relay accept them.
    tokio::time::sleep(Duration::from_millis(200)).await;
    held
}

async fn can_reach(relay: &Identity, addr: std::net::SocketAddr) -> bool {
    let d = RelayDialer::new(addr.to_string(), relay.node_id(), Identity::generate())
        .pair_wait(Duration::from_millis(400));
    // A refusal at accept shows as an error before pairing; a pair timeout shows as a refusal
    // by the relay after the handshake, which is also a connection that got through.
    match d.connect(Identity::generate().node_id(), new_token()).await {
        Ok(_) => true,
        Err(e) => {
            e.kind() == std::io::ErrorKind::PermissionDenied
                || e.kind() == std::io::ErrorKind::TimedOut
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn the_unauthenticated_budget_has_defaults_below_the_connection_limit() {
    let cfg = RelayConfig::default();
    assert_eq!(cfg.max_preauth, 64);
    assert_eq!(cfg.max_preauth_per_ip, 4);
    assert_eq!(cfg.max_waiting, 4096);
    assert_eq!(cfg.handshake_timeout, Duration::from_secs(4));
    assert!(cfg.max_preauth < cfg.max_connections);
}

#[tokio::test(flavor = "multi_thread")]
async fn one_source_address_cannot_hold_more_unauthenticated_connections_than_its_cap() {
    let cfg = RelayConfig {
        max_preauth_per_ip: 3,
        handshake_timeout: Duration::from_secs(2),
        ..RelayConfig::default()
    };
    let (relay_id, addr) = start_relay(open(), cfg).await;
    let held = silent(addr, 3).await;
    assert!(
        !can_reach(&relay_id, addr).await,
        "a fourth connection from the same address got through"
    );
    // The silent ones are closed at the pre-auth timeout, and the address is served again.
    tokio::time::sleep(Duration::from_millis(2300)).await;
    assert!(can_reach(&relay_id, addr).await);
    drop(held);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_unauthenticated_budget_is_global() {
    let cfg = RelayConfig {
        max_preauth: 2,
        max_preauth_per_ip: 100,
        handshake_timeout: Duration::from_secs(2),
        ..RelayConfig::default()
    };
    let (relay_id, addr) = start_relay(open(), cfg).await;
    let held = silent(addr, 2).await;
    assert!(!can_reach(&relay_id, addr).await);
    drop(held);
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(
        can_reach(&relay_id, addr).await,
        "the budget was not released"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn the_waiting_pairs_are_capped_as_a_whole() {
    let cfg = RelayConfig {
        max_waiting: 2,
        ..RelayConfig::default()
    };
    let (relay_id, addr) = start_relay(open(), cfg).await;
    let mut waits = Vec::new();
    for _ in 0..2 {
        let d = RelayDialer::new(addr.to_string(), relay_id.node_id(), Identity::generate());
        let peer = Identity::generate().node_id();
        waits.push(tokio::spawn(
            async move { d.connect(peer, new_token()).await },
        ));
    }
    tokio::time::sleep(Duration::from_millis(300)).await;
    let d = RelayDialer::new(addr.to_string(), relay_id.node_id(), Identity::generate());
    let err = d
        .connect(Identity::generate().node_id(), new_token())
        .await
        .expect_err("a third waiting pair is refused");
    assert_eq!(err.kind(), std::io::ErrorKind::PermissionDenied, "{err}");
}
