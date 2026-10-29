//! Pins SPEC.md letify-link, Session item 6 (UDP paths) and Paths and the path manager item 3
//! (Kinds) over real UDP and TCP on loopback. A lossy UDP proxy stands in for a bad path.

mod common;

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use common::*;
use letify_link::hysteresis::Policy;
use letify_link::udp::{UdpConfig, UdpListener, UdpPath};
use letify_link::{
    Acceptor, Candidate, Config, Identity, ManagerConfig, PathEvent, PathManager, PeerPolicy,
    Session,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio::time::{Instant, timeout};

const CHUNK: usize = 1 << 20;
const MIB: f64 = 1048576.0;

/// One chunk of test data: a fixed pattern with the chunk index stamped in front.
fn chunk(base: &[u8], i: u64) -> Vec<u8> {
    let mut c = base.to_vec();
    c[..8].copy_from_slice(&i.to_le_bytes());
    c
}

/// Send `mib` MiB on a new stream of `from`, pausing `pace` after each MiB, and check every
/// byte on the other side. Returns the time the last byte arrived.
async fn transfer(from: &Session, to: &Session, mib: u64, pace: Duration) -> Instant {
    let base = pattern(CHUNK, 7);
    let mut tx = from.open_stream().await.unwrap();
    let writer = tokio::spawn({
        let base = base.clone();
        async move {
            for i in 0..mib {
                tx.write_all(&chunk(&base, i)).await.unwrap();
                if !pace.is_zero() {
                    tokio::time::sleep(pace).await;
                }
            }
            tx.shutdown().await.unwrap();
        }
    });
    let mut rx = to.accept_stream().await.unwrap();
    let mut got = vec![0u8; CHUNK];
    for i in 0..mib {
        rx.read_exact(&mut got).await.unwrap();
        assert!(got == chunk(&base, i), "chunk {i} differs");
    }
    assert_eq!(rx.read(&mut got).await.unwrap(), 0, "the stream ends");
    writer.await.unwrap();
    Instant::now()
}

fn quick_udp() -> UdpConfig {
    UdpConfig {
        idle_timeout: Duration::from_secs(2),
        keep_alive: Duration::from_millis(500),
        ..UdpConfig::default()
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_session_runs_over_a_udp_path() {
    let (a, b, _proxy) = udp_session(Config::default(), Duration::ZERO, 0.0, None).await;
    timeout(
        Duration::from_secs(60),
        transfer(&a, &b, 16, Duration::ZERO),
    )
    .await
    .unwrap();
    timeout(
        Duration::from_secs(60),
        transfer(&b, &a, 16, Duration::ZERO),
    )
    .await
    .unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn udp_path_recovers_from_loss() {
    let (a, b, _proxy) = udp_session(Config::default(), Duration::from_millis(5), 0.05, None).await;
    timeout(
        Duration::from_secs(120),
        transfer(&a, &b, 16, Duration::ZERO),
    )
    .await
    .unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_udp_listener_refuses_a_source_it_was_not_given() {
    let server = udp_socket();
    let server_addr = server.local_addr().unwrap();
    let wrong: SocketAddr = "127.0.0.1:9".parse().unwrap();
    let mut listener = UdpListener::new(server, Some(wrong), &UdpConfig::default()).unwrap();
    let accepted = tokio::spawn(async move { listener.accept().await });
    let cfg = UdpConfig {
        connect_timeout: Duration::from_millis(800),
        ..UdpConfig::default()
    };
    let connected = UdpPath::connect_with(udp_socket(), server_addr, &cfg).await;
    assert!(connected.is_err(), "a source that was not given is refused");
    assert!(
        timeout(Duration::from_millis(300), accepted).await.is_err(),
        "the listener keeps waiting for the given peer"
    );
}

/// Both kinds of candidate toward one server. The TCP one runs through a throttle.
struct Duo {
    client: Identity,
    server_id: letify_link::NodeId,
    candidates: Vec<Candidate>,
    sessions: mpsc::Receiver<Session>,
    udp_proxy: UdpProxy,
    _throttle: Throttle,
}

async fn duo(tcp_rate: f64, udp_rate: f64, udp_dial_delay: Duration) -> Duo {
    let client = Identity::generate();
    let server = Identity::generate();
    let tcp = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let tcp_addr = tcp.local_addr().unwrap();
    let udp_sock = udp_socket();
    let udp_addr = udp_sock.local_addr().unwrap();
    let listener = UdpListener::new(udp_sock, None, &UdpConfig::default()).unwrap();
    let server_id = server.node_id();
    let acceptor = Acceptor::new(
        server,
        PeerPolicy::allow(vec![client.node_id()]),
        Config::default(),
    );
    let sessions = serve(acceptor, Some(tcp), Some(listener));
    let throttle = Throttle::to(tcp_addr, tcp_rate).await;
    let udp_proxy = UdpProxy::limited(udp_addr, Duration::ZERO, 0.0, Some(udp_rate)).await;
    let (tcp_front, udp_front) = (throttle.addr, udp_proxy.addr);
    let candidates = vec![
        Candidate::tcp("tcp", tcp_front),
        Candidate::udp("udp", move || async move {
            tokio::time::sleep(udp_dial_delay).await;
            UdpPath::connect_with(udp_socket(), udp_front, &quick_udp()).await
        }),
    ];
    Duo {
        client,
        server_id,
        candidates,
        sessions,
        udp_proxy,
        _throttle: throttle,
    }
}

fn mover() -> ManagerConfig {
    ManagerConfig {
        measure_interval: Duration::from_millis(100),
        redial_initial: Duration::from_millis(50),
        redial_max: Duration::from_millis(400),
        dial_timeout: Duration::from_secs(3),
        policy: Policy {
            dwell: Duration::from_millis(100),
            confirmations: 2,
            spacing: Duration::from_millis(100),
            ..Policy::default()
        },
        ..ManagerConfig::default()
    }
}

/// Events with the time they were read, collected by a task that owns the manager.
fn collect(mut pm: PathManager) -> mpsc::UnboundedReceiver<(Instant, PathEvent)> {
    let (tx, rx) = mpsc::unbounded_channel();
    tokio::spawn(async move {
        while let Some(ev) = pm.next_event().await {
            if tx.send((Instant::now(), ev)).is_err() {
                break;
            }
        }
    });
    rx
}

/// Read events until `want` matches one, and return when it was read.
async fn until(
    events: &mut mpsc::UnboundedReceiver<(Instant, PathEvent)>,
    seen: &mut Vec<(Instant, PathEvent)>,
    want: impl Fn(&PathEvent) -> bool,
) -> Instant {
    loop {
        let (at, ev) = timeout(Duration::from_secs(30), events.recv())
            .await
            .expect("the awaited event arrives")
            .expect("the manager runs");
        let hit = want(&ev);
        seen.push((at, ev));
        if hit {
            return at;
        }
    }
}

fn is_moved(ev: &PathEvent, name: &str) -> bool {
    matches!(ev, PathEvent::Lost { name: n, reason } if n == name && reason == "moved")
}

#[tokio::test(flavor = "multi_thread")]
async fn a_session_moves_from_tcp_to_udp_without_losing_bytes() {
    let mut duo = duo(2.0 * MIB, 12.0 * MIB, Duration::from_millis(300)).await;
    let pm = PathManager::start(
        duo.client.clone(),
        duo.server_id,
        duo.candidates.clone(),
        Config::default(),
        mover(),
    )
    .await
    .unwrap();
    let a = Arc::new(pm.session().downgrade().upgrade().unwrap());
    let mut events = collect(pm);
    let b = Arc::new(
        timeout(Duration::from_secs(10), duo.sessions.recv())
            .await
            .unwrap()
            .unwrap(),
    );
    // 64 MiB over a 2 MiB/s TCP path and a 12 MiB/s UDP path takes about 5 s, and the move
    // takes about 1 s, so it happens while bytes are still flowing.
    let moved_at = tokio::spawn(async move {
        let mut seen = Vec::new();
        let at = until(&mut events, &mut seen, |ev| is_moved(ev, "tcp")).await;
        (at, seen)
    });
    let done = transfer(&a, &b, 64, Duration::ZERO).await;
    let (moved_at, seen) = moved_at.await.unwrap();
    assert!(
        moved_at < done,
        "the old subflow closed during the transfer"
    );
    let preferred = seen.iter().position(|(_, ev)| {
        matches!(ev, PathEvent::Preferred { from: Some(f), to } if f == "tcp" && to == "udp")
    });
    let lost = seen.iter().position(|(_, ev)| is_moved(ev, "tcp"));
    assert!(preferred.is_some(), "Preferred names the move: {seen:?}");
    assert!(preferred < lost, "Preferred comes before Lost");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_session_moves_from_udp_to_tcp_when_udp_dies() {
    let mut duo = duo(4.0 * MIB, 16.0 * MIB, Duration::from_millis(100)).await;
    let pm = PathManager::start(
        duo.client.clone(),
        duo.server_id,
        duo.candidates.clone(),
        Config::default(),
        mover(),
    )
    .await
    .unwrap();
    let a = Arc::new(pm.session().downgrade().upgrade().unwrap());
    let mut events = collect(pm);
    let b = Arc::new(
        timeout(Duration::from_secs(10), duo.sessions.recv())
            .await
            .unwrap()
            .unwrap(),
    );
    // Wait for the move to UDP, so TCP is on standby when UDP dies. A little paced traffic
    // gives the manager rates to compare.
    let warm = tokio::spawn({
        let (a, b) = (a.clone(), b.clone());
        async move { transfer(&a, &b, 24, Duration::ZERO).await }
    });
    let mut seen = Vec::new();
    until(&mut events, &mut seen, |ev| is_moved(ev, "tcp")).await;
    warm.await.unwrap();

    let knobs = duo.udp_proxy.knobs.clone();
    let cut = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(1200)).await;
        knobs.set_loss(1.0);
    });
    timeout(
        Duration::from_secs(90),
        transfer(&a, &b, 32, Duration::ZERO),
    )
    .await
    .expect("the transfer completes over TCP after UDP died");
    cut.await.unwrap();
    // Measured events keep arriving, so read for a fixed time instead of until silence.
    let end = Instant::now() + Duration::from_millis(500);
    while let Ok(Some(item)) = tokio::time::timeout_at(end, events.recv()).await {
        seen.push(item);
    }
    let died = seen.iter().any(|(_, ev)| {
        matches!(ev, PathEvent::Lost { name, reason } if name == "udp" && reason == "the subflow died")
    });
    let tcp_back = seen
        .iter()
        .rev()
        .any(|(_, ev)| matches!(ev, PathEvent::Added { name } if name == "tcp"));
    assert!(died, "UDP was reported lost: {seen:?}");
    assert!(tcp_back, "TCP was dialed again and added: {seen:?}");
}
