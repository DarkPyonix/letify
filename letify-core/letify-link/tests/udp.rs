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
        if i % 8 == 7 {
            eprintln!("transfer of {mib} MiB: {} MiB received", i + 1);
        }
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
            if !matches!(ev, PathEvent::Measured { .. }) {
                eprintln!("path event: {ev:?}");
            }
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
    // Measured events keep arriving, so a per-event timeout never fires. The deadline is for
    // the whole wait, and a miss shows what was seen.
    let end = Instant::now() + Duration::from_secs(60);
    loop {
        let (at, ev) = tokio::time::timeout_at(end, events.recv())
            .await
            .unwrap_or_else(|_| panic!("the awaited event arrives; saw {seen:?}"))
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
    // Windows debug builds move far fewer UDP bytes a second through the shaping proxy, so
    // the TCP path is slowed to keep the same ratio and the transfer is shortened to match.
    let (tcp_rate, mib) = if cfg!(windows) {
        (0.5 * MIB, 16)
    } else {
        (2.0 * MIB, 64)
    };
    let mut duo = duo(tcp_rate, 12.0 * MIB, Duration::from_millis(300)).await;
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
    let done = timeout(
        Duration::from_secs(120),
        transfer(&a, &b, mib, Duration::ZERO),
    )
    .await
    .expect("the transfer completes");
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
    // Windows debug builds move far fewer UDP bytes a second, see the move to UDP above.
    let (tcp_rate, warm_mib, after_mib) = if cfg!(windows) {
        (0.5 * MIB, 16, 24)
    } else {
        (4.0 * MIB, 24, 32)
    };
    let mut duo = duo(tcp_rate, 16.0 * MIB, Duration::from_millis(100)).await;
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
        async move { transfer(&a, &b, warm_mib, Duration::ZERO).await }
    });
    let mut seen = Vec::new();
    until(&mut events, &mut seen, |ev| is_moved(ev, "tcp")).await;
    timeout(Duration::from_secs(120), warm)
        .await
        .expect("the warm transfer completes")
        .unwrap();

    let knobs = duo.udp_proxy.knobs.clone();
    let cut = tokio::spawn(async move {
        // The transfer must still be running when UDP is cut. The UDP path carried 16 to
        // 28 MB/s on windows-latest, so 16 MiB took about 1 s and ended before a cut at
        // 1.2 s. The cut comes at 0.5 s and the Windows transfer is 24 MiB.
        tokio::time::sleep(Duration::from_millis(500)).await;
        knobs.set_loss(1.0);
    });
    timeout(
        Duration::from_secs(90),
        transfer(&a, &b, after_mib, Duration::ZERO),
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

/// Pins SPEC.md Paths and the path manager, Kinds compete by their sum.
/// One UDP candidate and several TCP candidates toward one server, every path delayed. The
/// UDP path is faster than any single TCP path and slower than all of them together.
async fn swarm(tcp_paths: usize, tcp_rate: f64, udp_rate: f64) -> (Duo, Vec<Throttle>) {
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
    let udp_proxy =
        UdpProxy::limited(udp_addr, Duration::from_millis(40), 0.0, Some(udp_rate)).await;
    let udp_front = udp_proxy.addr;
    let mut candidates = vec![Candidate::udp("udp", move || async move {
        UdpPath::connect_with(udp_socket(), udp_front, &quick_udp()).await
    })];
    let mut throttles = Vec::new();
    for i in 0..tcp_paths {
        let t = Throttle::delayed(tcp_addr, tcp_rate, Duration::from_millis(60)).await;
        candidates.push(Candidate::tcp(format!("tcp{i}"), t.addr));
        throttles.push(t);
    }
    let duo = Duo {
        client,
        server_id,
        candidates,
        sessions,
        udp_proxy,
        _throttle: Throttle::to(tcp_addr, 1e9).await,
    };
    (duo, throttles)
}

#[tokio::test(flavor = "multi_thread")]
async fn tcp_subflows_that_together_beat_the_udp_path_are_not_closed_by_a_move_to_it() {
    let (mut duo, _throttles) = swarm(4, 3.0 * MIB, 5.0 * MIB).await;
    let pm = PathManager::start(
        duo.client.clone(),
        duo.server_id,
        duo.candidates.clone(),
        Config::default(),
        ManagerConfig {
            probe_idle: Duration::from_millis(300),
            probe_window: Duration::from_millis(500),
            policy: Policy {
                // Once the burst ends, the rate of each TCP subflow decays within a second
                // and is replaced by its probe, and the probes go round the four subflows one
                // per probe_idle, so for about 1.2 s the sum of the TCP kind is made of probes
                // and decayed rates, and can fall below the UDP probe. A confirmation spacing
                // shorter than that cycle (100 ms in `mover`) lets this transient count as two
                // confirmations, which happened in 2 of 6 runs under CPU load and once on
                // ubuntu-latest. The default spacing is 10 s, far above the cycle, so this
                // test spaces the confirmations by 2 s, and the 8 s idle below still holds
                // three of them, enough to catch a kind comparison made on single probes.
                spacing: Duration::from_secs(2),
                ..mover().policy
            },
            ..mover()
        },
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
    // A bulk burst, then idle, then another transfer, as a probe, a build and a call do.
    transfer(&a, &b, 24, Duration::ZERO).await;
    tokio::time::sleep(Duration::from_secs(8)).await;
    transfer(&a, &b, 24, Duration::ZERO).await;
    let mut seen = Vec::new();
    let end = Instant::now() + Duration::from_millis(200);
    while let Ok(Some(item)) = tokio::time::timeout_at(end, events.recv()).await {
        seen.push(item);
    }
    // The session starts on the UDP path, which answers first, so `Preferred` names it once
    // at the start. A move to the TCP kind may follow, because four TCP subflows at 3 MiB/s
    // beat the UDP path at 5 MiB/s, and it closes the UDP subflow as "moved" (SPEC.md Paths
    // and the path manager, Kinds). What must never happen is a TCP subflow closed, or a move
    // from the TCP kind to the UDP path.
    assert!(
        !seen.iter().any(|(_, ev)| matches!(
            ev,
            PathEvent::Lost { name, .. } if name.starts_with("tcp")
        )),
        "no TCP subflow was lost: {seen:?}"
    );
    assert!(
        !seen.iter().any(|(_, ev)| matches!(
            ev,
            PathEvent::Preferred { from: Some(from), to } if from.starts_with("tcp") && to == "udp"
        )),
        "the session did not move from the TCP kind to UDP: {seen:?}"
    );
    let udp_lost = seen.iter().any(|(_, ev)| is_moved(ev, "udp"));
    assert_eq!(
        a.subflow_count(),
        if udp_lost { 4 } else { 5 },
        "the four TCP subflows are open, and the UDP subflow too unless the TCP kind won: {seen:?}"
    );
}
