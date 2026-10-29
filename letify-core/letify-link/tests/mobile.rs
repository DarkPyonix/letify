//! Pins SPEC.md letify-link, Mobile and power states: the mobile profile, the keepalive and
//! probe setters, and the resumption that `resume_now` and `invalidate_paths` ask for.

mod common;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::*;
use letify_link::{
    Acceptor, Candidate, Config, Identity, ManagerConfig, PathManager, PeerPolicy, PowerState,
    ProbePolicy, Session,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::time::{Instant, sleep, timeout};

/// Two sessions over one in-memory subflow with a configuration for each side.
async fn pair_with(a_cfg: Config, b_cfg: Config) -> (Session, Session) {
    let a = Identity::generate();
    let b = Identity::generate();
    let (ta, tb) = duplex_pair();
    let responder = tokio::spawn({
        let (b, a_id) = (b.clone(), a.node_id());
        async move { Session::accept(tb, &b, PeerPolicy::allow(vec![a_id]), b_cfg).await }
    });
    let initiator = Session::connect(ta, &a, b.node_id(), a_cfg).await.unwrap();
    (initiator, responder.await.unwrap().unwrap())
}

fn keepalive(ms: u64) -> Config {
    Config {
        keepalive_interval: Duration::from_millis(ms),
        ..Config::default()
    }
}

/// Bytes `session` has read from its first subflow so far.
fn read_by(session: &Session) -> u64 {
    session.subflow_stats()[0].received_bytes
}

#[test]
fn the_mobile_profile_holds_less_and_pings_less_often() {
    let (mobile, normal) = (Config::mobile(), Config::default());
    assert_eq!(mobile.max_unacked_bytes, 4 << 20);
    assert_eq!(mobile.max_reorder_bytes, 8 << 20);
    assert_eq!(mobile.stream_window, 2 << 20);
    assert_eq!(mobile.subflow_window, 1 << 20);
    assert!(mobile.max_reorder_bytes < normal.max_reorder_bytes);
    assert_eq!(mobile.keepalive_interval, Duration::from_secs(30));
    assert!(mobile.keepalive_interval > normal.keepalive_interval);
    // The states: 15 s in the foreground and 120 s in the background by default.
    assert_eq!(normal.keepalive_interval, Duration::from_secs(15));
    assert_eq!(
        normal.background_keepalive_interval,
        Duration::from_secs(120)
    );
    assert_eq!(
        mobile.background_keepalive_interval,
        Duration::from_secs(120)
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_mobile_session_carries_a_transfer() {
    let (a, b) = pair_with(Config::mobile(), Config::mobile()).await;
    let data = pattern(6 << 20, 3);
    let mut tx = a.open_stream().await.unwrap();
    let writer = tokio::spawn({
        let data = data.clone();
        async move {
            tx.write_all(&data).await.unwrap();
            tx.shutdown().await.unwrap();
        }
    });
    let mut rx = b.accept_stream().await.unwrap();
    let mut got = Vec::new();
    timeout(Duration::from_secs(30), rx.read_to_end(&mut got))
        .await
        .unwrap()
        .unwrap();
    writer.await.unwrap();
    assert!(got == data);
    assert!(a.unacked_bytes() <= 4 << 20);
}

#[tokio::test(flavor = "multi_thread")]
async fn set_keepalive_interval_changes_how_often_an_idle_subflow_pings() {
    // Only a pings, every 100 ms; b sends none and counts what it reads.
    let (a, b) = pair_with(keepalive(100), keepalive(0)).await;
    sleep(Duration::from_millis(150)).await;
    let before = read_by(&b);
    sleep(Duration::from_millis(600)).await;
    let steady = read_by(&b) - before;
    assert!(steady >= 4 * 37, "{steady} bytes at 100 ms");

    a.set_keepalive_interval(Duration::ZERO);
    assert_eq!(a.keepalive_interval(), Duration::ZERO);
    sleep(Duration::from_millis(200)).await;
    let before = read_by(&b);
    sleep(Duration::from_millis(600)).await;
    assert_eq!(read_by(&b), before, "a pinged with keepalives off");
    assert_eq!(b.subflow_count(), 1, "b dropped the quiet subflow");

    a.set_keepalive_interval(Duration::from_millis(50));
    sleep(Duration::from_millis(200)).await;
    let before = read_by(&b);
    sleep(Duration::from_millis(600)).await;
    let fast = read_by(&b) - before;
    assert!(
        fast >= 2 * steady / 2,
        "{fast} bytes at 50 ms, {steady} at 100 ms"
    );
    assert!(fast > steady, "{fast} bytes at 50 ms, {steady} at 100 ms");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_peer_is_told_a_longer_interval_so_it_keeps_the_subflow() {
    // b sends no keepalives, so it declares a subflow dead after 3 x a's interval, 300 ms.
    let (a, b) = pair_with(keepalive(100), keepalive(0)).await;
    a.set_keepalive_interval(Duration::from_millis(600));
    sleep(Duration::from_millis(1500)).await;
    assert_eq!((a.subflow_count(), b.subflow_count()), (1, 1));
}

#[tokio::test(flavor = "multi_thread")]
async fn power_states_set_the_keepalive_interval() {
    let cfg = Config {
        keepalive_interval: Duration::from_millis(100),
        background_keepalive_interval: Duration::from_millis(900),
        ..Config::default()
    };
    let (a, _b) = pair_with(cfg.clone(), cfg).await;
    assert_eq!(a.keepalive_interval(), Duration::from_millis(100));
    a.set_power_state(PowerState::Background);
    assert_eq!(a.keepalive_interval(), Duration::from_millis(900));
    a.set_power_state(PowerState::PowerSave);
    assert_eq!(a.keepalive_interval(), Duration::ZERO);
    a.set_power_state(PowerState::Foreground);
    assert_eq!(a.keepalive_interval(), Duration::from_millis(100));
}

/// A server that echoes every stream of every session, and the listener's address.
async fn echo_server(config: Config) -> (Identity, std::net::SocketAddr) {
    let server = Identity::generate();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let acceptor = Acceptor::new(server.clone(), PeerPolicy::from_fn(|_| true), config);
    let mut sessions = spawn_server(listener, acceptor);
    tokio::spawn(async move {
        while let Some(session) = sessions.recv().await {
            tokio::spawn(async move {
                while let Ok(mut s) = session.accept_stream().await {
                    tokio::spawn(async move {
                        let mut buf = [0u8; 1];
                        while s.read_exact(&mut buf).await.is_ok() {
                            if s.write_all(&buf).await.is_err() {
                                break;
                            }
                        }
                    });
                }
            });
        }
    });
    (server, addr)
}

/// A client path manager whose subflows can be black-holed: the gates of every transport it
/// dialed, and a switch that makes the next dials fail.
struct Client {
    pm: PathManager,
    gates: Arc<Mutex<Vec<Gate>>>,
    down: Arc<AtomicBool>,
}

impl Client {
    async fn start(server: &Identity, addr: std::net::SocketAddr, cfg: ManagerConfig) -> Client {
        let gates: Arc<Mutex<Vec<Gate>>> = Arc::new(Mutex::new(Vec::new()));
        let down = Arc::new(AtomicBool::new(false));
        let dial = {
            let (gates, down) = (gates.clone(), down.clone());
            move || {
                let (gates, down) = (gates.clone(), down.clone());
                async move {
                    if down.load(Ordering::SeqCst) {
                        return Err(std::io::Error::other("the network is down"));
                    }
                    let tcp = TcpStream::connect(addr).await?;
                    tcp.set_nodelay(true)?;
                    let (gated, gate) = Gated::new(tcp);
                    gates.lock().unwrap().push(gate);
                    Ok(gated)
                }
            }
        };
        let pm = PathManager::start(
            Identity::generate(),
            server.node_id(),
            vec![Candidate::new("tcp", dial)],
            Config::default(),
            cfg,
        )
        .await
        .unwrap();
        Client { pm, gates, down }
    }

    /// Black-hole every subflow dialed so far, as a suspended phone's connections are.
    fn black_hole(&self) {
        self.gates.lock().unwrap().iter().for_each(Gate::close);
    }
}

async fn echo_once(s: &mut letify_link::Stream) -> Duration {
    let t = Instant::now();
    s.write_all(&[7]).await.unwrap();
    let mut got = [0u8; 1];
    timeout(Duration::from_secs(10), s.read_exact(&mut got))
        .await
        .expect("the echo arrives")
        .unwrap();
    t.elapsed()
}

#[tokio::test(flavor = "multi_thread")]
async fn resume_now_resumes_within_one_second_at_p95() {
    let (server, addr) = echo_server(Config::default()).await;
    let client = Client::start(&server, addr, ManagerConfig::default()).await;
    let mut s = client.pm.session().open_stream().await.unwrap();
    echo_once(&mut s).await;
    let mut times = Vec::new();
    for _ in 0..20 {
        let old: Vec<u64> = client
            .pm
            .session()
            .subflow_stats()
            .iter()
            .map(|x| x.id)
            .collect();
        client.black_hole();
        let t = Instant::now();
        client.pm.session().resume_now();
        let took = echo_once(&mut s).await;
        assert!(t.elapsed() >= took);
        let now: Vec<u64> = client
            .pm
            .session()
            .subflow_stats()
            .iter()
            .map(|x| x.id)
            .collect();
        assert!(
            now.iter().all(|id| !old.contains(id)),
            "{old:?} still alive in {now:?}"
        );
        times.push(took);
    }
    times.sort();
    let p95 = times[(times.len() as f64 * 0.95).ceil() as usize - 1];
    eprintln!(
        "resume_now: median {:?}, p95 {p95:?}, worst {:?}",
        times[10], times[19]
    );
    assert!(p95 < Duration::from_secs(1), "p95 {p95:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_suspended_session_without_resume_now_stays_on_its_dead_subflow() {
    let (server, addr) = echo_server(Config::default()).await;
    let client = Client::start(&server, addr, ManagerConfig::default()).await;
    let mut s = client.pm.session().open_stream().await.unwrap();
    echo_once(&mut s).await;
    client.black_hole();
    // Nothing tells the session that the subflow is dead, and the 15 s keepalive has not fired.
    s.write_all(&[7]).await.unwrap();
    let mut got = [0u8; 1];
    let echoed = timeout(Duration::from_millis(1500), s.read_exact(&mut got)).await;
    assert!(echoed.is_err());
    // This is what resume_now is for.
    client.pm.session().resume_now();
    timeout(Duration::from_secs(2), s.read_exact(&mut got))
        .await
        .expect("the echo arrives after resume_now")
        .unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn resume_now_skips_the_redial_backoff() {
    let (server, addr) = echo_server(Config::default()).await;
    let cfg = ManagerConfig {
        redial_initial: Duration::from_millis(200),
        redial_max: Duration::from_secs(30),
        ..ManagerConfig::default()
    };
    let client = Client::start(&server, addr, cfg).await;
    let mut s = client.pm.session().open_stream().await.unwrap();
    echo_once(&mut s).await;
    // The network goes away: dials fail and the wait between them grows to 1.6 s and beyond.
    client.down.store(true, Ordering::SeqCst);
    client.black_hole();
    client.pm.session().resume_now();
    sleep(Duration::from_millis(2500)).await;
    // The network is back. Without the hint the candidate sleeps up to 3.2 s more.
    client.down.store(false, Ordering::SeqCst);
    let t = Instant::now();
    client.pm.session().resume_now();
    echo_once(&mut s).await;
    assert!(t.elapsed() < Duration::from_secs(1), "{:?}", t.elapsed());
}

#[tokio::test(flavor = "multi_thread")]
async fn invalidate_paths_replaces_every_subflow_at_once() {
    let (server, addr) = echo_server(Config::default()).await;
    let client = Client::start(&server, addr, ManagerConfig::default()).await;
    let mut s = client.pm.session().open_stream().await.unwrap();
    echo_once(&mut s).await;
    let old: Vec<u64> = client
        .pm
        .session()
        .subflow_stats()
        .iter()
        .map(|x| x.id)
        .collect();
    client.black_hole();
    let t = Instant::now();
    client.pm.invalidate_paths();
    echo_once(&mut s).await;
    assert!(t.elapsed() < Duration::from_secs(1), "{:?}", t.elapsed());
    let now: Vec<u64> = client
        .pm
        .session()
        .subflow_stats()
        .iter()
        .map(|x| x.id)
        .collect();
    assert!(now.iter().all(|id| !old.contains(id)));
}

#[tokio::test(flavor = "multi_thread")]
async fn set_probe_policy_turns_idle_probes_off_and_on() {
    let (server, addr) = echo_server(Config::default()).await;
    let probing = ProbePolicy {
        idle: Duration::from_millis(100),
        window: Duration::from_millis(500),
        min_bytes: 8 << 10,
    };
    let cfg = ManagerConfig {
        measure_interval: Duration::from_millis(50),
        probe_idle: probing.idle,
        probe_window: probing.window,
        probe_min_bytes: probing.min_bytes,
        ..ManagerConfig::default()
    };
    let client = Client::start(&server, addr, cfg).await;
    sleep(Duration::from_millis(1200)).await;
    assert!(
        client.pm.probe_bytes_spent() > 0,
        "the idle subflow was never probed"
    );

    client.pm.set_probe_policy(ProbePolicy::off());
    sleep(Duration::from_millis(300)).await;
    let off = client.pm.probe_bytes_spent();
    sleep(Duration::from_millis(1200)).await;
    assert_eq!(
        client.pm.probe_bytes_spent(),
        off,
        "probes continued while off"
    );

    client.pm.set_probe_policy(probing);
    sleep(Duration::from_millis(1500)).await;
    assert!(
        client.pm.probe_bytes_spent() > off,
        "probes did not come back"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn the_power_states_of_the_path_manager_set_keepalives_and_probes() {
    let (server, addr) = echo_server(Config::default()).await;
    let cfg = ManagerConfig {
        measure_interval: Duration::from_millis(50),
        probe_idle: Duration::from_millis(100),
        probe_window: Duration::from_millis(500),
        probe_min_bytes: 8 << 10,
        ..ManagerConfig::default()
    };
    let client = Client::start(&server, addr, cfg).await;
    client.pm.set_power_state(PowerState::PowerSave);
    assert_eq!(client.pm.session().keepalive_interval(), Duration::ZERO);
    sleep(Duration::from_millis(300)).await;
    let saved = client.pm.probe_bytes_spent();
    sleep(Duration::from_millis(1200)).await;
    assert_eq!(client.pm.probe_bytes_spent(), saved, "power save probed");

    client.pm.set_power_state(PowerState::Foreground);
    assert_eq!(
        client.pm.session().keepalive_interval(),
        Config::default().keepalive_interval
    );
    sleep(Duration::from_millis(1500)).await;
    assert!(
        client.pm.probe_bytes_spent() > saved,
        "the foreground did not probe"
    );
}
