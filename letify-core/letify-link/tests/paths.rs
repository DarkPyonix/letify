//! Pins SPEC.md letify-link, Paths and the path manager items 1, 2, 3 (active probes), 4, 6
//! (below the floor) and 8 (events) over real TCP on loopback.

mod common;

use std::time::Duration;

use common::*;
use letify_link::{
    Acceptor, Candidate, Config, Identity, ManagerConfig, PathEvent, PathManager, PeerPolicy,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::time::timeout;

struct Rig {
    client: Identity,
    server: Identity,
    target: std::net::SocketAddr,
    sessions: tokio::sync::mpsc::Receiver<letify_link::Session>,
}

async fn rig(config: Config) -> Rig {
    let client = Identity::generate();
    let server = Identity::generate();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let target = listener.local_addr().unwrap();
    let acceptor = Acceptor::new(
        server.clone(),
        PeerPolicy::allow(vec![client.node_id()]),
        config,
    );
    Rig {
        client,
        server,
        target,
        sessions: spawn_server(listener, acceptor),
    }
}

fn quick() -> ManagerConfig {
    ManagerConfig {
        measure_interval: Duration::from_millis(100),
        redial_initial: Duration::from_millis(20),
        redial_max: Duration::from_millis(200),
        ..ManagerConfig::default()
    }
}

async fn next_added(pm: &mut PathManager) -> String {
    loop {
        let ev = timeout(Duration::from_secs(10), pm.next_event())
            .await
            .expect("an event arrives")
            .expect("the manager is running");
        if let PathEvent::Added { name } = ev {
            return name;
        }
    }
}

fn closed_port() -> std::net::SocketAddr {
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    l.local_addr().unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn the_session_starts_on_the_first_candidate_that_answers() {
    let mut rig = rig(Config::default()).await;
    let slow_target = rig.target;
    let candidates = vec![
        Candidate::tcp("refused", closed_port()),
        Candidate::new("slow", move || async move {
            tokio::time::sleep(Duration::from_millis(400)).await;
            letify_link::tcp::connect(slow_target).await
        }),
        Candidate::tcp("fast", rig.target),
    ];
    let mut pm = PathManager::start(
        rig.client.clone(),
        rig.server.node_id(),
        candidates,
        Config::default(),
        quick(),
    )
    .await
    .unwrap();
    assert_eq!(next_added(&mut pm).await, "fast");

    let server_session = timeout(Duration::from_secs(5), rig.sessions.recv())
        .await
        .unwrap()
        .unwrap();
    let mut s = pm.session().open_stream().await.unwrap();
    s.write_all(b"ping").await.unwrap();
    let mut sb = server_session.accept_stream().await.unwrap();
    let mut got = [0u8; 4];
    sb.read_exact(&mut got).await.unwrap();
    assert_eq!(&got, b"ping");
    // The slow candidate joined the same session; it did not start another.
    assert!(
        timeout(Duration::from_millis(1500), rig.sessions.recv())
            .await
            .is_err()
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn extra_candidates_join_as_subflows() {
    let mut rig = rig(Config::default()).await;
    let candidates = vec![
        Candidate::tcp("one", rig.target),
        Candidate::tcp("two", rig.target),
        Candidate::tcp("three", rig.target),
    ];
    let pm = PathManager::start(
        rig.client.clone(),
        rig.server.node_id(),
        candidates,
        Config::default(),
        quick(),
    )
    .await
    .unwrap();
    let server_session = rig.sessions.recv().await.unwrap();
    timeout(Duration::from_secs(10), async {
        while pm.session().subflow_count() < 3 || server_session.subflow_count() < 3 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("all three candidates became subflows");
}

#[tokio::test(flavor = "multi_thread")]
async fn path_events_report_lost_and_measured_paths() {
    let mut rig = rig(Config::default()).await;
    let proxy = Proxy::to(rig.target).await;
    let pm_candidates = vec![Candidate::tcp("only", proxy.addr)];
    let mut pm = PathManager::start(
        rig.client.clone(),
        rig.server.node_id(),
        pm_candidates,
        Config::default(),
        quick(),
    )
    .await
    .unwrap();
    let _server = rig.sessions.recv().await.unwrap();
    let mut saw_measured = false;
    let mut saw_lost = false;
    let mut cut = false;
    timeout(Duration::from_secs(10), async {
        while !(saw_measured && saw_lost) {
            match pm.next_event().await.unwrap() {
                PathEvent::Measured { name, .. } => {
                    assert_eq!(name, "only");
                    saw_measured = true;
                    if !cut {
                        cut = true;
                        proxy.drop_connections();
                    }
                }
                PathEvent::Lost { name, .. } => {
                    assert_eq!(name, "only");
                    saw_lost = true;
                }
                _ => {}
            }
        }
    })
    .await
    .expect("measured and lost events arrive");
}

#[tokio::test(flavor = "multi_thread")]
async fn when_every_subflow_dies_the_manager_redials_and_the_stream_continues() {
    let config = Config {
        resume_window: Duration::from_secs(60),
        ..Config::default()
    };
    let mut rig = rig(config.clone()).await;
    let p1 = Proxy::to(rig.target).await;
    let p2 = Proxy::to(rig.target).await;
    let candidates = vec![Candidate::tcp("p1", p1.addr), Candidate::tcp("p2", p2.addr)];
    let pm = PathManager::start(
        rig.client.clone(),
        rig.server.node_id(),
        candidates,
        config,
        quick(),
    )
    .await
    .unwrap();
    let server_session = rig.sessions.recv().await.unwrap();

    let total = 8 << 20;
    let data = pattern(total, 5);
    let mut s = pm.session().open_stream().await.unwrap();
    let writer = tokio::spawn({
        let data = data.clone();
        async move {
            s.write_all(&data).await.unwrap();
            s.shutdown().await.unwrap();
        }
    });
    let mut sb = server_session.accept_stream().await.unwrap();
    let mut got = vec![0u8; 2 << 20];
    sb.read_exact(&mut got).await.unwrap();

    // Kill every connection in the middle of the transfer.
    p1.drop_connections();
    p2.drop_connections();

    let mut rest = Vec::new();
    timeout(Duration::from_secs(60), sb.read_to_end(&mut rest))
        .await
        .expect("the transfer finishes after the manager redials")
        .unwrap();
    got.extend_from_slice(&rest);
    writer.await.unwrap();
    assert_eq!(got.len(), total);
    assert!(got == data, "bytes differ after the redial");
}

/// A TCP forwarder that limits the client to server direction to about `chunk / delay`.
async fn throttled(
    target: std::net::SocketAddr,
    chunk: usize,
    delay: Duration,
) -> std::net::SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        while let Ok((front, _)) = listener.accept().await {
            tokio::spawn(async move {
                let Ok(back) = tokio::net::TcpStream::connect(target).await else {
                    return;
                };
                let (mut fr, mut fw) = front.into_split();
                let (mut br, mut bw) = back.into_split();
                let up = async move {
                    let mut buf = vec![0u8; chunk];
                    while let Ok(n) = fr.read(&mut buf).await {
                        if n == 0 || bw.write_all(&buf[..n]).await.is_err() {
                            break;
                        }
                        tokio::time::sleep(delay).await;
                    }
                };
                let down = async move {
                    let _ = tokio::io::copy(&mut br, &mut fw).await;
                };
                tokio::join!(up, down);
            });
        }
    });
    addr
}

fn floor_cfg() -> ManagerConfig {
    ManagerConfig {
        measure_interval: Duration::from_millis(50),
        redial_initial: Duration::from_millis(20),
        redial_max: Duration::from_millis(200),
        floor: Some(1_000_000.0),
        floor_backoff: Duration::from_secs(30),
        // The session is idle, so probes measure the subflows: one 16 KiB probe per 200 ms.
        probe_window: Duration::from_millis(200),
        probe_idle: Duration::from_millis(50),
        policy: letify_link::hysteresis::Policy {
            spacing: Duration::from_millis(100),
            ..Default::default()
        },
        ..ManagerConfig::default()
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_subflow_below_the_floor_is_closed_while_another_is_above_it() {
    let mut rig = rig(Config::default()).await;
    let slow = throttled(rig.target, 4096, Duration::from_millis(20)).await;
    let candidates = vec![
        Candidate::tcp("slow", slow),
        Candidate::tcp("fast", rig.target),
    ];
    let mut pm = PathManager::start(
        rig.client.clone(),
        rig.server.node_id(),
        candidates,
        Config::default(),
        floor_cfg(),
    )
    .await
    .unwrap();
    let server_session = rig.sessions.recv().await.unwrap();
    let mut lost = None;
    timeout(Duration::from_secs(20), async {
        while lost.is_none() {
            if let PathEvent::Lost { name, reason } = pm.next_event().await.unwrap() {
                lost = Some((name, reason));
            }
        }
    })
    .await
    .expect("the slow subflow is closed");
    let (name, reason) = lost.unwrap();
    assert_eq!(name, "slow");
    assert_eq!(reason, "below the floor");
    // The manager does not redial the candidate during the back-off.
    let again = timeout(Duration::from_secs(1), async {
        loop {
            if let Some(PathEvent::Added { name }) = pm.next_event().await
                && name == "slow"
            {
                return;
            }
        }
    })
    .await;
    assert!(again.is_err(), "the slow candidate was redialed");
    assert_eq!(pm.session().subflow_count(), 1);
    // The session works on the remaining subflow.
    let mut s = pm.session().open_stream().await.unwrap();
    s.write_all(b"ok").await.unwrap();
    let mut sb = server_session.accept_stream().await.unwrap();
    let mut got = [0u8; 2];
    sb.read_exact(&mut got).await.unwrap();
    assert_eq!(&got, b"ok");
}

/// The same rule with the session under heavy traffic, so the rates come from data and not
/// from probes. The slow subflow carries only what its throttle lets through, and the fast one
/// carries the rest, so the slow one is measured below the floor and closed.
#[tokio::test(flavor = "multi_thread")]
async fn a_throttled_subflow_under_heavy_traffic_is_measured_below_the_floor_and_closed() {
    let mut rig = rig(Config::default()).await;
    let slow = throttled(rig.target, 4096, Duration::from_millis(20)).await;
    let candidates = vec![
        Candidate::tcp("slow", slow),
        Candidate::tcp("fast", rig.target),
    ];
    let mut pm = PathManager::start(
        rig.client.clone(),
        rig.server.node_id(),
        candidates,
        Config::default(),
        floor_cfg(),
    )
    .await
    .unwrap();
    let server_session = rig.sessions.recv().await.unwrap();
    let mut stream = pm.session().open_stream().await.unwrap();
    let writer = tokio::spawn(async move {
        let chunk = pattern(1 << 20, 11);
        while stream.write_all(&chunk).await.is_ok() {}
    });
    let reader = tokio::spawn(async move {
        let mut sb = server_session.accept_stream().await.unwrap();
        let mut buf = vec![0u8; 1 << 20];
        while let Ok(n) = sb.read(&mut buf).await {
            if n == 0 {
                break;
            }
        }
        sb
    });
    let mut lost = None;
    timeout(Duration::from_secs(30), async {
        while lost.is_none() {
            if let PathEvent::Lost { name, reason } = pm.next_event().await.unwrap() {
                lost = Some((name, reason));
            }
        }
    })
    .await
    .expect("the slow subflow is closed while the session is busy");
    let (name, reason) = lost.unwrap();
    assert_eq!(name, "slow");
    assert_eq!(reason, "below the floor");
    assert_eq!(pm.session().subflow_count(), 1);
    writer.abort();
    reader.abort();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_sole_slow_subflow_is_never_closed_for_being_slow() {
    let mut rig = rig(Config::default()).await;
    let slow = throttled(rig.target, 4096, Duration::from_millis(20)).await;
    let mut pm = PathManager::start(
        rig.client.clone(),
        rig.server.node_id(),
        vec![Candidate::tcp("slow", slow)],
        Config::default(),
        floor_cfg(),
    )
    .await
    .unwrap();
    let _server = rig.sessions.recv().await.unwrap();
    let lost = timeout(Duration::from_secs(3), async {
        loop {
            if let Some(PathEvent::Lost { .. }) = pm.next_event().await {
                return;
            }
        }
    })
    .await;
    assert!(lost.is_err(), "the sole subflow was closed");
    assert_eq!(pm.session().subflow_count(), 1);
    // The probes did measure it, and below the floor.
    let rate = pm.session().subflow_stats()[0].probe_rate.expect("probed");
    assert!(rate < 1_000_000.0, "{rate}");
}

#[tokio::test(flavor = "multi_thread")]
async fn lifecycle_events_are_not_dropped_when_the_reader_is_slow() {
    let mut rig = rig(Config {
        resume_window: Duration::from_secs(60),
        ..Config::default()
    })
    .await;
    let proxy = Proxy::to(rig.target).await;
    let mut pm = PathManager::start(
        rig.client.clone(),
        rig.server.node_id(),
        vec![Candidate::tcp("only", proxy.addr)],
        Config {
            resume_window: Duration::from_secs(60),
            ..Config::default()
        },
        ManagerConfig {
            measure_interval: Duration::from_millis(10),
            event_capacity: 4,
            ..quick()
        },
    )
    .await
    .unwrap();
    let _server = rig.sessions.recv().await.unwrap();
    // The caller reads nothing while the subflow is lost and redialed three times, and
    // measurements keep arriving, far more than the capacity of 4.
    for _ in 0..3 {
        tokio::time::sleep(Duration::from_millis(300)).await;
        proxy.drop_connections();
        tokio::time::sleep(Duration::from_millis(150)).await;
        timeout(Duration::from_secs(10), async {
            while pm.session().subflow_count() < 1 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("the manager redialed");
    }
    tokio::time::sleep(Duration::from_millis(300)).await;
    let (mut added, mut lost, mut measured) = (0, 0, 0);
    while let Ok(Some(ev)) = timeout(Duration::ZERO, pm.next_event()).await {
        match ev {
            PathEvent::Added { .. } => added += 1,
            PathEvent::Lost { .. } => lost += 1,
            PathEvent::Measured { .. } => measured += 1,
            _ => {}
        }
    }
    assert_eq!(added, 4, "one Added at the start and one per redial");
    assert_eq!(lost, 3);
    // About 150 measurements were produced. At most 4 were unread at any time, and a few more
    // arrive while the queue is being read.
    assert!(measured < 20, "{measured} measurements were kept");
}

#[tokio::test(flavor = "multi_thread")]
async fn an_idle_subflow_is_measured_by_probes_within_the_budget() {
    let mut rig = rig(Config::default()).await;
    let window = Duration::from_secs(1);
    let min = 8 << 10;
    let pm = PathManager::start(
        rig.client.clone(),
        rig.server.node_id(),
        vec![
            Candidate::tcp("one", rig.target),
            Candidate::tcp("two", rig.target),
        ],
        Config::default(),
        ManagerConfig {
            measure_interval: Duration::from_millis(50),
            probe_window: window,
            probe_min_bytes: min,
            probe_idle: Duration::from_millis(100),
            ..quick()
        },
    )
    .await
    .unwrap();
    let server_session = rig.sessions.recv().await.unwrap();
    let begun = tokio::time::Instant::now();
    timeout(Duration::from_secs(10), async {
        while pm.session().subflow_count() < 2 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    tokio::time::sleep(Duration::from_millis(3200)).await;
    let elapsed = begun.elapsed();

    // Every idle subflow got a rate from a probe.
    let stats = pm.session().subflow_stats();
    assert_eq!(stats.len(), 2);
    for st in &stats {
        assert!(
            st.probe_acked_bytes > 0,
            "subflow {} was never probed",
            st.id
        );
        assert!(st.probe_rate.is_some_and(|r| r > 0.0));
    }
    // The spend stays within the budget of each window, with one probe of slack.
    let spent = pm.probe_bytes_spent();
    let windows = (elapsed.as_secs_f64() / window.as_secs_f64()).ceil() as u64;
    assert!(spent > 0);
    assert!(
        spent <= windows * min as u64 + min as u64,
        "spent {spent} bytes in {windows} windows"
    );
    // The receiver discarded the filler: nothing reached a stream and the session still works.
    let mut s = pm.session().open_stream().await.unwrap();
    s.write_all(b"ok").await.unwrap();
    let mut sb = server_session.accept_stream().await.unwrap();
    let mut got = [0u8; 2];
    sb.read_exact(&mut got).await.unwrap();
    assert_eq!(&got, b"ok");
    assert_eq!(server_session.dropped_records(), 0);
}
