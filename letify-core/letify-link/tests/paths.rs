//! Pins SPEC.md letify-link, Paths and the path manager items 1, 2 and 4 over real TCP on
//! loopback.

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
                PathEvent::Lost { name } => {
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
