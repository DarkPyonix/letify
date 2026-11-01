//! Pins SPEC.md letify-link, Sidecar interface item 10 (Trace): a UDP path has a congestion
//! gauge whether the session received it directly or wrapped in a `BoxTransport`, which is the
//! form a punched candidate takes through the sidecar.

mod common;

use std::time::Duration;

use common::*;
use letify_link::udp::{UdpConfig, UdpListener, UdpPath};
use letify_link::{Acceptor, BoxTransport, Candidate, Config, Identity, PathManager, PeerPolicy};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::time::timeout;

/// Send 2 MiB from `a` to `b` so that the transport has carried real traffic.
async fn carry(a: &letify_link::Session, b: &letify_link::Session) {
    let mut s = a.open_stream().await.unwrap();
    let writer = tokio::spawn(async move {
        s.write_all(&pattern(2 << 20, 5)).await.unwrap();
        s.shutdown().await.unwrap();
    });
    let mut sb = b.accept_stream().await.unwrap();
    let mut got = Vec::new();
    sb.read_to_end(&mut got).await.unwrap();
    writer.await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;
}

async fn server() -> (
    std::net::SocketAddr,
    Identity,
    tokio::sync::mpsc::Receiver<letify_link::Session>,
    Identity,
) {
    let (client, node) = (Identity::generate(), Identity::generate());
    let sock = udp_socket();
    let addr = sock.local_addr().unwrap();
    let listener = UdpListener::new(sock, None, &UdpConfig::default()).unwrap();
    let acceptor = Acceptor::new(
        node.clone(),
        PeerPolicy::allow(vec![client.node_id()]),
        Config::default(),
    );
    (addr, node, serve(acceptor, None, Some(listener)), client)
}

#[tokio::test(flavor = "multi_thread")]
async fn a_udp_path_given_directly_has_a_gauge() {
    let (addr, node, mut sessions, client) = server().await;
    let path = UdpPath::connect(udp_socket(), addr).await.unwrap();
    let a = letify_link::Session::connect(path, &client, node.node_id(), Config::default())
        .await
        .unwrap();
    let b = timeout(Duration::from_secs(10), sessions.recv())
        .await
        .unwrap()
        .unwrap();
    carry(&a, &b).await;
    assert!(a.subflow_stats()[0].congestion.is_some());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_udp_path_in_a_box_transport_has_a_gauge() {
    let (addr, node, mut sessions, client) = server().await;
    let path = UdpPath::connect(udp_socket(), addr).await.unwrap();
    let boxed: BoxTransport = Box::new(path);
    let a = letify_link::Session::connect(boxed, &client, node.node_id(), Config::default())
        .await
        .unwrap();
    let b = timeout(Duration::from_secs(10), sessions.recv())
        .await
        .unwrap()
        .unwrap();
    carry(&a, &b).await;
    assert!(a.subflow_stats()[0].congestion.is_some());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_boxed_udp_subflow_added_to_a_session_has_a_gauge() {
    let (addr, node, mut sessions, client) = server().await;
    let first = UdpPath::connect(udp_socket(), addr).await.unwrap();
    let a = letify_link::Session::connect(first, &client, node.node_id(), Config::default())
        .await
        .unwrap();
    let b = timeout(Duration::from_secs(10), sessions.recv())
        .await
        .unwrap()
        .unwrap();
    let second: BoxTransport = Box::new(UdpPath::connect(udp_socket(), addr).await.unwrap());
    let id = a.add_subflow(second).await.unwrap();
    carry(&a, &b).await;
    let stats = a.subflow_stats();
    let added = stats.iter().find(|s| s.id == id).unwrap();
    assert!(added.congestion.is_some());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_punched_udp_candidate_has_a_gauge_through_the_path_manager() {
    // The sidecar hands a punched UdpPath over as a BoxTransport that the dial closure returns.
    let (addr, node, mut sessions, client) = server().await;
    let path = UdpPath::connect(udp_socket(), addr).await.unwrap();
    let slot = std::sync::Arc::new(std::sync::Mutex::new(Some(Box::new(path) as BoxTransport)));
    let dial = move || {
        let slot = slot.clone();
        async move {
            slot.lock().unwrap().take().ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::ConnectionAborted, "punched socket used")
            })
        }
    };
    let pm = PathManager::start(
        client,
        node.node_id(),
        vec![Candidate::udp("punched", dial)],
        Config::default(),
        Default::default(),
    )
    .await
    .unwrap();
    let b = timeout(Duration::from_secs(10), sessions.recv())
        .await
        .unwrap()
        .unwrap();
    carry(pm.session(), &b).await;
    assert!(pm.session().subflow_stats()[0].congestion.is_some());
}
