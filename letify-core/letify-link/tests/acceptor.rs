//! Pins SPEC.md letify-link, Paths and the path manager item 1 and Session item 4: the
//! acceptor routes a resuming subflow to its session and a new handshake to a new session.

mod common;

use std::sync::Arc;
use std::time::Duration;

use letify_link::{Accepted, Acceptor, Config, Error, Identity, PeerPolicy, Session};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::time::timeout;

struct Rig {
    client: Identity,
    server: Identity,
    acceptor: Arc<Acceptor>,
    listener: TcpListener,
}

async fn rig() -> Rig {
    let client = Identity::generate();
    let server = Identity::generate();
    let acceptor = Arc::new(Acceptor::new(
        server.clone(),
        PeerPolicy::allow(vec![client.node_id()]),
        Config::default(),
    ));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    Rig {
        client,
        server,
        acceptor,
        listener,
    }
}

/// Connect a client session and let the acceptor produce the server session.
async fn pair(rig: &Rig) -> (Session, Session) {
    let addr = rig.listener.local_addr().unwrap();
    let accept = async {
        let (tcp, _) = rig.listener.accept().await.unwrap();
        rig.acceptor.accept(tcp).await.unwrap()
    };
    let connect = async {
        let tcp = TcpStream::connect(addr).await.unwrap();
        Session::connect(tcp, &rig.client, rig.server.node_id(), Config::default())
            .await
            .unwrap()
    };
    let (accepted, client) = tokio::join!(accept, connect);
    match accepted {
        Accepted::Session(s) => (client, s),
        Accepted::Subflow(_) => panic!("a handshake is not a subflow"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_new_handshake_at_the_acceptor_yields_a_new_session() {
    let rig = rig().await;
    let (a, b) = pair(&rig).await;
    assert_eq!(b.peer(), rig.client.node_id());
    let mut s = a.open_stream().await.unwrap();
    s.write_all(b"hello").await.unwrap();
    let mut sb = b.accept_stream().await.unwrap();
    let mut got = [0u8; 5];
    sb.read_exact(&mut got).await.unwrap();
    assert_eq!(&got, b"hello");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_resuming_subflow_is_routed_to_its_session_by_the_acceptor() {
    let rig = rig().await;
    let (a, b) = pair(&rig).await;
    // A second session on the same acceptor must not receive the subflow.
    let (a2, b2) = pair(&rig).await;

    let addr = rig.listener.local_addr().unwrap();
    let accept = async {
        let (tcp, _) = rig.listener.accept().await.unwrap();
        rig.acceptor.accept(tcp).await.unwrap()
    };
    let dial = async { a.add_subflow(TcpStream::connect(addr).await.unwrap()).await };
    let (accepted, added) = tokio::join!(accept, dial);
    added.unwrap();
    assert!(matches!(accepted, Accepted::Subflow(id) if id == b.session_id()));
    assert_eq!(a.subflow_count(), 2);
    assert_eq!(b.subflow_count(), 2);
    assert_eq!(a2.subflow_count(), 1);
    assert_eq!(b2.subflow_count(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_subflow_naming_an_unknown_session_is_dropped() {
    let rig = rig().await;
    let addr = rig.listener.local_addr().unwrap();
    let mut rogue = TcpStream::connect(addr).await.unwrap();
    let mut bytes = b"LLNS".to_vec();
    bytes.extend_from_slice(&[7u8; 48]);
    rogue.write_all(&bytes).await.unwrap();
    let (tcp, _) = rig.listener.accept().await.unwrap();
    let err = rig.acceptor.accept(tcp).await.err().expect("refused");
    assert!(matches!(err, Error::Subflow(_)), "{err}");
    // The acceptor closed the transport.
    let mut buf = [0u8; 1];
    let n = timeout(Duration::from_secs(2), rogue.read(&mut buf))
        .await
        .unwrap()
        .unwrap_or(0);
    assert_eq!(n, 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_closed_session_is_forgotten_by_the_acceptor() {
    let rig = rig().await;
    let (a, b) = pair(&rig).await;
    // A prefix that was valid while the session lived.
    let prefix = join_through_mitm(&a, &rig.acceptor).await;
    drop(b);
    drop(a);
    tokio::time::sleep(Duration::from_millis(50)).await;
    let addr = rig.listener.local_addr().unwrap();
    let mut rogue = TcpStream::connect(addr).await.unwrap();
    rogue.write_all(&prefix).await.unwrap();
    let (tcp, _) = rig.listener.accept().await.unwrap();
    assert!(rig.acceptor.accept(tcp).await.is_err());
}

/// The 52 bytes a dialer sends first on a new subflow of `a`, through `acceptor`, and the
/// subflow it added.
async fn join_through_mitm(a: &Session, acceptor: &Acceptor) -> Vec<u8> {
    let (ta, m_a) = common::duplex_pair();
    let (m_b, tb) = common::duplex_pair();
    let mitm = common::Mitm::new(m_a, m_b, common::Prefix::Subflow);
    let (added, routed) = tokio::join!(a.add_subflow(ta), acceptor.accept(tb));
    added.unwrap();
    assert!(matches!(routed, Ok(Accepted::Subflow(_))));
    let prefix = mitm.prefix.lock().unwrap().clone();
    assert_eq!(prefix.len(), 52);
    prefix
}

#[tokio::test(flavor = "multi_thread")]
async fn join_tokens_differ_per_subflow_and_do_not_reveal_the_session() {
    let rig = rig().await;
    let (a, b) = pair(&rig).await;
    let one = join_through_mitm(&a, &rig.acceptor).await;
    let two = join_through_mitm(&a, &rig.acceptor).await;
    assert_eq!(&one[..4], b"LLNS");
    assert_ne!(one[4..36], two[4..36], "the tokens must differ");
    assert_ne!(one[4..20], two[4..20], "no part of a token may repeat");
    assert_ne!(one[36..], two[36..], "the challenges must differ");
    let id = b.session_id();
    for prefix in [&one, &two] {
        assert!(
            !prefix.windows(16).any(|w| w == id),
            "the session id must not be on the wire"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_copied_join_on_another_transport_is_refused_and_the_genuine_join_succeeds() {
    let rig = rig().await;
    let (a, b) = pair(&rig).await;
    let a = Arc::new(a);
    // The genuine dialer's far end is held by the test, which plays the network.
    let (d_cli, mut d_net) = common::duplex_pair();
    let dial = tokio::spawn({
        let a = a.clone();
        async move { a.add_subflow(d_cli).await }
    });
    // The observer sees everything the dialer sends before it hears anything.
    let mut seen = vec![0u8; 52];
    d_net.read_exact(&mut seen).await.unwrap();
    let mut more = vec![0u8; 4096];
    if let Ok(Ok(n)) = timeout(Duration::from_millis(300), d_net.read(&mut more)).await {
        seen.extend_from_slice(&more[..n]);
    }

    // It copies those bytes onto its own connection, which reaches the acceptor first.
    let (mut att, att_srv) = common::duplex_pair();
    att.write_all(&seen).await.unwrap();
    let copied = tokio::spawn({
        let acceptor = rig.acceptor.clone();
        async move { acceptor.accept(att_srv).await }
    });
    tokio::time::sleep(Duration::from_millis(200)).await;

    // The genuine bytes then arrive on the genuine transport.
    let (mut g_cli, g_srv) = common::duplex_pair();
    g_cli.write_all(&seen).await.unwrap();
    let genuine = tokio::spawn({
        let acceptor = rig.acceptor.clone();
        async move { acceptor.accept(g_srv).await }
    });
    let (mut g_r, mut g_w) = tokio::io::split(g_cli);
    let (mut n_r, mut n_w) = tokio::io::split(d_net);
    tokio::spawn(async move { tokio::io::copy(&mut g_r, &mut n_w).await });
    tokio::spawn(async move { tokio::io::copy(&mut n_r, &mut g_w).await });

    let routed = timeout(Duration::from_secs(5), genuine)
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(routed, Ok(Accepted::Subflow(_))), "genuine join");
    timeout(Duration::from_secs(5), dial)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(
        b.subflow_count(),
        2,
        "only the genuine transport was attached"
    );
    drop(att);
    let copied = timeout(Duration::from_secs(5), copied)
        .await
        .unwrap()
        .unwrap();
    assert!(copied.is_err(), "the copied join must not attach");
    assert_eq!(b.subflow_count(), 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_session_has_at_most_eight_joins_in_progress() {
    let rig = rig().await;
    let (a, _b) = pair(&rig).await;
    let prefix = join_through_mitm(&a, &rig.acceptor).await;
    let mut held = Vec::new();
    for _ in 0..8 {
        let (mut att, srv) = common::duplex_pair();
        att.write_all(&prefix).await.unwrap();
        let acceptor = rig.acceptor.clone();
        held.push((att, tokio::spawn(async move { acceptor.accept(srv).await })));
    }
    tokio::time::sleep(Duration::from_millis(200)).await;
    let (mut att, srv) = common::duplex_pair();
    att.write_all(&prefix).await.unwrap();
    let ninth = timeout(Duration::from_secs(2), rig.acceptor.accept(srv))
        .await
        .expect("the ninth is refused at once");
    assert!(matches!(ninth, Err(Error::Subflow(_))));
}

#[tokio::test(flavor = "multi_thread")]
async fn half_open_handshakes_are_capped() {
    let client = Identity::generate();
    let server = Identity::generate();
    let config = Config {
        max_half_open: 2,
        handshake_timeout: Duration::from_secs(10),
        ..Config::default()
    };
    let acceptor = Arc::new(Acceptor::new(
        server.clone(),
        PeerPolicy::allow(vec![client.node_id()]),
        config.clone(),
    ));
    let mut silent = Vec::new();
    for _ in 0..2 {
        let (att, srv) = common::duplex_pair();
        let acceptor = acceptor.clone();
        silent.push((att, tokio::spawn(async move { acceptor.accept(srv).await })));
    }
    tokio::time::sleep(Duration::from_millis(200)).await;
    let (_att, srv) = common::duplex_pair();
    let third = timeout(Duration::from_secs(2), acceptor.accept(srv))
        .await
        .expect("a transport over the cap is closed at once");
    assert!(matches!(third, Err(Error::Handshake(_))));

    // A slot frees when a half-open transport ends, and a real handshake then succeeds.
    let (att, task) = silent.pop().unwrap();
    drop(att);
    let _ = task.await;
    let (ta, tb) = common::duplex_pair();
    let (connected, accepted) = tokio::join!(
        Session::connect(ta, &client, server.node_id(), config),
        acceptor.accept(tb)
    );
    connected.unwrap();
    assert!(matches!(accepted, Ok(Accepted::Session(_))));
}
