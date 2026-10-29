//! Pins SPEC.md letify-link, Session item 9 (Revocation): a session can be closed with a
//! reason, and a peer policy that changes at run time refuses a revoked node on its next
//! handshake or resume and closes its live sessions.

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use common::*;
use letify_link::{Accepted, Acceptor, Config, Error, Identity, PeerPolicy, Session};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::time::{sleep, timeout};

/// Wait until `cond` holds, for at most `secs` seconds.
async fn until(secs: u64, mut cond: impl FnMut() -> bool) -> bool {
    let end = tokio::time::Instant::now() + Duration::from_secs(secs);
    while tokio::time::Instant::now() < end {
        if cond() {
            return true;
        }
        sleep(Duration::from_millis(10)).await;
    }
    cond()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_session_closed_with_a_reason_tells_the_peer_the_reason() {
    let (a, b) = session_pair(Config::default()).await;
    let mut s = a.open_stream().await.unwrap();
    s.write_all(b"hi").await.unwrap();
    let mut sb = b.accept_stream().await.unwrap();
    let mut two = [0u8; 2];
    sb.read_exact(&mut two).await.unwrap();
    a.close_with_reason("maintenance window").await;
    assert!(until(5, || b.is_closed()).await);
    assert_eq!(a.close_reason().as_deref(), Some("maintenance window"));
    assert_eq!(b.close_reason().as_deref(), Some("maintenance window"));
    assert!(matches!(b.accept_stream().await, Err(Error::Closed)));
    let mut rest = Vec::new();
    assert_eq!(sb.read_to_end(&mut rest).await.unwrap(), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_reason_is_cut_to_256_bytes() {
    let (a, b) = session_pair(Config::default()).await;
    a.close_with_reason(&"x".repeat(1000)).await;
    assert!(until(5, || b.is_closed()).await);
    assert_eq!(b.close_reason().unwrap().len(), 256);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_revoked_node_is_refused_at_the_handshake() {
    let (a, b) = (Identity::generate(), Identity::generate());
    let policy = PeerPolicy::allow(vec![a.node_id()]);
    policy.revoke(a.node_id());
    let (ta, tb) = duplex_pair();
    let responder = tokio::spawn({
        let (b, policy) = (b.clone(), policy.clone());
        async move { Session::accept(tb, &b, policy, Config::default()).await }
    });
    let initiator = Session::connect(ta, &a, b.node_id(), Config::default()).await;
    assert!(initiator.is_err());
    assert!(matches!(
        responder.await.unwrap(),
        Err(Error::UnknownPeer(id)) if id == a.node_id()
    ));
    // Reinstated, the same node is accepted again.
    policy.reinstate(a.node_id());
    let (ta, tb) = duplex_pair();
    let responder = tokio::spawn({
        let (b, policy) = (b.clone(), policy.clone());
        async move { Session::accept(tb, &b, policy, Config::default()).await }
    });
    Session::connect(ta, &a, b.node_id(), Config::default())
        .await
        .unwrap();
    responder.await.unwrap().unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn revoking_a_node_closes_its_open_session_at_once() {
    let (a, b) = (Identity::generate(), Identity::generate());
    let policy = PeerPolicy::allow(vec![a.node_id()]);
    let (ta, tb) = duplex_pair();
    let responder = tokio::spawn({
        let (b, policy) = (b.clone(), policy.clone());
        async move { Session::accept(tb, &b, policy, Config::default()).await }
    });
    let ia = Session::connect(ta, &a, b.node_id(), Config::default())
        .await
        .unwrap();
    let rb = responder.await.unwrap().unwrap();
    assert!(!rb.is_closed());
    policy.revoke(a.node_id());
    assert!(until(2, || rb.is_closed()).await, "the responder stays open");
    assert!(until(2, || ia.is_closed()).await, "the initiator stays open");
    assert_eq!(rb.close_reason().as_deref(), Some("revoked"));
    assert_eq!(ia.close_reason().as_deref(), Some("revoked"));
}

/// An acceptor with a TCP listener, two clients and a session for each.
struct Rig {
    acceptor: Arc<Acceptor>,
    listener: TcpListener,
    server: Identity,
    gate: Arc<AtomicBool>,
}

async fn rig(clients: &[&Identity]) -> Rig {
    let server = Identity::generate();
    let ids: Vec<_> = clients.iter().map(|c| c.node_id()).collect();
    // The callback can be flipped without any revoke call, to isolate the refusal on resume.
    let gate = Arc::new(AtomicBool::new(true));
    let policy = PeerPolicy::from_fn({
        let gate = gate.clone();
        move |id| gate.load(Ordering::SeqCst) && ids.contains(id)
    });
    Rig {
        acceptor: Arc::new(Acceptor::new(server.clone(), policy, Config::default())),
        listener: TcpListener::bind("127.0.0.1:0").await.unwrap(),
        server,
        gate,
    }
}

async fn open(rig: &Rig, client: &Identity) -> (Session, Session) {
    let addr = rig.listener.local_addr().unwrap();
    let accept = async {
        let (tcp, _) = rig.listener.accept().await.unwrap();
        rig.acceptor.accept(tcp).await.unwrap()
    };
    let connect = async {
        let tcp = TcpStream::connect(addr).await.unwrap();
        Session::connect(tcp, client, rig.server.node_id(), Config::default())
            .await
            .unwrap()
    };
    let (accepted, session) = tokio::join!(accept, connect);
    let Accepted::Session(served) = accepted else {
        panic!("a handshake is not a subflow");
    };
    (session, served)
}

#[tokio::test(flavor = "multi_thread")]
async fn revoking_at_the_acceptor_closes_only_the_sessions_to_that_node() {
    let (c1, c2) = (Identity::generate(), Identity::generate());
    let rig = rig(&[&c1, &c2]).await;
    let (a1, b1) = open(&rig, &c1).await;
    let (a2, b2) = open(&rig, &c2).await;
    rig.acceptor.revoke(c1.node_id());
    assert!(until(2, || b1.is_closed() && a1.is_closed()).await);
    assert_eq!(a1.close_reason().as_deref(), Some("revoked"));
    sleep(Duration::from_millis(200)).await;
    assert!(!b2.is_closed() && !a2.is_closed());
    // The revoked node cannot start a new session either.
    let addr = rig.listener.local_addr().unwrap();
    let (served, again) = tokio::join!(
        async {
            let (tcp, _) = rig.listener.accept().await.unwrap();
            rig.acceptor.accept(tcp).await
        },
        async {
            let tcp = TcpStream::connect(addr).await.unwrap();
            Session::connect(tcp, &c1, rig.server.node_id(), Config::default()).await
        }
    );
    assert!(again.is_err());
    assert!(matches!(served, Err(Error::UnknownPeer(_))));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_revoked_node_is_refused_when_it_resumes() {
    let c = Identity::generate();
    let rig = rig(&[&c]).await;
    let (a, b) = open(&rig, &c).await;
    // The acceptor's policy turns against the node without a revoke call, so the session
    // stays open and only the resume can notice.
    rig.gate.store(false, Ordering::SeqCst);
    let addr = rig.listener.local_addr().unwrap();
    let accept = async {
        let (tcp, _) = rig.listener.accept().await.unwrap();
        rig.acceptor.accept(tcp).await
    };
    let dial = async { a.add_subflow(TcpStream::connect(addr).await.unwrap()).await };
    let (accepted, added) = timeout(Duration::from_secs(5), async { tokio::join!(accept, dial) })
        .await
        .unwrap();
    assert!(matches!(accepted, Err(Error::UnknownPeer(_))), "{accepted:?}");
    assert!(added.is_err());
    assert_eq!(b.subflow_count(), 1);
}
