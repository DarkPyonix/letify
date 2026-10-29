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
    bytes.extend_from_slice(&[7u8; 16]);
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
    let id = b.session_id();
    drop(b);
    drop(a);
    tokio::time::sleep(Duration::from_millis(50)).await;
    let addr = rig.listener.local_addr().unwrap();
    let mut rogue = TcpStream::connect(addr).await.unwrap();
    let mut bytes = b"LLNS".to_vec();
    bytes.extend_from_slice(&id);
    rogue.write_all(&bytes).await.unwrap();
    let (tcp, _) = rig.listener.accept().await.unwrap();
    assert!(rig.acceptor.accept(tcp).await.is_err());
}
