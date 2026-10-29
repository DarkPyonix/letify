//! Pins SPEC.md letify-link, Session item 5: resumption on a new subflow and the resume window.

mod common;

use std::time::Duration;

use common::*;
use letify_link::{Config, Error, Identity, PeerPolicy, Session};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::time::timeout;

#[tokio::test(flavor = "multi_thread")]
async fn a_session_survives_its_only_subflow_dying_and_resumes_on_a_new_one_without_losing_bytes() {
    let a_id = Identity::generate();
    let b_id = Identity::generate();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let target = listener.local_addr().unwrap();
    let proxy = Proxy::to(target).await;
    let config = Config {
        resume_window: Duration::from_secs(60),
        ..Config::default()
    };

    let responder = tokio::spawn({
        let (b_id, a_node, config) = (b_id.clone(), a_id.node_id(), config.clone());
        async move {
            let (tcp, _) = listener.accept().await.unwrap();
            let session = Session::accept(tcp, &b_id, PeerPolicy::allow(vec![a_node]), config)
                .await
                .unwrap();
            (session, listener)
        }
    });
    let tcp = TcpStream::connect(proxy.addr).await.unwrap();
    let a = Session::connect(tcp, &a_id, b_id.node_id(), config)
        .await
        .unwrap();
    let (b, listener) = responder.await.unwrap();

    let total = 8 << 20;
    let data = pattern(total, 11);
    let mut s = a.open_stream().await.unwrap();
    let writer = tokio::spawn({
        let data = data.clone();
        async move {
            s.write_all(&data).await.unwrap();
            s.shutdown().await.unwrap();
        }
    });
    let mut sb = b.accept_stream().await.unwrap();
    let mut got = vec![0u8; 2 << 20];
    sb.read_exact(&mut got).await.unwrap();

    // Kill the only path in the middle of the transfer.
    proxy.cut();
    timeout(Duration::from_secs(5), async {
        while a.subflow_count() > 0 || b.subflow_count() > 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("both ends notice the dead path");

    // A new TCP connection, direct this time, resumes the same session.
    let accept = tokio::spawn(async move { listener.accept().await.unwrap().0 });
    let new_a = TcpStream::connect(target).await.unwrap();
    let new_b = accept.await.unwrap();
    let (ra, rb) = tokio::join!(a.add_subflow(new_a), b.add_subflow(new_b));
    ra.unwrap();
    rb.unwrap();

    // Every byte arrives, and the writer saw no error at any point.
    let mut rest = Vec::new();
    timeout(Duration::from_secs(60), sb.read_to_end(&mut rest))
        .await
        .unwrap()
        .unwrap();
    got.extend_from_slice(&rest);
    writer.await.unwrap();
    assert_eq!(got.len(), total);
    assert!(got == data, "bytes differ after the resume");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_session_that_gets_no_new_subflow_within_the_window_fails_streams_with_a_connection_lost_error()
 {
    let a_id = Identity::generate();
    let b_id = Identity::generate();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let proxy = Proxy::to(listener.local_addr().unwrap()).await;
    let config = Config {
        resume_window: Duration::from_millis(300),
        ..Config::default()
    };
    let responder = tokio::spawn({
        let (b_id, a_node, config) = (b_id.clone(), a_id.node_id(), config.clone());
        async move {
            let (tcp, _) = listener.accept().await.unwrap();
            Session::accept(tcp, &b_id, PeerPolicy::allow(vec![a_node]), config)
                .await
                .unwrap()
        }
    });
    let tcp = TcpStream::connect(proxy.addr).await.unwrap();
    let a = Session::connect(tcp, &a_id, b_id.node_id(), config)
        .await
        .unwrap();
    let b = responder.await.unwrap();

    let mut s = a.open_stream().await.unwrap();
    s.write_all(b"before").await.unwrap();
    let mut sb = b.accept_stream().await.unwrap();
    let mut buf = [0u8; 6];
    sb.read_exact(&mut buf).await.unwrap();

    proxy.cut();
    // Within the window the stream only pauses: the read is still pending.
    assert!(
        timeout(Duration::from_millis(100), sb.read(&mut buf))
            .await
            .is_err()
    );

    // After the window it fails with an error that says the connection was lost.
    let err = timeout(Duration::from_secs(5), sb.read(&mut buf))
        .await
        .unwrap()
        .unwrap_err();
    assert!(
        matches!(Error::from_io(&err), Some(Error::ConnectionLost(_))),
        "{err:?}"
    );
    let err = timeout(Duration::from_secs(5), s.write_all(b"after"))
        .await
        .unwrap()
        .unwrap_err();
    assert!(
        matches!(Error::from_io(&err), Some(Error::ConnectionLost(_))),
        "{err:?}"
    );
    assert!(matches!(
        a.accept_stream().await,
        Err(Error::ConnectionLost(_))
    ));
    assert!(Error::ConnectionLost("x".into()).is_infrastructure());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_subflow_for_another_session_is_refused() {
    let (a, _b) = session_pair(Config::default()).await;
    let (c, d) = session_pair(Config::default()).await;
    let (ta, tb) = duplex_pair();
    let (ra, rb) = tokio::join!(a.add_subflow(ta), d.add_subflow(tb));
    assert!(ra.is_err() && rb.is_err());
    drop(c);
}
