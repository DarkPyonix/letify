//! Pins SPEC.md letify-link, Session item 1 (handshake) and Identity.

mod common;

use common::*;
use letify_link::{Config, Error, Identity, PeerPolicy, Session};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[tokio::test]
async fn a_session_between_two_known_keys_carries_a_stream_both_ways() {
    let (a, b) = session_pair(Config::default()).await;
    let mut sa = a.open_stream().await.unwrap();
    sa.write_all(b"ping").await.unwrap();
    let mut sb = b.accept_stream().await.unwrap();
    let mut buf = [0u8; 4];
    sb.read_exact(&mut buf).await.unwrap();
    assert_eq!(&buf, b"ping");
    sb.write_all(b"pong").await.unwrap();
    sa.read_exact(&mut buf).await.unwrap();
    assert_eq!(&buf, b"pong");
}

#[tokio::test]
async fn each_side_learns_the_other_sides_node_id() {
    let a = Identity::generate();
    let b = Identity::generate();
    let (ta, tb) = duplex_pair();
    let (bb, a_id) = (b.clone(), a.node_id());
    let responder = tokio::spawn(async move {
        Session::accept(tb, &bb, PeerPolicy::allow(vec![a_id]), Config::default()).await
    });
    let initiator = Session::connect(ta, &a, b.node_id(), Config::default()).await.unwrap();
    let responder = responder.await.unwrap().unwrap();
    assert_eq!(initiator.peer(), b.node_id());
    assert_eq!(responder.peer(), a.node_id());
}

#[tokio::test]
async fn an_unknown_initiator_key_is_refused() {
    let known = Identity::generate();
    let stranger = Identity::generate();
    let b = Identity::generate();
    let (ta, tb) = duplex_pair();
    let bb = b.clone();
    let responder = tokio::spawn(async move {
        Session::accept(tb, &bb, PeerPolicy::allow(vec![known.node_id()]), Config::default()).await
    });
    let initiator = Session::connect(ta, &stranger, b.node_id(), Config::default()).await;
    assert!(initiator.is_err());
    match responder.await.unwrap() {
        Err(Error::UnknownPeer(id)) => assert_eq!(id, stranger.node_id()),
        other => panic!("expected UnknownPeer, got {:?}", other.map(|_| ())),
    }
}

#[tokio::test]
async fn a_policy_callback_decides_which_initiators_are_accepted() {
    let a = Identity::generate();
    let b = Identity::generate();
    let (ta, tb) = duplex_pair();
    let (bb, a_id) = (b.clone(), a.node_id());
    let responder = tokio::spawn(async move {
        Session::accept(tb, &bb, PeerPolicy::from_fn(move |id| *id == a_id), Config::default())
            .await
    });
    Session::connect(ta, &a, b.node_id(), Config::default()).await.unwrap();
    responder.await.unwrap().unwrap();
}

#[tokio::test]
async fn an_initiator_that_expects_another_responder_key_fails() {
    let a = Identity::generate();
    let b = Identity::generate();
    let other = Identity::generate();
    let (ta, tb) = duplex_pair();
    let a_id = a.node_id();
    let responder = tokio::spawn(async move {
        Session::accept(tb, &b, PeerPolicy::allow(vec![a_id]), Config::default()).await
    });
    let initiator = Session::connect(ta, &a, other.node_id(), Config::default()).await;
    assert!(initiator.is_err());
    assert!(responder.await.unwrap().is_err());
}

#[tokio::test]
async fn a_caller_supplied_ed25519_secret_key_gives_the_same_node_id_every_time() {
    let secret = [7u8; 32];
    let one = Identity::from_secret_bytes(secret);
    let two = Identity::from_secret_bytes(secret);
    assert_eq!(one.node_id(), two.node_id());
    assert_eq!(one.secret_bytes(), secret);
    // The caller's key is a working identity: a session can be built on it.
    let b = Identity::generate();
    let (ta, tb) = duplex_pair();
    let (bb, id) = (b.clone(), one.node_id());
    let responder = tokio::spawn(async move {
        Session::accept(tb, &bb, PeerPolicy::allow(vec![id]), Config::default()).await
    });
    Session::connect(ta, &one, b.node_id(), Config::default()).await.unwrap();
    responder.await.unwrap().unwrap();
}

#[tokio::test]
async fn a_tampered_version_header_fails_the_handshake() {
    let a = Identity::generate();
    let b = Identity::generate();
    let (ta, tb) = duplex_pair();
    // Bytes 0..4 are the magic, 4..6 the lowest version, 6..8 the highest. The initiator offers
    // 2..=3 and the tamper turns the highest into 2. The responder still finds a common
    // version, so only the prologue binding can reject the handshake.
    let ta = Tamper::new(ta, 7);
    let offered = Config { versions: 2..=3, ..Config::default() };
    let supported = Config { versions: 1..=3, ..Config::default() };
    let (bb, a_id) = (b.clone(), a.node_id());
    let responder = tokio::spawn(async move {
        Session::accept(tb, &bb, PeerPolicy::allow(vec![a_id]), supported).await
    });
    let initiator = Session::connect(ta, &a, b.node_id(), offered).await;
    assert!(initiator.is_err());
    assert!(matches!(responder.await.unwrap(), Err(Error::Handshake(_))));
}

#[tokio::test]
async fn mismatched_versions_are_refused_naming_both() {
    let a = Identity::generate();
    let b = Identity::generate();
    let (ta, tb) = duplex_pair();
    let ours = Config { versions: 3..=4, ..Config::default() };
    let theirs = Config { versions: 1..=2, ..Config::default() };
    let (bb, a_id) = (b.clone(), a.node_id());
    let responder = tokio::spawn(async move {
        Session::accept(tb, &bb, PeerPolicy::allow(vec![a_id]), theirs).await
    });
    let err = match Session::connect(ta, &a, b.node_id(), ours).await {
        Err(e) => e,
        Ok(_) => panic!("the handshake must fail"),
    };
    match &err {
        Error::VersionMismatch { ours, theirs } => {
            assert_eq!(ours, &(3..=4));
            assert_eq!(theirs, &(1..=2));
        }
        other => panic!("expected VersionMismatch, got {other:?}"),
    }
    let text = err.to_string();
    assert!(text.contains("3..=4") && text.contains("1..=2"), "{text}");
    assert!(matches!(responder.await.unwrap(), Err(Error::VersionMismatch { .. })));
}

#[tokio::test]
async fn the_highest_common_version_is_chosen() {
    let a = Identity::generate();
    let b = Identity::generate();
    let (ta, tb) = duplex_pair();
    let ours = Config { versions: 1..=5, ..Config::default() };
    let theirs = Config { versions: 3..=9, ..Config::default() };
    let (bb, a_id) = (b.clone(), a.node_id());
    let responder = tokio::spawn(async move {
        Session::accept(tb, &bb, PeerPolicy::allow(vec![a_id]), theirs).await
    });
    let initiator = Session::connect(ta, &a, b.node_id(), ours).await.unwrap();
    let responder = responder.await.unwrap().unwrap();
    assert_eq!(initiator.version(), 5);
    assert_eq!(responder.version(), 5);
}
