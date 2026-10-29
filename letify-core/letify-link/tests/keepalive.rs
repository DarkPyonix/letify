//! Pins SPEC.md letify-link, Paths and the path manager item 5: keepalive and dead path
//! detection.

mod common;

use std::time::Duration;

use common::*;
use letify_link::{Config, Identity, PeerPolicy, Session};

fn with_interval(ms: u64) -> Config {
    Config {
        keepalive_interval: Duration::from_millis(ms),
        ..fast()
    }
}

/// An initiator with interval `a_ms` whose reads can be stalled, and a responder with `b_ms`.
async fn mixed_pair(a_ms: u64, b_ms: u64) -> (Session, Session, Gate) {
    let a = Identity::generate();
    let b = Identity::generate();
    let (ta, tb) = duplex_pair();
    let (ta, gate) = Gated::new(ta);
    let responder = tokio::spawn({
        let (b, a_id) = (b.clone(), a.node_id());
        async move { Session::accept(tb, &b, PeerPolicy::allow(vec![a_id]), with_interval(b_ms)).await }
    });
    let initiator = Session::connect(ta, &a, b.node_id(), with_interval(a_ms))
        .await
        .unwrap();
    (initiator, responder.await.unwrap().unwrap(), gate)
}

fn fast() -> Config {
    Config {
        keepalive_interval: Duration::from_millis(50),
        resume_window: Duration::from_secs(60),
        ..Config::default()
    }
}

async fn gated_pair() -> (Session, Session, Gate) {
    let a = Identity::generate();
    let b = Identity::generate();
    let (ta, tb) = duplex_pair();
    let (ta, gate) = Gated::new(ta);
    let responder = tokio::spawn({
        let (b, a_id) = (b.clone(), a.node_id());
        async move { Session::accept(tb, &b, PeerPolicy::allow(vec![a_id]), fast()).await }
    });
    let initiator = Session::connect(ta, &a, b.node_id(), fast()).await.unwrap();
    (initiator, responder.await.unwrap().unwrap(), gate)
}

#[tokio::test(flavor = "multi_thread")]
async fn an_idle_subflow_stays_alive_on_keepalives() {
    let (a, b, _gate) = gated_pair().await;
    // Six intervals pass with no application traffic; each side sees records all along.
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(a.subflow_count(), 1);
    assert_eq!(b.subflow_count(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_dead_subflow_is_detected_by_missed_keepalives() {
    let (a, b, gate) = gated_pair().await;
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(a.subflow_count(), 1);
    // The transport stays open but nothing reaches `a` any more.
    gate.close();
    tokio::time::timeout(Duration::from_secs(2), async {
        while a.subflow_count() > 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("three missed intervals declare the subflow dead");
    drop(b);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_pong_gives_each_subflow_a_round_trip_estimate() {
    let (a, _b, _gate) = gated_pair().await;
    tokio::time::sleep(Duration::from_millis(200)).await;
    let stats = a.subflow_stats();
    assert_eq!(stats.len(), 1);
    let rtt = stats[0].rtt.expect("a pong arrived");
    assert!(rtt < Duration::from_millis(500), "rtt {rtt:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_zero_interval_turns_keepalives_off() {
    let cfg = Config {
        keepalive_interval: Duration::ZERO,
        ..fast()
    };
    let (a, _b) = session_pair(cfg).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(a.subflow_count(), 1);
    assert!(a.subflow_stats()[0].rtt.is_none());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_peer_with_a_longer_keepalive_interval_is_not_declared_dead() {
    // The peer pings every 400 ms, so the dead time of both sides is 3 x 400 ms = 1200 ms,
    // not 3 x 50 ms = 150 ms.
    let (a, b, gate) = mixed_pair(50, 400).await;
    gate.close();
    tokio::time::sleep(Duration::from_millis(600)).await;
    assert_eq!(
        a.subflow_count(),
        1,
        "declared dead after less than 1200 ms"
    );
    tokio::time::timeout(Duration::from_secs(4), async {
        while a.subflow_count() > 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("declared dead after 3 x the longer interval");
    drop(b);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_peer_with_keepalives_off_leaves_the_dead_time_to_our_interval() {
    let (a, b, gate) = mixed_pair(50, 0).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(a.subflow_count(), 1);
    gate.close();
    tokio::time::timeout(Duration::from_secs(2), async {
        while a.subflow_count() > 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("3 x our own interval applies when the peer advertises 0");
    drop(b);
}

#[tokio::test(flavor = "multi_thread")]
async fn when_both_sides_advertise_zero_only_transport_errors_kill_a_subflow() {
    let (a, b, gate) = mixed_pair(0, 0).await;
    gate.close();
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert_eq!(a.subflow_count(), 1);
    assert_eq!(b.subflow_count(), 1);
}
