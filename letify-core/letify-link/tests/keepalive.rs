//! Pins SPEC.md letify-link, Paths and the path manager item 5: keepalive and dead path
//! detection.

mod common;

use std::time::Duration;

use common::*;
use letify_link::{Config, Identity, PeerPolicy, Session};

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
