//! Pins SPEC.md letify-link, Session: a record that was accepted once is never acted on again.

mod common;

use std::sync::atomic::Ordering;
use std::time::Duration;

use common::*;
use letify_link::{Config, REPLAY_WINDOW};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::time::{sleep, timeout};

#[tokio::test]
async fn a_replayed_record_is_dropped() {
    let (a, b, mitm) = session_pair_through_mitm(Config::default()).await;
    let mut s = a.open_stream().await.unwrap();
    s.write_all(b"hello").await.unwrap();
    let mut sb = b.accept_stream().await.unwrap();
    let mut buf = [0u8; 5];
    sb.read_exact(&mut buf).await.unwrap();
    let seen = mitm.recorded();
    assert!(!seen.is_empty());
    for record in &seen {
        mitm.inject(record).await;
    }
    sleep(Duration::from_millis(200)).await;
    assert_eq!(b.dropped_records(), seen.len() as u64);
    // The replay is not a session error: the stream still works.
    s.write_all(b"again").await.unwrap();
    sb.read_exact(&mut buf).await.unwrap();
    assert_eq!(&buf, b"again");
}

#[tokio::test]
async fn a_replayed_stream_close_does_not_end_the_session() {
    let (a, b, mitm) = session_pair_through_mitm(Config::default()).await;
    let mut s = a.open_stream().await.unwrap();
    s.write_all(b"x").await.unwrap();
    s.shutdown().await.unwrap();
    let mut sb = b.accept_stream().await.unwrap();
    let mut all = Vec::new();
    sb.read_to_end(&mut all).await.unwrap();
    for record in mitm.recorded() {
        mitm.inject(&record).await;
    }
    sleep(Duration::from_millis(200)).await;
    let mut next = a.open_stream().await.unwrap();
    next.write_all(b"ok").await.unwrap();
    let mut nb = b.accept_stream().await.unwrap();
    let mut two = [0u8; 2];
    nb.read_exact(&mut two).await.unwrap();
    assert_eq!(&two, b"ok");
    assert!(b.dropped_records() > 0);
}

#[tokio::test]
async fn records_arriving_out_of_order_within_the_window_are_accepted() {
    let (a, b, mitm) = session_pair_through_mitm(Config::default()).await;
    mitm.withhold.store(true, Ordering::SeqCst);
    let mut s = a.open_stream().await.unwrap();
    for part in [b"aa", b"bb", b"cc"] {
        s.write_all(part).await.unwrap();
        sleep(Duration::from_millis(30)).await;
    }
    let held = mitm.recorded();
    assert!(held.len() >= 4);
    for record in held.iter().rev() {
        mitm.inject(record).await;
    }
    let mut sb = timeout(Duration::from_secs(5), b.accept_stream())
        .await
        .unwrap()
        .unwrap();
    let mut got = [0u8; 6];
    timeout(Duration::from_secs(5), sb.read_exact(&mut got))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(&got, b"aabbcc");
    assert_eq!(b.dropped_records(), 0);
}

#[tokio::test]
async fn a_record_older_than_the_window_is_dropped() {
    let (a, b, mitm) = session_pair_through_mitm(Config::default()).await;
    let mut s = a.open_stream().await.unwrap();
    let count = REPLAY_WINDOW as usize + 500;
    for _ in 0..count {
        s.write_all(b"z").await.unwrap();
    }
    let mut sb = b.accept_stream().await.unwrap();
    let mut got = vec![0u8; count];
    timeout(Duration::from_secs(30), sb.read_exact(&mut got))
        .await
        .unwrap()
        .unwrap();
    let seen = mitm.recorded();
    assert!(
        seen.len() > REPLAY_WINDOW as usize + 100,
        "{} records",
        seen.len()
    );
    assert_eq!(b.dropped_records(), 0);
    mitm.inject(&seen[0]).await;
    sleep(Duration::from_millis(200)).await;
    assert_eq!(
        b.dropped_records(),
        1,
        "a record below the window is dropped"
    );
}

#[tokio::test]
async fn a_replayed_resume_hello_is_refused() {
    let (a, b) = session_pair(Config::default()).await;
    // Add a subflow legitimately through an attacker that records the initiator's side.
    let (ta, m_a) = duplex_pair();
    let (m_b, tb) = duplex_pair();
    let mitm = Mitm::new(m_a, m_b, Prefix::Subflow);
    let (ra, rb) = tokio::join!(a.add_subflow(ta), b.add_subflow(tb));
    ra.unwrap();
    rb.unwrap();
    sleep(Duration::from_millis(100)).await;
    let records = mitm.recorded();
    assert_eq!(
        records.len(),
        1,
        "the Hello is the first record on the subflow"
    );
    let mut replay = mitm.prefix.lock().unwrap().clone();
    replay.extend_from_slice(&records[0]);

    // The attacker opens a new connection to b and replays the recorded Hello.
    let (att, victim) = duplex_pair();
    let attacker = tokio::spawn(async move {
        let mut att = att;
        att.write_all(&replay).await.unwrap();
        let mut sink = Vec::new();
        let _ = att.read_to_end(&mut sink).await;
    });
    let before = b.subflow_count();
    let refused = b.add_subflow(victim).await;
    assert!(
        matches!(refused, Err(letify_link::Error::Subflow(_))),
        "{refused:?}"
    );
    assert_eq!(b.subflow_count(), before);
    attacker.abort();
}
