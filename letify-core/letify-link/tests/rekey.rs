//! Pins SPEC.md letify-link, Session item 8 (Rekey): the keys of a direction rotate after a
//! record count or an age, records of an older epoch still open, and a session resumes across
//! epochs.

mod common;

use std::sync::atomic::Ordering;
use std::time::Duration;

use common::*;
use letify_link::Config;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::time::{sleep, timeout};

fn tiny_limit(records: u64) -> Config {
    Config {
        rekey_after_records: records,
        ..Config::default()
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_session_rekeys_after_the_record_limit() {
    let (a, b) = session_pair(tiny_limit(16)).await;
    assert_eq!((a.send_epoch(), b.receive_epoch()), (0, 0));
    let data = pattern(3 << 20, 5);
    let mut tx = a.open_stream().await.unwrap();
    let writer = tokio::spawn({
        let data = data.clone();
        async move {
            tx.write_all(&data).await.unwrap();
            tx.shutdown().await.unwrap();
        }
    });
    let mut rx = b.accept_stream().await.unwrap();
    let mut got = Vec::new();
    timeout(Duration::from_secs(30), rx.read_to_end(&mut got))
        .await
        .unwrap()
        .unwrap();
    writer.await.unwrap();
    assert!(got == data, "bytes differ across the rekeys");
    assert!(a.send_epoch() > 0, "the sender never rekeyed");
    assert!(b.receive_epoch() > 0, "the receiver never saw a new epoch");
    assert_eq!(
        b.dropped_records(),
        0,
        "one subflow loses no record to a rekey"
    );
    // The other direction rotates by itself, and does not move the first.
    let before = a.send_epoch();
    let back = pattern(1 << 20, 6);
    let mut tx = b.open_stream().await.unwrap();
    let writer = tokio::spawn({
        let back = back.clone();
        async move {
            tx.write_all(&back).await.unwrap();
            tx.shutdown().await.unwrap();
        }
    });
    let mut rx = a.accept_stream().await.unwrap();
    let mut got = Vec::new();
    timeout(Duration::from_secs(30), rx.read_to_end(&mut got))
        .await
        .unwrap()
        .unwrap();
    writer.await.unwrap();
    assert!(got == back);
    assert!(b.send_epoch() > 0 && a.receive_epoch() > 0);
    assert!(a.send_epoch() >= before);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_session_rekeys_after_the_age_limit() {
    let config = Config {
        rekey_after: Duration::from_millis(200),
        ..Config::default()
    };
    let (a, b) = session_pair(config).await;
    let mut tx = a.open_stream().await.unwrap();
    let mut rx = {
        tx.write_all(b"1").await.unwrap();
        b.accept_stream().await.unwrap()
    };
    let mut one = [0u8; 1];
    rx.read_exact(&mut one).await.unwrap();
    assert_eq!(a.send_epoch(), 0);
    sleep(Duration::from_millis(300)).await;
    tx.write_all(b"2").await.unwrap();
    rx.read_exact(&mut one).await.unwrap();
    assert_eq!(&one, b"2");
    assert_eq!(a.send_epoch(), 1);
    assert_eq!(b.receive_epoch(), 1);
}

/// Add `n` more in-memory subflows to a running pair.
async fn add_subflows(a: &letify_link::Session, b: &letify_link::Session, n: usize) {
    for _ in 0..n {
        let (ta, tb) = duplex_pair();
        let (ra, rb) = tokio::join!(a.add_subflow(ta), b.add_subflow(tb));
        ra.unwrap();
        rb.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn records_in_flight_across_a_rekey_are_recovered_by_retransmission() {
    // Four subflows and a limit of 16 records: records of the old epoch are still in flight on
    // some subflows when another subflow carries the first record of the new one. The receiver
    // drops those, and the session resends their frames in the new epoch.
    let (a, b) = session_pair(tiny_limit(16)).await;
    add_subflows(&a, &b, 3).await;
    let data = pattern(8 << 20, 3);
    let mut tx = a.open_stream().await.unwrap();
    let writer = tokio::spawn({
        let data = data.clone();
        async move {
            tx.write_all(&data).await.unwrap();
            tx.shutdown().await.unwrap();
        }
    });
    let mut rx = b.accept_stream().await.unwrap();
    let mut got = Vec::new();
    timeout(Duration::from_secs(60), rx.read_to_end(&mut got))
        .await
        .unwrap()
        .unwrap();
    writer.await.unwrap();
    assert!(got == data, "bytes were lost across the rekeys");
    assert!(a.send_epoch() >= 4, "epoch {}", a.send_epoch());
    assert!(b.receive_epoch() >= 4, "epoch {}", b.receive_epoch());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_forged_epoch_header_does_not_switch_keys() {
    let (a, b, mitm) = session_pair_through_mitm(tiny_limit(1000)).await;
    let mut tx = a.open_stream().await.unwrap();
    tx.write_all(b"one").await.unwrap();
    let mut rx = b.accept_stream().await.unwrap();
    let mut three = [0u8; 3];
    rx.read_exact(&mut three).await.unwrap();
    let real = mitm.recorded().pop().expect("the initiator sent a record");
    for epoch in [1u16, 2, 60000] {
        // The same record with its header tagged for another epoch, before any announcement.
        let mut forged = real.clone();
        forged[2..4].copy_from_slice(&epoch.to_be_bytes());
        mitm.inject(&forged).await;
    }
    sleep(Duration::from_millis(200)).await;
    assert_eq!(b.receive_epoch(), 0, "a header alone moved the keys");
    assert_eq!(b.dropped_records(), 3);
    // The session carries on in epoch 0.
    tx.write_all(b"two").await.unwrap();
    timeout(Duration::from_secs(5), rx.read_exact(&mut three))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(&three, b"two");
    assert_eq!(b.dropped_records(), 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_session_resumes_across_a_rekey() {
    // The only subflow runs through an attacker who withholds everything after the handshake,
    // so the receiver sees none of the announcements. The initiator rekeys several times,
    // then a new subflow resumes the session: its Hello carries the announcements, and the
    // frames the old subflow never delivered are sent again.
    let (a, b, mitm) = session_pair_through_mitm(tiny_limit(4)).await;
    mitm.withhold.store(true, Ordering::SeqCst);
    let mut tx = a.open_stream().await.unwrap();
    let mut sent = Vec::new();
    for i in 0..30u8 {
        let part = [i; 50];
        tx.write_all(&part).await.unwrap();
        sent.extend_from_slice(&part);
        sleep(Duration::from_millis(5)).await;
    }
    assert!(a.send_epoch() >= 2, "epoch {}", a.send_epoch());
    assert_eq!(b.receive_epoch(), 0);
    let (ta, tb) = duplex_pair();
    let (old_a, old_b) = (a.subflow_stats()[0].id, b.subflow_stats()[0].id);
    let (ra, rb) = tokio::join!(a.add_subflow(ta), b.add_subflow(tb));
    ra.unwrap();
    rb.unwrap();
    assert_eq!(
        b.receive_epoch(),
        a.send_epoch(),
        "the Hello brought b up to date"
    );
    a.close_subflow(old_a);
    b.close_subflow(old_b);
    let mut rx = timeout(Duration::from_secs(10), b.accept_stream())
        .await
        .unwrap()
        .unwrap();
    let mut got = vec![0u8; sent.len()];
    timeout(Duration::from_secs(10), rx.read_exact(&mut got))
        .await
        .unwrap()
        .unwrap();
    assert!(got == sent);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_session_resumes_across_epochs() {
    let (a, b) = session_pair(tiny_limit(8)).await;
    let mut tx = a.open_stream().await.unwrap();
    let data = pattern(1 << 20, 9);
    tx.write_all(&data[..512 << 10]).await.unwrap();
    let mut rx = b.accept_stream().await.unwrap();
    let mut got = vec![0u8; 512 << 10];
    rx.read_exact(&mut got).await.unwrap();
    assert!(a.send_epoch() > 0);
    // Replace the only subflow: the Hello of the new one is sealed in the current epoch.
    let (ta, tb) = duplex_pair();
    let (ra, rb) = tokio::join!(a.add_subflow(ta), b.add_subflow(tb));
    let (old_a, old_b) = (a.subflow_stats()[0].id, b.subflow_stats()[0].id);
    ra.unwrap();
    rb.unwrap();
    a.close_subflow(old_a);
    b.close_subflow(old_b);
    tx.write_all(&data[512 << 10..]).await.unwrap();
    tx.shutdown().await.unwrap();
    let mut rest = Vec::new();
    timeout(Duration::from_secs(30), rx.read_to_end(&mut rest))
        .await
        .unwrap()
        .unwrap();
    got.extend_from_slice(&rest);
    assert!(got == data, "bytes differ after resuming across epochs");
}
