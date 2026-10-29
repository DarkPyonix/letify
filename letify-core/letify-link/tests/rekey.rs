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
    assert_eq!(b.dropped_records(), 0);
    // The other direction rotates by itself.
    let mut tx = b.open_stream().await.unwrap();
    tx.write_all(b"back").await.unwrap();
    let mut rx = a.accept_stream().await.unwrap();
    let mut four = [0u8; 4];
    rx.read_exact(&mut four).await.unwrap();
    assert_eq!(&four, b"back");
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

#[tokio::test(flavor = "multi_thread")]
async fn records_in_flight_across_a_rekey_are_accepted() {
    // The initiator seals 4 records per epoch. The attacker on the path holds every record,
    // so they cross two rekeys, then delivers them newest first.
    let (a, b, mitm) = session_pair_through_mitm(tiny_limit(4)).await;
    mitm.withhold.store(true, Ordering::SeqCst);
    let mut s = a.open_stream().await.unwrap();
    for part in [
        b"aa", b"bb", b"cc", b"dd", b"ee", b"ff", b"gg", b"hh", b"ii",
    ] {
        s.write_all(part).await.unwrap();
        sleep(Duration::from_millis(30)).await;
    }
    let held = mitm.recorded();
    assert!(held.len() >= 10, "{} records", held.len());
    assert!(a.send_epoch() >= 2, "epoch {}", a.send_epoch());
    for record in held.iter().rev() {
        mitm.inject(record).await;
    }
    let mut sb = timeout(Duration::from_secs(5), b.accept_stream())
        .await
        .unwrap()
        .unwrap();
    let mut got = [0u8; 18];
    timeout(Duration::from_secs(5), sb.read_exact(&mut got))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(&got, b"aabbccddeeffgghhii");
    assert_eq!(b.dropped_records(), 0);
    assert_eq!(b.receive_epoch(), a.send_epoch());
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
