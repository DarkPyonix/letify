//! Pins SPEC.md letify-link, Session item 7: stream scheduling. Small messages are not queued
//! behind a bulk stream, bulk streams share a link fairly, and fairness costs bulk throughput
//! at most 10 percent. Everything runs over real loopback TCP behind a throttling proxy.

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use common::*;
use letify_link::{Config, Priority, Session};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::time::Instant;

const MIB: f64 = 1024.0 * 1024.0;

/// The peer: streams that start with 0 are read and counted in `counters[n]` where n is the
/// second byte, and streams that start with 1 are echoed.
fn spawn_server(b: Session, counters: Vec<Arc<AtomicU64>>) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        while let Ok(mut s) = b.accept_stream().await {
            let counters = counters.clone();
            tokio::spawn(async move {
                let mut head = [0u8; 2];
                if s.read_exact(&mut head).await.is_err() {
                    return;
                }
                if head[0] == 0 {
                    let mut buf = vec![0u8; 256 << 10];
                    while let Ok(n) = s.read(&mut buf).await {
                        if n == 0 {
                            break;
                        }
                        counters[head[1] as usize].fetch_add(n as u64, Ordering::Relaxed);
                    }
                } else {
                    let mut msg = [0u8; 64];
                    while s.read_exact(&mut msg).await.is_ok() {
                        if s.write_all(&msg).await.is_err() {
                            break;
                        }
                    }
                }
            });
        }
    })
}

/// A bulk stream tagged `tag`, which the server counts.
async fn bulk_tagged(a: &Session, tag: u8, stop: Arc<AtomicBool>) -> tokio::task::JoinHandle<()> {
    let mut s = a.open_stream_with(Priority::Bulk).await.unwrap();
    tokio::spawn(async move {
        let _ = s.write_all(&[0, tag]).await;
        let chunk = vec![0x33u8; 64 << 10];
        while !stop.load(Ordering::Relaxed) {
            if s.write_all(&chunk).await.is_err() {
                break;
            }
        }
    })
}

#[tokio::test(flavor = "multi_thread")]
async fn a_small_message_is_not_queued_behind_a_bulk_stream() {
    let rate = 8.0 * MIB;
    let (a, b, _p) = throttled_session(&[rate], Config::default(), Config::default()).await;
    let counters = vec![Arc::new(AtomicU64::new(0))];
    let server = spawn_server(b, counters.clone());
    let stop = Arc::new(AtomicBool::new(false));
    let bulk = bulk_tagged(&a, 0, stop.clone()).await;
    let mut s = a.open_stream().await.unwrap();
    s.write_all(&[1, 0]).await.unwrap();
    // Let the bulk stream saturate the link before measuring.
    tokio::time::sleep(Duration::from_secs(1)).await;
    let mut rtts = Vec::new();
    let end = Instant::now() + Duration::from_secs(8);
    let mut msg = [7u8; 64];
    while Instant::now() < end {
        let t = Instant::now();
        s.write_all(&msg).await.unwrap();
        s.read_exact(&mut msg).await.unwrap();
        rtts.push(t.elapsed());
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    stop.store(true, Ordering::Relaxed);
    bulk.abort();
    server.abort();
    rtts.sort();
    let p99 = rtts[((rtts.len() - 1) as f64 * 0.99).round() as usize];
    let p50 = rtts[rtts.len() / 2];
    eprintln!("small message under a saturating bulk stream: p50 {p50:?}, p99 {p99:?}");
    // 4 x R + one frame time + margin, as the spec states: 80 ms + 6 ms + 14 ms.
    assert!(
        p99 < Duration::from_millis(100),
        "p99 {p99:?} with p50 {p50:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn bulk_streams_share_bandwidth_fairly() {
    let rate = 8.0 * MIB;
    let (a, b, _p) = throttled_session(&[rate], Config::default(), Config::default()).await;
    let counters: Vec<_> = (0..2).map(|_| Arc::new(AtomicU64::new(0))).collect();
    let server = spawn_server(b, counters.clone());
    let stop = Arc::new(AtomicBool::new(false));
    let t0 = bulk_tagged(&a, 0, stop.clone()).await;
    let t1 = bulk_tagged(&a, 1, stop.clone()).await;
    tokio::time::sleep(Duration::from_secs(1)).await;
    let before: Vec<u64> = counters.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    tokio::time::sleep(Duration::from_secs(5)).await;
    let got: Vec<f64> = counters
        .iter()
        .zip(&before)
        .map(|(c, b)| (c.load(Ordering::Relaxed) - b) as f64 / 5.0 / MIB)
        .collect();
    stop.store(true, Ordering::Relaxed);
    t0.abort();
    t1.abort();
    server.abort();
    eprintln!("two bulk streams: {:.2} and {:.2} MiB/s", got[0], got[1]);
    let (hi, lo) = (got[0].max(got[1]), got[0].min(got[1]));
    assert!(lo > 0.0 && (hi - lo) / hi < 0.2, "{got:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn bulk_throughput_is_not_reduced_by_more_than_10_percent_by_fairness() {
    let rate = 16.0 * MIB;
    let (a, b, _p) = throttled_session(&[rate], Config::default(), Config::default()).await;
    let counters = vec![Arc::new(AtomicU64::new(0))];
    let server = spawn_server(b, counters.clone());
    let stop = Arc::new(AtomicBool::new(false));
    let bulk = bulk_tagged(&a, 0, stop.clone()).await;
    // An interactive stream that is open and idle must not cost the bulk stream anything.
    let mut idle = a.open_stream().await.unwrap();
    idle.write_all(&[1, 0]).await.unwrap();
    tokio::time::sleep(Duration::from_secs(2)).await;
    let before = counters[0].load(Ordering::Relaxed);
    tokio::time::sleep(Duration::from_secs(5)).await;
    let got = (counters[0].load(Ordering::Relaxed) - before) as f64 / 5.0 / MIB;
    stop.store(true, Ordering::Relaxed);
    bulk.abort();
    server.abort();
    eprintln!(
        "bulk stream: {got:.2} MiB/s of a {:.0} MiB/s link",
        rate / MIB
    );
    assert!(got >= 0.9 * rate / MIB, "{got:.2} MiB/s");
}
