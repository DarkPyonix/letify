//! Pins SPEC.md letify-link, Session item 4: subflow acknowledgement, scheduling and
//! reinjection, over real loopback TCP behind throttling proxies.

mod common;

use std::time::{Duration, Instant};

use common::*;
use letify_link::Config;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::time::timeout;

const MIB: f64 = 1024.0 * 1024.0;

fn blocks() -> Vec<Vec<u8>> {
    (0..16).map(|i| pattern(1 << 20, i + 1)).collect()
}

/// Send `total` MiB over the first stream and read it on the other side, checking the bytes.
/// Returns MiB per second measured from the first byte to the last.
async fn transfer(a: &letify_link::Session, b: &letify_link::Session, total: usize) -> f64 {
    let blocks = std::sync::Arc::new(blocks());
    let mut s = a.open_stream().await.unwrap();
    let writer = tokio::spawn({
        let blocks = blocks.clone();
        async move {
            for i in 0..total {
                s.write_all(&blocks[i % 16]).await.unwrap();
            }
            s.shutdown().await.unwrap();
        }
    });
    let mut sb = b.accept_stream().await.unwrap();
    let start = Instant::now();
    let mut buf = vec![0u8; 1 << 20];
    for i in 0..total {
        timeout(Duration::from_secs(120), sb.read_exact(&mut buf))
            .await
            .expect("the transfer finishes")
            .unwrap();
        assert!(buf == blocks[i % 16], "block {i} differs");
    }
    let secs = start.elapsed().as_secs_f64();
    writer.await.unwrap();
    total as f64 / secs
}

async fn measure(label: &str, rates: &[f64], total: usize) -> f64 {
    let (a, b, _p) = throttled_session(rates, Config::default(), Config::default()).await;
    let got = transfer(&a, &b, total).await;
    let sum: f64 = rates.iter().sum::<f64>() / MIB;
    eprintln!("{label}: {got:.1} MiB/s of {sum:.1} MiB/s (sum of the subflows)");
    got / sum
}

#[tokio::test(flavor = "multi_thread")]
async fn aggregate_throughput_is_close_to_the_sum_of_the_subflows_not_the_slowest() {
    let rates = [2.0 * MIB, 20.0 * MIB, 20.0 * MIB, 20.0 * MIB];
    let share = measure("3 fast + 1 slow", &rates, 64).await;
    assert!(share >= 0.7, "aggregate is {:.0}% of the sum", share * 100.0);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "prints reference numbers"]
async fn reference_rates() {
    measure("1 fast", &[20.0 * MIB], 32).await;
    measure("4 fast", &[20.0 * MIB; 4], 64).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_stalled_subflow_does_not_stall_delivery() {
    let rates = [20.0 * MIB; 4];
    let (a, b, proxies) = throttled_session(&rates, Config::default(), Config::default()).await;
    let blocks = std::sync::Arc::new(blocks());
    let total = 400usize;
    let mut s = a.open_stream().await.unwrap();
    let writer = tokio::spawn({
        let blocks = blocks.clone();
        async move {
            for i in 0..total {
                s.write_all(&blocks[i % 16]).await.unwrap();
            }
            s.shutdown().await.unwrap();
        }
    });
    let mut sb = b.accept_stream().await.unwrap();
    let got = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let reader = tokio::spawn({
        let (blocks, got) = (blocks.clone(), got.clone());
        async move {
            let mut buf = vec![0u8; 1 << 20];
            for i in 0..total {
                sb.read_exact(&mut buf).await.unwrap();
                assert!(buf == blocks[i % 16], "block {i} differs");
                got.store(i + 1, std::sync::atomic::Ordering::SeqCst);
            }
        }
    });
    tokio::time::sleep(Duration::from_millis(300)).await;
    let before = got.load(std::sync::atomic::Ordering::SeqCst);
    let sent_before = a.subflow_stats()[0].sent_bytes;
    proxies[0].pause();
    tokio::time::sleep(Duration::from_secs(2)).await;
    let during = got.load(std::sync::atomic::Ordering::SeqCst) - before;
    let sent_during = a.subflow_stats()[0].sent_bytes - sent_before;
    eprintln!("stalled subflow: {during} MiB delivered in the 2 s pause");
    proxies[0].resume();
    // 3 subflows at 20 MiB/s would give about 120 MiB in 2 s. Anything near the 8 MiB of the
    // retransmit buffer would mean delivery waited for the stalled subflow.
    assert!(during >= 40, "only {during} MiB arrived while a subflow was paused");
    // The stalled subflow's window bounds what it was handed.
    assert!(
        sent_during <= 2 * 1024 * 1024 + 128 * 1024,
        "{sent_during} bytes were written to a stalled subflow"
    );
    timeout(Duration::from_secs(60), reader)
        .await
        .expect("delivery finishes after the subflow resumes")
        .unwrap();
    writer.await.unwrap();
}
