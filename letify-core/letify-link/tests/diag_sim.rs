//! Diagnostic reproduction (SPEC.md letify-link, Session item 4): one fast subflow and four
//! slow ones behind queues, 180 ms round trip. Prints the windows and the rate.
mod common;
use common::*;
use letify_link::Config;
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const MIB: f64 = 1048576.0;

fn cfg() -> Config {
    let mut c = Config::default();
    if let Some(m) = std::env::var("SIM_UNACKED_MIB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        c.max_unacked_bytes = m << 20;
        c.max_reorder_bytes = (2 * m) << 20;
        c.stream_window = ((2 * m) << 20) as u32;
    }
    c
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn window_collapse() {
    let rates: Vec<f64> = std::env::var("SIM_RATES")
        .unwrap_or("20,3,3,3,3".into())
        .split(',')
        .map(|r| r.parse::<f64>().unwrap() * MIB)
        .collect();
    let rtt_ms: u64 = std::env::var("SIM_RTT_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(180);
    let (a, b, _p) = delayed_session(&rates, Duration::from_millis(rtt_ms / 2), cfg(), cfg()).await;
    let total = 128usize;
    let mut s = if std::env::var("SIM_BULK").is_ok() {
        a.open_stream_with(letify_link::Priority::Bulk)
            .await
            .unwrap()
    } else {
        a.open_stream().await.unwrap()
    };
    let blk = pattern(1 << 20, 1);
    let w = tokio::spawn(async move {
        for _ in 0..total {
            s.write_all(&blk).await.unwrap();
        }
        s.shutdown().await.unwrap();
    });
    let mut sb = b.accept_stream().await.unwrap();
    let start = Instant::now();
    let mon = tokio::spawn({
        let a = a.downgrade();
        async move {
            loop {
                tokio::time::sleep(Duration::from_secs(2)).await;
                let Some(a) = a.upgrade() else { return };
                let f = a.flow_snapshot();
                let st: Vec<String> = a
                    .subflow_stats()
                    .iter()
                    .map(|s| format!("{}K/{}K", s.window >> 10, s.inflight_bytes >> 10))
                    .collect();
                eprintln!(
                    "t{:>5.1}s unacked {}K reinj {} windows/inflight {:?}",
                    start.elapsed().as_secs_f64(),
                    f.unacked_bytes >> 10,
                    f.reinjected,
                    st
                );
            }
        }
    });
    let mut buf = vec![0u8; 1 << 20];
    for _ in 0..total {
        sb.read_exact(&mut buf).await.unwrap();
    }
    let secs = start.elapsed().as_secs_f64();
    w.await.unwrap();
    mon.abort();
    eprintln!(
        "RATE {:.1} MiB/s of {:.1} sum; fastest alone {:.1}; reinjected {}",
        total as f64 / secs,
        rates.iter().sum::<f64>() / MIB,
        rates[0] / MIB,
        a.reinjected_frames()
    );
}
