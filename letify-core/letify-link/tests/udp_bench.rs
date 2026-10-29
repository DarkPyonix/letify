//! Measurements of SPEC.md letify-link, Session item 6 (UDP paths) and the M2 head-of-line
//! question. Every test here is ignored; run them in release mode, one at a time:
//!
//! `cargo test --release -p letify-link --test udp_bench -- --ignored --nocapture --test-threads=1`
//!
//! UDP loss and delay come from a userspace proxy. TCP loss cannot be injected without
//! privileges, so the TCP side runs through `TcpModel`, which delays bytes and models the
//! cost of loss. The TCP numbers under loss are therefore a model and not a measurement.

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use common::*;
use letify_link::{Config, Identity, PeerPolicy, Session};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::time::Instant;

fn big() -> Config {
    Config {
        max_unacked_bytes: 64 << 20,
        stream_window: 32 << 20,
        max_reorder_bytes: 64 << 20,
        subflow_window: 16 << 20,
        ..Config::default()
    }
}

/// A session over `n` TCP subflows through the lossy path model.
async fn tcp_session(
    n: usize,
    one_way: Duration,
    loss: f64,
    config: Config,
) -> (Session, Session, TcpModel) {
    let (a_id, b_id) = (Identity::generate(), Identity::generate());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let model = TcpModel::to(listener.local_addr().unwrap(), one_way, loss).await;
    let responder = tokio::spawn({
        let (b_id, a_node, config) = (b_id.clone(), a_id.node_id(), config.clone());
        async move {
            let (tcp, _) = listener.accept().await.unwrap();
            letify_link::tcp::tune(&tcp).unwrap();
            let b = Session::accept(tcp, &b_id, PeerPolicy::allow(vec![a_node]), config)
                .await
                .unwrap();
            for _ in 1..n {
                let (tcp, _) = listener.accept().await.unwrap();
                letify_link::tcp::tune(&tcp).unwrap();
                b.add_subflow(tcp).await.unwrap();
            }
            b
        }
    });
    let connect = || async { letify_link::tcp::connect(model.addr).await.unwrap() };
    let a = Session::connect(connect().await, &a_id, b_id.node_id(), config)
        .await
        .unwrap();
    for _ in 1..n {
        a.add_subflow(connect().await).await.unwrap();
    }
    (a, responder.await.unwrap(), model)
}

/// Bytes per second that a stream from `a` to `b` delivers after a warm-up, in MiB/s.
async fn throughput(a: &Session, b: &Session, warm: Duration, measure: Duration) -> f64 {
    let stop = Arc::new(AtomicBool::new(false));
    let mut tx = a.open_stream().await.unwrap();
    let writer = tokio::spawn({
        let stop = stop.clone();
        async move {
            let chunk = vec![0x5au8; 64 << 10];
            while !stop.load(Ordering::Relaxed) {
                if tx.write_all(&chunk).await.is_err() {
                    break;
                }
            }
        }
    });
    let mut rx = b.accept_stream().await.unwrap();
    let mut buf = vec![0u8; 256 << 10];
    let start = Instant::now();
    let mut counted = 0u64;
    let mut from = None;
    while start.elapsed() < warm + measure {
        let n = rx.read(&mut buf).await.unwrap();
        if n == 0 {
            break;
        }
        if start.elapsed() >= warm {
            from.get_or_insert(Instant::now());
            counted += n as u64;
        }
    }
    let secs = from.map_or(1.0, |t| t.elapsed().as_secs_f64());
    stop.store(true, Ordering::Relaxed);
    writer.abort();
    counted as f64 / secs / 1048576.0
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "a measurement, about 60 s"]
async fn bench_throughput_udp_vs_tcp_with_100ms_rtt_and_1_percent_loss() {
    let one_way = Duration::from_millis(50);
    let (warm, measure) = (Duration::from_secs(3), Duration::from_secs(10));
    for loss in [0.0, 0.01] {
        let (a, b, _model) = tcp_session(1, one_way, loss, big()).await;
        let tcp = throughput(&a, &b, warm, measure).await;
        println!(
            "RESULT throughput, rtt 100 ms, loss {:>4.1}%: 1 TCP subflow (modelled) {tcp:6.2} MiB/s",
            loss * 100.0
        );
        let (a, b, _proxy) = udp_session(big(), one_way, loss, None).await;
        let udp = throughput(&a, &b, warm, measure).await;
        println!(
            "RESULT throughput, rtt 100 ms, loss {:>4.1}%: 1 UDP path (quinn, BBR)  {udp:6.2} MiB/s",
            loss * 100.0
        );
    }
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

/// 32 echo streams send a small message every 50 ms while one stream sends bulk data.
/// Returns the round trip times of the small messages, one list per stream.
async fn latencies(
    a: Arc<Session>,
    b: Arc<Session>,
    run: Duration,
    with_bulk: bool,
) -> Vec<Vec<Duration>> {
    const STREAMS: usize = 32;
    // The server echoes every stream that starts with tag 1 and discards the one with tag 0.
    let server = tokio::spawn(async move {
        while let Ok(mut s) = b.accept_stream().await {
            tokio::spawn(async move {
                let mut tag = [0u8; 1];
                if s.read_exact(&mut tag).await.is_err() {
                    return;
                }
                let mut buf = vec![0u8; 64 << 10];
                if tag[0] == 0 {
                    while matches!(s.read(&mut buf).await, Ok(n) if n > 0) {}
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
    });
    let stop = Arc::new(AtomicBool::new(false));
    let sent = Arc::new(AtomicU64::new(0));
    let mut bulk = a
        .open_stream_with(letify_link::Priority::Bulk)
        .await
        .unwrap();
    let bulk_task = tokio::spawn({
        let (stop, sent) = (stop.clone(), sent.clone());
        async move {
            if !with_bulk {
                return;
            }
            let _ = bulk.write_all(&[0]).await;
            let chunk = vec![0x33u8; 64 << 10];
            while !stop.load(Ordering::Relaxed) {
                if bulk.write_all(&chunk).await.is_err() {
                    break;
                }
                sent.fetch_add(chunk.len() as u64, Ordering::Relaxed);
            }
        }
    });
    let started = Instant::now();
    let mut tasks = Vec::new();
    for i in 0..STREAMS {
        let mut s = a.open_stream().await.unwrap();
        let stop = stop.clone();
        tasks.push(tokio::spawn(async move {
            let _ = s.write_all(&[1]).await;
            let mut out = Vec::new();
            // Stagger the streams over one period so the messages do not leave together.
            tokio::time::sleep(Duration::from_millis(50 * i as u64 / STREAMS as u64)).await;
            let mut msg = [7u8; 64];
            while !stop.load(Ordering::Relaxed) {
                let t = Instant::now();
                if s.write_all(&msg).await.is_err() || s.read_exact(&mut msg).await.is_err() {
                    break;
                }
                out.push(t.elapsed());
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            out
        }));
    }
    tokio::time::sleep(run).await;
    for st in a.subflow_stats() {
        println!(
            "    subflow {}: cap {} KiB, rate {:?} B/s, srtt {:?}",
            st.id,
            st.bulk_cap >> 10,
            st.delivery_rate.map(|r| r as u64),
            st.srtt
        );
    }
    stop.store(true, Ordering::Relaxed);
    let mut all = Vec::new();
    for t in tasks {
        // A message in flight when the run ends may take a while on a bad path.
        if let Ok(Ok(v)) = tokio::time::timeout(Duration::from_secs(30), t).await {
            all.push(v);
        }
    }
    println!(
        "    bulk stream delivered {:.1} MiB in {:.1} s",
        sent.load(Ordering::Relaxed) as f64 / 1048576.0,
        started.elapsed().as_secs_f64()
    );
    bulk_task.abort();
    server.abort();
    all
}

fn report(label: &str, per_stream: Vec<Vec<Duration>>) {
    let mut all: Vec<Duration> = per_stream.iter().flatten().copied().collect();
    all.sort();
    let mut p99s: Vec<Duration> = per_stream
        .iter()
        .filter(|v| !v.is_empty())
        .map(|v| {
            let mut v = v.clone();
            v.sort();
            percentile(&v, 0.99)
        })
        .collect();
    p99s.sort();
    println!(
        "RESULT {label}: {} messages, p50 {:?}, p99 {:?}, worst stream p99 {:?}, median stream p99 {:?}",
        all.len(),
        percentile(&all, 0.5),
        percentile(&all, 0.99),
        p99s.last().unwrap(),
        p99s[p99s.len() / 2],
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "a measurement, about 2 min"]
async fn bench_head_of_line_32_streams_at_2_percent_loss() {
    let one_way = Duration::from_millis(10);
    let loss = 0.02;
    let run = Duration::from_secs(10);
    for with_bulk in [false, true] {
        println!(
            "32 streams, rtt 20 ms, loss 2%, {}",
            if with_bulk {
                "one bulk stream in the background"
            } else {
                "no background transfer"
            }
        );
        // The default session settings, so the queue behind the bulk stream is the one a
        // user gets.
        let (a, b, _model) = tcp_session(4, one_way, loss, Config::default()).await;
        let r = latencies(Arc::new(a), Arc::new(b), run, with_bulk).await;
        report("4 TCP subflows (modelled)", r);
        let rate = Some(20.0 * 1048576.0);
        let (a, b, _proxy) = udp_session(Config::default(), one_way, loss, rate).await;
        let r = latencies(Arc::new(a), Arc::new(b), run, with_bulk).await;
        report("1 UDP path (quinn, BBR)   ", r);
    }
}

/// The two bulk cases of the head-of-line bench alone: one bulk stream beside 32 small ones,
/// over 4 modelled TCP subflows and over one UDP path, both at 2% loss and 20 ms round trip.
/// A shorter run than the full bench, for tuning the bulk cap.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "a measurement, about 30 s"]
async fn bench_bulk_beside_32_streams_at_2_percent_loss() {
    let one_way = Duration::from_millis(10);
    let run = Duration::from_secs(10);
    let (a, b, _model) = tcp_session(4, one_way, 0.02, Config::default()).await;
    let r = latencies(Arc::new(a), Arc::new(b), run, true).await;
    report("4 TCP subflows (modelled)", r);
    let rate = Some(20.0 * 1048576.0);
    let (a, b, _proxy) = udp_session(Config::default(), one_way, 0.02, rate).await;
    let r = latencies(Arc::new(a), Arc::new(b), run, true).await;
    report("1 UDP path (quinn, BBR)   ", r);
}
