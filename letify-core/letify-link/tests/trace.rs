//! Pins SPEC.md letify-link, Sidecar interface item 10: the diagnostic trace of a live session.

mod common;

use std::time::{Duration, Instant};

use common::*;
use letify_link::Config;
use letify_link::trace::{TraceConfig, spawn};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[tokio::test(start_paused = true)]
async fn a_traced_session_appends_a_line_with_its_subflows_while_it_carries_data() {
    let (a, b) = session_pair(Config::default()).await;
    let dir = std::env::temp_dir().join(format!("ll-trace-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("trace.jsonl");
    let _ = std::fs::remove_file(&path);
    let config = TraceConfig::parse(path.to_str(), Some("50"), None).unwrap();
    let task = spawn(config, 7, a.downgrade(), Instant::now(), None);

    let mut s = a.open_stream().await.unwrap();
    let writer = tokio::spawn(async move {
        s.write_all(&pattern(4 << 20, 3)).await.unwrap();
        s.shutdown().await.unwrap();
    });
    let mut sb = b.accept_stream().await.unwrap();
    let mut got = Vec::new();
    sb.read_to_end(&mut got).await.unwrap();
    writer.await.unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;
    task.abort();

    let text = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<serde_json::Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert!(!lines.is_empty(), "no trace line was written");
    let last = lines.last().unwrap();
    assert_eq!(last["session"], 7);
    let subflows = last["subflows"].as_array().unwrap();
    assert_eq!(subflows.len(), 1);
    assert!(subflows[0]["acked"].as_u64().unwrap() >= 4 << 20);
    std::fs::remove_dir_all(&dir).unwrap();
}

fn read_lines(path: &std::path::Path) -> Vec<serde_json::Value> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn trace_file(name: &str) -> (std::path::PathBuf, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!("ll-trace-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("trace.jsonl");
    let _ = std::fs::remove_file(&path);
    (dir, path)
}

#[tokio::test(start_paused = true)]
async fn a_tcp_like_subflow_line_names_its_kind_and_that_it_has_no_gauge() {
    let (a, b) = session_pair(Config::default()).await;
    let (dir, path) = trace_file("kind");
    let config = TraceConfig::parse(path.to_str(), Some("50"), None).unwrap();
    let task = spawn(config, 1, a.downgrade(), Instant::now(), None);
    let mut s = a.open_stream().await.unwrap();
    let w = tokio::spawn(async move {
        s.write_all(&pattern(1 << 20, 1)).await.unwrap();
        s.shutdown().await.unwrap();
    });
    let mut sb = b.accept_stream().await.unwrap();
    let mut got = Vec::new();
    sb.read_to_end(&mut got).await.unwrap();
    w.await.unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;
    task.abort();
    let lines = read_lines(&path);
    let sub = &lines.last().unwrap()["subflows"][0];
    assert_eq!(sub["kind"], "tcp");
    assert_eq!(sub["gauged"], false);
    assert!(sub["window_limit"].as_u64().unwrap() > 0);
    assert!(sub.get("min_rtt_ms").is_some());
    assert!(sub.get("cwnd").is_none());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test(start_paused = true)]
async fn closing_a_subflow_writes_one_event_line_with_the_reason() {
    let (a, _b) = session_pair(Config::default()).await;
    let (dir, path) = trace_file("closed");
    let config = TraceConfig::parse(path.to_str(), Some("50"), None).unwrap();
    let task = spawn(config, 2, a.downgrade(), Instant::now(), None);
    let id = a.subflow_stats()[0].id;
    a.close_subflow(id);
    tokio::time::sleep(Duration::from_millis(400)).await;
    task.abort();
    let events: Vec<_> = read_lines(&path)
        .into_iter()
        .filter(|l| l["event"] == "subflow_closed")
        .collect();
    assert_eq!(events.len(), 1, "{events:?}");
    assert_eq!(events[0]["subflow"], id);
    assert_eq!(events[0]["session"], 2);
    assert_eq!(events[0]["reason"], "closed by caller");
    assert_eq!(events[0]["kind"], "tcp");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test(start_paused = true)]
async fn a_failed_session_writes_an_event_line_with_the_failure_text() {
    let config = Config {
        resume_window: Duration::from_millis(100),
        ..Config::default()
    };
    let (a, _b) = session_pair(config).await;
    let (dir, path) = trace_file("failed");
    let tc = TraceConfig::parse(path.to_str(), Some("50"), None).unwrap();
    let task = spawn(tc, 3, a.downgrade(), Instant::now(), None);
    a.close_subflow(a.subflow_stats()[0].id);
    tokio::time::sleep(Duration::from_millis(600)).await;
    task.abort();
    let failed: Vec<_> = read_lines(&path)
        .into_iter()
        .filter(|l| l["event"] == "session_failed")
        .collect();
    assert_eq!(failed.len(), 1, "{failed:?}");
    assert!(
        failed[0]["reason"]
            .as_str()
            .unwrap()
            .contains("resume window ended")
    );
    std::fs::remove_dir_all(&dir).unwrap();
}
