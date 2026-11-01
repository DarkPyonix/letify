//! Pins SPEC.md letify-link, Sidecar interface item 10: the diagnostic trace of a live session.

mod common;

use std::time::{Duration, Instant};

use common::*;
use letify_link::Config;
use letify_link::trace::{TraceConfig, spawn};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[tokio::test]
async fn a_traced_session_appends_a_line_with_its_subflows_while_it_carries_data() {
    let (a, b) = session_pair(Config::default()).await;
    let dir = std::env::temp_dir().join(format!("ll-trace-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("trace.jsonl");
    let _ = std::fs::remove_file(&path);
    let config = TraceConfig::parse(path.to_str(), Some("50"), None).unwrap();
    let task = spawn(config, 7, a.downgrade(), Instant::now());

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
