//! Pins SPEC.md letify-link, Session items 2 to 4: streams, sequencing and subflows.

mod common;

use std::time::Duration;

use common::*;
use letify_link::{Config, Identity, PeerPolicy, Session};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::time::timeout;

#[tokio::test(flavor = "multi_thread")]
async fn many_streams_on_one_session_deliver_in_order_independently() {
    let (a, b) = session_pair(Config::default()).await;
    let streams = 16u64;
    let size = 300_000usize;
    let receiver = tokio::spawn(async move {
        let mut tasks = Vec::new();
        for _ in 0..streams {
            let mut s = b.accept_stream().await.unwrap();
            tasks.push(tokio::spawn(async move {
                let mut all = Vec::new();
                s.read_to_end(&mut all).await.unwrap();
                all
            }));
        }
        let mut got = Vec::new();
        for t in tasks {
            got.push(t.await.unwrap());
        }
        (b, got)
    });
    let mut writers = Vec::new();
    for i in 0..streams {
        let mut s = a.open_stream().await.unwrap();
        writers.push(tokio::spawn(async move {
            s.write_all(&pattern(size, i + 1)).await.unwrap();
            s.shutdown().await.unwrap();
        }));
    }
    for w in writers {
        w.await.unwrap();
    }
    let (_b, got) = timeout(Duration::from_secs(30), receiver)
        .await
        .unwrap()
        .unwrap();
    // Streams are accepted in the order they were opened, so stream i carries pattern i + 1.
    for (i, bytes) in got.iter().enumerate() {
        assert_eq!(bytes.len(), size);
        assert!(*bytes == pattern(size, i as u64 + 1), "stream {i} differs");
    }
}

#[tokio::test]
async fn a_stream_close_is_separate_from_the_session_close() {
    let (a, b) = session_pair(Config::default()).await;
    let mut first = a.open_stream().await.unwrap();
    first.write_all(b"done").await.unwrap();
    first.shutdown().await.unwrap();
    let mut first_b = b.accept_stream().await.unwrap();
    let mut text = Vec::new();
    first_b.read_to_end(&mut text).await.unwrap();
    assert_eq!(text, b"done");
    // The session is still usable after one stream ended.
    let mut second = a.open_stream().await.unwrap();
    second.write_all(b"more").await.unwrap();
    let mut second_b = b.accept_stream().await.unwrap();
    let mut buf = [0u8; 4];
    second_b.read_exact(&mut buf).await.unwrap();
    assert_eq!(&buf, b"more");
}

#[tokio::test]
async fn either_side_can_open_a_stream() {
    let (a, b) = session_pair(Config::default()).await;
    let mut from_b = b.open_stream().await.unwrap();
    from_b.write_all(b"hi").await.unwrap();
    let mut at_a = a.accept_stream().await.unwrap();
    let mut buf = [0u8; 2];
    at_a.read_exact(&mut buf).await.unwrap();
    assert_eq!(&buf, b"hi");
}

#[tokio::test]
async fn a_closed_session_ends_the_peer_with_a_graceful_end() {
    let (a, b) = session_pair(Config::default()).await;
    let mut s = a.open_stream().await.unwrap();
    s.write_all(b"last").await.unwrap();
    let mut sb = b.accept_stream().await.unwrap();
    a.close().await;
    let mut all = Vec::new();
    sb.read_to_end(&mut all).await.unwrap();
    assert_eq!(all, b"last");
    assert!(b.accept_stream().await.is_err());
}

#[tokio::test]
async fn a_stream_writer_waits_for_window_when_the_reader_is_slow() {
    let config = Config {
        stream_window: 64 * 1024,
        ..Config::default()
    };
    let (a, b) = session_pair(config).await;
    let mut s = a.open_stream().await.unwrap();
    let data = pattern(1 << 20, 9);
    let to_send = data.clone();
    let writer = tokio::spawn(async move {
        s.write_all(&to_send).await.unwrap();
        s.shutdown().await.unwrap();
    });
    let mut sb = b.accept_stream().await.unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(
        !writer.is_finished(),
        "a 1 MiB write must wait for a 64 KiB window"
    );
    let mut all = Vec::new();
    sb.read_to_end(&mut all).await.unwrap();
    writer.await.unwrap();
    assert!(all == data);
}

/// Two sessions joined by `n` in-memory subflows.
async fn session_pair_with_subflows(n: usize, config: Config) -> (Session, Session) {
    let (a, b) = session_pair(config).await;
    for _ in 1..n {
        let (ta, tb) = duplex_pair();
        let (ra, rb) = tokio::join!(a.add_subflow(ta), b.add_subflow(tb));
        ra.unwrap();
        rb.unwrap();
    }
    (a, b)
}

#[tokio::test(flavor = "multi_thread")]
async fn frames_spread_over_several_subflows_arrive_in_order() {
    let config = Config {
        subflow_window: 128 * 1024,
        ..Config::default()
    };
    let (a, b) = session_pair_with_subflows(4, config).await;
    assert_eq!(a.subflow_count(), 4);
    let data = pattern(16 << 20, 3);
    let mut s = a.open_stream().await.unwrap();
    let to_send = data.clone();
    let writer = tokio::spawn(async move {
        s.write_all(&to_send).await.unwrap();
        s.shutdown().await.unwrap();
    });
    let mut sb = b.accept_stream().await.unwrap();
    let mut all = Vec::new();
    timeout(Duration::from_secs(60), sb.read_to_end(&mut all))
        .await
        .unwrap()
        .unwrap();
    writer.await.unwrap();
    assert!(all == data);
    let used = a
        .subflow_sent_bytes()
        .iter()
        .filter(|b| **b > 1_000_000)
        .count();
    assert!(
        used >= 2,
        "frames used {used} subflows: {:?}",
        a.subflow_sent_bytes()
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn the_unacknowledged_buffer_is_bounded_in_bytes() {
    let bound = 256 * 1024;
    let config = Config {
        max_unacked_bytes: bound,
        stream_window: 32 << 20,
        ..Config::default()
    };
    let a_id = Identity::generate();
    let b_id = Identity::generate();
    let (ta, tb) = duplex_pair();
    let (tb, gate) = Gated::new(tb);
    let (bb, a_node, cfg) = (b_id.clone(), a_id.node_id(), config.clone());
    let responder = tokio::spawn(async move {
        Session::accept(tb, &bb, PeerPolicy::allow(vec![a_node]), cfg).await
    });
    let a = Session::connect(ta, &a_id, b_id.node_id(), config)
        .await
        .unwrap();
    let b = responder.await.unwrap().unwrap();
    // From here the responder reads nothing, so nothing is acknowledged.
    gate.close();
    let data = pattern(4 << 20, 5);
    let mut s = a.open_stream().await.unwrap();
    let writer = tokio::spawn({
        let data = data.clone();
        async move {
            s.write_all(&data).await.unwrap();
            s.shutdown().await.unwrap();
        }
    });
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert!(
        !writer.is_finished(),
        "the writer must block when the buffer is full"
    );
    let held = a.unacked_bytes();
    assert!(
        held <= bound,
        "unacked {held} bytes exceeds the bound {bound}"
    );
    assert!(
        held > bound / 2,
        "unacked {held} bytes: the buffer should be nearly full"
    );
    // Acknowledgements free the buffer and the writer finishes.
    gate.open();
    let mut sb = b.accept_stream().await.unwrap();
    let mut all = Vec::new();
    timeout(Duration::from_secs(30), sb.read_to_end(&mut all))
        .await
        .unwrap()
        .unwrap();
    writer.await.unwrap();
    assert!(all == data);
}
