//! Pins SPEC.md letify-link, Session item 4 (Receive buffer): the reorder buffer is bounded in
//! bytes and the sender honours the bound the peer advertised.

mod common;

use common::*;
use letify_link::Config;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[tokio::test(flavor = "multi_thread")]
async fn the_receive_buffer_stays_within_its_bound() {
    let bound = 1 << 20;
    let server = Config {
        max_reorder_bytes: bound,
        ..Config::default()
    };
    let mib = 1024.0 * 1024.0;
    let (a, b, _p) = throttled_session(
        &[2.0 * mib, 20.0 * mib, 20.0 * mib],
        Config::default(),
        server,
    )
    .await;
    let data = pattern(16 << 20, 7);
    let mut s = a.open_stream().await.unwrap();
    let writer = tokio::spawn({
        let data = data.clone();
        async move {
            s.write_all(&data).await.unwrap();
            s.shutdown().await.unwrap();
        }
    });
    let mut sb = b.accept_stream().await.unwrap();
    let mut all = Vec::new();
    sb.read_to_end(&mut all).await.unwrap();
    writer.await.unwrap();
    assert!(all == data);
    let peak = b.reorder_peak_bytes();
    assert!(peak > 0, "the slow subflow should have caused reordering");
    assert!(
        peak <= bound,
        "the reorder buffer held {peak} bytes, bound {bound}"
    );
    // The sender took the peer's bound as its own limit on unacknowledged bytes.
    assert!(a.unacked_bytes() <= bound);
}
