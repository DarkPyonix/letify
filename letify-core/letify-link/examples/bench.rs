//! Loopback throughput of one session over 1 and over 4 TCP subflows.
//!
//! Run with `cargo run --release -p letify-link --example bench`.

use std::time::Instant;

use letify_link::{Config, Identity, PeerPolicy, Session};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

const TOTAL: usize = 256 << 20;

async fn run(subflows: usize) -> f64 {
    let a_id = Identity::generate();
    let b_id = Identity::generate();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let config = Config {
        max_unacked_bytes: 32 << 20,
        stream_window: 16 << 20,
        ..Config::default()
    };
    let responder = tokio::spawn({
        let (b_id, a_node, config) = (b_id.clone(), a_id.node_id(), config.clone());
        async move {
            let (tcp, _) = listener.accept().await.unwrap();
            let b = Session::accept(tcp, &b_id, PeerPolicy::allow(vec![a_node]), config)
                .await
                .unwrap();
            for _ in 1..subflows {
                let (tcp, _) = listener.accept().await.unwrap();
                b.add_subflow(tcp).await.unwrap();
            }
            b
        }
    });
    let a = Session::connect(
        TcpStream::connect(addr).await.unwrap(),
        &a_id,
        b_id.node_id(),
        config,
    )
    .await
    .unwrap();
    for _ in 1..subflows {
        a.add_subflow(TcpStream::connect(addr).await.unwrap())
            .await
            .unwrap();
    }
    let b = responder.await.unwrap();

    let start = Instant::now();
    let mut s = a.open_stream().await.unwrap();
    let writer = tokio::spawn(async move {
        let chunk = vec![0x5au8; 1 << 20];
        for _ in 0..TOTAL / chunk.len() {
            s.write_all(&chunk).await.unwrap();
        }
        s.shutdown().await.unwrap();
    });
    let mut sb = b.accept_stream().await.unwrap();
    let mut buf = vec![0u8; 1 << 20];
    let mut seen = 0;
    while seen < TOTAL {
        let n = sb.read(&mut buf).await.unwrap();
        assert!(n > 0, "stream ended early");
        seen += n;
    }
    writer.await.unwrap();
    (TOTAL as f64 / (1 << 20) as f64) / start.elapsed().as_secs_f64()
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    for subflows in [1, 4] {
        let rate = run(subflows).await;
        println!(
            "{subflows} subflow(s): {rate:.0} MiB/s over {} MiB",
            TOTAL >> 20
        );
    }
}
