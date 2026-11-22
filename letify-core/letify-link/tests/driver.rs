//! The public punch driver, used the way an external crate uses it: only `letify_link`'s public
//! API, with a `MemoryRendezvous` standing in for the embedder's own control channel.
//! Spec "Punch driver".

use std::sync::Arc;
use std::time::Duration;

use letify_link::driver::{Endpoint, PunchErrorKind, PunchOptions, punch_session};
use letify_link::rendezvous::MemoryHub;
use letify_link::{Acceptor, Config, Identity, PeerPolicy};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

fn quick(tcp: usize, udp: usize) -> PunchOptions {
    PunchOptions {
        tcp_sockets: tcp,
        udp_sockets: udp,
        lead: Duration::from_millis(300),
        window: Duration::from_secs(8),
        ..PunchOptions::default()
    }
}

async fn exchange_over(opts: PunchOptions) {
    let (a, b) = (Identity::generate(), Identity::generate());
    let (a_id, b_id) = (a.node_id(), b.node_id());
    let hub = MemoryHub::new();
    let (rdv_a, rdv_b) = (hub.endpoint(a_id), hub.endpoint(b_id));
    let acceptor = Arc::new(Acceptor::new(
        b,
        PeerPolicy::from_fn(|_| true),
        Config::default(),
    ));
    let opts_b = opts.clone();
    let responder = tokio::spawn(async move {
        let got = punch_session(&rdv_b, a_id, Endpoint::Responder { acceptor }, &opts_b)
            .await
            .expect("responder punch");
        let mut stream = got.session.accept_stream().await.unwrap();
        let mut buf = [0u8; 5];
        stream.read_exact(&mut buf).await.unwrap();
        stream.write_all(b"world").await.unwrap();
        stream.flush().await.unwrap();
        (buf, got)
    });
    let initiator = Endpoint::Initiator {
        identity: &a,
        config: Config::default(),
    };
    let got = punch_session(&rdv_a, b_id, initiator, &opts)
        .await
        .expect("initiator punch");
    assert!(got.info.report.starts_with("here: NAT unknown"));
    let mut stream = got.session.open_stream().await.unwrap();
    stream.write_all(b"hello").await.unwrap();
    stream.flush().await.unwrap();
    let mut back = [0u8; 5];
    stream.read_exact(&mut back).await.unwrap();
    assert_eq!(&back, b"world");
    let (heard, _keep) = responder.await.unwrap();
    assert_eq!(&heard, b"hello");
}

#[tokio::test]
async fn two_nodes_punch_over_tcp_and_exchange_bytes_through_the_public_driver() {
    exchange_over(quick(2, 0)).await;
}

#[tokio::test]
async fn two_nodes_punch_over_udp_and_exchange_bytes_through_the_public_driver() {
    exchange_over(quick(0, 1)).await;
}

#[tokio::test]
async fn a_punch_to_a_peer_that_never_answers_fails_with_the_report_line() {
    let (a, b) = (Identity::generate(), Identity::generate());
    let hub = MemoryHub::new();
    let rdv_a = hub.endpoint(a.node_id());
    // The peer has an endpoint, so the offer is delivered, and nothing reads it.
    let _silent = hub.endpoint(b.node_id());
    let opts = PunchOptions {
        lead: Duration::from_millis(100),
        window: Duration::from_millis(300),
        ..PunchOptions::default()
    };
    let initiator = Endpoint::Initiator {
        identity: &a,
        config: Config::default(),
    };
    let err = match punch_session(&rdv_a, b.node_id(), initiator, &opts).await {
        Ok(_) => panic!("a silent peer cannot be punched"),
        Err(e) => e,
    };
    let report = "here: NAT unknown, offered tcp 4 (0 predicted) udp 0; there: no answer";
    assert_eq!(err.kind(), PunchErrorKind::Infrastructure);
    assert_eq!(err.report(), Some(report));
    assert_eq!(
        err.to_string(),
        format!("the punch window closed before the peer answered; {report}")
    );
}

#[tokio::test]
async fn a_punch_with_no_sockets_is_a_usage_error() {
    let a = Identity::generate();
    let hub = MemoryHub::new();
    let rdv = hub.endpoint(a.node_id());
    let initiator = Endpoint::Initiator {
        identity: &a,
        config: Config::default(),
    };
    let err = match punch_session(&rdv, a.node_id(), initiator, &quick(0, 0)).await {
        Ok(_) => panic!("no sockets cannot punch"),
        Err(e) => e,
    };
    assert_eq!(err.kind(), PunchErrorKind::Usage);
    assert_eq!(err.report(), None);
}
