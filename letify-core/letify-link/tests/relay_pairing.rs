//! Pins SPEC.md letify-link, Relay item 15: the answerer of a `RelayOffer` dials the relay,
//! is paired, and gets the session from its acceptor.
#![cfg(feature = "relay")]

use std::sync::Arc;
use std::time::Duration;

use letify_link::relay::{Relay, RelayConfig, RelayDialer, RelayPairing, new_token};
use letify_link::rendezvous::{MemoryHub, Message, recv_message, send_message};
use letify_link::{Accepted, Acceptor, Config, Identity, PeerPolicy, Session};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::time::timeout;

struct World {
    relay_id: Identity,
    relay_addr: String,
    offerer: Identity,
    answerer: Identity,
    hub: MemoryHub,
}

async fn world() -> World {
    let relay_id = Identity::generate();
    let relay = Relay::bind(
        "127.0.0.1:0",
        relay_id.clone(),
        PeerPolicy::from_fn(|_| true),
        RelayConfig::default(),
    )
    .await
    .unwrap();
    let relay_addr = relay.local_addr().to_string();
    tokio::spawn(relay.run());
    World {
        relay_id,
        relay_addr,
        offerer: Identity::generate(),
        answerer: Identity::generate(),
        hub: MemoryHub::new(),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn an_answerer_that_trusts_the_relay_offer_gets_a_session_through_the_relay() {
    let w = world().await;
    let off_ep = w.hub.endpoint(w.offerer.node_id());
    let ans_ep = Arc::new(w.hub.endpoint(w.answerer.node_id()));
    let acceptor = Arc::new(Acceptor::new(
        w.answerer.clone(),
        PeerPolicy::allow(vec![w.offerer.node_id()]),
        Config::default(),
    ));
    let relay_node = w.relay_id.node_id();
    let pairing = RelayPairing::new(ans_ep, w.answerer.clone(), acceptor, move |_, relay, _| {
        *relay == relay_node
    });
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    tokio::spawn(async move {
        let _ = pairing
            .run(move |from, accepted| {
                let _ = tx.send((from, accepted));
            })
            .await;
    });

    let token = new_token();
    send_message(
        &off_ep,
        w.answerer.node_id(),
        &Message::RelayOffer {
            token,
            relay: w.relay_id.node_id(),
            address: w.relay_addr.clone(),
        },
    )
    .await
    .unwrap();
    let dialer = RelayDialer::new(
        w.relay_addr.clone(),
        w.relay_id.node_id(),
        w.offerer.clone(),
    );
    let tcp = dialer.connect(w.answerer.node_id(), token).await.unwrap();
    let client = Session::connect(tcp, &w.offerer, w.answerer.node_id(), Config::default())
        .await
        .unwrap();

    let (from, accepted) = timeout(Duration::from_secs(10), rx.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(from, w.offerer.node_id());
    let Accepted::Session(server) = accepted else {
        panic!("a handshake is not a subflow");
    };
    let mut s = client.open_stream().await.unwrap();
    s.write_all(b"via relay").await.unwrap();
    let mut got = [0u8; 9];
    server
        .accept_stream()
        .await
        .unwrap()
        .read_exact(&mut got)
        .await
        .unwrap();
    assert_eq!(&got, b"via relay");

    // The offerer also got the answer, which says the answerer would connect.
    let (_, msg) = recv_message(&off_ep).await.unwrap();
    assert_eq!(
        msg,
        Message::RelayAnswer {
            token,
            accepted: true
        }
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn an_answerer_that_does_not_trust_the_offer_declines_and_dials_nothing() {
    let w = world().await;
    let off_ep = w.hub.endpoint(w.offerer.node_id());
    let ans_ep = Arc::new(w.hub.endpoint(w.answerer.node_id()));
    let acceptor = Arc::new(Acceptor::new(
        w.answerer.clone(),
        PeerPolicy::from_fn(|_| true),
        Config::default(),
    ));
    let pairing = RelayPairing::new(ans_ep, w.answerer.clone(), acceptor, |_, _, _| false);
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    tokio::spawn(async move {
        let _ = pairing
            .run(move |_, accepted| {
                let _ = tx.send(accepted);
            })
            .await;
    });
    let token = new_token();
    send_message(
        &off_ep,
        w.answerer.node_id(),
        &Message::RelayOffer {
            token,
            relay: w.relay_id.node_id(),
            address: w.relay_addr.clone(),
        },
    )
    .await
    .unwrap();
    let (_, msg) = timeout(Duration::from_secs(5), recv_message(&off_ep))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        msg,
        Message::RelayAnswer {
            token,
            accepted: false
        }
    );
    // Nothing dialed the relay, so a dialer for the same token is not paired.
    let dialer = RelayDialer::new(
        w.relay_addr.clone(),
        w.relay_id.node_id(),
        w.offerer.clone(),
    )
    .pair_wait(Duration::from_millis(500));
    assert!(dialer.connect(w.answerer.node_id(), token).await.is_err());
    assert!(rx.try_recv().is_err());
}
