//! Pins SPEC.md letify-link, Rendezvous trait (the message kinds, 4 KiB limit) and Relay item
//! 15 (meeting at a relay).

use std::net::SocketAddr;

use letify_link::Identity;
use letify_link::NodeId;
use letify_link::rendezvous::{MAX_MESSAGE, Message, Proto, Punch};

fn punch(mappings: Vec<SocketAddr>) -> Punch {
    Punch {
        proto: Proto::Tcp,
        token: [7; 16],
        start_at_ms: 1,
        until_ms: 2,
        mappings,
    }
}

fn relay_offer(address: &str) -> Message {
    Message::RelayOffer {
        token: [9; 16],
        relay: Identity::generate().node_id(),
        address: address.to_string(),
    }
}

#[test]
fn a_relay_offer_survives_the_wire() {
    let m = relay_offer("relay.example.org:443");
    assert_eq!(Message::decode(&m.encode()).unwrap(), m);
}

#[test]
fn a_relay_answer_survives_the_wire() {
    for accepted in [true, false] {
        let m = Message::RelayAnswer {
            token: [3; 16],
            accepted,
        };
        assert_eq!(Message::decode(&m.encode()).unwrap(), m);
    }
}

#[test]
fn a_relay_offer_fits_the_rendezvous_limit_at_its_largest() {
    let m = relay_offer(&"h".repeat(255));
    assert!(m.encode().len() <= MAX_MESSAGE);
    assert_eq!(Message::decode(&m.encode()).unwrap(), m);
}

#[test]
fn a_truncated_or_padded_relay_message_is_refused() {
    let mut offer = relay_offer("relay.example.org:443").encode();
    let answer = Message::RelayAnswer {
        token: [3; 16],
        accepted: true,
    }
    .encode();
    assert!(Message::decode(&offer[..offer.len() - 1]).is_err());
    offer.push(0);
    assert!(Message::decode(&offer).is_err());
    assert!(Message::decode(&answer[..answer.len() - 1]).is_err());
    // An empty address is not a relay.
    assert!(Message::decode(&relay_offer("").encode()).is_err());
}

#[test]
fn a_relay_offer_with_an_address_that_is_not_utf8_is_refused() {
    let mut bytes = relay_offer("ab").encode();
    let n = bytes.len();
    bytes[n - 2] = 0xff;
    bytes[n - 1] = 0xfe;
    assert!(Message::decode(&bytes).is_err());
}

#[test]
fn a_punch_offer_with_no_mappings_is_not_a_relay_request() {
    let m = Message::PunchOffer(punch(vec![]));
    assert_eq!(Message::decode(&m.encode()).unwrap(), m);
    assert!(!matches!(
        Message::decode(&m.encode()).unwrap(),
        Message::RelayOffer { .. }
    ));
}

#[tokio::test]
async fn sending_a_relay_offer_with_an_address_over_255_bytes_is_refused() {
    let hub = letify_link::rendezvous::MemoryHub::new();
    let a = Identity::generate();
    let b: NodeId = Identity::generate().node_id();
    let ep = hub.endpoint(a.node_id());
    let err = letify_link::rendezvous::send_message(&ep, b, &relay_offer(&"h".repeat(256)))
        .await
        .unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
}
