//! The rendezvous: how two nodes exchange small messages before any path exists. It owns the
//! [`Rendezvous`] trait, the message kinds the library sends over it and an in-memory
//! implementation for tests. It does not own a real transport (letify's bootstraps, a signalling
//! service) and it gives no secrecy: the session handshake authenticates every peer.

use std::collections::HashMap;
use std::future::Future;
use std::io;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::mpsc;

use crate::identity::NodeId;

/// The largest message a rendezvous has to carry.
pub const MAX_MESSAGE: usize = 4096;
/// How long before the start of a punch the offer is sent, so the answer can come back first.
pub const DEFAULT_LEAD: Duration = Duration::from_secs(10);
/// How long a punch keeps trying after its start.
pub const DEFAULT_WINDOW: Duration = Duration::from_secs(150);

/// A channel for small messages between nodes. Delivery may be delayed by seconds and need not
/// be ordered, but a message arrives whole or not at all.
pub trait Rendezvous: Send + Sync {
    /// Send `msg` (at most [`MAX_MESSAGE`] bytes) to node `to`.
    fn send(&self, to: NodeId, msg: Vec<u8>) -> impl Future<Output = io::Result<()>> + Send;
    /// Wait for the next message from any node, with the sender.
    fn recv(&self) -> impl Future<Output = io::Result<(NodeId, Vec<u8>)>> + Send;
}

/// Which socket type a punch is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Proto {
    /// TCP simultaneous open.
    Tcp,
    /// UDP probes.
    Udp,
}

/// The messages the library sends over a [`Rendezvous`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Message {
    /// First contact: the sender's public key, which must equal the node id the transport
    /// reports, and whether it wants a record in return.
    Hello {
        /// The ed25519 public key of the sender.
        key: [u8; 32],
        /// True when the sender asks for the receiver's address record.
        want_record: bool,
    },
    /// A signed address record, as [`crate::record`] encodes it.
    Record(Vec<u8>),
    /// The initiator proposes a punch.
    PunchOffer(Punch),
    /// The other side accepts and reports its own mappings and when it stops.
    PunchAnswer(Punch),
}

/// The terms of one punch. Times are unix milliseconds of the sender's clock; the two clocks
/// are assumed to agree within about a second, as with letify's Python punch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Punch {
    /// TCP or UDP.
    pub proto: Proto,
    /// The random token that pairs the two sides. Socket `i` of a TCP punch uses `token + [i]`.
    pub token: [u8; 16],
    /// When both sides start to dial or probe. Set by the initiator and echoed by the answer.
    pub start_at_ms: u64,
    /// When the sender stops trying. Each side adopts the earlier of the two.
    pub until_ms: u64,
    /// The sender's public mapping of each socket, one per socket in order.
    pub mappings: Vec<SocketAddr>,
}

const K_HELLO: u8 = 1;
const K_RECORD: u8 = 2;
const K_OFFER: u8 = 3;
const K_ANSWER: u8 = 4;

impl Message {
    /// Encode as one kind byte followed by the fields.
    pub fn encode(&self) -> Vec<u8> {
        match self {
            Message::Hello { key, want_record } => {
                let mut out = vec![K_HELLO, u8::from(*want_record)];
                out.extend(key);
                out
            }
            Message::Record(bytes) => [&[K_RECORD][..], bytes].concat(),
            Message::PunchOffer(p) => p.encode(K_OFFER),
            Message::PunchAnswer(p) => p.encode(K_ANSWER),
        }
    }

    /// Parse a message; unknown kinds and truncation are errors.
    pub fn decode(data: &[u8]) -> io::Result<Message> {
        let bad =
            |why: &str| io::Error::new(io::ErrorKind::InvalidData, format!("rendezvous: {why}"));
        match data.first() {
            Some(&K_HELLO) if data.len() == 34 => Ok(Message::Hello {
                want_record: data[1] != 0,
                key: data[2..34].try_into().unwrap(),
            }),
            Some(&K_RECORD) => Ok(Message::Record(data[1..].to_vec())),
            Some(&k @ (K_OFFER | K_ANSWER)) => {
                let punch = Punch::decode(&data[1..]).ok_or_else(|| bad("bad punch message"))?;
                Ok(if k == K_OFFER {
                    Message::PunchOffer(punch)
                } else {
                    Message::PunchAnswer(punch)
                })
            }
            _ => Err(bad("unknown or malformed message")),
        }
    }
}

impl Punch {
    fn encode(&self, kind: u8) -> Vec<u8> {
        let mut out = vec![kind, u8::from(self.proto == Proto::Udp)];
        out.extend(self.token);
        out.extend(self.start_at_ms.to_be_bytes());
        out.extend(self.until_ms.to_be_bytes());
        out.push(self.mappings.len() as u8);
        for m in &self.mappings {
            match m.ip() {
                IpAddr::V4(ip) => {
                    out.push(4);
                    out.extend(ip.octets());
                }
                IpAddr::V6(ip) => {
                    out.push(6);
                    out.extend(ip.octets());
                }
            }
            out.extend(m.port().to_be_bytes());
        }
        out
    }

    fn decode(d: &[u8]) -> Option<Punch> {
        let proto = if *d.first()? == 0 {
            Proto::Tcp
        } else {
            Proto::Udp
        };
        let token = d.get(1..17)?.try_into().ok()?;
        let start_at_ms = u64::from_be_bytes(d.get(17..25)?.try_into().ok()?);
        let until_ms = u64::from_be_bytes(d.get(25..33)?.try_into().ok()?);
        let count = *d.get(33)? as usize;
        let mut at = 34;
        let mut mappings = Vec::with_capacity(count);
        for _ in 0..count {
            let size = match *d.get(at)? {
                4 => 4,
                6 => 16,
                _ => return None,
            };
            let raw = d.get(at + 1..at + 1 + size + 2)?;
            at += 1 + size + 2;
            let port = u16::from_be_bytes([raw[size], raw[size + 1]]);
            let ip = if size == 4 {
                IpAddr::V4(Ipv4Addr::from(<[u8; 4]>::try_from(&raw[..4]).ok()?))
            } else {
                IpAddr::V6(Ipv6Addr::from(<[u8; 16]>::try_from(&raw[..16]).ok()?))
            };
            mappings.push(SocketAddr::new(ip, port));
        }
        (at == d.len()).then_some(Punch {
            proto,
            token,
            start_at_ms,
            until_ms,
            mappings,
        })
    }
}

/// Send one [`Message`] through a rendezvous.
pub async fn send_message<R: Rendezvous>(r: &R, to: NodeId, msg: &Message) -> io::Result<()> {
    let bytes = msg.encode();
    if bytes.len() > MAX_MESSAGE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "rendezvous message over 4 KiB",
        ));
    }
    r.send(to, bytes).await
}

/// Receive the next [`Message`]. A `Hello` whose key is not the sender's node id is refused,
/// because the key exchange must not let a node speak for another.
pub async fn recv_message<R: Rendezvous>(r: &R) -> io::Result<(NodeId, Message)> {
    let (from, bytes) = r.recv().await?;
    let msg = Message::decode(&bytes)?;
    if let Message::Hello { key, .. } = &msg
        && key != from.as_bytes()
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "rendezvous: hello key differs from the sender",
        ));
    }
    Ok((from, msg))
}

type Mailbox = mpsc::UnboundedSender<(NodeId, Vec<u8>)>;

/// An in-memory network of rendezvous endpoints, for tests.
#[derive(Clone, Default)]
pub struct MemoryHub {
    boxes: Arc<Mutex<HashMap<NodeId, Mailbox>>>,
}

impl MemoryHub {
    /// A hub with no endpoints.
    pub fn new() -> Self {
        Self::default()
    }

    /// Attach the endpoint of `node`. Attaching twice replaces the first mailbox.
    pub fn endpoint(&self, node: NodeId) -> MemoryRendezvous {
        let (tx, rx) = mpsc::unbounded_channel();
        self.boxes.lock().unwrap().insert(node, tx);
        MemoryRendezvous {
            node,
            hub: self.clone(),
            inbox: tokio::sync::Mutex::new(rx),
        }
    }
}

/// One node's endpoint on a [`MemoryHub`].
pub struct MemoryRendezvous {
    node: NodeId,
    hub: MemoryHub,
    inbox: tokio::sync::Mutex<mpsc::UnboundedReceiver<(NodeId, Vec<u8>)>>,
}

impl Rendezvous for MemoryRendezvous {
    async fn send(&self, to: NodeId, msg: Vec<u8>) -> io::Result<()> {
        if msg.len() > MAX_MESSAGE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "message over 4 KiB",
            ));
        }
        let tx = self.hub.boxes.lock().unwrap().get(&to).cloned();
        tx.ok_or_else(|| io::Error::new(io::ErrorKind::NotConnected, format!("no endpoint {to}")))?
            .send((self.node, msg))
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "endpoint dropped"))
    }

    async fn recv(&self) -> io::Result<(NodeId, Vec<u8>)> {
        self.inbox
            .lock()
            .await
            .recv()
            .await
            .ok_or_else(|| io::Error::new(io::ErrorKind::BrokenPipe, "hub closed"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::Identity;

    fn punch(proto: Proto) -> Punch {
        Punch {
            proto,
            token: [3; 16],
            start_at_ms: 1_700_000_000_123,
            until_ms: 1_700_000_150_123,
            mappings: vec![
                "203.0.113.4:5000".parse().unwrap(),
                "[2001:db8::1]:6000".parse().unwrap(),
            ],
        }
    }

    #[test]
    fn every_message_kind_round_trips() {
        for msg in [
            Message::Hello {
                key: [9; 32],
                want_record: true,
            },
            Message::Record(vec![1, 2, 3]),
            Message::PunchOffer(punch(Proto::Tcp)),
            Message::PunchAnswer(punch(Proto::Udp)),
        ] {
            assert_eq!(Message::decode(&msg.encode()).unwrap(), msg);
        }
    }

    #[test]
    fn a_garbled_message_is_an_error() {
        let bytes = Message::PunchOffer(punch(Proto::Tcp)).encode();
        assert!(Message::decode(&bytes[..bytes.len() - 1]).is_err());
        assert!(Message::decode(&[]).is_err());
        assert!(Message::decode(&[99, 0]).is_err());
    }

    #[tokio::test]
    async fn a_message_reaches_the_named_node_with_the_sender() {
        let hub = MemoryHub::new();
        let (a, b) = (
            Identity::generate().node_id(),
            Identity::generate().node_id(),
        );
        let (ea, eb) = (hub.endpoint(a), hub.endpoint(b));
        send_message(&ea, b, &Message::Record(vec![5]))
            .await
            .unwrap();
        assert_eq!(
            recv_message(&eb).await.unwrap(),
            (a, Message::Record(vec![5]))
        );
        let unknown = Identity::generate().node_id();
        let err = ea.send(unknown, vec![1]).await.unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotConnected);
    }

    #[tokio::test]
    async fn a_hello_with_another_nodes_key_is_refused() {
        let hub = MemoryHub::new();
        let (a, b, c) = (
            Identity::generate().node_id(),
            Identity::generate().node_id(),
            Identity::generate().node_id(),
        );
        let (ea, eb) = (hub.endpoint(a), hub.endpoint(b));
        let lie = Message::Hello {
            key: *c.as_bytes(),
            want_record: false,
        };
        send_message(&ea, b, &lie).await.unwrap();
        assert!(recv_message(&eb).await.is_err());
    }

    #[tokio::test]
    async fn a_message_over_4_kib_is_refused() {
        let hub = MemoryHub::new();
        let (a, b) = (
            Identity::generate().node_id(),
            Identity::generate().node_id(),
        );
        let (ea, _eb) = (hub.endpoint(a), hub.endpoint(b));
        let err = ea.send(b, vec![0; MAX_MESSAGE + 1]).await.unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }
}
