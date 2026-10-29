//! Address observation: the trait for anything that reports the public address it sees a UDP
//! datagram come from, and the QAD exchange that a node can run for that purpose. It owns the
//! trait, the QAD request and reply and their signature. It does not own the STUN client (that
//! is `stun.rs`, which implements the trait) or the comparison of mappings (also `stun.rs`).
//!
//! QAD request, 128 bytes: `LQAD`, version byte 1, a 16 byte random nonce, zero padding. The
//! padding makes a reply never larger than its request, so a responder is no amplifier.
//!
//! QAD reply: `LQAD`, version byte, the nonce, the family byte (4 or 6), the observed port
//! (2 bytes big endian), the observed IP (4 or 16 bytes) and a 64 byte ed25519 signature by the
//! responder over `letify-link qad v1`, the nonce and every earlier byte of the reply.

use std::future::Future;
use std::io;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use tokio::net::UdpSocket;
use tokio::time::timeout;

use crate::identity::{Identity, NodeId};
use crate::stun::Mapping;

const MAGIC: &[u8; 4] = b"LQAD";
const VERSION: u8 = 1;
const NONCE_LEN: usize = 16;
const REQUEST_LEN: usize = 128;
const DOMAIN: &[u8] = b"letify-link qad v1";
const TRIES: u32 = 3;

/// A boxed future, so that the trait can be used as `dyn AddressObserver`.
pub type ObserveFuture<'a> = Pin<Box<dyn Future<Output = io::Result<Mapping>> + Send + 'a>>;

/// Something that reports the public address of a local UDP socket.
pub trait AddressObserver: Send + Sync {
    /// Send from `socket` and return the observer's own address with the public address it
    /// saw. `socket` is not connected, and the observer must not leave data queued on it.
    fn observe<'a>(&'a self, socket: &'a UdpSocket) -> ObserveFuture<'a>;
}

/// Answers QAD requests with the address they came from. Any node can run one over UDP.
pub struct QadResponder {
    socket: UdpSocket,
    identity: Identity,
}

impl QadResponder {
    /// Bind a UDP socket at `addr`. Replies are signed with `identity`.
    pub async fn bind(addr: &str, identity: Identity) -> io::Result<Self> {
        Ok(QadResponder {
            socket: UdpSocket::bind(addr).await?,
            identity,
        })
    }

    /// The address the responder listens on.
    pub fn local_addr(&self) -> SocketAddr {
        self.socket
            .local_addr()
            .expect("a bound socket has an address")
    }

    /// The node id observers pin to authenticate the replies.
    pub fn node_id(&self) -> NodeId {
        self.identity.node_id()
    }

    /// Answer requests until the task is dropped. A datagram that is not a request is ignored.
    pub async fn run(self) {
        let mut buf = [0u8; 2048];
        while let Ok((n, from)) = self.socket.recv_from(&mut buf).await {
            if n != REQUEST_LEN || &buf[..4] != MAGIC || buf[4] != VERSION {
                continue;
            }
            let nonce: [u8; NONCE_LEN] = buf[5..5 + NONCE_LEN].try_into().unwrap();
            let reply = encode_reply(&self.identity, &nonce, from);
            let _ = self.socket.send_to(&reply, from).await;
        }
    }
}

fn signed_bytes(nonce: &[u8; NONCE_LEN], body: &[u8]) -> Vec<u8> {
    let mut m = DOMAIN.to_vec();
    m.extend_from_slice(nonce);
    m.extend_from_slice(body);
    m
}

fn encode_reply(identity: &Identity, nonce: &[u8; NONCE_LEN], seen: SocketAddr) -> Vec<u8> {
    let mut out = MAGIC.to_vec();
    out.push(VERSION);
    out.extend_from_slice(nonce);
    match seen.ip() {
        IpAddr::V4(ip) => {
            out.push(4);
            out.extend_from_slice(&seen.port().to_be_bytes());
            out.extend_from_slice(&ip.octets());
        }
        IpAddr::V6(ip) => {
            out.push(6);
            out.extend_from_slice(&seen.port().to_be_bytes());
            out.extend_from_slice(&ip.octets());
        }
    }
    let signature = identity.sign(&signed_bytes(nonce, &out));
    out.extend_from_slice(&signature);
    out
}

/// Parse a reply for `nonce`. With `pin` the signature must verify under that node id.
fn decode_reply(
    data: &[u8],
    nonce: &[u8; NONCE_LEN],
    pin: Option<NodeId>,
) -> io::Result<SocketAddr> {
    let bad = |why: &str| io::Error::new(io::ErrorKind::InvalidData, format!("QAD: {why}"));
    if data.len() < 5 + NONCE_LEN + 3 + 64 || &data[..4] != MAGIC || data[4] != VERSION {
        return Err(bad("not a reply"));
    }
    if data[5..5 + NONCE_LEN] != nonce[..] {
        return Err(bad("reply is for another request"));
    }
    let at = 5 + NONCE_LEN;
    let (ip, end) = match data[at] {
        4 => {
            let b: [u8; 4] = data
                .get(at + 3..at + 7)
                .ok_or_else(|| bad("truncated"))?
                .try_into()
                .unwrap();
            (IpAddr::V4(Ipv4Addr::from(b)), at + 7)
        }
        6 => {
            let b: [u8; 16] = data
                .get(at + 3..at + 19)
                .ok_or_else(|| bad("truncated"))?
                .try_into()
                .unwrap();
            (IpAddr::V6(Ipv6Addr::from(b)), at + 19)
        }
        _ => return Err(bad("unknown address family")),
    };
    let port = u16::from_be_bytes([data[at + 1], data[at + 2]]);
    if data.len() != end + 64 {
        return Err(bad("wrong length"));
    }
    if let Some(node) = pin {
        let signature: [u8; 64] = data[end..].try_into().unwrap();
        if !node.verify(&signed_bytes(nonce, &data[..end]), &signature) {
            return Err(bad("signature does not verify"));
        }
    }
    Ok(SocketAddr::new(ip, port))
}

/// Asks one QAD responder for the public address of a socket.
#[derive(Clone, Debug)]
pub struct QadObserver {
    server: SocketAddr,
    pin: Option<NodeId>,
    wait: Duration,
}

impl QadObserver {
    /// An observer for the responder at `server`. With `pin` set to its node id, a reply is
    /// accepted only when it carries a valid signature by that node. Without it only the nonce
    /// is checked, which stops a stray datagram but not a forger on the path.
    pub fn new(server: SocketAddr, pin: Option<NodeId>) -> Self {
        QadObserver {
            server,
            pin,
            wait: Duration::from_secs(3),
        }
    }

    /// Total time for the request and its retries. Default 3 s.
    pub fn wait(mut self, wait: Duration) -> Self {
        self.wait = wait;
        self
    }

    async fn ask(&self, socket: &UdpSocket) -> io::Result<Mapping> {
        let mut nonce = [0u8; NONCE_LEN];
        getrandom::getrandom(&mut nonce).map_err(io::Error::other)?;
        let mut request = [0u8; REQUEST_LEN];
        request[..4].copy_from_slice(MAGIC);
        request[4] = VERSION;
        request[5..5 + NONCE_LEN].copy_from_slice(&nonce);
        let mut buf = [0u8; 2048];
        for _ in 0..TRIES {
            socket.send_to(&request, self.server).await?;
            let round = async {
                loop {
                    let (n, from) = socket.recv_from(&mut buf).await?;
                    if from != self.server {
                        continue;
                    }
                    if let Ok(mapped) = decode_reply(&buf[..n], &nonce, self.pin) {
                        return Ok::<_, io::Error>(mapped);
                    }
                }
            };
            if let Ok(found) = timeout(self.wait / TRIES, round).await {
                return Ok(Mapping {
                    server: self.server,
                    mapped: found?,
                });
            }
        }
        Err(io::Error::new(
            io::ErrorKind::TimedOut,
            format!("no QAD reply from {}", self.server),
        ))
    }
}

impl AddressObserver for QadObserver {
    fn observe<'a>(&'a self, socket: &'a UdpSocket) -> ObserveFuture<'a> {
        Box::pin(self.ask(socket))
    }
}

impl<T: AddressObserver + ?Sized> AddressObserver for Arc<T> {
    fn observe<'a>(&'a self, socket: &'a UdpSocket) -> ObserveFuture<'a> {
        (**self).observe(socket)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn responder() -> (SocketAddr, NodeId) {
        let r = QadResponder::bind("127.0.0.1:0", Identity::generate())
            .await
            .unwrap();
        let out = (r.local_addr(), r.node_id());
        tokio::spawn(r.run());
        out
    }

    #[tokio::test]
    async fn a_qad_responder_reports_the_source_address_it_saw() {
        let (addr, id) = responder().await;
        let socket = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let m = QadObserver::new(addr, Some(id))
            .observe(&socket)
            .await
            .unwrap();
        assert_eq!(m.server, addr);
        assert_eq!(m.mapped, socket.local_addr().unwrap());
    }

    #[tokio::test]
    async fn a_reply_signed_by_another_node_is_refused() {
        let (addr, _id) = responder().await;
        let socket = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let wrong = Identity::generate().node_id();
        let err = QadObserver::new(addr, Some(wrong))
            .wait(Duration::from_millis(300))
            .observe(&socket)
            .await
            .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::TimedOut);
    }

    #[tokio::test]
    async fn a_reply_for_another_nonce_is_refused() {
        let id = Identity::generate();
        let nonce = [1u8; NONCE_LEN];
        let reply = encode_reply(&id, &nonce, "10.0.0.1:5000".parse().unwrap());
        assert_eq!(
            decode_reply(&reply, &nonce, Some(id.node_id())).unwrap(),
            "10.0.0.1:5000".parse::<SocketAddr>().unwrap()
        );
        assert!(decode_reply(&reply, &[2u8; NONCE_LEN], None).is_err());
        let mut forged = reply.clone();
        forged[5 + NONCE_LEN + 1] ^= 1;
        assert!(decode_reply(&forged, &nonce, Some(id.node_id())).is_err());
        assert!(decode_reply(&forged, &nonce, None).is_ok());
    }

    #[test]
    fn an_ipv6_reply_round_trips_and_is_no_larger_than_the_request() {
        let id = Identity::generate();
        let nonce = [3u8; NONCE_LEN];
        let seen: SocketAddr = "[2001:db8::9]:777".parse().unwrap();
        let reply = encode_reply(&id, &nonce, seen);
        assert!(reply.len() <= REQUEST_LEN);
        assert_eq!(decode_reply(&reply, &nonce, Some(id.node_id())).unwrap(), seen);
    }

    #[tokio::test]
    async fn a_short_datagram_gets_no_reply() {
        let (addr, _) = responder().await;
        let socket = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        socket.send_to(b"LQAD", addr).await.unwrap();
        let mut buf = [0u8; 100];
        let got = timeout(Duration::from_millis(200), socket.recv_from(&mut buf)).await;
        assert!(got.is_err());
    }
}
