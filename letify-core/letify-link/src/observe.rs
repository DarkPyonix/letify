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
//!
//! TCP dial-back: request byte 21 is a flags byte whose bit 0 asks the responder to test one TCP
//! port, bytes 22 and 23 are that port (big endian). The responder connects once, to the
//! observed source IP and that port only, and its reply then carries one more byte before the
//! signature: 0 not reachable, 1 reachable, 2 not tested. A request without the flag, and a
//! responder that predates the flag, give the reply without that byte, which reads as not
//! tested. SPEC.md letify-link, NAT traversal item 6.

use std::collections::HashMap;
use std::future::Future;
use std::io;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use tokio::net::{TcpStream, UdpSocket};
use tokio::sync::Semaphore;
use tokio::time::{Instant, timeout};

use crate::identity::{Identity, NodeId};
use crate::stun::Mapping;

const MAGIC: &[u8; 4] = b"LQAD";
const VERSION: u8 = 1;
const NONCE_LEN: usize = 16;
const REQUEST_LEN: usize = 128;
const DOMAIN: &[u8] = b"letify-link qad v1";
const TRIES: u32 = 3;
const FLAGS_AT: usize = 5 + NONCE_LEN;
const FLAG_DIAL_BACK: u8 = 1;
/// How long the responder waits for the TCP connect of a dial-back.
const DIAL_TIMEOUT: Duration = Duration::from_secs(1);

/// Whether a TCP port of the requester answered a connect from the QAD responder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reachability {
    /// The responder connected to the port.
    Reachable,
    /// The responder tried and the connect was refused, failed or timed out.
    Unreachable,
    /// No test was made: not asked, port 0, the responder is busy or out of budget for this
    /// source, or the responder does not know dial-back.
    NotTested,
}

impl Reachability {
    fn byte(self) -> u8 {
        match self {
            Reachability::Unreachable => 0,
            Reachability::Reachable => 1,
            Reachability::NotTested => 2,
        }
    }

    fn from_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(Reachability::Unreachable),
            1 => Some(Reachability::Reachable),
            2 => Some(Reachability::NotTested),
            _ => None,
        }
    }
}

/// A boxed future, so that the trait can be used as `dyn AddressObserver`.
pub type ObserveFuture<'a> = Pin<Box<dyn Future<Output = io::Result<Mapping>> + Send + 'a>>;

/// Something that reports the public address of a local UDP socket.
pub trait AddressObserver: Send + Sync {
    /// Send from `socket` and return the observer's own address with the public address it
    /// saw. `socket` is not connected, and the observer must not leave data queued on it.
    fn observe<'a>(&'a self, socket: &'a UdpSocket) -> ObserveFuture<'a>;
}

/// How many requests a [`QadResponder`] signs. A request that finds a bucket empty is dropped.
#[derive(Clone, Copy, Debug)]
pub struct QadLimits {
    /// Requests per second, all sources together. Default 2000.
    pub global_per_s: f64,
    /// Burst of the global bucket. Default 2000.
    pub global_burst: u32,
    /// Requests per second from one source address. Default 10.
    pub source_per_s: f64,
    /// Burst of the bucket of one source address. Default 20.
    pub source_burst: u32,
    /// Tokens a TCP dial-back takes from the source bucket on top of the one every request
    /// takes. Default 4. A source that cannot pay is answered with the reply "not tested".
    pub dialback_cost: u32,
    /// Dial-back connects in flight at once, all sources together. Default 32. Over it a
    /// request is answered with the reply "not tested".
    pub dialback_concurrent: usize,
}

impl Default for QadLimits {
    fn default() -> Self {
        QadLimits {
            global_per_s: 2000.0,
            global_burst: 2000,
            source_per_s: 10.0,
            source_burst: 20,
            dialback_cost: 4,
            dialback_concurrent: 32,
        }
    }
}

/// A token bucket: `burst` tokens at most, refilled at `per_s` tokens a second.
struct TokenBucket {
    tokens: f64,
    burst: f64,
    per_s: f64,
    last: Instant,
}

impl TokenBucket {
    fn new(burst: u32, per_s: f64) -> Self {
        TokenBucket {
            tokens: f64::from(burst),
            burst: f64::from(burst),
            per_s,
            last: Instant::now(),
        }
    }

    fn take(&mut self, now: Instant) -> bool {
        self.take_n(1.0, now)
    }

    /// Take `n` tokens or none.
    fn take_n(&mut self, n: f64, now: Instant) -> bool {
        let dt = now.saturating_duration_since(self.last).as_secs_f64();
        self.tokens = (self.tokens + dt * self.per_s).min(self.burst);
        self.last = self.last.max(now);
        if self.tokens >= n {
            self.tokens -= n;
            true
        } else {
            false
        }
    }
}

/// Most source addresses the responder keeps a bucket for.
const MAX_SOURCES: usize = 4096;

struct Limiter {
    limits: QadLimits,
    global: TokenBucket,
    sources: HashMap<IpAddr, TokenBucket>,
}

impl Limiter {
    fn new(limits: QadLimits) -> Self {
        Limiter {
            global: TokenBucket::new(limits.global_burst, limits.global_per_s),
            sources: HashMap::new(),
            limits,
        }
    }

    /// Whether to answer a request from `from` now, and for a dial-back request whether the
    /// source can also pay the extra cost of the connect. `None` means drop the request. The
    /// source bucket is charged first, so one flooding source cannot spend the global tokens
    /// of the others.
    fn allow_with(&mut self, from: IpAddr, now: Instant, dial_back: bool) -> Option<bool> {
        if !self.sources.contains_key(&from) && self.sources.len() >= MAX_SOURCES {
            // Drop the source that was seen longest ago.
            if let Some(old) = self
                .sources
                .iter()
                .min_by_key(|(_, b)| b.last)
                .map(|(k, _)| *k)
            {
                self.sources.remove(&old);
            }
        }
        let limits = self.limits;
        let bucket = self
            .sources
            .entry(from)
            .or_insert_with(|| TokenBucket::new(limits.source_burst, limits.source_per_s));
        if !bucket.take(now) {
            return None;
        }
        let paid = dial_back && bucket.take_n(f64::from(limits.dialback_cost), now);
        self.global.take(now).then_some(paid)
    }
}

/// Answers QAD requests with the address they came from. Any node can run one over UDP.
pub struct QadResponder {
    socket: Arc<UdpSocket>,
    identity: Identity,
    limits: QadLimits,
}

impl QadResponder {
    /// Bind a UDP socket at `addr`. Replies are signed with `identity`.
    pub async fn bind(addr: &str, identity: Identity) -> io::Result<Self> {
        Ok(QadResponder {
            socket: Arc::new(UdpSocket::bind(addr).await?),
            identity,
            limits: QadLimits::default(),
        })
    }

    /// Replace the default limits on how many requests are signed.
    pub fn with_limits(mut self, limits: QadLimits) -> Self {
        self.limits = limits;
        self
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
    /// A request that asks for a TCP dial-back is answered after one connect to its observed
    /// source IP, and the connect runs in its own task so other requests are not held.
    pub async fn run(self) {
        let mut buf = [0u8; 2048];
        let mut limiter = Limiter::new(self.limits);
        let dials = Arc::new(Semaphore::new(self.limits.dialback_concurrent));
        while let Ok((n, from)) = self.socket.recv_from(&mut buf).await {
            if n != REQUEST_LEN || &buf[..4] != MAGIC || buf[4] != VERSION {
                continue;
            }
            let flagged = buf[FLAGS_AT] & FLAG_DIAL_BACK != 0;
            let Some(paid) = limiter.allow_with(from.ip(), Instant::now(), flagged) else {
                continue;
            };
            let nonce: [u8; NONCE_LEN] = buf[5..5 + NONCE_LEN].try_into().unwrap();
            if !flagged {
                let reply = encode_reply(&self.identity, &nonce, from, None);
                let _ = self.socket.send_to(&reply, from).await;
                continue;
            }
            // The target is the observed source IP and the port field, nothing else of the
            // request is read.
            let port = u16::from_be_bytes([buf[FLAGS_AT + 1], buf[FLAGS_AT + 2]]);
            let permit = if paid && port != 0 {
                dials.clone().try_acquire_owned().ok()
            } else {
                None
            };
            let (socket, identity) = (self.socket.clone(), self.identity.clone());
            tokio::spawn(async move {
                let verdict = match permit {
                    Some(_permit) => dial_back(SocketAddr::new(from.ip(), port)).await,
                    None => Reachability::NotTested,
                };
                let reply = encode_reply(&identity, &nonce, from, Some(verdict));
                let _ = socket.send_to(&reply, from).await;
            });
        }
    }
}

/// One TCP connect to `target`, closed at once.
async fn dial_back(target: SocketAddr) -> Reachability {
    match timeout(DIAL_TIMEOUT, TcpStream::connect(target)).await {
        Ok(Ok(_)) => Reachability::Reachable,
        _ => Reachability::Unreachable,
    }
}

fn signed_bytes(nonce: &[u8; NONCE_LEN], body: &[u8]) -> Vec<u8> {
    let mut m = DOMAIN.to_vec();
    m.extend_from_slice(nonce);
    m.extend_from_slice(body);
    m
}

fn encode_reply(
    identity: &Identity,
    nonce: &[u8; NONCE_LEN],
    seen: SocketAddr,
    reach: Option<Reachability>,
) -> Vec<u8> {
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
    if let Some(r) = reach {
        out.push(r.byte());
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
) -> io::Result<(SocketAddr, Reachability)> {
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
    let (reach, end) = match data.len() {
        n if n == end + 64 => (Reachability::NotTested, end),
        n if n == end + 65 => (
            Reachability::from_byte(data[end]).ok_or_else(|| bad("unknown reachability"))?,
            end + 1,
        ),
        _ => return Err(bad("wrong length")),
    };
    if let Some(node) = pin {
        let signature: [u8; 64] = data[end..].try_into().unwrap();
        if !node.verify(&signed_bytes(nonce, &data[..end]), &signature) {
            return Err(bad("signature does not verify"));
        }
    }
    Ok((SocketAddr::new(ip, port), reach))
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

    /// Like `observe`, and also ask the responder to connect once to `tcp_port` at the address
    /// it sees this request come from, to learn whether that port is reachable from outside.
    /// Publish a guessed TCP candidate only for [`Reachability::Reachable`]. A responder that
    /// does not know dial-back, or one that is out of budget for this source, gives
    /// [`Reachability::NotTested`]. Port 0 is never tested.
    pub async fn observe_tcp(
        &self,
        socket: &UdpSocket,
        tcp_port: u16,
    ) -> io::Result<(Mapping, Reachability)> {
        self.ask(socket, Some(tcp_port)).await
    }

    async fn ask(
        &self,
        socket: &UdpSocket,
        tcp_port: Option<u16>,
    ) -> io::Result<(Mapping, Reachability)> {
        let mut nonce = [0u8; NONCE_LEN];
        getrandom::getrandom(&mut nonce).map_err(io::Error::other)?;
        let mut request = [0u8; REQUEST_LEN];
        request[..4].copy_from_slice(MAGIC);
        request[4] = VERSION;
        request[5..5 + NONCE_LEN].copy_from_slice(&nonce);
        if let Some(port) = tcp_port {
            request[FLAGS_AT] = FLAG_DIAL_BACK;
            request[FLAGS_AT + 1..FLAGS_AT + 3].copy_from_slice(&port.to_be_bytes());
        }
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
                let (mapped, reach) = found?;
                return Ok((
                    Mapping {
                        server: self.server,
                        mapped,
                    },
                    reach,
                ));
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
        Box::pin(async move { Ok(self.ask(socket, None).await?.0) })
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
        let reply = encode_reply(&id, &nonce, "10.0.0.1:5000".parse().unwrap(), None);
        assert_eq!(
            decode_reply(&reply, &nonce, Some(id.node_id())).unwrap().0,
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
        let reply = encode_reply(&id, &nonce, seen, Some(Reachability::Reachable));
        assert!(reply.len() <= REQUEST_LEN);
        assert_eq!(
            decode_reply(&reply, &nonce, Some(id.node_id())).unwrap(),
            (seen, Reachability::Reachable)
        );
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

    async fn limited(limits: QadLimits) -> SocketAddr {
        let r = QadResponder::bind("127.0.0.1:0", Identity::generate())
            .await
            .unwrap()
            .with_limits(limits);
        let addr = r.local_addr();
        tokio::spawn(r.run());
        addr
    }

    /// Send `n` requests from `socket` and count the replies that arrive.
    async fn replies(socket: &UdpSocket, addr: SocketAddr, n: usize) -> usize {
        let mut request = [0u8; REQUEST_LEN];
        request[..4].copy_from_slice(MAGIC);
        request[4] = VERSION;
        for _ in 0..n {
            socket.send_to(&request, addr).await.unwrap();
        }
        let mut got = 0;
        let mut buf = [0u8; 256];
        while timeout(Duration::from_millis(300), socket.recv_from(&mut buf))
            .await
            .is_ok()
        {
            got += 1;
        }
        got
    }

    #[tokio::test]
    async fn a_source_that_floods_gets_only_its_burst_answered() {
        let addr = limited(QadLimits {
            source_burst: 3,
            source_per_s: 0.001,
            ..QadLimits::default()
        })
        .await;
        let flood = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        assert_eq!(replies(&flood, addr, 12).await, 3);
    }

    #[tokio::test]
    async fn all_sources_together_are_held_to_the_global_burst() {
        let addr = limited(QadLimits {
            global_burst: 4,
            global_per_s: 0.001,
            ..QadLimits::default()
        })
        .await;
        let mut total = 0;
        for _ in 0..3 {
            // Each socket has its own port, but one IP address, so the global bucket is the one
            // that stops them.
            let s = UdpSocket::bind("127.0.0.1:0").await.unwrap();
            total += replies(&s, addr, 3).await;
        }
        assert_eq!(total, 4);
    }

    #[test]
    fn a_bucket_refills_with_time() {
        let mut b = TokenBucket::new(2, 10.0);
        let t0 = Instant::now();
        assert!(b.take(t0) && b.take(t0) && !b.take(t0));
        assert!(b.take(t0 + Duration::from_millis(150)));
    }
}
