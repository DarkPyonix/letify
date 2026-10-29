//! The relay role (Cargo feature `relay`): a node that forwards the bytes of two authenticated
//! nodes to each other, and the dialer that reaches a peer through one. It owns the relay
//! handshake policy, the pairing of two requests, the splice and its limits. It does not own
//! the end-to-end session, which runs over the spliced connection as an ordinary subflow, and
//! it never reads what the clients write after pairing.
//!
//! Wire format after the Noise IK handshake of `handshake.rs`, with the relay as responder:
//!
//! - Request, client to relay: a 2 byte big endian length and the Noise transport ciphertext
//!   (nonce 0) of the wanted peer's node id (32 bytes) and the pairing token (16 bytes).
//! - Reply, relay to client: a 2 byte length and the ciphertext (nonce 0 of the other
//!   direction) of a status byte, 0 paired or 1 refused, then a short English reason.
//! - After status 0 both connections carry the session bytes verbatim.

use std::collections::HashMap;
use std::io;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use snow::StatelessTransportState;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Semaphore;
use tokio::time::Instant;

use crate::config::Config;
use crate::handshake;
use crate::identity::{Identity, NodeId, PeerPolicy};
use crate::paths::Candidate;

/// Length of the pairing token.
pub const TOKEN_LEN: usize = 16;

/// A fresh random pairing token. Both peers of a relayed path must use the same one.
pub fn new_token() -> [u8; TOKEN_LEN] {
    let mut t = [0u8; TOKEN_LEN];
    getrandom::getrandom(&mut t).expect("the operating system supplies random bytes");
    t
}

/// Settings of a relay, chosen by its operator.
#[derive(Clone, Debug)]
pub struct RelayConfig {
    /// Time a client has to finish the handshake and send its request. Default 10 s.
    pub handshake_timeout: Duration,
    /// Time a request waits for its counterpart. Default 30 s.
    pub pair_timeout: Duration,
    /// Waiting requests one client may hold. Default 8.
    pub max_pending: usize,
    /// Size of the buffer of one direction of a pair. Default 16 KiB.
    pub buffer_bytes: usize,
    /// Bytes per second the relay forwards from one client, summed over its pairs. Default
    /// none.
    pub rate_limit: Option<u64>,
    /// Connections the relay serves at once. Default 1024.
    pub max_connections: usize,
}

impl Default for RelayConfig {
    fn default() -> Self {
        RelayConfig {
            handshake_timeout: Duration::from_secs(10),
            pair_timeout: Duration::from_secs(30),
            max_pending: 8,
            buffer_bytes: 16 << 10,
            rate_limit: None,
            max_connections: 1024,
        }
    }
}

/// Spreads a byte rate over time: a caller waits until its bytes fit under the rate.
struct Limiter {
    rate: f64,
    next: tokio::sync::Mutex<Instant>,
}

impl Limiter {
    async fn acquire(&self, n: usize) {
        let wake = {
            let mut next = self.next.lock().await;
            let now = Instant::now();
            let start = (*next).max(now);
            *next = start + Duration::from_secs_f64(n as f64 / self.rate);
            start
        };
        tokio::time::sleep_until(wake).await;
    }
}

struct Conn {
    tcp: TcpStream,
    crypto: Arc<StatelessTransportState>,
    node: NodeId,
}

struct Waiting {
    serial: u64,
    conn: Conn,
}

type PairKey = (NodeId, NodeId, [u8; TOKEN_LEN]);

struct Inner {
    identity: Identity,
    policy: PeerPolicy,
    cfg: RelayConfig,
    link: Config,
    waiting: Mutex<HashMap<PairKey, Waiting>>,
    limiters: Mutex<HashMap<NodeId, Arc<Limiter>>>,
    serial: AtomicU64,
}

/// A relay bound to a TCP address.
pub struct Relay {
    listener: TcpListener,
    inner: Arc<Inner>,
}

impl Relay {
    /// Bind `addr`. `policy` decides which nodes may use the relay: `PeerPolicy::allow` for an
    /// allow list, `PeerPolicy::from_fn(|_| true)` for an open relay.
    pub async fn bind(
        addr: &str,
        identity: Identity,
        policy: PeerPolicy,
        cfg: RelayConfig,
    ) -> io::Result<Relay> {
        let listener = TcpListener::bind(addr).await?;
        let link = Config {
            handshake_timeout: cfg.handshake_timeout,
            ..Config::default()
        };
        Ok(Relay {
            listener,
            inner: Arc::new(Inner {
                identity,
                policy,
                cfg,
                link,
                waiting: Mutex::new(HashMap::new()),
                limiters: Mutex::new(HashMap::new()),
                serial: AtomicU64::new(0),
            }),
        })
    }

    /// The address the relay listens on.
    pub fn local_addr(&self) -> SocketAddr {
        self.listener
            .local_addr()
            .expect("a bound listener has an address")
    }

    /// The node id clients must register for this relay.
    pub fn node_id(&self) -> NodeId {
        self.inner.identity.node_id()
    }

    /// Serve until the task is dropped.
    pub async fn run(self) {
        let slots = Arc::new(Semaphore::new(self.inner.cfg.max_connections));
        loop {
            let Ok((tcp, _)) = self.listener.accept().await else {
                // Accept errors such as too many open files are transient.
                tokio::time::sleep(Duration::from_millis(50)).await;
                continue;
            };
            let Ok(permit) = slots.clone().try_acquire_owned() else {
                continue;
            };
            let inner = self.inner.clone();
            tokio::spawn(async move {
                let _ = serve(inner, tcp).await;
                drop(permit);
            });
        }
    }
}

fn io_err(why: impl std::fmt::Display) -> io::Error {
    io::Error::other(why.to_string())
}

async fn write_record(
    tcp: &mut TcpStream,
    crypto: &StatelessTransportState,
    plain: &[u8],
) -> io::Result<()> {
    let mut ct = vec![0u8; plain.len() + 16];
    let n = crypto.write_message(0, plain, &mut ct).map_err(io_err)?;
    let mut out = (n as u16).to_be_bytes().to_vec();
    out.extend_from_slice(&ct[..n]);
    tcp.write_all(&out).await?;
    tcp.flush().await
}

async fn read_record(tcp: &mut TcpStream, crypto: &StatelessTransportState) -> io::Result<Vec<u8>> {
    let mut len = [0u8; 2];
    tcp.read_exact(&mut len).await?;
    let mut ct = vec![0u8; u16::from_be_bytes(len) as usize];
    tcp.read_exact(&mut ct).await?;
    let mut plain = vec![0u8; ct.len()];
    let n = crypto.read_message(0, &ct, &mut plain).map_err(io_err)?;
    plain.truncate(n);
    Ok(plain)
}

async fn status(conn: &mut Conn, ok: bool, reason: &str) -> io::Result<()> {
    let mut plain = vec![if ok { 0 } else { 1 }];
    plain.extend_from_slice(reason.as_bytes());
    write_record(&mut conn.tcp, &conn.crypto, &plain).await
}

async fn serve(inner: Arc<Inner>, mut tcp: TcpStream) -> io::Result<()> {
    crate::tcp::tune(&tcp)?;
    // A client the policy refuses, or one that is not speaking the handshake, is just closed.
    let est = handshake::respond(&mut tcp, &inner.identity, &inner.policy, &inner.link)
        .await
        .map_err(io_err)?;
    let crypto = est.crypto;
    let request = tokio::time::timeout(inner.cfg.handshake_timeout, read_record(&mut tcp, &crypto))
        .await
        .map_err(|_| io_err("no request"))??;
    if request.len() != 32 + TOKEN_LEN {
        return Err(io_err("malformed request"));
    }
    let want = NodeId::from_bytes(request[..32].try_into().unwrap());
    let token: [u8; TOKEN_LEN] = request[32..].try_into().unwrap();
    let me = est.peer;
    let mine = Conn {
        tcp,
        crypto,
        node: me,
    };

    enum Next {
        Wait(u64),
        Pair(Conn, Conn),
        Refuse(Conn, &'static str),
    }
    let next = {
        let mut waiting = inner.waiting.lock().unwrap();
        if let Some(w) = waiting.remove(&(want, me, token)) {
            Next::Pair(w.conn, mine)
        } else if waiting.contains_key(&(me, want, token)) {
            Next::Refuse(
                mine,
                "a request with this peer and token is already waiting",
            )
        } else if waiting.keys().filter(|k| k.0 == me).count() >= inner.cfg.max_pending {
            Next::Refuse(mine, "too many waiting requests")
        } else {
            let serial = inner.serial.fetch_add(1, Ordering::Relaxed);
            // The waiting entry owns the connection until the counterpart arrives or it times out.
            waiting.insert((me, want, token), Waiting { serial, conn: mine });
            Next::Wait(serial)
        }
    };
    match next {
        Next::Refuse(mut conn, why) => {
            let _ = status(&mut conn, false, why).await;
        }
        Next::Wait(serial) => {
            tokio::time::sleep(inner.cfg.pair_timeout).await;
            let expired = {
                let mut waiting = inner.waiting.lock().unwrap();
                match waiting.get(&(me, want, token)) {
                    Some(w) if w.serial == serial => waiting.remove(&(me, want, token)),
                    _ => None,
                }
            };
            if let Some(mut w) = expired {
                let _ = status(&mut w.conn, false, "no counterpart arrived in time").await;
            }
        }
        Next::Pair(mut first, mut second) => {
            status(&mut first, true, "").await?;
            status(&mut second, true, "").await?;
            splice(&inner, first, second).await;
        }
    }
    Ok(())
}

fn limiter_for(inner: &Inner, node: NodeId) -> Option<Arc<Limiter>> {
    let rate = inner.cfg.rate_limit?;
    let mut limiters = inner.limiters.lock().unwrap();
    // Forget the nodes that have no pair running, so an open relay does not grow without bound.
    limiters.retain(|_, l| Arc::strong_count(l) > 1);
    Some(
        limiters
            .entry(node)
            .or_insert_with(|| {
                Arc::new(Limiter {
                    rate: rate.max(1) as f64,
                    next: tokio::sync::Mutex::new(Instant::now()),
                })
            })
            .clone(),
    )
}

/// Copy `from` to `to` through one buffer, waiting for the sender's rate limit before each write.
async fn pump(
    mut from: tokio::net::tcp::OwnedReadHalf,
    mut to: tokio::net::tcp::OwnedWriteHalf,
    buffer: usize,
    limiter: Option<Arc<Limiter>>,
) {
    let mut buf = vec![0u8; buffer.max(1)];
    loop {
        let n = match from.read(&mut buf).await {
            Ok(0) | Err(_) => return,
            Ok(n) => n,
        };
        if let Some(l) = &limiter {
            l.acquire(n).await;
        }
        if to.write_all(&buf[..n]).await.is_err() {
            return;
        }
    }
}

/// Copy both directions until either ends, then close both connections.
async fn splice(inner: &Inner, a: Conn, b: Conn) {
    let (la, lb) = (limiter_for(inner, a.node), limiter_for(inner, b.node));
    let (ar, aw) = a.tcp.into_split();
    let (br, bw) = b.tcp.into_split();
    let buffer = inner.cfg.buffer_bytes;
    tokio::select! {
        _ = pump(ar, bw, buffer, la) => {}
        _ = pump(br, aw, buffer, lb) => {}
    }
}

/// Reaches a peer through a relay the user registered.
#[derive(Clone)]
pub struct RelayDialer {
    address: String,
    relay: NodeId,
    identity: Identity,
    link: Config,
    pair_wait: Duration,
}

impl RelayDialer {
    /// A dialer for the relay at `address` (`host:port`) whose node id is `relay`.
    pub fn new(address: impl Into<String>, relay: NodeId, identity: Identity) -> Self {
        RelayDialer {
            address: address.into(),
            relay,
            identity,
            link: Config::default(),
            pair_wait: Duration::from_secs(35),
        }
    }

    /// How long to wait for the peer to arrive at the relay. Default 35 s, which is longer than
    /// the relay's default pair timeout, so the relay's refusal is what the caller sees.
    pub fn pair_wait(mut self, wait: Duration) -> Self {
        self.pair_wait = wait;
        self
    }

    /// Connect to the relay, authenticate, ask to be paired with `peer` under `token`, and
    /// return the transport once the peer has asked the same. The peer calls `connect` with this
    /// node's id and the same token.
    pub async fn connect(&self, peer: NodeId, token: [u8; TOKEN_LEN]) -> io::Result<TcpStream> {
        let mut tcp = TcpStream::connect(&self.address).await?;
        crate::tcp::tune(&tcp)?;
        let est = handshake::initiate(&mut tcp, &self.identity, self.relay, &self.link)
            .await
            .map_err(io::Error::from)?;
        let mut request = peer.as_bytes().to_vec();
        request.extend_from_slice(&token);
        write_record(&mut tcp, &est.crypto, &request).await?;
        let reply = tokio::time::timeout(self.pair_wait, read_record(&mut tcp, &est.crypto))
            .await
            .map_err(|_| {
                io::Error::new(io::ErrorKind::TimedOut, "the relay did not pair in time")
            })??;
        match reply.first() {
            Some(0) => Ok(tcp),
            Some(_) => Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!(
                    "the relay refused: {}",
                    String::from_utf8_lossy(&reply[1..])
                ),
            )),
            None => Err(io_err("empty reply from the relay")),
        }
    }

    /// The dialer as a path manager candidate named `name`, for `peer` and `token`.
    pub fn candidate(
        &self,
        name: impl Into<String>,
        peer: NodeId,
        token: [u8; TOKEN_LEN],
    ) -> Candidate {
        let dialer = self.clone();
        Candidate::new(name, move || {
            let dialer = dialer.clone();
            async move { dialer.connect(peer, token).await }
        })
    }
}
