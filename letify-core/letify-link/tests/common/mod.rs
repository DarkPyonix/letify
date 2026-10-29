//! Helpers shared by the integration tests: in-memory subflows, a gate that stalls reads, a
//! TCP forwarder that can be cut, and deterministic test data.
#![allow(dead_code)]

use std::io;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

use letify_link::{Config, Identity, PeerPolicy, Session};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;

/// Pseudo random bytes from a seed, so a test can compare what arrived with what was sent.
pub fn pattern(len: usize, seed: u64) -> Vec<u8> {
    let mut x = seed | 1;
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        out.extend_from_slice(&x.to_le_bytes());
    }
    out.truncate(len);
    out
}

/// One in-memory reliable byte transport, as two ends.
pub fn duplex_pair() -> (tokio::io::DuplexStream, tokio::io::DuplexStream) {
    tokio::io::duplex(256 * 1024)
}

/// Two sessions over one in-memory subflow. The responder accepts exactly the initiator.
pub async fn session_pair(config: Config) -> (Session, Session) {
    let a = Identity::generate();
    let b = Identity::generate();
    let (ta, tb) = duplex_pair();
    let responder = tokio::spawn({
        let (b, a_id, config) = (b.clone(), a.node_id(), config.clone());
        async move { Session::accept(tb, &b, PeerPolicy::allow(vec![a_id]), config).await }
    });
    let initiator = Session::connect(ta, &a, b.node_id(), config).await.unwrap();
    (initiator, responder.await.unwrap().unwrap())
}

/// Wraps a transport so that reads stall while the gate is closed.
pub struct Gated<T> {
    inner: T,
    gate: Gate,
}

#[derive(Clone, Default)]
pub struct Gate {
    open: Arc<AtomicBool>,
    waker: Arc<Mutex<Option<Waker>>>,
}

impl Gate {
    pub fn close(&self) {
        self.open.store(false, Ordering::SeqCst);
    }

    pub fn open(&self) {
        self.open.store(true, Ordering::SeqCst);
        if let Some(w) = self.waker.lock().unwrap().take() {
            w.wake();
        }
    }
}

impl<T> Gated<T> {
    pub fn new(inner: T) -> (Self, Gate) {
        let gate = Gate::default();
        gate.open.store(true, Ordering::SeqCst);
        (
            Self {
                inner,
                gate: gate.clone(),
            },
            gate,
        )
    }
}

impl<T: AsyncRead + Unpin> AsyncRead for Gated<T> {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        if !self.gate.open.load(Ordering::SeqCst) {
            *self.gate.waker.lock().unwrap() = Some(cx.waker().clone());
            if !self.gate.open.load(Ordering::SeqCst) {
                return Poll::Pending;
            }
        }
        Pin::new(&mut self.inner).poll_read(cx, buf)
    }
}

impl<T: AsyncWrite + Unpin> AsyncWrite for Gated<T> {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.inner).poll_write(cx, buf)
    }
    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_flush(cx)
    }
    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_shutdown(cx)
    }
}

/// Flips one byte of what the wrapped transport writes, at a fixed offset of the stream.
pub struct Tamper<T> {
    inner: T,
    offset: usize,
    written: usize,
}

impl<T> Tamper<T> {
    pub fn new(inner: T, offset: usize) -> Self {
        Self {
            inner,
            offset,
            written: 0,
        }
    }
}

impl<T: AsyncRead + Unpin> AsyncRead for Tamper<T> {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_read(cx, buf)
    }
}

impl<T: AsyncWrite + Unpin> AsyncWrite for Tamper<T> {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let start = self.written;
        let offset = self.offset;
        if start <= offset && offset < start + buf.len() {
            let mut copy = buf.to_vec();
            copy[offset - start] ^= 0x01;
            let n = match Pin::new(&mut self.inner).poll_write(cx, &copy) {
                Poll::Ready(Ok(n)) => n,
                other => return other,
            };
            self.written += n;
            return Poll::Ready(Ok(n));
        }
        let n = match Pin::new(&mut self.inner).poll_write(cx, buf) {
            Poll::Ready(Ok(n)) => n,
            other => return other,
        };
        self.written += n;
        Poll::Ready(Ok(n))
    }
    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_flush(cx)
    }
    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_shutdown(cx)
    }
}

/// A TCP forwarder on loopback. Dropping its connections, by `cut`, looks to both ends like
/// a path that died.
pub struct Proxy {
    pub addr: std::net::SocketAddr,
    conns: Arc<Mutex<Vec<JoinHandle<()>>>>,
    accept: JoinHandle<()>,
}

impl Proxy {
    pub async fn to(target: std::net::SocketAddr) -> Proxy {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let conns = Arc::new(Mutex::new(Vec::new()));
        let accept = tokio::spawn({
            let conns = conns.clone();
            async move {
                while let Ok((mut front, _)) = listener.accept().await {
                    let handle = tokio::spawn(async move {
                        if let Ok(mut back) = TcpStream::connect(target).await {
                            let _ = tokio::io::copy_bidirectional(&mut front, &mut back).await;
                        }
                    });
                    conns.lock().unwrap().push(handle);
                }
            }
        });
        Proxy {
            addr,
            conns,
            accept,
        }
    }

    pub fn cut(&self) {
        self.accept.abort();
        for handle in self.conns.lock().unwrap().drain(..) {
            handle.abort();
        }
    }
}

/// What precedes the records on the initiator's side of a transport.
pub enum Prefix {
    /// A session handshake: the header, then the first Noise message.
    Handshake,
    /// A subflow added to a running session: the magic and session id.
    Subflow,
}

/// An on-path attacker between two ends. It records every record the initiator sends, can
/// withhold them and can inject bytes toward the responder.
pub struct Mitm {
    pub records: Arc<Mutex<Vec<Vec<u8>>>>,
    pub withhold: Arc<AtomicBool>,
    pub prefix: Arc<Mutex<Vec<u8>>>,
    to_b: Arc<tokio::sync::Mutex<tokio::io::WriteHalf<tokio::io::DuplexStream>>>,
}

impl Mitm {
    pub fn new(
        a_end: tokio::io::DuplexStream,
        b_end: tokio::io::DuplexStream,
        prefix: Prefix,
    ) -> Mitm {
        use tokio::io::AsyncReadExt;
        use tokio::io::AsyncWriteExt;
        let (mut ar, mut aw) = tokio::io::split(a_end);
        let (mut br, bw) = tokio::io::split(b_end);
        let to_b = Arc::new(tokio::sync::Mutex::new(bw));
        let records = Arc::new(Mutex::new(Vec::new()));
        let withhold = Arc::new(AtomicBool::new(false));
        let prefix_seen = Arc::new(Mutex::new(Vec::new()));
        tokio::spawn(async move {
            let _ = tokio::io::copy(&mut br, &mut aw).await;
        });
        tokio::spawn({
            let (to_b, records, withhold, prefix_seen) = (
                to_b.clone(),
                records.clone(),
                withhold.clone(),
                prefix_seen.clone(),
            );
            async move {
                let mut head = match prefix {
                    Prefix::Handshake => vec![0u8; 10],
                    Prefix::Subflow => vec![0u8; 20],
                };
                if ar.read_exact(&mut head).await.is_err() {
                    return;
                }
                if let Prefix::Handshake = prefix {
                    let mut body = vec![0u8; u16::from_be_bytes([head[8], head[9]]) as usize];
                    if ar.read_exact(&mut body).await.is_err() {
                        return;
                    }
                    head.extend_from_slice(&body);
                }
                *prefix_seen.lock().unwrap() = head.clone();
                if to_b.lock().await.write_all(&head).await.is_err() {
                    return;
                }
                loop {
                    let mut rec = vec![0u8; 10];
                    if ar.read_exact(&mut rec).await.is_err() {
                        return;
                    }
                    let mut ct = vec![0u8; u16::from_be_bytes([rec[0], rec[1]]) as usize];
                    if ar.read_exact(&mut ct).await.is_err() {
                        return;
                    }
                    rec.extend_from_slice(&ct);
                    records.lock().unwrap().push(rec.clone());
                    if !withhold.load(Ordering::SeqCst)
                        && to_b.lock().await.write_all(&rec).await.is_err()
                    {
                        return;
                    }
                }
            }
        });
        Mitm {
            records,
            withhold,
            prefix: prefix_seen,
            to_b,
        }
    }

    pub async fn inject(&self, bytes: &[u8]) {
        use tokio::io::AsyncWriteExt;
        self.to_b.lock().await.write_all(bytes).await.unwrap();
    }

    pub fn recorded(&self) -> Vec<Vec<u8>> {
        self.records.lock().unwrap().clone()
    }
}

/// Two sessions whose initiator side runs through an on-path attacker.
pub async fn session_pair_through_mitm(config: Config) -> (Session, Session, Mitm) {
    let a = Identity::generate();
    let b = Identity::generate();
    let (ta, m_a) = duplex_pair();
    let (m_b, tb) = duplex_pair();
    let mitm = Mitm::new(m_a, m_b, Prefix::Handshake);
    let responder = tokio::spawn({
        let (b, a_id, config) = (b.clone(), a.node_id(), config.clone());
        async move { Session::accept(tb, &b, PeerPolicy::allow(vec![a_id]), config).await }
    });
    let initiator = Session::connect(ta, &a, b.node_id(), config).await.unwrap();
    (initiator, responder.await.unwrap().unwrap(), mitm)
}

impl Proxy {
    /// Drop every open connection but keep accepting new ones.
    pub fn drop_connections(&self) {
        for handle in self.conns.lock().unwrap().drain(..) {
            handle.abort();
        }
    }
}

/// A listener that runs an `Acceptor` on every incoming TCP connection. New sessions arrive on
/// the returned channel; subflows are routed to their sessions by the acceptor.
pub fn spawn_server(
    listener: TcpListener,
    acceptor: letify_link::Acceptor,
) -> tokio::sync::mpsc::Receiver<Session> {
    let (tx, rx) = tokio::sync::mpsc::channel(8);
    let acceptor = Arc::new(acceptor);
    tokio::spawn(async move {
        while let Ok((tcp, _)) = listener.accept().await {
            let (acceptor, tx) = (acceptor.clone(), tx.clone());
            tokio::spawn(async move {
                if let Ok(letify_link::Accepted::Session(s)) = acceptor.accept_tcp(tcp).await {
                    let _ = tx.send(s).await;
                }
            });
        }
    });
    rx
}

/// A TCP forwarder on loopback that limits the rate of the client to server direction with a
/// token bucket, and can pause both directions. It stands in for one path of a given speed.
pub struct Throttle {
    pub addr: std::net::SocketAddr,
    paused: Arc<AtomicBool>,
    resume: Arc<tokio::sync::Notify>,
}

impl Throttle {
    /// Forward to `target` at `rate` bytes per second toward the target.
    pub async fn to(target: std::net::SocketAddr, rate: f64) -> Throttle {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let paused = Arc::new(AtomicBool::new(false));
        let resume = Arc::new(tokio::sync::Notify::new());
        tokio::spawn({
            let (paused, resume) = (paused.clone(), resume.clone());
            async move {
                while let Ok((front, _)) = listener.accept().await {
                    let (paused, resume) = (paused.clone(), resume.clone());
                    tokio::spawn(async move {
                        let Ok(back) = TcpStream::connect(target).await else {
                            return;
                        };
                        let _ = front.set_nodelay(true);
                        let _ = back.set_nodelay(true);
                        let (fr, fw) = front.into_split();
                        let (br, bw) = back.into_split();
                        tokio::join!(
                            pump(fr, bw, Some(rate), paused.clone(), resume.clone()),
                            pump(br, fw, None, paused, resume),
                        );
                    });
                }
            }
        });
        Throttle {
            addr,
            paused,
            resume,
        }
    }

    /// Stop forwarding in both directions until `resume`.
    pub fn pause(&self) {
        self.paused.store(true, Ordering::SeqCst);
    }

    pub fn resume(&self) {
        self.paused.store(false, Ordering::SeqCst);
        self.resume.notify_waiters();
    }
}

async fn pump(
    mut from: tokio::net::tcp::OwnedReadHalf,
    mut to: tokio::net::tcp::OwnedWriteHalf,
    rate: Option<f64>,
    paused: Arc<AtomicBool>,
    resume: Arc<tokio::sync::Notify>,
) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut buf = vec![0u8; 16 * 1024];
    let mut start = tokio::time::Instant::now();
    let mut sent = 0f64;
    loop {
        // Wait while paused. The flag is checked again after registering for the wake up.
        loop {
            let woken = resume.notified();
            tokio::pin!(woken);
            woken.as_mut().enable();
            if !paused.load(Ordering::SeqCst) {
                break;
            }
            woken.await;
            // Time spent paused does not count as credit.
            start = tokio::time::Instant::now();
            sent = 0.0;
        }
        let Ok(n) = from.read(&mut buf).await else {
            break;
        };
        if n == 0 || to.write_all(&buf[..n]).await.is_err() {
            break;
        }
        if let Some(rate) = rate {
            sent += n as f64;
            let due = start + std::time::Duration::from_secs_f64(sent / rate);
            tokio::time::sleep_until(due).await;
        }
    }
    let _ = to.shutdown().await;
}

/// A session over `rates.len()` TCP subflows on loopback; subflow `i` runs through a
/// throttle of `rates[i]` bytes per second toward the responder. The first element of the
/// result is the initiator.
pub async fn throttled_session(
    rates: &[f64],
    client: Config,
    server: Config,
) -> (Session, Session, Vec<Throttle>) {
    let a_id = Identity::generate();
    let b_id = Identity::generate();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let target = listener.local_addr().unwrap();
    let n = rates.len();
    let responder = tokio::spawn({
        let (b_id, a_node) = (b_id.clone(), a_id.node_id());
        async move {
            let (tcp, _) = listener.accept().await.unwrap();
            let b = Session::accept(tcp, &b_id, PeerPolicy::allow(vec![a_node]), server)
                .await
                .unwrap();
            for _ in 1..n {
                let (tcp, _) = listener.accept().await.unwrap();
                b.add_subflow(tcp).await.unwrap();
            }
            b
        }
    });
    let mut proxies = Vec::new();
    for rate in rates {
        proxies.push(Throttle::to(target, *rate).await);
    }
    let a = Session::connect(
        TcpStream::connect(proxies[0].addr).await.unwrap(),
        &a_id,
        b_id.node_id(),
        client,
    )
    .await
    .unwrap();
    for p in &proxies[1..] {
        a.add_subflow(TcpStream::connect(p.addr).await.unwrap())
            .await
            .unwrap();
    }
    (a, responder.await.unwrap(), proxies)
}
