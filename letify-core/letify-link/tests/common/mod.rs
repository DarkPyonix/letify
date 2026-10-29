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
