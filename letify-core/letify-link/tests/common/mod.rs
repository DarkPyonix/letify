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
        (Self { inner, gate: gate.clone() }, gate)
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
        Self { inner, offset, written: 0 }
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
        Proxy { addr, conns, accept }
    }

    pub fn cut(&self) {
        self.accept.abort();
        for handle in self.conns.lock().unwrap().drain(..) {
            handle.abort();
        }
    }
}
