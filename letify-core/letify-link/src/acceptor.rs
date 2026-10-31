//! The listener side of a node: it reads what a new transport starts with and routes it. A
//! resuming subflow goes to the live session that its preamble names, a new handshake becomes a
//! new session that is returned to the caller. It owns the table of live sessions and nothing
//! about accepting sockets: the caller hands it transports one at a time.

use std::collections::HashMap;
use std::io;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, ReadBuf};
use tokio::net::TcpStream;

use crate::config::Config;
use crate::error::{Error, Result};
use crate::identity::{Identity, NodeId, PeerPolicy};
use crate::session::{JOIN_PREFIX, Session, SessionId, Transport, WeakSession};

/// Joins of one session that may be in progress at the same time.
const MAX_PENDING_JOINS: usize = 8;

/// What an incoming transport turned out to be.
pub enum Accepted {
    /// A new handshake completed. The caller owns the session.
    Session(Session),
    /// A subflow of the live session with this id. It was added to that session.
    Subflow(SessionId),
}

/// Routes incoming transports to sessions.
pub struct Acceptor {
    identity: Identity,
    policy: PeerPolicy,
    config: Config,
    sessions: Mutex<HashMap<SessionId, Entry>>,
    half_open: Arc<AtomicUsize>,
}

/// A live session of the table and the joins in progress for it.
struct Entry {
    session: WeakSession,
    pending: Arc<AtomicUsize>,
}

/// One slot of a counter, given back when dropped.
struct Slot(Arc<AtomicUsize>);

impl Slot {
    /// Take a slot unless `limit` are taken.
    fn take(counter: &Arc<AtomicUsize>, limit: usize) -> Option<Slot> {
        let taken = counter.fetch_add(1, Ordering::SeqCst);
        if taken >= limit {
            counter.fetch_sub(1, Ordering::SeqCst);
            return None;
        }
        Some(Slot(counter.clone()))
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Gives back the bytes the acceptor already read, then reads from the transport.
struct Prefixed<T> {
    prefix: Vec<u8>,
    pos: usize,
    inner: T,
}

impl<T: AsyncRead + Unpin> AsyncRead for Prefixed<T> {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        if self.pos < self.prefix.len() {
            let n = (self.prefix.len() - self.pos).min(buf.remaining());
            let start = self.pos;
            buf.put_slice(&self.prefix[start..start + n]);
            self.pos += n;
            return Poll::Ready(Ok(()));
        }
        Pin::new(&mut self.inner).poll_read(cx, buf)
    }
}

impl<T: AsyncWrite + Unpin> AsyncWrite for Prefixed<T> {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        b: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.inner).poll_write(cx, b)
    }
    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_flush(cx)
    }
    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_shutdown(cx)
    }
}

impl Acceptor {
    /// An acceptor for the node `identity`. `policy` decides which initiators may start a
    /// session; `config` applies to every session it creates.
    pub fn new(identity: Identity, policy: PeerPolicy, config: Config) -> Self {
        Acceptor {
            identity,
            policy,
            config,
            sessions: Mutex::new(HashMap::new()),
            half_open: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Refuse `node` from now on and close every live session this acceptor created to it,
    /// with the reason `revoked`. A handshake from it fails with `Error::UnknownPeer`, and so
    /// does a subflow that tries to resume one of its sessions. SPEC.md letify-link, Session
    /// item 9.
    pub fn revoke(&self, node: NodeId) {
        self.policy.revoke(node);
        let sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
        for session in sessions.values().filter_map(|e| e.session.upgrade()) {
            if session.peer().canonical() == node.canonical() {
                session.begin_close(Some("revoked"));
            }
        }
    }

    /// Accept `node` again after [`Acceptor::revoke`], subject to the policy as before. A
    /// session that was closed stays closed.
    pub fn reinstate(&self, node: NodeId) {
        self.policy.reinstate(node);
    }

    /// Tune an accepted TCP stream as [`crate::tcp::tune`] does, then accept it.
    pub async fn accept_tcp(&self, stream: TcpStream) -> Result<Accepted> {
        crate::tcp::tune(&stream)?;
        self.accept(stream).await
    }

    /// Read the start of `transport` and route it. A subflow that names no live session is
    /// dropped (the transport is closed) and reported as `Error::Subflow`.
    pub async fn accept<T: Transport>(&self, mut transport: T) -> Result<Accepted> {
        let _slot = Slot::take(&self.half_open, self.config.max_half_open)
            .ok_or_else(|| Error::Handshake("too many handshakes in progress".into()))?;
        let mut magic = [0u8; 4];
        tokio::time::timeout(
            self.config.handshake_timeout,
            transport.read_exact(&mut magic),
        )
        .await
        .map_err(|_| Error::Handshake("timed out waiting for the first bytes".into()))?
        .map_err(|e| Error::Handshake(format!("the transport closed at once: {e}")))?;
        match &magic {
            b"LLNK" => {
                let gauge = crate::session::gauge_of(&transport);
                let prefixed = Prefixed {
                    prefix: magic.to_vec(),
                    pos: 0,
                    inner: transport,
                };
                let session = Session::accept_gauged(
                    prefixed,
                    &self.identity,
                    self.policy.clone(),
                    self.config.clone(),
                    gauge,
                )
                .await?;
                let mut sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
                sessions.retain(|_, e| e.session.upgrade().is_some());
                sessions.insert(
                    session.session_id(),
                    Entry {
                        session: session.downgrade(),
                        pending: Arc::new(AtomicUsize::new(0)),
                    },
                );
                Ok(Accepted::Session(session))
            }
            b"LLNS" => {
                let mut rest = [0u8; JOIN_PREFIX - 4];
                tokio::time::timeout(
                    self.config.handshake_timeout,
                    transport.read_exact(&mut rest),
                )
                .await
                .map_err(|_| Error::Subflow("timed out reading the join prefix".into()))?
                .map_err(|e| Error::Subflow(format!("the transport closed: {e}")))?;
                let token: [u8; 32] = rest[..32].try_into().unwrap();
                let challenge: [u8; 16] = rest[32..].try_into().unwrap();
                // The token names no session to anyone without its secret, so each live
                // session is asked whether the token is its own.
                let (session, pending) = {
                    let mut sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
                    sessions.retain(|_, e| e.session.upgrade().is_some());
                    sessions
                        .values()
                        .filter_map(|e| Some((e.session.upgrade()?, e.pending.clone())))
                        .find(|(s, _)| s.owns_token(&token))
                        .ok_or_else(|| Error::Subflow("no live session owns this token".into()))?
                };
                if !self.policy.accepts(&session.peer()) {
                    return Err(Error::UnknownPeer(session.peer()));
                }
                let _join = Slot::take(&pending, MAX_PENDING_JOINS)
                    .ok_or_else(|| Error::Subflow("too many joins in progress".into()))?;
                session.add_subflow_routed(transport, challenge).await?;
                Ok(Accepted::Subflow(session.session_id()))
            }
            _ => Err(Error::Handshake(
                "the transport does not start with the letify-link magic".into(),
            )),
        }
    }
}
