//! The listener side of a node: it reads what a new transport starts with and routes it. A
//! resuming subflow goes to the live session that its preamble names, a new handshake becomes a
//! new session that is returned to the caller. It owns the table of live sessions and nothing
//! about accepting sockets: the caller hands it transports one at a time.

use std::collections::HashMap;
use std::io;
use std::pin::Pin;
use std::sync::Mutex;
use std::task::{Context, Poll};

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, ReadBuf};
use tokio::net::TcpStream;

use crate::config::Config;
use crate::error::{Error, Result};
use crate::identity::{Identity, PeerPolicy};
use crate::session::{Session, SessionId, Transport, WeakSession};

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
    sessions: Mutex<HashMap<SessionId, WeakSession>>,
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
        }
    }

    /// Tune an accepted TCP stream as [`crate::tcp::tune`] does, then accept it.
    pub async fn accept_tcp(&self, stream: TcpStream) -> Result<Accepted> {
        crate::tcp::tune(&stream)?;
        self.accept(stream).await
    }

    /// Read the start of `transport` and route it. A subflow that names no live session is
    /// dropped (the transport is closed) and reported as `Error::Subflow`.
    pub async fn accept<T: Transport>(&self, mut transport: T) -> Result<Accepted> {
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
                let prefixed = Prefixed {
                    prefix: magic.to_vec(),
                    pos: 0,
                    inner: transport,
                };
                let session = Session::accept(
                    prefixed,
                    &self.identity,
                    self.policy.clone(),
                    self.config.clone(),
                )
                .await?;
                let mut sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
                sessions.retain(|_, weak| weak.upgrade().is_some());
                sessions.insert(session.session_id(), session.downgrade());
                Ok(Accepted::Session(session))
            }
            b"LLNS" => {
                let mut id = [0u8; 16];
                tokio::time::timeout(self.config.handshake_timeout, transport.read_exact(&mut id))
                    .await
                    .map_err(|_| Error::Subflow("timed out reading the session id".into()))?
                    .map_err(|e| Error::Subflow(format!("the transport closed: {e}")))?;
                let session = {
                    let mut sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
                    let live = sessions.get(&id).and_then(WeakSession::upgrade);
                    if live.is_none() {
                        sessions.remove(&id);
                    }
                    live
                };
                let session =
                    session.ok_or_else(|| Error::Subflow("no live session has this id".into()))?;
                session.add_subflow_routed(transport).await?;
                Ok(Accepted::Subflow(id))
            }
            _ => Err(Error::Handshake(
                "the transport does not start with the letify-link magic".into(),
            )),
        }
    }
}
