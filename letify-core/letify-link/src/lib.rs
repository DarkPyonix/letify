//! letify-link: the connection layer of letify 3.0, session part.
//!
//! # What this crate owns today
//!
//! - Identity: an ed25519 node key, the node id as its public key, and the X25519 static key
//!   derived from it ([`Identity`], [`NodeId`], [`PeerPolicy`]).
//! - The handshake: Noise_IK_25519_ChaChaPoly_BLAKE2s behind a plaintext header that carries
//!   the protocol version, bound into the Noise prologue.
//! - The session ([`Session`]): encrypted frames with 64-bit sequence numbers, cumulative
//!   acks, a byte bounded retransmission buffer, streams with their own flow control window
//!   ([`Stream`], tokio `AsyncRead` and `AsyncWrite`), several subflows at once, and resumption
//!   on a new subflow after every subflow died.
//!
//! - Keepalive: an idle subflow sends a Ping, the peer answers with a Pong that gives a round
//!   trip estimate, and a subflow with no record for three intervals is declared dead.
//! - Keepalive agreement: each side advertises its interval in the handshake, and the dead time
//!   is three times the larger one.
//! - Probes: a Probe frame measures an idle subflow within the probe budget.
//! - The [`Acceptor`], which routes incoming transports to a live session or to a new one.
//! - The [`PathManager`], which dials candidate paths, starts the session on the first that
//!   answers, adds the others as subflows and redials when every subflow died, and the pure
//!   switching rules in [`hysteresis`]. It moves a session between path kinds: when the
//!   switching rules prefer a UDP path over TCP ones (or the reverse), it closes the old kind
//!   once the new path carried acknowledged traffic.
//! - UDP paths ([`udp`]): a QUIC connection with BBR over a punched UDP socket, whose one stream
//!   is a subflow transport.
//!
//! - The [`rendezvous`] trait with its messages and an in-memory implementation, the signed
//!   address [`record`], the [`stun`] client and NAT classification, [`punch`] (TCP simultaneous
//!   open over several sockets, UDP probing, port prediction, the many-port attempt) and the
//!   [`policing`] check.
//!
//! - The [`sidecar`], the control protocol that the `letify-link serve` executable speaks to
//!   the pure-Python package.
//!
//! A subflow is any reliable byte transport: a TCP connection, a UDP path, a pipe or an
//! in-memory duplex.
//!
//! # What it does not own yet
//!
//! The relay role is the Cargo feature `relay` ([`relay`]). The caller supplies the candidate
//! dialers.
//!
//! # Embedding
//!
//! The crate is a library first. An application links it and runs it in its own process. The
//! `letify-link` executable is one embedder of it, for the Python package. There is no `Node`
//! object: a node is the pieces below, owned by the embedder.
//!
//! **Who owns the runtime.** The caller does. The crate never creates a tokio runtime and
//! never blocks a thread. Every function is async, and the tasks that a [`Session`], a
//! [`PathManager`] or a relay start are spawned with `tokio::spawn`, so they land on the runtime
//! that is current when the function is called. An embedder with its own runtime calls the
//! crate from inside it, with `handle.spawn(..)` or inside `handle.enter()`. An embedder with
//! no runtime (a C caller, an iOS app) builds one and keeps it alive for as long as it holds a
//! session: a multi-thread runtime with `enable_all()` when the `relay` feature is on, a
//! current-thread one otherwise. The `letify-link` binary does exactly that. A later
//! `letify-link-ffi` crate will own a runtime for C callers; this crate does not.
//!
//! **Lifecycle.**
//!
//! 1. *Start.* Load or generate an [`Identity`]. The node id is `identity.node_id()`. Build an
//!    [`Acceptor`] with it to take incoming transports, and call [`PathManager::start`] (or
//!    [`Session::connect`] on a transport of your own) to reach a peer.
//! 2. *Run.* Feed incoming TCP connections and UDP paths to [`Acceptor::accept_tcp`]. It returns
//!    [`Accepted::Session`] for a new peer and adds a subflow to a live session otherwise. Open
//!    and accept streams on the [`Session`]. Watch [`Session::subflow_changes`] and the
//!    [`PathManager`] events for paths.
//! 3. *Stop.* Call [`Session::close`], which writes the queued frames first, then drop the
//!    [`PathManager`], which aborts its dialers. Dropping every handle of a session ends it.
//!    Shut the runtime down last. A relay stops when its `run` task is dropped.
//!
//! ```no_run
//! use letify_link::{Acceptor, Accepted, Config, Identity, PeerPolicy};
//! use tokio::io::AsyncWriteExt;
//! use tokio::net::TcpListener;
//!
//! # async fn demo(rt: tokio::runtime::Handle) -> std::io::Result<()> {
//! let identity = Identity::generate();
//! let acceptor = std::sync::Arc::new(Acceptor::new(
//!     identity,
//!     PeerPolicy::from_fn(|_| true),
//!     Config::default(),
//! ));
//! let listener = TcpListener::bind("0.0.0.0:0").await?;
//! // The caller's handle decides which runtime runs the session.
//! rt.spawn(async move {
//!     while let Ok((tcp, _)) = listener.accept().await {
//!         let acceptor = acceptor.clone();
//!         tokio::spawn(async move {
//!             if let Ok(Accepted::Session(session)) = acceptor.accept_tcp(tcp).await {
//!                 if let Ok(mut stream) = session.accept_stream().await {
//!                     let _ = stream.write_all(b"hello").await;
//!                 }
//!                 session.close().await;
//!             }
//!         });
//!     }
//! });
//! # Ok(()) }
//! ```
//!
//! **Build targets.** The crate is pure Rust (rustls uses the ring backend) and is checked in
//! CI on Linux, macOS and Windows, and for `aarch64-apple-ios` and `aarch64-linux-android`. It
//! builds as an `rlib` only. A C ABI needs `extern "C"` exports and a header, which belong in a
//! separate `letify-link-ffi` crate with `staticlib` and `cdylib` outputs; adding those
//! crate types here would build artifacts that export nothing.
//!
//! # Errors
//!
//! A stream whose session lost every subflow for longer than the resume window fails with an
//! `io::Error` that wraps [`Error::ConnectionLost`]. Recover it with [`Error::from_io`]. It is
//! an infrastructure failure and never an application error.

mod acceptor;
mod config;
mod error;
mod frame;
mod handshake;
pub mod hysteresis;
mod identity;
mod keys;
pub mod observe;
mod paths;
pub mod policing;
pub mod punch;
pub mod record;
#[cfg(feature = "relay")]
pub mod relay;
pub mod rendezvous;
mod replay;
mod session;
pub mod sidecar;
mod stream;
pub mod stun;
pub mod tcp;
pub mod udp;

pub use acceptor::{Accepted, Acceptor};
pub use config::{Config, PROTOCOL_VERSION, PowerState};
pub use error::{Error, Result};
pub use identity::{Identity, NodeId, PeerPolicy};
pub use paths::{
    BoxTransport, Candidate, ManagerConfig, PathEvent, PathKind, PathManager, ProbePolicy,
};
pub use replay::REPLAY_WINDOW;
pub use session::{
    CongestionStats, Priority, Session, SessionId, SubflowId, SubflowStats, Transport, WeakSession,
};
pub use stream::Stream;
