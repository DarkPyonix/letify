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
//! A subflow is any reliable byte transport: a TCP connection, a pipe or an in-memory duplex.
//!
//! # What it does not own yet
//!
//! Rendezvous, NAT traversal and punching, UDP paths, the path manager, the relay role, the
//! `letify-link` sidecar and the Python binding. The caller finds paths and hands each new
//! transport to [`Session::add_subflow`].
//!
//! # Errors
//!
//! A stream whose session lost every subflow for longer than the resume window fails with an
//! `io::Error` that wraps [`Error::ConnectionLost`]. Recover it with [`Error::from_io`]. It is
//! an infrastructure failure and never an application error.

mod config;
mod error;
mod frame;
mod handshake;
mod identity;
mod replay;
mod session;
mod stream;

pub use config::{Config, PROTOCOL_VERSION};
pub use error::{Error, Result};
pub use identity::{Identity, NodeId, PeerPolicy};
pub use replay::REPLAY_WINDOW;
pub use session::{Session, Transport};
pub use stream::Stream;
