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
