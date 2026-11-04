//! The error type of the crate. It owns the split between infrastructure failure and
//! everything else; it does not know about application errors, which a stream never carries.

use std::fmt;
use std::io;
use std::ops::RangeInclusive;

use crate::identity::NodeId;

/// Everything that can go wrong in a session.
#[derive(Debug)]
pub enum Error {
    /// The handshake failed: a malformed message, a failed decryption, a tampered header or a
    /// peer that closed the transport.
    Handshake(String),
    /// The responder was not told to accept this initiator key.
    UnknownPeer(NodeId),
    /// The two sides share no protocol version.
    VersionMismatch {
        /// The versions the local side supports.
        ours: RangeInclusive<u16>,
        /// The versions the remote side supports.
        theirs: RangeInclusive<u16>,
    },
    /// The session lost every subflow and the resume window ended, or resumption was
    /// impossible. This is an infrastructure failure and may be retried on a new session.
    ConnectionLost(String),
    /// The session was closed on purpose, by this side or by the peer.
    Closed,
    /// A subflow could not be added to the session.
    Subflow(String),
    /// An I/O error outside a session: while reading or writing a handshake.
    Io(io::Error),
}

impl Error {
    /// True when the failure is in the infrastructure and a new session may succeed.
    pub fn is_infrastructure(&self) -> bool {
        matches!(
            self,
            Error::ConnectionLost(_) | Error::Io(_) | Error::Subflow(_)
        )
    }

    /// Recover the session error that a stream wrapped into an `io::Error`, if it holds one.
    pub fn from_io(err: &io::Error) -> Option<&Error> {
        err.get_ref()?.downcast_ref::<Error>()
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Handshake(why) => write!(f, "handshake failed: {why}"),
            Error::UnknownPeer(id) => write!(f, "initiator {id} is not accepted"),
            Error::VersionMismatch { ours, theirs } => write!(
                f,
                "no common protocol version: this side supports {ours:?}, the other side supports {theirs:?}"
            ),
            Error::ConnectionLost(why) => write!(f, "connection lost: {why}"),
            Error::Closed => write!(f, "session closed"),
            Error::Subflow(why) => write!(f, "subflow refused: {why}"),
            Error::Io(err) => write!(f, "i/o error: {err}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Self {
        Error::Io(err)
    }
}

impl From<Error> for io::Error {
    fn from(err: Error) -> Self {
        let kind = match err {
            Error::ConnectionLost(_) => io::ErrorKind::ConnectionAborted,
            Error::Closed => io::ErrorKind::BrokenPipe,
            _ => io::ErrorKind::Other,
        };
        io::Error::new(kind, err)
    }
}

/// Result alias used across the crate.
pub type Result<T> = std::result::Result<T, Error>;
