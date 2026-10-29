//! Caller settings of a session. Every field has a default; none is read from a file.

use std::ops::RangeInclusive;
use std::time::Duration;

/// The protocol version this crate speaks.
pub const PROTOCOL_VERSION: u16 = 1;

/// Settings for one session.
#[derive(Clone, Debug)]
pub struct Config {
    /// The protocol versions this side supports. The responder picks the highest common one.
    pub versions: RangeInclusive<u16>,
    /// Upper bound, in bytes, on the frames kept until the peer acknowledges them. A stream
    /// writer waits while the bound is reached. Default 8 MiB.
    pub max_unacked_bytes: usize,
    /// Bytes a stream may receive before its reader consumes them, advertised to the peer.
    /// Default 4 MiB.
    pub stream_window: u32,
    /// Bytes that may be in flight on one subflow, sent and not yet acknowledged on that
    /// subflow. Default 2 MiB.
    pub subflow_window: usize,
    /// Bytes of frames the receiver holds ahead of a gap, advertised to the peer, which limits
    /// its unacknowledged bytes to the smaller of this and its own `max_unacked_bytes`. Values
    /// below 128 KiB are raised to 128 KiB. Default 16 MiB.
    pub max_reorder_bytes: usize,
    /// How long a session without any subflow waits for a new one before its streams fail
    /// with `Error::ConnectionLost`. Default 24 h.
    pub resume_window: Duration,
    /// A subflow that sent nothing for this long sends a keepalive. Zero turns this side's
    /// keepalives off. Each side advertises the value in the handshake, and a subflow that
    /// received no record for three times the larger of the two values is declared dead. When
    /// one side advertises zero the other's value is used, and when both do only a transport
    /// error ends a subflow. Default 15 s.
    pub keepalive_interval: Duration,
    /// Time limit for a handshake or for adding a subflow. Default 30 s.
    pub handshake_timeout: Duration,
    /// This side moves its sending keys to the next epoch after sealing this many records in
    /// one epoch. Zero turns the count limit off. Default 2^32.
    pub rekey_after_records: u64,
    /// This side moves its sending keys to the next epoch when the epoch is this old and a
    /// record is sealed. Zero turns the age limit off. Default 24 h.
    pub rekey_after: Duration,
}

impl Config {
    /// The reorder bound in KiB as the handshake carries it, after the 128 KiB least.
    pub(crate) fn reorder_kib(&self) -> u32 {
        (self.max_reorder_bytes.max(128 << 10) >> 10).min(u32::MAX as usize) as u32
    }
}

impl Default for Config {
    fn default() -> Self {
        Config {
            versions: PROTOCOL_VERSION..=PROTOCOL_VERSION,
            max_unacked_bytes: 8 << 20,
            stream_window: 4 << 20,
            subflow_window: 2 << 20,
            max_reorder_bytes: 16 << 20,
            resume_window: Duration::from_secs(24 * 3600),
            keepalive_interval: Duration::from_secs(15),
            handshake_timeout: Duration::from_secs(30),
            rekey_after_records: 1 << 32,
            rekey_after: Duration::from_secs(24 * 3600),
        }
    }
}
