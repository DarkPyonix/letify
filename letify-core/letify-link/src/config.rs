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
    /// Bytes that may be in flight on one subflow, sent and not yet acknowledged. Default
    /// 2 MiB.
    pub subflow_window: usize,
    /// How long a session without any subflow waits for a new one before its streams fail
    /// with `Error::ConnectionLost`. Default 24 h.
    pub resume_window: Duration,
    /// Time limit for a handshake or for adding a subflow. Default 30 s.
    pub handshake_timeout: Duration,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            versions: PROTOCOL_VERSION..=PROTOCOL_VERSION,
            max_unacked_bytes: 8 << 20,
            stream_window: 4 << 20,
            subflow_window: 2 << 20,
            resume_window: Duration::from_secs(24 * 3600),
            handshake_timeout: Duration::from_secs(30),
        }
    }
}
