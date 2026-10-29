//! Caller settings of a session. Every field has a default; none is read from a file.

use std::ops::RangeInclusive;
use std::time::Duration;

/// The protocol version this crate speaks.
pub const PROTOCOL_VERSION: u16 = 1;

/// What the application tells the library about the device. It sets the keepalive interval of
/// a session and the probing of a path manager, see SPEC.md letify-link, Mobile and power
/// states.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PowerState {
    /// The application is in use. Keepalives every `keepalive_interval`.
    Foreground,
    /// The application is in the background. Keepalives every `background_keepalive_interval`.
    Background,
    /// The device saves power. No keepalives, no probes: the session relies on resumption.
    PowerSave,
}

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
    /// The keepalive interval of the `Background` power state. Zero turns keepalives off.
    /// Default 120 s.
    pub background_keepalive_interval: Duration,
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
    /// The keepalive interval of a power state: `keepalive_interval` in the foreground,
    /// `background_keepalive_interval` in the background and none in power save.
    pub fn keepalive_for(&self, state: PowerState) -> Duration {
        match state {
            PowerState::Foreground => self.keepalive_interval,
            PowerState::Background => self.background_keepalive_interval,
            PowerState::PowerSave => Duration::ZERO,
        }
    }

    /// A profile for a phone or a tablet: less memory in flight and fewer pings than the
    /// default. `max_unacked_bytes` 4 MiB, `max_reorder_bytes` 8 MiB, `stream_window` 2 MiB,
    /// `subflow_window` 1 MiB and `keepalive_interval` 30 s. Every other field is the default.
    pub fn mobile() -> Config {
        Config {
            max_unacked_bytes: 4 << 20,
            stream_window: 2 << 20,
            subflow_window: 1 << 20,
            max_reorder_bytes: 8 << 20,
            keepalive_interval: Duration::from_secs(30),
            ..Config::default()
        }
    }

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
            background_keepalive_interval: Duration::from_secs(120),
            handshake_timeout: Duration::from_secs(30),
            rekey_after_records: 1 << 32,
            rekey_after: Duration::from_secs(24 * 3600),
        }
    }
}
