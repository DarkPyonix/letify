//! The plaintext frames carried inside encrypted records, and the record layout on a subflow.
//! It owns encoding and decoding only; ordering and flow control live in the session.

use crate::error::{Error, Result};

/// Frame header: type (1), sequence (8), stream id (4).
pub(crate) const HDR: usize = 13;
/// Largest payload of one data frame.
pub(crate) const CHUNK: usize = 48 * 1024;
/// Record header on a subflow: ciphertext length (2), key epoch (2) and nonce (8).
pub(crate) const RECORD_HDR: usize = 12;
/// Size of the Noise authentication tag.
pub(crate) const TAG: usize = 16;

const T_OPEN: u8 = 1;
const T_DATA: u8 = 2;
const T_FIN: u8 = 3;
const T_CREDIT: u8 = 4;
const T_CLOSE: u8 = 5;
const T_ACK: u8 = 0x10;
const T_HELLO: u8 = 0x11;
const T_PING: u8 = 0x12;
const T_PONG: u8 = 0x13;
const T_PROBE: u8 = 0x14;
const T_PROBE_ACK: u8 = 0x15;
const T_KEEPALIVE: u8 = 0x16;

/// What a sequenced frame does.
#[derive(Debug)]
pub(crate) enum Kind {
    Open,
    /// The whole plaintext is kept and the payload starts at `HDR`.
    Data(Vec<u8>),
    Fin,
    Credit(u64),
    /// The session is closed, with the closer's reason when it gave one.
    Close(Option<String>),
}

#[derive(Debug)]
pub(crate) enum Frame {
    /// Delivered in order, kept by the sender until acknowledged.
    Seq { seq: u64, stream: u32, kind: Kind },
    /// All frames below `next` have been delivered (the session ack), and `sub` bytes of
    /// sequenced frames arrived on the subflow that carries this frame (the subflow ack).
    Ack { next: u64, sub: u64 },
    /// The first record of a subflow added to a running session: same meaning as `Ack`.
    Hello { next: u64 },
    /// Keepalive. `stamp` is the sender's clock in microseconds, returned by the `Pong`.
    Ping { stamp: u64 },
    /// The answer to a `Ping`, carrying its stamp back.
    Pong { stamp: u64 },
    /// Filler that measures a subflow. It has no sequence number and is never delivered.
    Probe,
    /// The answer to a `Probe`: the size of the probe's record in bytes.
    ProbeAck { bytes: u64 },
    /// The sender's keepalive interval changed to `ms` milliseconds, 0 for off.
    Keepalive { ms: u64 },
}

fn head(t: u8, seq: u64, stream: u32, extra: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(HDR + extra);
    out.push(t);
    out.extend_from_slice(&seq.to_be_bytes());
    out.extend_from_slice(&stream.to_be_bytes());
    out
}

pub(crate) fn encode_open(seq: u64, stream: u32) -> Vec<u8> {
    head(T_OPEN, seq, stream, 0)
}

pub(crate) fn encode_data(seq: u64, stream: u32, payload: &[u8]) -> Vec<u8> {
    let mut out = head(T_DATA, seq, stream, payload.len());
    out.extend_from_slice(payload);
    out
}

pub(crate) fn encode_fin(seq: u64, stream: u32) -> Vec<u8> {
    head(T_FIN, seq, stream, 0)
}

pub(crate) fn encode_credit(seq: u64, stream: u32, bytes: u64) -> Vec<u8> {
    let mut out = head(T_CREDIT, seq, stream, 8);
    out.extend_from_slice(&bytes.to_be_bytes());
    out
}

/// Longest reason a Close frame carries, in bytes.
pub(crate) const MAX_REASON: usize = 256;

/// A Close frame. The reason is cut to `MAX_REASON` bytes on a character boundary.
pub(crate) fn encode_close(seq: u64, reason: Option<&str>) -> Vec<u8> {
    let reason = reason.map(cut_reason).unwrap_or_default();
    let mut out = head(T_CLOSE, seq, 0, reason.len());
    out.extend_from_slice(reason.as_bytes());
    out
}

/// `reason` cut to at most `MAX_REASON` bytes without splitting a character.
pub(crate) fn cut_reason(reason: &str) -> &str {
    let mut end = reason.len().min(MAX_REASON);
    while !reason.is_char_boundary(end) {
        end -= 1;
    }
    &reason[..end]
}

pub(crate) fn encode_ack(next: u64, sub: u64) -> Vec<u8> {
    let mut out = vec![T_ACK];
    out.extend_from_slice(&next.to_be_bytes());
    out.extend_from_slice(&sub.to_be_bytes());
    out
}

pub(crate) fn encode_hello(next: u64) -> Vec<u8> {
    let mut out = vec![T_HELLO];
    out.extend_from_slice(&next.to_be_bytes());
    out
}

pub(crate) fn encode_ping(stamp: u64) -> Vec<u8> {
    let mut out = vec![T_PING];
    out.extend_from_slice(&stamp.to_be_bytes());
    out
}

/// A probe whose sealed record is `record_bytes` long.
pub(crate) fn encode_probe(record_bytes: usize) -> Vec<u8> {
    let mut out = vec![0u8; record_bytes.saturating_sub(RECORD_HDR + TAG).max(1)];
    out[0] = T_PROBE;
    out
}

pub(crate) fn encode_probe_ack(bytes: u64) -> Vec<u8> {
    let mut out = vec![T_PROBE_ACK];
    out.extend_from_slice(&bytes.to_be_bytes());
    out
}

pub(crate) fn encode_keepalive(ms: u64) -> Vec<u8> {
    let mut out = vec![T_KEEPALIVE];
    out.extend_from_slice(&ms.to_be_bytes());
    out
}

pub(crate) fn encode_pong(stamp: u64) -> Vec<u8> {
    let mut out = vec![T_PONG];
    out.extend_from_slice(&stamp.to_be_bytes());
    out
}

fn bad(why: &str) -> Error {
    Error::ConnectionLost(format!("malformed frame: {why}"))
}

pub(crate) fn decode(buf: Vec<u8>) -> Result<Frame> {
    let t = *buf.first().ok_or_else(|| bad("empty"))?;
    if t == T_PROBE {
        return Ok(Frame::Probe);
    }
    if t == T_ACK {
        let next = u64::from_be_bytes(
            buf.get(1..9)
                .ok_or_else(|| bad("short ack"))?
                .try_into()
                .unwrap(),
        );
        let sub = u64::from_be_bytes(
            buf.get(9..17)
                .ok_or_else(|| bad("short ack"))?
                .try_into()
                .unwrap(),
        );
        return Ok(Frame::Ack { next, sub });
    }
    if t == T_HELLO || t == T_PING || t == T_PONG || t == T_PROBE_ACK || t == T_KEEPALIVE {
        let raw: [u8; 8] = buf
            .get(1..9)
            .ok_or_else(|| bad("short ack"))?
            .try_into()
            .unwrap();
        let next = u64::from_be_bytes(raw);
        return Ok(match t {
            T_HELLO => Frame::Hello { next },
            T_PING => Frame::Ping { stamp: next },
            T_PROBE_ACK => Frame::ProbeAck { bytes: next },
            T_KEEPALIVE => Frame::Keepalive { ms: next },
            _ => Frame::Pong { stamp: next },
        });
    }
    if buf.len() < HDR {
        return Err(bad("short header"));
    }
    let seq = u64::from_be_bytes(buf[1..9].try_into().unwrap());
    let stream = u32::from_be_bytes(buf[9..13].try_into().unwrap());
    let kind = match t {
        T_OPEN => Kind::Open,
        T_DATA => Kind::Data(buf),
        T_FIN => Kind::Fin,
        T_CREDIT => {
            let raw: [u8; 8] = buf
                .get(HDR..HDR + 8)
                .ok_or_else(|| bad("short credit"))?
                .try_into()
                .unwrap();
            Kind::Credit(u64::from_be_bytes(raw))
        }
        T_CLOSE => {
            let text = String::from_utf8_lossy(&buf[HDR..]);
            Kind::Close((!text.is_empty()).then(|| cut_reason(&text).to_string()))
        }
        _ => return Err(bad("unknown type")),
    };
    Ok(Frame::Seq { seq, stream, kind })
}
