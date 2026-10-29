//! The address record: one signed, documented binary description of how to reach a node. It
//! owns the encoding, the signature and the age check. It does not own transport (that is the
//! [`Rendezvous`](crate::rendezvous::Rendezvous)) or the choice of candidates.
//!
//! # Wire format (version 1, all integers big endian)
//!
//! ```text
//! offset  size  field
//! 0       1     flags: bits 7..4 format version (1), bit 0 relay present
//! 1       32    node id (ed25519 public key)
//! 33      4     record version, a counter the node raises with every new record
//! 37      4     timestamp, unix seconds
//! 41      1     NAT type: 0 unknown, 1 endpoint independent, 2 symmetric
//! 42      2     port step of a symmetric NAT as i16 (0 otherwise)
//! 44      4     candidate counts: TCP IPv4, TCP IPv6, UDP IPv4, UDP IPv6
//! 48      ...   TCP IPv4 (4 byte address, 2 byte port), TCP IPv6 (16 + 2), UDP IPv4, UDP IPv6
//!         ...   relay, when flagged: 1 byte family (4 or 6), then address and port
//!         64    ed25519 signature over "letify-link record v1" and every byte before it
//! ```
//!
//! The record is 112 bytes with no candidates, plus 6 bytes per IPv4 candidate and 18 per IPv6
//! candidate, so 8 TCP and 8 UDP IPv6 candidates take 400 bytes. A record is valid for
//! [`MAX_AGE`] after its timestamp.

use std::fmt;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::identity::{Identity, NodeId};
use crate::stun::NatType;

/// How long after its timestamp a record is still accepted.
pub const MAX_AGE: Duration = Duration::from_secs(600);
/// How far ahead of the receiver's clock a timestamp may be, to allow for clock skew.
pub const MAX_SKEW: Duration = Duration::from_secs(60);
/// Size of the encoded record without candidates and relay.
pub const BASE_SIZE: usize = 112;

const FORMAT: u8 = 1;
const DOMAIN: &[u8] = b"letify-link record v1";
const HEADER: usize = 48;

/// Why a record was refused.
#[derive(Debug, PartialEq, Eq)]
pub enum RecordError {
    /// The bytes do not parse as a record.
    Malformed(&'static str),
    /// The signature does not verify under the node id in the record.
    BadSignature,
    /// The timestamp is older than [`MAX_AGE`].
    Expired,
    /// The timestamp is more than [`MAX_SKEW`] ahead of the receiver's clock.
    FromTheFuture,
}

impl fmt::Display for RecordError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RecordError::Malformed(why) => write!(f, "malformed address record: {why}"),
            RecordError::BadSignature => write!(f, "address record signature does not verify"),
            RecordError::Expired => write!(f, "address record is older than {MAX_AGE:?}"),
            RecordError::FromTheFuture => write!(f, "address record is dated in the future"),
        }
    }
}

impl std::error::Error for RecordError {}

/// How to reach a node. Fields are public so the caller fills them in; [`AddressRecord::encode`]
/// signs and [`AddressRecord::decode`] verifies.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AddressRecord {
    /// The node this record describes. It also verifies the signature.
    pub node: NodeId,
    /// Raised by the node with every new record, so a receiver keeps the highest.
    pub version: u32,
    /// Creation time, unix seconds.
    pub timestamp: u32,
    /// The NAT type of the node.
    pub nat: NatType,
    /// Addresses a peer may dial or punch over TCP.
    pub tcp: Vec<SocketAddr>,
    /// Addresses a peer may send UDP to.
    pub udp: Vec<SocketAddr>,
    /// A relay the node can be reached through, when it has registered one.
    pub relay: Option<SocketAddr>,
}

/// Unix seconds now.
pub fn unix_now() -> u32 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as u32)
        .unwrap_or(0)
}

impl AddressRecord {
    /// An empty record for `node` stamped with the current time.
    pub fn new(node: NodeId, version: u32) -> Self {
        AddressRecord {
            node,
            version,
            timestamp: unix_now(),
            nat: NatType::Unknown,
            tcp: Vec::new(),
            udp: Vec::new(),
            relay: None,
        }
    }

    /// Encode and sign with `identity`, which must be the key of `self.node`.
    pub fn encode(&self, identity: &Identity) -> Vec<u8> {
        assert_eq!(
            identity.node_id(),
            self.node,
            "a record is signed by its own node"
        );
        let (nat, step) = match self.nat {
            NatType::Unknown => (0u8, 0i16),
            NatType::EndpointIndependent => (1, 0),
            NatType::Symmetric { step } => (2, step),
        };
        let split = |list: &[SocketAddr]| {
            let v4: Vec<_> = list.iter().filter(|a| a.is_ipv4()).copied().collect();
            let v6: Vec<_> = list.iter().filter(|a| a.is_ipv6()).copied().collect();
            (v4, v6)
        };
        let (tcp4, tcp6) = split(&self.tcp);
        let (udp4, udp6) = split(&self.udp);
        let mut out = Vec::with_capacity(BASE_SIZE + 6 * 16);
        out.push(FORMAT << 4 | u8::from(self.relay.is_some()));
        out.extend(self.node.as_bytes());
        out.extend(self.version.to_be_bytes());
        out.extend(self.timestamp.to_be_bytes());
        out.push(nat);
        out.extend(step.to_be_bytes());
        for group in [&tcp4, &tcp6, &udp4, &udp6] {
            out.push(u8::try_from(group.len()).expect("at most 255 candidates per group"));
        }
        for group in [&tcp4, &tcp6, &udp4, &udp6] {
            group.iter().for_each(|a| put_addr(&mut out, a, false));
        }
        if let Some(relay) = &self.relay {
            put_addr(&mut out, relay, true);
        }
        let signature = identity.sign(&signed_bytes(&out));
        out.extend(signature);
        out
    }

    /// Parse `bytes`, verify the signature and check the age against `now` (unix seconds).
    pub fn decode(bytes: &[u8], now: u32) -> Result<Self, RecordError> {
        let short = RecordError::Malformed("truncated");
        if bytes.len() < BASE_SIZE {
            return Err(short);
        }
        let (body, sig) = bytes.split_at(bytes.len() - 64);
        if body[0] >> 4 != FORMAT {
            return Err(RecordError::Malformed("unknown format version"));
        }
        let node = NodeId::from_bytes(body[1..33].try_into().unwrap());
        let signature: [u8; 64] = sig.try_into().unwrap();
        if !node.verify(&signed_bytes(body), &signature) {
            return Err(RecordError::BadSignature);
        }
        let version = u32::from_be_bytes(body[33..37].try_into().unwrap());
        let timestamp = u32::from_be_bytes(body[37..41].try_into().unwrap());
        let step = i16::from_be_bytes(body[42..44].try_into().unwrap());
        let nat = match body[41] {
            0 => NatType::Unknown,
            1 => NatType::EndpointIndependent,
            2 => NatType::Symmetric { step },
            _ => return Err(RecordError::Malformed("unknown NAT type")),
        };
        let counts = [
            body[44] as usize,
            body[45] as usize,
            body[46] as usize,
            body[47] as usize,
        ];
        let mut at = HEADER;
        let mut groups: Vec<Vec<SocketAddr>> = Vec::new();
        for (i, count) in counts.iter().enumerate() {
            let mut group = Vec::with_capacity(*count);
            for _ in 0..*count {
                group.push(get_addr(body, &mut at, Some(i % 2 == 1)).ok_or(short_err())?);
            }
            groups.push(group);
        }
        let relay = if body[0] & 1 == 1 {
            Some(get_addr(body, &mut at, None).ok_or(short_err())?)
        } else {
            None
        };
        if at != body.len() {
            return Err(RecordError::Malformed("trailing bytes"));
        }
        let age = now as i64 - timestamp as i64;
        if age > MAX_AGE.as_secs() as i64 {
            return Err(RecordError::Expired);
        }
        if -age > MAX_SKEW.as_secs() as i64 {
            return Err(RecordError::FromTheFuture);
        }
        let mut it = groups.into_iter();
        let (t4, t6, u4, u6) = (
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
        );
        Ok(AddressRecord {
            node,
            version,
            timestamp,
            nat,
            tcp: [t4, t6].concat(),
            udp: [u4, u6].concat(),
            relay,
        })
    }
}

fn short_err() -> RecordError {
    RecordError::Malformed("truncated")
}

fn signed_bytes(body: &[u8]) -> Vec<u8> {
    [DOMAIN, body].concat()
}

/// Append an address; `tagged` adds the family byte (used by the relay only).
fn put_addr(out: &mut Vec<u8>, addr: &SocketAddr, tagged: bool) {
    match addr.ip() {
        IpAddr::V4(ip) => {
            if tagged {
                out.push(4);
            }
            out.extend(ip.octets());
        }
        IpAddr::V6(ip) => {
            if tagged {
                out.push(6);
            }
            out.extend(ip.octets());
        }
    }
    out.extend(addr.port().to_be_bytes());
}

/// Read an address at `*at`. `v6` is the family when known, `None` when a family byte leads.
fn get_addr(body: &[u8], at: &mut usize, v6: Option<bool>) -> Option<SocketAddr> {
    let v6 = match v6 {
        Some(v) => v,
        None => {
            let family = *body.get(*at)?;
            *at += 1;
            match family {
                4 => false,
                6 => true,
                _ => return None,
            }
        }
    };
    let size = if v6 { 16 } else { 4 };
    let raw = body.get(*at..*at + size + 2)?;
    *at += size + 2;
    let port = u16::from_be_bytes([raw[size], raw[size + 1]]);
    let ip = if v6 {
        IpAddr::V6(Ipv6Addr::from(<[u8; 16]>::try_from(&raw[..16]).ok()?))
    } else {
        IpAddr::V4(Ipv4Addr::from(<[u8; 4]>::try_from(&raw[..4]).ok()?))
    };
    Some(SocketAddr::new(ip, port))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v6(n: u16) -> SocketAddr {
        SocketAddr::new(
            IpAddr::V6(Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, n)),
            4000 + n,
        )
    }

    fn sample(identity: &Identity) -> AddressRecord {
        let mut r = AddressRecord::new(identity.node_id(), 7);
        r.nat = NatType::Symmetric { step: -2 };
        r.tcp = vec!["203.0.113.5:4001".parse().unwrap(), v6(1)];
        r.udp = vec!["203.0.113.5:4002".parse().unwrap()];
        r.relay = Some("198.51.100.9:7000".parse().unwrap());
        r
    }

    #[test]
    fn a_record_round_trips_with_every_field() {
        let id = Identity::generate();
        let rec = sample(&id);
        let bytes = rec.encode(&id);
        let back = AddressRecord::decode(&bytes, rec.timestamp).unwrap();
        // Candidates are grouped by family on the wire, IPv4 first.
        assert_eq!(back.tcp, rec.tcp);
        assert_eq!(back, rec);
    }

    #[test]
    fn an_empty_record_is_112_bytes_and_16_ipv6_candidates_fit_in_400() {
        let id = Identity::generate();
        let empty = AddressRecord::new(id.node_id(), 1);
        assert_eq!(empty.encode(&id).len(), BASE_SIZE);
        let mut full = empty.clone();
        full.tcp = (0..8).map(v6).collect();
        full.udp = (0..8).map(v6).collect();
        assert_eq!(full.encode(&id).len(), 400);
        let mut v4 = empty;
        v4.tcp = vec!["1.2.3.4:5".parse().unwrap(); 3];
        assert_eq!(v4.encode(&id).len(), BASE_SIZE + 18);
    }

    #[test]
    fn a_tampered_record_is_refused_wherever_the_byte_is_changed() {
        let id = Identity::generate();
        let rec = sample(&id);
        let bytes = rec.encode(&id);
        for i in 0..bytes.len() {
            let mut bad = bytes.clone();
            bad[i] ^= 0x01;
            assert!(
                AddressRecord::decode(&bad, rec.timestamp).is_err(),
                "byte {i}"
            );
        }
    }

    #[test]
    fn a_record_signed_by_another_key_is_refused() {
        let (a, b) = (Identity::generate(), Identity::generate());
        let mut bytes = sample(&a).encode(&a);
        // Claim to be node b while keeping a's signature.
        bytes[1..33].copy_from_slice(b.node_id().as_bytes());
        assert_eq!(
            AddressRecord::decode(&bytes, unix_now()),
            Err(RecordError::BadSignature)
        );
    }

    #[test]
    fn a_record_older_than_ten_minutes_is_expired() {
        let id = Identity::generate();
        let rec = sample(&id);
        let bytes = rec.encode(&id);
        let t = rec.timestamp;
        assert!(AddressRecord::decode(&bytes, t + 600).is_ok());
        assert_eq!(
            AddressRecord::decode(&bytes, t + 601),
            Err(RecordError::Expired)
        );
        assert!(AddressRecord::decode(&bytes, t - 60).is_ok());
        assert_eq!(
            AddressRecord::decode(&bytes, t - 61),
            Err(RecordError::FromTheFuture)
        );
    }

    #[test]
    fn truncated_input_is_refused_without_panicking() {
        let id = Identity::generate();
        let bytes = sample(&id).encode(&id);
        for len in 0..bytes.len() {
            assert!(AddressRecord::decode(&bytes[..len], unix_now()).is_err());
        }
    }
}
