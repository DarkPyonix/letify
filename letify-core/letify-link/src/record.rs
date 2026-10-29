//! The address record: one signed, documented binary description of how to reach a node. It
//! owns the encoding, the signature and the age check. It does not own transport (that is the
//! [`Rendezvous`](crate::rendezvous::Rendezvous)) or the choice of candidates.
//!
//! # Wire format (version 2, all integers big endian)
//!
//! ```text
//! offset  size  field
//! 0       1     flags: bits 7..4 format version (2), bit 0 relay present
//! 1       32    node id (ed25519 public key)
//! 33      4     record version, a counter the node raises with every new record
//! 37      4     timestamp, unix seconds
//! 41      1     NAT type: 0 unknown, 1 endpoint independent, 2 symmetric
//! 42      2     port step of a symmetric NAT as i16 (0 otherwise)
//! 44      4     candidate counts: TCP IPv4, TCP IPv6, UDP IPv4, UDP IPv6
//! 48      ...   TCP IPv4 (4 byte address, 2 byte port), TCP IPv6 (16 + 2), UDP IPv4, UDP IPv6
//!         ...   relay, when flagged: 1 byte family (4 or 6), then address and port
//!         ...   transports: 1 byte count (at most 4), then per transport a 1 byte length
//!               (1 to 32) and that many printable ASCII bytes
//!         64    ed25519 signature over "letify-link record v2" and every byte before it
//! ```
//!
//! The record is 113 bytes with no candidates, plus 6 bytes per IPv4 candidate and 18 per IPv6
//! candidate, so 8 TCP and 8 UDP IPv6 candidates take 401 bytes. At most [`MAX_CANDIDATES`]
//! of each protocol are carried, chosen by [`prioritize`]. A record is valid for [`MAX_AGE`]
//! after its timestamp. [`to_pkarr_txt`] and [`from_pkarr_txt`] carry a signed record in the
//! TXT strings of a pkarr packet.

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
pub const BASE_SIZE: usize = 113;
/// Most candidates of each protocol (TCP, UDP) a record carries.
pub const MAX_CANDIDATES: usize = 8;
/// Most transport identifiers a record carries.
pub const MAX_TRANSPORTS: usize = 4;
/// Longest transport identifier, in bytes.
pub const MAX_TRANSPORT_LEN: usize = 32;
/// The transport a record with no `transports` stands for.
pub const DEFAULT_TRANSPORT: &str = "letify-link/1";
/// Longest string of a DNS TXT record.
pub const TXT_CHUNK: usize = 255;

const FORMAT: u8 = 2;
const DOMAIN: &[u8] = b"letify-link record v2";
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
    /// Identifiers of the connection backends the node accepts, in its order of preference.
    /// Empty means [`DEFAULT_TRANSPORT`] only.
    pub transports: Vec<String>,
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
            transports: Vec::new(),
        }
    }

    /// Encode and sign with `identity`, which must be the key of `self.node`. It keeps the first
    /// [`MAX_CANDIDATES`] candidates of each protocol, in the order given (see [`prioritize`]),
    /// and panics on a transport identifier that [`valid_transport`] refuses or on more than
    /// [`MAX_TRANSPORTS`] of them, which are programming errors.
    pub fn encode(&self, identity: &Identity) -> Vec<u8> {
        assert_eq!(
            identity.node_id(),
            self.node,
            "a record is signed by its own node"
        );
        assert!(
            self.transports.len() <= MAX_TRANSPORTS
                && self.transports.iter().all(|t| valid_transport(t))
        );
        let (nat, step) = match self.nat {
            NatType::Unknown => (0u8, 0i16),
            NatType::EndpointIndependent => (1, 0),
            NatType::Symmetric { step } => (2, step),
        };
        let split = |list: &[SocketAddr]| {
            let list = &list[..list.len().min(MAX_CANDIDATES)];
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
        out.push(self.transports.len() as u8);
        for t in &self.transports {
            out.push(t.len() as u8);
            out.extend(t.as_bytes());
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
        if counts[0] + counts[1] > MAX_CANDIDATES || counts[2] + counts[3] > MAX_CANDIDATES {
            return Err(RecordError::Malformed("too many candidates"));
        }
        let count = *body.get(at).ok_or(short_err())? as usize;
        at += 1;
        if count > MAX_TRANSPORTS {
            return Err(RecordError::Malformed("too many transports"));
        }
        let mut transports = Vec::with_capacity(count);
        for _ in 0..count {
            let len = *body.get(at).ok_or(short_err())? as usize;
            let raw = body.get(at + 1..at + 1 + len).ok_or(short_err())?;
            at += 1 + len;
            let name = std::str::from_utf8(raw)
                .ok()
                .filter(|t| valid_transport(t))
                .ok_or(RecordError::Malformed("bad transport identifier"))?;
            transports.push(name.to_owned());
        }
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
            transports,
        })
    }

    /// Whether the record says the node accepts `transport`.
    pub fn speaks(&self, transport: &str) -> bool {
        if self.transports.is_empty() {
            transport == DEFAULT_TRANSPORT
        } else {
            self.transports.iter().any(|t| t == transport)
        }
    }

    /// The first of `mine`, in my order of preference, that the node also accepts.
    pub fn choose_transport<'a>(&self, mine: &[&'a str]) -> Option<&'a str> {
        mine.iter().copied().find(|t| self.speaks(t))
    }

    /// [`AddressRecord::encode`] as TXT strings, see [`to_pkarr_txt`].
    pub fn encode_txt(&self, identity: &Identity) -> Vec<String> {
        to_pkarr_txt(&self.encode(identity))
    }

    /// [`from_pkarr_txt`] then [`AddressRecord::decode`].
    pub fn decode_txt<S: AsRef<str>>(strings: &[S], now: u32) -> Result<Self, RecordError> {
        Self::decode(&from_pkarr_txt(strings)?, now)
    }
}

/// Whether `name` is a legal transport identifier: 1 to [`MAX_TRANSPORT_LEN`] printable ASCII
/// characters, no space.
pub fn valid_transport(name: &str) -> bool {
    (1..=MAX_TRANSPORT_LEN).contains(&name.len())
        && name.bytes().all(|b| (0x21..=0x7e).contains(&b))
}

/// A candidate address with what the node knows about it, to be ordered by [`prioritize`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Endpoint {
    /// The address.
    pub addr: SocketAddr,
    /// Unix seconds at which the address last carried traffic or answered a probe, or 0 when it
    /// never did.
    pub verified_at: u32,
}

/// Whether `ip` is loopback, link-local, private, carrier-grade NAT or unique-local.
pub fn is_private(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            let o = v4.octets();
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_unspecified()
                || (o[0] == 100 && (o[1] & 0xc0) == 64)
        }
        IpAddr::V6(v6) => {
            let seg = v6.segments();
            v6.is_loopback()
                || v6.is_unspecified()
                || (seg[0] & 0xfe00) == 0xfc00
                || (seg[0] & 0xffc0) == 0xfe80
        }
    }
}

/// Order candidates for a record, best first: the most recently verified, then public before
/// private, then IPv4 before IPv6, then the order given. The caller keeps the first
/// [`MAX_CANDIDATES`].
pub fn prioritize(endpoints: &[Endpoint]) -> Vec<SocketAddr> {
    let mut list = endpoints.to_vec();
    list.sort_by_key(|e| {
        (
            std::cmp::Reverse(e.verified_at),
            is_private(&e.addr.ip()),
            e.addr.is_ipv6(),
        )
    });
    list.into_iter().map(|e| e.addr).collect()
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

/// The signed record `bytes` as unpadded base64url text split into strings of at most
/// [`TXT_CHUNK`] characters, for the character-strings of one DNS TXT record.
pub fn to_pkarr_txt(bytes: &[u8]) -> Vec<String> {
    let mut text = String::with_capacity(bytes.len() * 4 / 3 + 3);
    for chunk in bytes.chunks(3) {
        let n = chunk.iter().fold(0u32, |a, b| a << 8 | *b as u32) << (8 * (3 - chunk.len()));
        for i in 0..chunk.len() + 1 {
            text.push(B64[(n >> (18 - 6 * i) & 63) as usize] as char);
        }
    }
    text.as_bytes()
        .chunks(TXT_CHUNK)
        .map(|c| String::from_utf8(c.to_vec()).expect("base64url is ASCII"))
        .collect()
}

/// Join the TXT strings written by [`to_pkarr_txt`] back into the signed record bytes.
pub fn from_pkarr_txt<S: AsRef<str>>(strings: &[S]) -> Result<Vec<u8>, RecordError> {
    let bad = RecordError::Malformed("not base64url text");
    let mut out = Vec::new();
    let (mut acc, mut bits) = (0u32, 0u32);
    let mut total = 0usize;
    for s in strings {
        if s.as_ref().len() > TXT_CHUNK {
            return Err(RecordError::Malformed("TXT string longer than 255"));
        }
        for b in s.as_ref().bytes() {
            let v = B64
                .iter()
                .position(|c| *c == b)
                .ok_or(RecordError::Malformed("not base64url text"))? as u32;
            acc = acc << 6 | v;
            bits += 6;
            total += 1;
            if bits >= 8 {
                bits -= 8;
                out.push((acc >> bits) as u8);
                acc &= (1 << bits) - 1;
            }
        }
    }
    // A length of 1 mod 4 cannot come from whole bytes, and leftover bits must be zero.
    if total % 4 == 1 || acc != 0 {
        return Err(bad);
    }
    Ok(out)
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
        r.transports = vec!["letify-link/1".into(), "iroh/1".into()];
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
    fn an_empty_record_is_113_bytes_and_16_ipv6_candidates_fit_in_401() {
        let id = Identity::generate();
        let empty = AddressRecord::new(id.node_id(), 1);
        assert_eq!(empty.encode(&id).len(), BASE_SIZE);
        let mut full = empty.clone();
        full.tcp = (0..8).map(v6).collect();
        full.udp = (0..8).map(v6).collect();
        assert_eq!(full.encode(&id).len(), 401);
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

    fn ep(addr: &str, verified_at: u32) -> Endpoint {
        Endpoint {
            addr: addr.parse().unwrap(),
            verified_at,
        }
    }

    #[test]
    fn candidates_are_ordered_by_recency_then_public_then_ipv4() {
        let order = prioritize(&[
            ep("10.0.0.2:1", 100),
            ep("[2001:db8::1]:1", 100),
            ep("203.0.113.1:1", 100),
            ep("203.0.113.2:1", 90),
            ep("192.168.1.1:1", 200),
            ep("203.0.113.3:1", 0),
        ]);
        let want: Vec<SocketAddr> = [
            "192.168.1.1:1",
            "203.0.113.1:1",
            "[2001:db8::1]:1",
            "10.0.0.2:1",
            "203.0.113.2:1",
            "203.0.113.3:1",
        ]
        .iter()
        .map(|a| a.parse().unwrap())
        .collect();
        assert_eq!(order, want);
    }

    #[test]
    fn private_ranges_are_recognised() {
        for ip in [
            "127.0.0.1",
            "10.1.1.1",
            "172.16.0.1",
            "192.168.0.1",
            "169.254.1.1",
            "100.64.0.1",
            "fd00::1",
            "fe80::1",
            "::1",
        ] {
            assert!(is_private(&ip.parse().unwrap()), "{ip}");
        }
        for ip in ["8.8.8.8", "100.128.0.1", "2001:db8::1"] {
            assert!(!is_private(&ip.parse().unwrap()), "{ip}");
        }
    }

    #[test]
    fn encode_keeps_the_first_eight_candidates_of_each_protocol() {
        let id = Identity::generate();
        let mut r = AddressRecord::new(id.node_id(), 1);
        r.tcp = (0..12).map(v6).collect();
        r.udp = (0..9).map(v6).collect();
        let back = AddressRecord::decode(&r.encode(&id), r.timestamp).unwrap();
        assert_eq!(back.tcp, r.tcp[..8]);
        assert_eq!(back.udp, r.udp[..8]);
    }

    #[test]
    fn a_record_with_more_than_eight_candidates_on_the_wire_is_refused() {
        // Build the bytes by hand: 9 IPv4 TCP candidates, signed correctly.
        let id = Identity::generate();
        let mut out = vec![FORMAT << 4];
        out.extend(id.node_id().as_bytes());
        out.extend(1u32.to_be_bytes());
        out.extend(unix_now().to_be_bytes());
        out.extend([0, 0, 0, 9, 0, 0, 0]);
        for _ in 0..9 {
            out.extend([1, 2, 3, 4, 0, 5]);
        }
        out.push(0);
        let sig = id.sign(&signed_bytes(&out));
        out.extend(sig);
        assert_eq!(
            AddressRecord::decode(&out, unix_now()),
            Err(RecordError::Malformed("too many candidates"))
        );
    }

    #[test]
    fn a_peer_chooses_the_first_transport_both_speak() {
        let id = Identity::generate();
        let mut r = AddressRecord::new(id.node_id(), 1);
        assert!(r.speaks("letify-link/1") && !r.speaks("iroh/1"));
        r.transports = vec!["iroh/1".into(), "letify-link/1".into()];
        let back = AddressRecord::decode(&r.encode(&id), r.timestamp).unwrap();
        assert_eq!(back.transports, r.transports);
        assert_eq!(
            back.choose_transport(&["letify-link/1", "iroh/1"]),
            Some("letify-link/1")
        );
        assert_eq!(back.choose_transport(&["quic/9"]), None);
    }

    #[test]
    fn transport_identifiers_are_limited() {
        assert!(valid_transport("letify-link/1"));
        assert!(!valid_transport(""));
        assert!(!valid_transport("has space"));
        assert!(!valid_transport("naïve"));
        assert!(!valid_transport(&"x".repeat(33)));
    }

    #[test]
    fn the_largest_record_fits_a_pkarr_packet_and_round_trips_through_txt() {
        let id = Identity::generate();
        let mut r = AddressRecord::new(id.node_id(), 1);
        r.tcp = (0..8).map(v6).collect();
        r.udp = (0..8).map(v6).collect();
        r.relay = Some(v6(99));
        r.nat = NatType::Symmetric { step: 4 };
        r.transports = (0..4).map(|i| format!("{i}{}", "t".repeat(31))).collect();
        let bytes = r.encode(&id);
        assert_eq!(bytes.len(), 552);
        let txt = r.encode_txt(&id);
        assert_eq!(txt.len(), 3);
        assert!(txt.iter().all(|s| s.len() <= TXT_CHUNK && s.is_ascii()));
        assert_eq!(txt.iter().map(String::len).sum::<usize>(), 736);
        assert!(
            736 + txt.len() + 100 < 1000,
            "room for the packet around it"
        );
        assert_eq!(AddressRecord::decode_txt(&txt, r.timestamp).unwrap(), r);
    }

    #[test]
    fn txt_encoding_round_trips_every_length_and_refuses_bad_text() {
        for len in 0..40usize {
            let data: Vec<u8> = (0..len).map(|i| (i * 37 + 11) as u8).collect();
            assert_eq!(from_pkarr_txt(&to_pkarr_txt(&data)).unwrap(), data, "{len}");
        }
        assert!(from_pkarr_txt(&["ab+d"]).is_err());
        assert!(from_pkarr_txt(&["a"]).is_err());
        assert!(from_pkarr_txt(&["x".repeat(256)]).is_err());
        // A tampered text fails the record signature, not just the text.
        let id = Identity::generate();
        let r = sample(&id);
        let mut txt = r.encode_txt(&id);
        let flipped = if txt[0].starts_with('A') { 'B' } else { 'A' };
        txt[0].replace_range(10..11, &flipped.to_string());
        assert!(AddressRecord::decode_txt(&txt, r.timestamp).is_err());
    }

    #[test]
    fn an_older_record_than_the_one_held_is_refused_as_stale() {
        let id = Identity::generate();
        let cache = RecordCache::new();
        let at = |version, ts| {
            let mut r = AddressRecord::new(id.node_id(), version);
            r.timestamp = ts;
            r.encode(&id)
        };
        let now = unix_now();
        assert!(cache.accept(&at(5, now), now).is_ok());
        assert_eq!(cache.accept(&at(4, now), now), Err(RecordError::Stale));
        // The same version with an older timestamp is also a rollback.
        assert_eq!(cache.accept(&at(5, now - 10), now), Err(RecordError::Stale));
        // The same record again, and a newer one, are taken.
        assert!(cache.accept(&at(5, now), now).is_ok());
        assert_eq!(cache.accept(&at(6, now), now).unwrap().version, 6);
        assert_eq!(cache.get(&id.node_id()).unwrap().version, 6);
        // Another node has its own entry.
        let other = Identity::generate();
        let r = AddressRecord::new(other.node_id(), 1);
        assert!(cache.accept(&r.encode(&other), now).is_ok());
    }

    #[test]
    fn a_record_that_fails_the_signature_does_not_enter_the_cache() {
        let id = Identity::generate();
        let cache = RecordCache::new();
        let mut bytes = AddressRecord::new(id.node_id(), 9).encode(&id);
        let n = bytes.len();
        bytes[n - 1] ^= 1;
        assert!(cache.accept(&bytes, unix_now()).is_err());
        assert!(cache.get(&id.node_id()).is_none());
    }

    #[test]
    fn the_cache_is_bounded_and_drops_the_oldest_entry() {
        let cache = RecordCache::with_capacity(2);
        let now = unix_now();
        let mut ids = Vec::new();
        for age in [30u32, 20, 10] {
            let id = Identity::generate();
            let mut r = AddressRecord::new(id.node_id(), 1);
            r.timestamp = now - age;
            cache.accept(&r.encode(&id), now).unwrap();
            ids.push(id.node_id());
        }
        assert!(cache.get(&ids[0]).is_none());
        assert!(cache.get(&ids[1]).is_some() && cache.get(&ids[2]).is_some());
    }
}
