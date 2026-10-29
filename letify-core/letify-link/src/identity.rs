//! Node identity: an ed25519 key pair, its public key as the node id, and the X25519 static
//! key that the Noise handshake uses, derived from the same key. It does not store keys.

use std::collections::HashSet;
use std::fmt;

use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use tokio::sync::watch;

use crate::error::{Error, Result};

/// The identity of a node: its ed25519 public key.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId([u8; 32]);

impl NodeId {
    /// Wrap the 32 bytes of an ed25519 public key without checking them. A value made here may
    /// not be a valid node id: see [`NodeId::try_from_bytes`]. Every check of a peer, a
    /// signature or a policy refuses an invalid one.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        NodeId(bytes)
    }

    /// Wrap the 32 bytes of an ed25519 public key, or refuse bytes that are not a valid node
    /// id: see [`NodeId::validate`].
    pub fn try_from_bytes(bytes: [u8; 32]) -> Result<Self> {
        let id = NodeId(bytes);
        id.validate()?;
        Ok(id)
    }

    /// Check that this is the canonical encoding of an ed25519 point that is neither of small
    /// order nor outside the prime order subgroup. Other points have several encodings or
    /// several keys that convert to the same X25519 key, so a revoke on the bytes could be
    /// bypassed with a sibling of the node's key.
    pub fn validate(&self) -> Result<()> {
        let bad = |why: &str| Error::Handshake(format!("not a valid node id ({why}): {self}"));
        let key = VerifyingKey::from_bytes(&self.0).map_err(|_| bad("not a curve point"))?;
        let point = key.to_edwards();
        if point.compress().to_bytes() != self.0 {
            return Err(bad("not the canonical encoding"));
        }
        if key.is_weak() {
            return Err(bad("small order"));
        }
        if !point.is_torsion_free() {
            return Err(bad("not in the prime order subgroup"));
        }
        Ok(())
    }

    /// The id that stands for this node in every allow, revoke and policy lookup: the
    /// encoding with the sign bit of the last byte cleared. The points A and -A, which differ
    /// only in that bit, give the same X25519 key, so they are one node.
    pub fn canonical(&self) -> NodeId {
        let mut bytes = self.0;
        bytes[31] &= 0x7f;
        NodeId(bytes)
    }

    /// The 32 bytes of the public key.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Check an ed25519 signature by this node over `message`.
    pub(crate) fn verify(&self, message: &[u8], signature: &[u8; 64]) -> bool {
        self.validate().is_ok()
            && VerifyingKey::from_bytes(&self.0)
                .map(|key| {
                    key.verify_strict(message, &Signature::from_bytes(signature))
                        .is_ok()
                })
                .unwrap_or(false)
    }

    /// The X25519 public key that corresponds to this node id, by the standard conversion of
    /// the Edwards point to a Montgomery point.
    pub(crate) fn x25519(&self) -> Result<[u8; 32]> {
        self.validate()?;
        let key = VerifyingKey::from_bytes(&self.0)
            .map_err(|e| Error::Handshake(format!("node id is not an ed25519 key: {e}")))?;
        Ok(key.to_montgomery().to_bytes())
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.iter().try_for_each(|b| write!(f, "{b:02x}"))
    }
}

impl std::str::FromStr for NodeId {
    type Err = Error;

    /// Parse the 64 hex characters that `Display` writes.
    fn from_str(s: &str) -> Result<Self> {
        let bad = || Error::Handshake(format!("not a node id: {s:?}"));
        if s.len() != 64 || !s.is_ascii() {
            return Err(bad());
        }
        let mut out = [0u8; 32];
        for (i, byte) in out.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).map_err(|_| bad())?;
        }
        NodeId::try_from_bytes(out)
    }
}

impl fmt::Debug for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NodeId({self})")
    }
}

/// An ed25519 node key. The caller may supply its own secret key.
#[derive(Clone)]
pub struct Identity {
    key: SigningKey,
}

impl Identity {
    /// Generate a fresh key from the operating system's random source.
    pub fn generate() -> Self {
        let mut seed = [0u8; 32];
        getrandom::getrandom(&mut seed).expect("the operating system has a random source");
        Self::from_secret_bytes(seed)
    }

    /// Use an ed25519 secret key (its 32 byte seed) that the caller holds.
    pub fn from_secret_bytes(secret: [u8; 32]) -> Self {
        Identity {
            key: SigningKey::from_bytes(&secret),
        }
    }

    /// The 32 byte seed of the ed25519 secret key.
    pub fn secret_bytes(&self) -> [u8; 32] {
        self.key.to_bytes()
    }

    /// The node id: the public key.
    pub fn node_id(&self) -> NodeId {
        NodeId(self.key.verifying_key().to_bytes())
    }

    /// Sign `message` with the node key.
    pub(crate) fn sign(&self, message: &[u8]) -> [u8; 64] {
        self.key.sign(message).to_bytes()
    }

    /// The X25519 static secret key, derived from the ed25519 key.
    pub(crate) fn x25519_secret(&self) -> [u8; 32] {
        self.key.to_scalar_bytes()
    }
}

impl fmt::Debug for Identity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Identity({})", self.node_id())
    }
}

/// The nodes revoked at run time, shared by every clone of a policy.
struct Revoked {
    set: std::sync::RwLock<HashSet<NodeId>>,
    /// Counts every change, so a session can wait for one.
    changes: watch::Sender<u64>,
}

/// Which initiators a responder accepts. The clones of a policy share one set of revoked
/// nodes, which can change while sessions run: see [`PeerPolicy::revoke`].
#[derive(Clone)]
pub struct PeerPolicy {
    /// Keyed on canonical ids: see [`NodeId::canonical`].
    check: std::sync::Arc<dyn Fn(&NodeId) -> bool + Send + Sync>,
    revoked: std::sync::Arc<Revoked>,
}

impl PeerPolicy {
    /// Accept exactly the listed node ids.
    pub fn allow(ids: Vec<NodeId>) -> Self {
        let ids: HashSet<NodeId> = ids.iter().map(NodeId::canonical).collect();
        Self::from_fn(move |id| ids.contains(&id.canonical()))
    }

    /// Accept the node ids for which the callback returns true.
    pub fn from_fn(check: impl Fn(&NodeId) -> bool + Send + Sync + 'static) -> Self {
        PeerPolicy {
            check: std::sync::Arc::new(check),
            revoked: std::sync::Arc::new(Revoked {
                set: std::sync::RwLock::new(HashSet::new()),
                changes: watch::channel(0).0,
            }),
        }
    }

    /// Refuse `id` from now on, whatever the allow list or callback says. A handshake from it
    /// fails, a subflow that resumes one of its sessions is refused, and a session accepted
    /// under this policy (or one of its clones) closes with the reason `revoked`.
    pub fn revoke(&self, id: NodeId) {
        let mut set = self.revoked.set.write().unwrap_or_else(|e| e.into_inner());
        if set.insert(id.canonical()) {
            self.revoked.changes.send_modify(|n| *n += 1);
        }
    }

    /// Remove `id` from the revoked set. The allow list or callback applies to it again.
    pub fn reinstate(&self, id: NodeId) {
        let mut set = self.revoked.set.write().unwrap_or_else(|e| e.into_inner());
        if set.remove(&id.canonical()) {
            self.revoked.changes.send_modify(|n| *n += 1);
        }
    }

    /// Whether `id` was revoked and not reinstated.
    pub fn is_revoked(&self, id: &NodeId) -> bool {
        self.revoked
            .set
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .contains(&id.canonical())
    }

    /// A counter that changes whenever the revoked set does.
    pub(crate) fn changes(&self) -> watch::Receiver<u64> {
        self.revoked.changes.subscribe()
    }

    pub(crate) fn accepts(&self, id: &NodeId) -> bool {
        id.validate().is_ok() && !self.is_revoked(id) && (self.check)(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The compressed encoding of a point of order 8.
    const ORDER_8: [u8; 32] = [
        0xc7, 0x17, 0x6a, 0x70, 0x3d, 0x4d, 0xd8, 0x4f, 0xba, 0x3c, 0x0b, 0x76, 0x0d, 0x10, 0x67,
        0x0f, 0x2a, 0x20, 0x53, 0xfa, 0x2c, 0x39, 0xcc, 0xc6, 0x4e, 0xc7, 0xfd, 0x77, 0x92, 0xac,
        0x03, 0x7a,
    ];

    fn flipped(id: &NodeId) -> NodeId {
        let mut b = *id.as_bytes();
        b[31] ^= 0x80;
        NodeId::from_bytes(b)
    }

    /// An identity whose public key has the sign bit set, so `canonical` changes it.
    fn identity_with_sign(set: bool) -> Identity {
        loop {
            let i = Identity::generate();
            if (i.node_id().as_bytes()[31] & 0x80 != 0) == set {
                return i;
            }
        }
    }

    #[test]
    fn a_generated_node_id_is_valid() {
        for _ in 0..8 {
            Identity::generate().node_id().validate().unwrap();
        }
    }

    #[test]
    fn a_sign_flipped_node_id_is_refused() {
        let me = identity_with_sign(false);
        let policy = PeerPolicy::allow(vec![me.node_id()]);
        let other = flipped(&me.node_id());
        assert_eq!(other.canonical(), me.node_id());
        // Revoking one encoding refuses the other.
        policy.revoke(me.node_id());
        assert!(!policy.accepts(&other));
        assert!(!policy.accepts(&me.node_id()));
        assert!(policy.is_revoked(&other));
        policy.reinstate(other);
        assert!(policy.accepts(&me.node_id()));
        // A flipped id is no longer the id bytes a signature was made for.
        let sig = me.sign(b"m");
        assert!(me.node_id().verify(b"m", &sig));
        assert!(!other.verify(b"m", &sig));
    }

    #[test]
    fn a_weak_node_id_is_refused() {
        let mut identity_point = [0u8; 32];
        identity_point[0] = 1;
        let mut minus_one = [0xffu8; 32];
        minus_one[0] = 0xec;
        minus_one[31] = 0x7f;
        let mut noncanonical = [0xffu8; 32];
        noncanonical[0] = 0xee;
        noncanonical[31] = 0x7f;
        for bytes in [identity_point, minus_one, noncanonical, ORDER_8] {
            let id = NodeId::from_bytes(bytes);
            assert!(NodeId::try_from_bytes(bytes).is_err(), "{id}");
            assert!(id.x25519().is_err(), "{id}");
            assert!(id.to_string().parse::<NodeId>().is_err(), "{id}");
            assert!(!PeerPolicy::from_fn(|_| true).accepts(&id), "{id}");
        }
    }

    #[test]
    fn a_node_id_with_a_torsion_component_is_refused() {
        let me = Identity::generate().node_id();
        let key = VerifyingKey::from_bytes(me.as_bytes()).unwrap();
        let small = VerifyingKey::from_bytes(&ORDER_8).unwrap();
        let mixed = (key.to_edwards() + small.to_edwards()).compress().to_bytes();
        assert!(VerifyingKey::from_bytes(&mixed).is_ok());
        assert!(NodeId::try_from_bytes(mixed).is_err());
        assert!(!PeerPolicy::from_fn(|_| true).accepts(&NodeId::from_bytes(mixed)));
    }

    #[test]
    fn revoking_a_node_also_covers_its_alternate_encodings() {
        let me = identity_with_sign(true);
        let policy = PeerPolicy::from_fn(|_| true);
        policy.revoke(flipped(&me.node_id()));
        assert!(policy.is_revoked(&me.node_id()));
        assert!(!policy.accepts(&me.node_id()));
        assert!(!policy.accepts(&flipped(&me.node_id())));
    }

    #[test]
    fn the_allow_list_admits_both_encodings_of_a_listed_node() {
        let me = identity_with_sign(true);
        let policy = PeerPolicy::allow(vec![flipped(&me.node_id())]);
        assert!(policy.accepts(&me.node_id()));
    }
}
