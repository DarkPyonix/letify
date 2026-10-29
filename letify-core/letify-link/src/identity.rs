//! Node identity: an ed25519 key pair, its public key as the node id, and the X25519 static
//! key that the Noise handshake uses, derived from the same key. It does not store keys.

use std::collections::HashSet;
use std::fmt;

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use tokio::sync::watch;

use crate::error::{Error, Result};

/// The identity of a node: its ed25519 public key.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId([u8; 32]);

impl NodeId {
    /// Wrap the 32 bytes of an ed25519 public key.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        NodeId(bytes)
    }

    /// The 32 bytes of the public key.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Check an ed25519 signature by this node over `message`.
    pub(crate) fn verify(&self, message: &[u8], signature: &[u8; 64]) -> bool {
        VerifyingKey::from_bytes(&self.0)
            .map(|key| {
                key.verify(message, &Signature::from_bytes(signature))
                    .is_ok()
            })
            .unwrap_or(false)
    }

    /// The X25519 public key that corresponds to this node id, by the standard conversion of
    /// the Edwards point to a Montgomery point.
    pub(crate) fn x25519(&self) -> Result<[u8; 32]> {
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
        Ok(NodeId(out))
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
    check: std::sync::Arc<dyn Fn(&NodeId) -> bool + Send + Sync>,
    revoked: std::sync::Arc<Revoked>,
}

impl PeerPolicy {
    /// Accept exactly the listed node ids.
    pub fn allow(ids: Vec<NodeId>) -> Self {
        Self::from_fn(move |id| ids.contains(id))
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
        if set.insert(id) {
            self.revoked.changes.send_modify(|n| *n += 1);
        }
    }

    /// Remove `id` from the revoked set. The allow list or callback applies to it again.
    pub fn reinstate(&self, id: NodeId) {
        let mut set = self.revoked.set.write().unwrap_or_else(|e| e.into_inner());
        if set.remove(&id) {
            self.revoked.changes.send_modify(|n| *n += 1);
        }
    }

    /// Whether `id` was revoked and not reinstated.
    pub fn is_revoked(&self, id: &NodeId) -> bool {
        self.revoked
            .set
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .contains(id)
    }

    /// A counter that changes whenever the revoked set does.
    pub(crate) fn changes(&self) -> watch::Receiver<u64> {
        self.revoked.changes.subscribe()
    }

    pub(crate) fn accepts(&self, id: &NodeId) -> bool {
        !self.is_revoked(id) && (self.check)(id)
    }
}
