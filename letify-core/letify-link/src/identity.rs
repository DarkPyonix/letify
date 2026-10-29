//! Node identity: an ed25519 key pair, its public key as the node id, and the X25519 static
//! key that the Noise handshake uses, derived from the same key. It does not store keys.

use std::fmt;

use ed25519_dalek::{SigningKey, VerifyingKey};

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

/// Which initiators a responder accepts.
#[derive(Clone)]
pub struct PeerPolicy {
    check: std::sync::Arc<dyn Fn(&NodeId) -> bool + Send + Sync>,
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
        }
    }

    pub(crate) fn accepts(&self, id: &NodeId) -> bool {
        (self.check)(id)
    }
}
