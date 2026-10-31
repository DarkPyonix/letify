//! The Noise_IK handshake with its plaintext version header. It owns version negotiation,
//! the allow list check and the keys of the resulting session; it does not run the session.
//!
//! Wire format, all integers big endian:
//!
//! - Initiator: `"LLNK"`, lowest version (2), highest version (2), length (2), Noise message 1.
//!   The first eight bytes are the Noise prologue, so a tampered header fails the handshake.
//!   Message 1 carries the initiator's ed25519 node id (32), its stream window (4), its
//!   keepalive interval in milliseconds (4, 0 means off) and its reorder bound in KiB (4).
//! - Responder, accepted: `"LLNK"`, 0, chosen version (2), length (2), Noise message 2.
//!   Message 2 carries the chosen version again (2), the responder's stream window (4), its
//!   keepalive interval (4) and its reorder bound in KiB (4).
//! - Responder, refused for versions: `"LLNK"`, 1, lowest (2), highest (2), text length (2),
//!   text. A refusal for a key it does not accept is a closed transport and nothing else.

use std::time::Duration;

use snow::StatelessTransportState;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use crate::config::Config;
use crate::error::{Error, Result};
use crate::identity::{Identity, NodeId, PeerPolicy};

const MAGIC: &[u8; 4] = b"LLNK";
const PARAMS: &str = "Noise_IK_25519_ChaChaPoly_BLAKE2s";

/// What a finished handshake yields.
pub(crate) struct Established {
    pub crypto: StatelessTransportState,
    pub peer: NodeId,
    pub version: u16,
    pub peer_window: u32,
    /// The keepalive interval the peer advertised; zero means the peer sends none.
    pub peer_keepalive: Duration,
    /// The reorder bound the peer advertised, in bytes.
    pub peer_reorder: usize,
    pub session_id: [u8; 16],
}

fn hs_err(why: impl std::fmt::Display) -> Error {
    Error::Handshake(why.to_string())
}

async fn read_exact<T: AsyncRead + Unpin>(t: &mut T, buf: &mut [u8]) -> Result<()> {
    t.read_exact(buf).await.map(|_| ()).map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            hs_err("the other side closed the transport, it may not accept this key")
        } else {
            Error::Io(e)
        }
    })
}

/// The keepalive interval as the handshake carries it: whole milliseconds, 0 for off.
fn keepalive_ms(cfg: &Config) -> u32 {
    cfg.keepalive_interval.as_millis().min(u32::MAX as u128) as u32
}

fn prologue(low: u16, high: u16) -> [u8; 8] {
    let mut p = [0u8; 8];
    p[..4].copy_from_slice(MAGIC);
    p[4..6].copy_from_slice(&low.to_be_bytes());
    p[6..8].copy_from_slice(&high.to_be_bytes());
    p
}

fn builder<'a>(secret: &'a [u8; 32], prologue: &'a [u8]) -> snow::Builder<'a> {
    let params = PARAMS.parse().expect("the Noise pattern name is valid");
    snow::Builder::new(params)
        .prologue(prologue)
        .local_private_key(secret)
}

/// Run the initiator side over `t`.
pub(crate) async fn initiate<T: AsyncRead + AsyncWrite + Unpin>(
    t: &mut T,
    id: &Identity,
    remote: NodeId,
    cfg: &Config,
) -> Result<Established> {
    tokio::time::timeout(cfg.handshake_timeout, initiate_inner(t, id, remote, cfg))
        .await
        .map_err(|_| hs_err("timed out"))?
}

async fn initiate_inner<T: AsyncRead + AsyncWrite + Unpin>(
    t: &mut T,
    id: &Identity,
    remote: NodeId,
    cfg: &Config,
) -> Result<Established> {
    let (low, high) = (*cfg.versions.start(), *cfg.versions.end());
    let pro = prologue(low, high);
    let secret = id.x25519_secret();
    let remote_key = remote.x25519()?;
    let mut hs = builder(&secret, &pro)
        .remote_public_key(&remote_key)
        .build_initiator()
        .map_err(hs_err)?;
    let mut payload = Vec::with_capacity(40);
    payload.extend_from_slice(id.node_id().as_bytes());
    payload.extend_from_slice(&cfg.stream_window.to_be_bytes());
    payload.extend_from_slice(&keepalive_ms(cfg).to_be_bytes());
    payload.extend_from_slice(&cfg.reorder_kib().to_be_bytes());
    let mut msg = vec![0u8; 1024];
    let n = hs.write_message(&payload, &mut msg).map_err(hs_err)?;
    let mut out = pro.to_vec();
    out.extend_from_slice(&(n as u16).to_be_bytes());
    out.extend_from_slice(&msg[..n]);
    t.write_all(&out).await?;
    t.flush().await?;

    let mut head = [0u8; 5];
    read_exact(t, &mut head).await?;
    if &head[..4] != MAGIC {
        return Err(hs_err(
            "the reply does not start with the letify-link magic",
        ));
    }
    if head[4] == 1 {
        let mut r = [0u8; 6];
        read_exact(t, &mut r).await?;
        let theirs = u16::from_be_bytes([r[0], r[1]])..=u16::from_be_bytes([r[2], r[3]]);
        let mut text = vec![0u8; u16::from_be_bytes([r[4], r[5]]) as usize];
        read_exact(t, &mut text).await?;
        return Err(Error::VersionMismatch {
            ours: cfg.versions.clone(),
            theirs,
        });
    }
    if head[4] != 0 {
        return Err(hs_err("unknown reply status"));
    }
    let mut r = [0u8; 4];
    read_exact(t, &mut r).await?;
    let chosen = u16::from_be_bytes([r[0], r[1]]);
    let mut reply = vec![0u8; u16::from_be_bytes([r[2], r[3]]) as usize];
    read_exact(t, &mut reply).await?;
    let mut body = vec![0u8; 1024];
    let n = hs.read_message(&reply, &mut body).map_err(hs_err)?;
    if n != 14
        || u16::from_be_bytes([body[0], body[1]]) != chosen
        || !cfg.versions.contains(&chosen)
    {
        return Err(hs_err("the chosen version is not the one the header named"));
    }
    let peer_window = u32::from_be_bytes([body[2], body[3], body[4], body[5]]);
    let peer_keepalive =
        Duration::from_millis(u32::from_be_bytes([body[6], body[7], body[8], body[9]]).into());
    let peer_reorder = u32::from_be_bytes(body[10..14].try_into().unwrap()) as usize;
    finish(
        hs,
        remote,
        chosen,
        peer_window,
        peer_keepalive,
        peer_reorder << 10,
    )
}

/// Run the responder side over `t`.
pub(crate) async fn respond<T: AsyncRead + AsyncWrite + Unpin>(
    t: &mut T,
    id: &Identity,
    policy: &PeerPolicy,
    cfg: &Config,
) -> Result<Established> {
    tokio::time::timeout(cfg.handshake_timeout, respond_inner(t, id, policy, cfg))
        .await
        .map_err(|_| hs_err("timed out"))?
}

async fn respond_inner<T: AsyncRead + AsyncWrite + Unpin>(
    t: &mut T,
    id: &Identity,
    policy: &PeerPolicy,
    cfg: &Config,
) -> Result<Established> {
    let mut head = [0u8; 10];
    read_exact(t, &mut head).await?;
    if &head[..4] != MAGIC {
        return Err(hs_err(
            "the header does not start with the letify-link magic",
        ));
    }
    let (low, high) = (
        u16::from_be_bytes([head[4], head[5]]),
        u16::from_be_bytes([head[6], head[7]]),
    );
    let mut first = vec![0u8; u16::from_be_bytes([head[8], head[9]]) as usize];
    read_exact(t, &mut first).await?;

    let common_low = low.max(*cfg.versions.start());
    let common_high = high.min(*cfg.versions.end());
    if common_low > common_high {
        let theirs = low..=high;
        let err = Error::VersionMismatch {
            ours: cfg.versions.clone(),
            theirs,
        };
        let text = err.to_string();
        let mut out = MAGIC.to_vec();
        out.push(1);
        out.extend_from_slice(&cfg.versions.start().to_be_bytes());
        out.extend_from_slice(&cfg.versions.end().to_be_bytes());
        out.extend_from_slice(&(text.len() as u16).to_be_bytes());
        out.extend_from_slice(text.as_bytes());
        t.write_all(&out).await?;
        t.flush().await?;
        return Err(err);
    }

    // The prologue is what the header said, so a header changed in transit fails here.
    let secret = id.x25519_secret();
    let pro = prologue(low, high);
    let mut hs = builder(&secret, &pro).build_responder().map_err(hs_err)?;
    let mut body = vec![0u8; 1024];
    let n = hs.read_message(&first, &mut body).map_err(hs_err)?;
    if n != 44 {
        return Err(hs_err("the first message carries an unexpected payload"));
    }
    let peer = NodeId::from_bytes(body[..32].try_into().unwrap());
    let peer_window = u32::from_be_bytes(body[32..36].try_into().unwrap());
    let peer_keepalive =
        Duration::from_millis(u32::from_be_bytes(body[36..40].try_into().unwrap()).into());
    let peer_reorder = u32::from_be_bytes(body[40..44].try_into().unwrap()) as usize;
    let proven = hs
        .get_remote_static()
        .ok_or_else(|| hs_err("no remote static key"))?;
    if peer.x25519()?.as_slice() != proven {
        return Err(hs_err(
            "the node id does not match the key that authenticated",
        ));
    }
    if !policy.accepts(&peer) {
        return Err(Error::UnknownPeer(peer));
    }

    let chosen = common_high;
    let mut payload = chosen.to_be_bytes().to_vec();
    payload.extend_from_slice(&cfg.stream_window.to_be_bytes());
    payload.extend_from_slice(&keepalive_ms(cfg).to_be_bytes());
    payload.extend_from_slice(&cfg.reorder_kib().to_be_bytes());
    let mut msg = vec![0u8; 1024];
    let n = hs.write_message(&payload, &mut msg).map_err(hs_err)?;
    let mut out = MAGIC.to_vec();
    out.push(0);
    out.extend_from_slice(&chosen.to_be_bytes());
    out.extend_from_slice(&(n as u16).to_be_bytes());
    out.extend_from_slice(&msg[..n]);
    t.write_all(&out).await?;
    t.flush().await?;
    finish(
        hs,
        peer,
        chosen,
        peer_window,
        peer_keepalive,
        peer_reorder << 10,
    )
}

fn finish(
    hs: snow::HandshakeState,
    peer: NodeId,
    version: u16,
    peer_window: u32,
    peer_keepalive: Duration,
    peer_reorder: usize,
) -> Result<Established> {
    let mut session_id = [0u8; 16];
    session_id.copy_from_slice(&hs.get_handshake_hash()[..16]);
    let crypto = hs.into_stateless_transport_mode().map_err(hs_err)?;
    Ok(Established {
        crypto,
        peer,
        version,
        peer_window,
        peer_keepalive,
        peer_reorder,
        session_id,
    })
}
