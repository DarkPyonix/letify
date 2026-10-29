//! The keys of a session: the epochs of each direction, sealing and opening records, and the
//! anti-replay window of every epoch. It owns key rotation (SPEC.md letify-link, Session item
//! 8). It does not own framing, the handshake that delivers the secret, or any subflow.
//!
//! Epoch 0 of both directions is the transport state that the Noise handshake produced. The key
//! of epoch n >= 1 is derived from the rekey secret, so a receiver can derive any epoch it
//! sees and no message announces a switch.

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use blake2::{Blake2s256, Digest};
use chacha20poly1305::aead::{AeadInPlace, KeyInit};
use chacha20poly1305::{ChaCha20Poly1305, Nonce};
use snow::StatelessTransportState;

use crate::error::{Error, Result};
use crate::frame::{RECORD_HDR, TAG};
use crate::replay::ReplayWindow;

/// A receiver accepts the newest epoch it has seen and the `KEEP - 1` before it.
const KEEP: usize = 4;
/// A record this many epochs ahead of the newest one seen is dropped.
const MAX_AHEAD: u16 = 16;

/// The secret that later epochs derive from: both sides' random bytes and the handshake hash.
pub(crate) fn rekey_secret(initiator: &[u8; 32], responder: &[u8; 32], hash: &[u8]) -> [u8; 32] {
    Blake2s256::new()
        .chain_update(b"letify-link rekey root v1")
        .chain_update(initiator)
        .chain_update(responder)
        .chain_update(hash)
        .finalize()
        .into()
}

/// The key of `epoch` for the direction that `from_initiator` names.
fn epoch_key(secret: &[u8; 32], from_initiator: bool, epoch: u16) -> [u8; 32] {
    Blake2s256::new()
        .chain_update(b"letify-link rekey v1")
        .chain_update(secret)
        .chain_update([u8::from(!from_initiator)])
        .chain_update(epoch.to_be_bytes())
        .finalize()
        .into()
}

/// The nonce of ChaChaPoly in Noise: four zero bytes and the counter, little endian.
fn nonce12(nonce: u64) -> [u8; 12] {
    let mut out = [0u8; 12];
    out[4..].copy_from_slice(&nonce.to_le_bytes());
    out
}

/// What seals and opens the records of one epoch.
enum Cipher {
    /// Epoch 0: the transport state of the handshake, for both directions.
    Noise(Arc<StatelessTransportState>),
    /// A later epoch: one key per direction.
    Derived {
        send: ChaCha20Poly1305,
        recv: ChaCha20Poly1305,
    },
}

impl Cipher {
    fn derived(secret: &[u8; 32], initiator: bool, epoch: u16) -> Cipher {
        let key = |from_initiator| {
            ChaCha20Poly1305::new(&epoch_key(secret, from_initiator, epoch).into())
        };
        Cipher::Derived {
            send: key(initiator),
            recv: key(!initiator),
        }
    }

    /// Seal `plain` into `out`, which holds `plain.len() + TAG` bytes. Returns the length.
    fn seal(&self, nonce: u64, plain: &[u8], out: &mut [u8]) -> Result<usize> {
        let lost = |e: &dyn std::fmt::Display| Error::ConnectionLost(format!("encrypt: {e}"));
        match self {
            Cipher::Noise(state) => state.write_message(nonce, plain, out).map_err(|e| lost(&e)),
            Cipher::Derived { send, .. } => {
                let (body, tag) = out.split_at_mut(plain.len());
                body.copy_from_slice(plain);
                let made = send
                    .encrypt_in_place_detached(Nonce::from_slice(&nonce12(nonce)), b"", body)
                    .map_err(|e| lost(&e))?;
                tag[..TAG].copy_from_slice(&made);
                Ok(plain.len() + TAG)
            }
        }
    }

    /// Open `ct` into `out`, which holds `ct.len()` bytes. Returns the plaintext length.
    fn open(&self, nonce: u64, ct: &[u8], out: &mut [u8]) -> Result<usize> {
        let lost = |e: &dyn std::fmt::Display| Error::ConnectionLost(format!("decrypt: {e}"));
        match self {
            Cipher::Noise(state) => state.read_message(nonce, ct, out).map_err(|e| lost(&e)),
            Cipher::Derived { recv, .. } => {
                let body_len = ct
                    .len()
                    .checked_sub(TAG)
                    .ok_or_else(|| lost(&"short record"))?;
                let (body, tag) = ct.split_at(body_len);
                out[..body_len].copy_from_slice(body);
                recv.decrypt_in_place_detached(
                    Nonce::from_slice(&nonce12(nonce)),
                    b"",
                    &mut out[..body_len],
                    tag.into(),
                )
                .map_err(|e| lost(&e))?;
                Ok(body_len)
            }
        }
    }
}

struct Sending {
    epoch: u16,
    cipher: Arc<Cipher>,
    /// The next nonce, which is also the number of records sealed in this epoch.
    nonce: u64,
    since: Instant,
}

struct Receiving {
    epoch: u16,
    cipher: Arc<Cipher>,
    replay: ReplayWindow,
}

struct Inner {
    sending: Sending,
    /// The epochs seen and still accepted, oldest first.
    receiving: Vec<Receiving>,
}

/// The keys of one session, shared by every subflow.
pub(crate) struct Keys {
    inner: Mutex<Inner>,
    secret: [u8; 32],
    initiator: bool,
    after_records: u64,
    after: Duration,
}

impl Keys {
    /// Epoch 0 over the Noise `state`. A limit of 0 turns that limit off.
    pub fn new(
        state: Arc<StatelessTransportState>,
        secret: [u8; 32],
        initiator: bool,
        after_records: u64,
        after: Duration,
    ) -> Keys {
        let cipher = Arc::new(Cipher::Noise(state));
        Keys {
            inner: Mutex::new(Inner {
                sending: Sending {
                    epoch: 0,
                    cipher: cipher.clone(),
                    nonce: 0,
                    since: Instant::now(),
                },
                receiving: vec![Receiving {
                    epoch: 0,
                    cipher,
                    replay: ReplayWindow::new(),
                }],
            }),
            secret,
            initiator,
            after_records,
            after,
        }
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The epoch this side seals records in.
    pub fn send_epoch(&self) -> u16 {
        self.lock().sending.epoch
    }

    /// The newest epoch of the peer that a record authenticated in.
    pub fn receive_epoch(&self) -> u16 {
        self.lock().receiving.last().map_or(0, |r| r.epoch)
    }

    /// Encrypt `plain` into a record: length, epoch, nonce, ciphertext. The sending key rotates
    /// first when the epoch is worn.
    pub fn seal(&self, plain: &[u8]) -> Result<Vec<u8>> {
        let (epoch, cipher, nonce) = {
            let mut inner = self.lock();
            let s = &mut inner.sending;
            let worn = (self.after_records != 0 && s.nonce >= self.after_records)
                || (!self.after.is_zero() && s.since.elapsed() >= self.after);
            if worn {
                s.epoch = s.epoch.checked_add(1).ok_or_else(|| {
                    Error::ConnectionLost("the key epochs of the session are used up".into())
                })?;
                s.cipher = Arc::new(Cipher::derived(&self.secret, self.initiator, s.epoch));
                s.nonce = 0;
                s.since = Instant::now();
            }
            let nonce = s.nonce;
            s.nonce += 1;
            (s.epoch, s.cipher.clone(), nonce)
        };
        let mut out = vec![0u8; RECORD_HDR + plain.len() + TAG];
        let n = cipher.seal(nonce, plain, &mut out[RECORD_HDR..])?;
        out.truncate(RECORD_HDR + n);
        out[..2].copy_from_slice(&(n as u16).to_be_bytes());
        out[2..4].copy_from_slice(&epoch.to_be_bytes());
        out[4..RECORD_HDR].copy_from_slice(&nonce.to_be_bytes());
        Ok(out)
    }

    /// Decrypt a record. `None` means the record is dropped without an error: its nonce was
    /// already accepted, it is older than the replay window, or its epoch is no longer kept or
    /// is too far ahead.
    pub fn open(&self, epoch: u16, nonce: u64, ct: &[u8]) -> Result<Option<Vec<u8>>> {
        let (cipher, derived) = {
            let inner = self.lock();
            match inner.receiving.iter().find(|r| r.epoch == epoch) {
                Some(r) => (r.cipher.clone(), false),
                None => {
                    let newest = inner.receiving.last().map_or(0, |r| r.epoch);
                    let oldest = newest.saturating_sub(KEEP as u16 - 1);
                    if epoch < oldest || epoch - newest.min(epoch) > MAX_AHEAD || epoch == 0 {
                        return Ok(None);
                    }
                    (
                        Arc::new(Cipher::derived(&self.secret, self.initiator, epoch)),
                        true,
                    )
                }
            }
        };
        let mut out = vec![0u8; ct.len()];
        let n = cipher.open(nonce, ct, &mut out)?;
        out.truncate(n);
        let mut inner = self.lock();
        if derived && !inner.receiving.iter().any(|r| r.epoch == epoch) {
            // The record authenticated, so the epoch is real.
            let at = inner.receiving.partition_point(|r| r.epoch < epoch);
            inner.receiving.insert(
                at,
                Receiving {
                    epoch,
                    cipher,
                    replay: ReplayWindow::new(),
                },
            );
            // Only the epochs within `KEEP - 1` of the newest stay.
            let newest = inner.receiving.last().map_or(0, |r| r.epoch);
            inner
                .receiving
                .retain(|r| u32::from(r.epoch) + KEEP as u32 > u32::from(newest));
        }
        let fresh = inner
            .receiving
            .iter_mut()
            .find(|r| r.epoch == epoch)
            .is_some_and(|r| r.replay.accept(nonce));
        Ok(fresh.then_some(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two `Keys` that share a rekey secret, as the two ends of a session do. Epoch 0 is a
    /// Noise state made by a throwaway handshake, because a test has no session.
    fn pair(after_records: u64) -> (Keys, Keys) {
        let params: snow::params::NoiseParams =
            "Noise_NN_25519_ChaChaPoly_BLAKE2s".parse().unwrap();
        let mut i = snow::Builder::new(params.clone())
            .build_initiator()
            .unwrap();
        let mut r = snow::Builder::new(params).build_responder().unwrap();
        let (mut m, mut p) = ([0u8; 256], [0u8; 256]);
        let n = i.write_message(&[], &mut m).unwrap();
        r.read_message(&m[..n], &mut p).unwrap();
        let n = r.write_message(&[], &mut m).unwrap();
        i.read_message(&m[..n], &mut p).unwrap();
        let (si, sr) = (
            Arc::new(i.into_stateless_transport_mode().unwrap()),
            Arc::new(r.into_stateless_transport_mode().unwrap()),
        );
        let secret = rekey_secret(&[1; 32], &[2; 32], b"hash");
        let long = Duration::ZERO;
        (
            Keys::new(si, secret, true, after_records, long),
            Keys::new(sr, secret, false, after_records, long),
        )
    }

    fn split(record: &[u8]) -> (u16, u64, &[u8]) {
        (
            u16::from_be_bytes([record[2], record[3]]),
            u64::from_be_bytes(record[4..RECORD_HDR].try_into().unwrap()),
            &record[RECORD_HDR..],
        )
    }

    #[test]
    fn records_in_flight_across_a_rekey_are_accepted() {
        let (a, b) = pair(2);
        let sealed: Vec<Vec<u8>> = (0..6u8).map(|i| a.seal(&[i; 40]).unwrap()).collect();
        assert_eq!(a.send_epoch(), 2);
        // Newest first: the older epochs are derived when their records arrive.
        for (i, rec) in sealed.iter().enumerate().rev() {
            let (epoch, nonce, ct) = split(rec);
            assert_eq!(epoch as usize, i / 2);
            let plain = b.open(epoch, nonce, ct).unwrap();
            assert_eq!(plain, Some(vec![i as u8; 40]));
        }
        assert_eq!(b.receive_epoch(), 2);
    }

    #[test]
    fn each_epoch_has_its_own_replay_window() {
        let (a, b) = pair(1);
        let first = a.seal(b"x").unwrap();
        let second = a.seal(b"x").unwrap();
        let (e0, n0, c0) = split(&first);
        let (e1, n1, c1) = split(&second);
        assert_eq!((e0, n0, e1, n1), (0, 0, 1, 0));
        assert!(b.open(e0, n0, c0).unwrap().is_some());
        assert!(b.open(e1, n1, c1).unwrap().is_some());
        assert!(b.open(e0, n0, c0).unwrap().is_none());
        assert!(b.open(e1, n1, c1).unwrap().is_none());
    }

    #[test]
    fn a_record_older_than_the_kept_epochs_is_dropped() {
        let (a, b) = pair(1);
        let sealed: Vec<Vec<u8>> = (0..6).map(|_| a.seal(b"y").unwrap()).collect();
        let (e, n, c) = split(&sealed[5]);
        assert!(b.open(e, n, c).unwrap().is_some());
        let (e, n, c) = split(&sealed[0]);
        assert!(b.open(e, n, c).unwrap().is_none());
        let (e, n, c) = split(&sealed[2]);
        assert!(b.open(e, n, c).unwrap().is_some());
    }

    #[test]
    fn a_forged_epoch_creates_no_state() {
        let (a, b) = pair(100);
        let rec = a.seal(b"z").unwrap();
        let (_, n, c) = split(&rec);
        assert!(
            b.open(7, n, c).is_err(),
            "a wrong epoch fails to authenticate"
        );
        assert_eq!(b.receive_epoch(), 0);
        assert!(b.open(500, n, c).unwrap().is_none(), "too far ahead");
        assert!(b.open(0, n, c).unwrap().is_some());
    }

    #[test]
    fn the_epochs_of_the_two_directions_use_different_keys() {
        let (a, b) = pair(1);
        let from_a = (a.seal(b"p").unwrap(), a.seal(b"p").unwrap());
        let from_b = (b.seal(b"p").unwrap(), b.seal(b"p").unwrap());
        assert_ne!(from_a.1[RECORD_HDR..], from_b.1[RECORD_HDR..]);
        // A record of b's direction does not open as one of a's.
        let (e, n, c) = split(&from_b.1);
        assert!(b.open(e, n, c).is_err());
        let (e, n, c) = split(&from_a.1);
        assert!(b.open(e, n, c).unwrap().is_some());
    }

    #[test]
    fn an_exhausted_epoch_space_fails_the_seal() {
        let (a, _) = pair(1);
        a.lock().sending.epoch = u16::MAX;
        a.lock().sending.nonce = 1;
        assert!(a.seal(b"q").is_err());
    }
}
