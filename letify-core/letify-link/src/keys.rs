//! The keys of a session: the epochs of each direction, sealing and opening records, and the
//! anti-replay window of the current receiving epoch. It owns key rotation (SPEC.md letify-link,
//! Session item 8). It does not own framing, the handshake that makes the epoch 0 keys, or any
//! subflow.
//!
//! Epoch 0 of both directions is the transport state that the Noise handshake produced. Each
//! later epoch is Noise's REKEY of the one before, done by snow's `rekey_outgoing` and
//! `rekey_incoming`. REKEY cannot be undone and the peer cannot derive it ahead, so a sender
//! announces a switch with a sealed `Rekey` frame and a receiver switches only after it.

use std::collections::VecDeque;
use std::sync::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::{Duration, Instant};

use snow::StatelessTransportState;

use crate::error::{Error, Result};
use crate::frame::{self, RECORD_HDR, TAG};
use crate::replay::ReplayWindow;

struct Inner {
    /// The epoch this side seals records in.
    send_epoch: u16,
    /// The next nonce, which is also the number of records sealed in this epoch.
    nonce: u64,
    since: Instant,
    /// The sealed `Rekey` records, oldest first. Entry `i` was sealed in epoch `base + i` and
    /// announces epoch `base + i + 1`, so `base + retained.len() == send_epoch`.
    retained: VecDeque<Vec<u8>>,
    base: u16,
    /// The epoch the peer's records are opened in.
    recv_epoch: u16,
    /// The peer announced `recv_epoch + 1` in a record that authenticated, with this nonce.
    announced: Option<u64>,
    replay: ReplayWindow,
}

/// The keys of one session, shared by every subflow.
pub(crate) struct Keys {
    /// Sealing and opening take the read side. A rekey of either direction takes the write
    /// side, so no record is sealed or opened across a switch.
    state: RwLock<StatelessTransportState>,
    inner: Mutex<Inner>,
    after_records: u64,
    after: Duration,
}

impl Keys {
    /// Epoch 0 over the Noise `state`. A limit of 0 turns that limit off.
    pub fn new(state: StatelessTransportState, after_records: u64, after: Duration) -> Keys {
        Keys {
            state: RwLock::new(state),
            inner: Mutex::new(Inner {
                send_epoch: 0,
                nonce: 0,
                since: Instant::now(),
                retained: VecDeque::new(),
                base: 0,
                recv_epoch: 0,
                announced: None,
                replay: ReplayWindow::new(),
            }),
            after_records,
            after,
        }
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn read(&self) -> RwLockReadGuard<'_, StatelessTransportState> {
        self.state.read().unwrap_or_else(|e| e.into_inner())
    }

    fn write(&self) -> RwLockWriteGuard<'_, StatelessTransportState> {
        self.state.write().unwrap_or_else(|e| e.into_inner())
    }

    /// The epoch this side seals records in.
    pub fn send_epoch(&self) -> u16 {
        self.lock().send_epoch
    }

    /// The epoch of the peer's keys that records are opened in.
    pub fn receive_epoch(&self) -> u16 {
        self.lock().recv_epoch
    }

    fn worn(&self, inner: &Inner) -> bool {
        (self.after_records != 0 && inner.nonce >= self.after_records)
            || (!self.after.is_zero() && inner.since.elapsed() >= self.after)
    }

    /// One record: length, epoch, nonce, ciphertext.
    fn record(
        state: &StatelessTransportState,
        epoch: u16,
        nonce: u64,
        plain: &[u8],
    ) -> Result<Vec<u8>> {
        let mut out = vec![0u8; RECORD_HDR + plain.len() + TAG];
        let n = state
            .write_message(nonce, plain, &mut out[RECORD_HDR..])
            .map_err(|e| Error::ConnectionLost(format!("encrypt: {e}")))?;
        out.truncate(RECORD_HDR + n);
        out[..2].copy_from_slice(&(n as u16).to_be_bytes());
        out[2..4].copy_from_slice(&epoch.to_be_bytes());
        out[4..RECORD_HDR].copy_from_slice(&nonce.to_be_bytes());
        Ok(out)
    }

    /// Announce the next epoch and move the sending direction to it, unless another caller
    /// already did.
    fn rotate(&self) -> Result<()> {
        let mut state = self.write();
        let mut inner = self.lock();
        if !self.worn(&inner) {
            return Ok(());
        }
        let next = inner.send_epoch.checked_add(1).ok_or_else(|| {
            Error::ConnectionLost("the key epochs of the session are used up".into())
        })?;
        let nonce = inner.nonce;
        let announcement = Self::record(
            &state,
            inner.send_epoch,
            nonce,
            &frame::encode_rekey(next.into()),
        )?;
        state.rekey_outgoing();
        inner.retained.push_back(announcement);
        inner.send_epoch = next;
        inner.nonce = 0;
        inner.since = Instant::now();
        Ok(())
    }

    /// Encrypt `plain` into a record. The sending key rotates first when the epoch is worn.
    ///
    /// `announced` is the epoch up to which the subflow that will carry the bytes has been
    /// sent the announcements. The result starts with the announcements it has not been sent,
    /// so the receiver sees each before the first record of the epoch it announces. `announced`
    /// is updated.
    pub fn seal(&self, plain: &[u8], announced: &mut u16) -> Result<Vec<u8>> {
        loop {
            let state = self.read();
            let mut inner = self.lock();
            if self.worn(&inner) {
                drop(inner);
                drop(state);
                self.rotate()?;
                continue;
            }
            let (epoch, nonce) = (inner.send_epoch, inner.nonce);
            inner.nonce += 1;
            let mut out = Vec::new();
            for k in (*announced).max(inner.base)..epoch {
                out.extend_from_slice(&inner.retained[usize::from(k - inner.base)]);
            }
            *announced = epoch;
            drop(inner);
            out.extend_from_slice(&Self::record(&state, epoch, nonce, plain)?);
            return Ok(out);
        }
    }

    /// The peer announced that its records move to epoch `next`, in the record with `nonce`
    /// that opened in the current epoch.
    pub fn announce(&self, next: u64, nonce: u64) {
        let mut inner = self.lock();
        if inner.recv_epoch.checked_add(1).map(u64::from) == Some(next) {
            inner.announced = Some(nonce);
        }
    }

    /// Move the receiving direction past `from`, if it is still there and was announced.
    fn advance(&self, from: u16) {
        let mut state = self.write();
        let mut inner = self.lock();
        if inner.recv_epoch == from && inner.announced.is_some() {
            state.rekey_incoming();
            inner.recv_epoch = from + 1;
            inner.announced = None;
            inner.replay = ReplayWindow::new();
        }
    }

    /// Whether a record that `open` dropped is one a resuming peer is expected to repeat: a
    /// record of an epoch already left, or the announcement already held.
    pub fn is_stale(&self, epoch: u16, nonce: u64) -> bool {
        let inner = self.lock();
        epoch < inner.recv_epoch || (epoch == inner.recv_epoch && inner.announced == Some(nonce))
    }

    /// The peer has switched to `epoch` of this side's keys, so the announcements below it are
    /// no longer needed.
    pub fn prune(&self, epoch: u16) {
        let mut inner = self.lock();
        while inner.base < epoch && !inner.retained.is_empty() {
            inner.retained.pop_front();
            inner.base += 1;
        }
    }

    /// Decrypt a record. `None` means the record is dropped without an error: its nonce was
    /// already accepted, it is older than the replay window, or its epoch is neither the
    /// current one nor the announced next one.
    pub fn open(&self, epoch: u16, nonce: u64, ct: &[u8]) -> Result<Option<Vec<u8>>> {
        loop {
            let state = self.read();
            let inner = self.lock();
            if epoch != inner.recv_epoch {
                if inner.recv_epoch.checked_add(1) == Some(epoch) && inner.announced.is_some() {
                    let from = inner.recv_epoch;
                    drop(inner);
                    drop(state);
                    self.advance(from);
                    continue;
                }
                return Ok(None);
            }
            drop(inner);
            let mut out = vec![0u8; ct.len()];
            let n = state
                .read_message(nonce, ct, &mut out)
                .map_err(|e| Error::ConnectionLost(format!("decrypt: {e}")))?;
            out.truncate(n);
            // The read guard is still held, so the epoch cannot have changed.
            let fresh = self.lock().replay.accept(nonce);
            return Ok(fresh.then_some(out));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A handshake with fixed keys, so that two runs make the same transport keys.
    fn handshake() -> (StatelessTransportState, StatelessTransportState) {
        let params: snow::params::NoiseParams =
            "Noise_NN_25519_ChaChaPoly_BLAKE2s".parse().unwrap();
        let (ei, er) = ([7u8; 32], [9u8; 32]);
        let mut i = snow::Builder::new(params.clone())
            .fixed_ephemeral_key_for_testing_only(&ei)
            .build_initiator()
            .unwrap();
        let mut r = snow::Builder::new(params)
            .fixed_ephemeral_key_for_testing_only(&er)
            .build_responder()
            .unwrap();
        let (mut m, mut p) = ([0u8; 256], [0u8; 256]);
        let n = i.write_message(&[], &mut m).unwrap();
        r.read_message(&m[..n], &mut p).unwrap();
        let n = r.write_message(&[], &mut m).unwrap();
        i.read_message(&m[..n], &mut p).unwrap();
        (
            i.into_stateless_transport_mode().unwrap(),
            r.into_stateless_transport_mode().unwrap(),
        )
    }

    fn pair(after_records: u64) -> (Keys, Keys) {
        let (i, r) = handshake();
        let long = Duration::ZERO;
        (
            Keys::new(i, after_records, long),
            Keys::new(r, after_records, long),
        )
    }

    /// The (epoch, nonce, ciphertext) of each record in `bytes`.
    fn records(bytes: &[u8]) -> Vec<(u16, u64, &[u8])> {
        let mut out = Vec::new();
        let mut at = 0;
        while at < bytes.len() {
            let len = u16::from_be_bytes([bytes[at], bytes[at + 1]]) as usize;
            let epoch = u16::from_be_bytes([bytes[at + 2], bytes[at + 3]]);
            let nonce = u64::from_be_bytes(bytes[at + 4..at + RECORD_HDR].try_into().unwrap());
            out.push((epoch, nonce, &bytes[at + RECORD_HDR..at + RECORD_HDR + len]));
            at += RECORD_HDR + len;
        }
        out
    }

    /// Open every record of `bytes` in order, handling announcements as the session does.
    fn open_all(keys: &Keys, bytes: &[u8]) -> Vec<Option<Vec<u8>>> {
        records(bytes)
            .into_iter()
            .map(|(e, n, c)| {
                let plain = keys.open(e, n, c).unwrap();
                if let Some(p) = &plain
                    && let Ok(frame::Frame::Rekey { next }) = frame::decode(p.clone())
                {
                    keys.announce(next, n);
                }
                plain
            })
            .collect()
    }

    #[test]
    fn the_rekeyed_key_is_the_noise_rekey_of_snow() {
        // Oracle: the same handshake, rekeyed on both ends by snow, seals and opens what the
        // keys of a session do after one rotation.
        let (oi, mut or) = handshake();
        let mut oi = oi;
        oi.rekey_outgoing();
        or.rekey_incoming();
        let mut sealed = [0u8; 64];
        let n = oi
            .write_message(0, b"after the rekey", &mut sealed)
            .unwrap();

        let (a, b) = pair(1);
        let mut announced = 0;
        a.seal(b"first", &mut announced).unwrap();
        let bytes = a.seal(b"after the rekey", &mut announced).unwrap();
        let recs = records(&bytes);
        // The announcement of epoch 1, then the record of epoch 1 with nonce 0.
        assert_eq!(recs.len(), 2);
        assert_eq!((recs[1].0, recs[1].1), (1, 0));
        assert_eq!(recs[1].2, &sealed[..n], "same key, same nonce, same bytes");
        // And the oracle's receiver opens what the session sealed.
        let mut out = [0u8; 64];
        let m = or.read_message(0, recs[1].2, &mut out).unwrap();
        assert_eq!(&out[..m], b"after the rekey");
        // The session's own receiver follows the announcement.
        let got = open_all(&b, &bytes);
        assert_eq!(got[1].as_deref(), Some(&b"after the rekey"[..]));
        assert_eq!(b.receive_epoch(), 1);
    }

    #[test]
    fn a_record_of_the_next_epoch_is_dropped_before_the_announcement() {
        let (a, b) = pair(1);
        let mut announced = 0;
        a.seal(b"x", &mut announced).unwrap();
        let bytes = a.seal(b"y", &mut announced).unwrap();
        let recs = records(&bytes);
        let (e, n, c) = recs[1];
        assert_eq!(e, 1);
        assert!(b.open(e, n, c).unwrap().is_none());
        assert_eq!(b.receive_epoch(), 0, "a header alone switches nothing");
        // The announcement still works afterwards.
        let got = open_all(&b, &bytes);
        assert_eq!(got[1].as_deref(), Some(&b"y"[..]));
    }

    #[test]
    fn an_epoch_other_than_the_current_or_next_is_dropped() {
        let (a, b) = pair(100);
        let mut announced = 0;
        let bytes = a.seal(b"z", &mut announced).unwrap();
        let (_, n, c) = records(&bytes)[0];
        assert!(b.open(2, n, c).unwrap().is_none());
        assert!(b.open(500, n, c).unwrap().is_none());
        assert_eq!(b.receive_epoch(), 0);
        assert!(b.open(0, n, c).unwrap().is_some());
    }

    #[test]
    fn after_the_switch_a_late_record_of_the_old_epoch_is_dropped() {
        let (a, b) = pair(2);
        let mut announced = 0;
        let late = a.seal(b"late", &mut announced).unwrap();
        a.seal(b"two", &mut announced).unwrap();
        let bytes = a.seal(b"three", &mut announced).unwrap();
        let got = open_all(&b, &bytes);
        assert_eq!(got[1].as_deref(), Some(&b"three"[..]));
        assert_eq!(b.receive_epoch(), 1);
        assert!(open_all(&b, &late)[0].is_none());
    }

    #[test]
    fn each_epoch_starts_a_fresh_replay_window() {
        let (a, b) = pair(1);
        let mut announced = 0;
        let first = a.seal(b"x", &mut announced).unwrap();
        let second = a.seal(b"x", &mut announced).unwrap();
        assert!(open_all(&b, &first)[0].is_some());
        let got = open_all(&b, &second);
        assert!(got[1].is_some(), "nonce 0 of epoch 1 is new");
        // The repeated announcement is a replay.
        assert!(open_all(&b, &second)[0].is_none());
    }

    #[test]
    fn a_subflow_that_has_not_been_announced_to_gets_the_announcements_first() {
        let (a, b) = pair(1);
        let mut one = 0;
        a.seal(b"p", &mut one).unwrap();
        a.seal(b"q", &mut one).unwrap();
        a.seal(b"r", &mut one).unwrap();
        assert_eq!(a.send_epoch(), 2);
        let mut other = 0;
        // The limit of 1 rotates once more before this record, so three announcements lead.
        let bytes = a.seal(b"s", &mut other).unwrap();
        assert_eq!(a.send_epoch(), 3);
        assert_eq!(
            records(&bytes).len(),
            4,
            "three announcements and the record"
        );
        let got = open_all(&b, &bytes);
        assert_eq!(got[3].as_deref(), Some(&b"s"[..]));
        assert_eq!(b.receive_epoch(), 3);
    }

    #[test]
    fn the_two_directions_rekey_on_their_own() {
        let (a, b) = pair(1);
        let (mut x, mut y) = (0, 0);
        a.seal(b"1", &mut x).unwrap();
        let from_a = a.seal(b"2", &mut x).unwrap();
        assert_eq!((a.send_epoch(), b.send_epoch()), (1, 0));
        open_all(&b, &from_a);
        let from_b = b.seal(b"3", &mut y).unwrap();
        assert_eq!(records(&from_b).len(), 1);
        assert!(open_all(&a, &from_b)[0].is_some());
    }

    #[test]
    fn an_exhausted_epoch_space_fails_the_seal() {
        let (a, _) = pair(1);
        {
            let mut inner = a.lock();
            inner.send_epoch = u16::MAX;
            inner.base = u16::MAX;
            inner.nonce = 1;
        }
        assert!(a.seal(b"q", &mut 0).is_err());
    }
}
