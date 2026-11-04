//! Anti-replay window for the record nonces of one direction, as WireGuard and IPsec keep one.
//! It owns the accept or drop decision for a nonce and nothing else.

/// Width of the window in records. A record whose nonce is this far or more below the highest
/// accepted nonce is dropped, and so is any nonce already accepted.
///
/// 8192 is four times `MAX_UNACKED_FRAMES`. A frame that a slow subflow delays can be passed by
/// at most that many data frames, plus about as many acknowledgements going the other way, so
/// a record that is only late never falls out of the window.
pub const REPLAY_WINDOW: u64 = 8192;

/// Most sequenced frames the sender keeps unacknowledged, whatever their size.
pub(crate) const MAX_UNACKED_FRAMES: usize = 2048;

const WORDS: usize = (REPLAY_WINDOW / 64) as usize;

/// A sliding bitmap over the nonces `highest - REPLAY_WINDOW + 1 ..= highest`, kept as a ring
/// indexed by `nonce % REPLAY_WINDOW`.
pub(crate) struct ReplayWindow {
    started: bool,
    highest: u64,
    bits: [u64; WORDS],
}

impl ReplayWindow {
    pub fn new() -> Self {
        ReplayWindow {
            started: false,
            highest: 0,
            bits: [0; WORDS],
        }
    }

    fn get(&self, n: u64) -> bool {
        let i = (n % REPLAY_WINDOW) as usize;
        self.bits[i / 64] >> (i % 64) & 1 == 1
    }

    fn set(&mut self, n: u64, on: bool) {
        let i = (n % REPLAY_WINDOW) as usize;
        if on {
            self.bits[i / 64] |= 1 << (i % 64);
        } else {
            self.bits[i / 64] &= !(1 << (i % 64));
        }
    }

    /// Accept `nonce` and mark it, or return false when it was seen or is older than the
    /// window. Call it only for a record that authenticated, so a forgery cannot use up nonces.
    pub fn accept(&mut self, nonce: u64) -> bool {
        if !self.started {
            self.started = true;
            self.highest = nonce;
            self.set(nonce, true);
            return true;
        }
        if nonce > self.highest {
            let shift = nonce - self.highest;
            if shift >= REPLAY_WINDOW {
                self.bits = [0; WORDS];
            } else {
                for n in self.highest + 1..=nonce {
                    self.set(n, false);
                }
            }
            self.highest = nonce;
            self.set(nonce, true);
            return true;
        }
        if self.highest - nonce >= REPLAY_WINDOW || self.get(nonce) {
            return false;
        }
        self.set(nonce, true);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_nonce_is_accepted_once() {
        let mut w = ReplayWindow::new();
        assert!(w.accept(5));
        assert!(!w.accept(5));
    }

    #[test]
    fn nonces_inside_the_window_are_accepted_in_any_order() {
        let mut w = ReplayWindow::new();
        assert!(w.accept(100));
        assert!(w.accept(50));
        assert!(w.accept(99));
        assert!(!w.accept(50));
    }

    #[test]
    fn a_nonce_a_full_window_below_the_highest_is_dropped() {
        let mut w = ReplayWindow::new();
        assert!(w.accept(0));
        assert!(w.accept(REPLAY_WINDOW));
        assert!(!w.accept(0));
        assert!(w.accept(1));
    }

    #[test]
    fn a_jump_beyond_the_window_forgets_the_old_marks() {
        let mut w = ReplayWindow::new();
        assert!(w.accept(10));
        assert!(w.accept(10 + 5 * REPLAY_WINDOW));
        assert!(!w.accept(10));
        assert!(w.accept(10 + 5 * REPLAY_WINDOW - 1));
    }
}
