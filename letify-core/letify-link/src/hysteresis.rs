//! The path switching rules as a pure decision: dwell, confirmation, no downgrade, handover and
//! probe budget. It owns the rules and their state; it owns no clock, no socket and no task, so
//! the caller passes the time in and a test needs no waiting.

use std::time::Duration;

/// Index of a path in the caller's table.
pub type PathId = usize;

/// The caller settings of the switching rules. The defaults are those of SPEC.md.
#[derive(Clone, Debug)]
pub struct Policy {
    /// No move for this long after the last switch. Default 30 s.
    pub dwell: Duration,
    /// Consecutive confirming measurements a candidate needs. Default 3.
    pub confirmations: u32,
    /// Minimum time between two counted measurements. Default 10 s.
    pub spacing: Duration,
    /// A candidate beats the current path when its rate exceeds the current rate times
    /// `1 + threshold`. Default 0.25.
    pub threshold: f64,
    /// A current path below this rate (bytes per second) is treated as lost. Default 0.
    pub floor: f64,
    /// Largest share of the traffic that background probes may use. Default 0.02.
    pub probe_budget: f64,
}

impl Default for Policy {
    fn default() -> Self {
        Policy {
            dwell: Duration::from_secs(30),
            confirmations: 3,
            spacing: Duration::from_secs(10),
            threshold: 0.25,
            floor: 0.0,
            probe_budget: 0.02,
        }
    }
}

/// One path as measured now.
#[derive(Clone, Copy, Debug)]
pub struct PathView {
    /// The path.
    pub id: PathId,
    /// Measured throughput in bytes per second.
    pub rate: f64,
    /// False once the path died.
    pub alive: bool,
}

/// Why a move happens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    /// There was no current path.
    Start,
    /// A candidate beat the current path in enough consecutive measurements.
    Faster,
    /// The current path died or fell below the floor.
    CurrentLost,
}

/// The outcome of one observation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decision {
    /// Stay on the current path.
    Hold,
    /// Make `to` the current path.
    Move {
        /// The new current path.
        to: PathId,
        /// Why.
        reason: Reason,
    },
}

/// The state of the switching rules.
#[derive(Debug)]
pub struct Selector {
    policy: Policy,
    current: Option<PathId>,
    last_switch: Option<Duration>,
    /// The candidate being confirmed, how many measurements counted, and when the last did.
    pending: Option<(PathId, u32, Duration)>,
    /// A move that waits for acknowledged traffic on the new path: (new, old).
    handover: Option<(PathId, PathId)>,
}

impl Selector {
    /// A selector with no current path.
    pub fn new(policy: Policy) -> Self {
        Selector {
            policy,
            current: None,
            last_switch: None,
            pending: None,
            handover: None,
        }
    }

    /// The current path.
    pub fn current(&self) -> Option<PathId> {
        self.current
    }

    fn switch(&mut self, now: Duration, to: PathId, reason: Reason) -> Decision {
        if let Some(old) = self.current
            && old != to
        {
            self.handover = Some((to, old));
        }
        self.current = Some(to);
        self.last_switch = Some(now);
        self.pending = None;
        Decision::Move { to, reason }
    }

    /// Feed one set of measurements taken at `now`, a time on any clock that only grows.
    pub fn observe(&mut self, now: Duration, paths: &[PathView]) -> Decision {
        let best = |skip: Option<PathId>, min_rate: f64| {
            paths
                .iter()
                .filter(|p| p.alive && Some(p.id) != skip && p.rate >= min_rate)
                .max_by(|a, b| a.rate.total_cmp(&b.rate).then(b.id.cmp(&a.id)))
                .copied()
        };
        let Some(cur) = self.current else {
            return match best(None, 0.0) {
                Some(p) => self.switch(now, p.id, Reason::Start),
                None => Decision::Hold,
            };
        };
        let cur_view = paths.iter().find(|p| p.id == cur);
        let cur_ok = cur_view.is_some_and(|p| p.alive && p.rate >= self.policy.floor);
        if !cur_ok {
            // A lost path cannot wait for the dwell, and a slower path is then acceptable.
            // Prefer a path that meets the floor.
            let pick = best(Some(cur), self.policy.floor).or_else(|| best(Some(cur), 0.0));
            return match pick {
                Some(p) => self.switch(now, p.id, Reason::CurrentLost),
                None => Decision::Hold,
            };
        }
        let cur_rate = cur_view.map_or(0.0, |p| p.rate);
        if self
            .last_switch
            .is_some_and(|t| now < t + self.policy.dwell)
        {
            self.pending = None;
            return Decision::Hold;
        }
        let Some(cand) = best(Some(cur), 0.0) else {
            self.pending = None;
            return Decision::Hold;
        };
        if cand.rate <= cur_rate * (1.0 + self.policy.threshold) {
            self.pending = None;
            return Decision::Hold;
        }
        let count = match self.pending {
            Some((id, count, at)) if id == cand.id => {
                if now < at + self.policy.spacing {
                    return Decision::Hold;
                }
                count + 1
            }
            _ => 1,
        };
        if count >= self.policy.confirmations {
            return self.switch(now, cand.id, Reason::Faster);
        }
        self.pending = Some((cand.id, count, now));
        Decision::Hold
    }

    /// Tell the selector that `path` carried acknowledged traffic. When `path` is the target
    /// of a move, this returns the old path, which may now be closed.
    pub fn carried_acked(&mut self, path: PathId) -> Option<PathId> {
        match self.handover {
            Some((new, old)) if new == path => {
                self.handover = None;
                Some(old)
            }
            _ => None,
        }
    }
}

/// True when a background probe of the session may use more bytes: `probe_bytes` already sent
/// as probes stay under `probe_budget` of the `total_bytes` of traffic.
pub fn probe_allowed(policy: &Policy, probe_bytes: u64, total_bytes: u64) -> bool {
    (probe_bytes as f64) < policy.probe_budget * total_bytes as f64
}
