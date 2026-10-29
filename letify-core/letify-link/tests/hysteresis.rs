//! Pins SPEC.md letify-link, Paths and the path manager item 3: dwell, confirmation, no
//! downgrade, handover and probe budget. Time is injected, so no test waits.

use std::time::Duration;

use letify_link::hysteresis::{Decision, PathView, Policy, Reason, Selector, probe_allowed};

fn s(n: u64) -> Duration {
    Duration::from_secs(n)
}

fn view(id: usize, rate: f64) -> PathView {
    PathView {
        id,
        rate,
        alive: true,
    }
}

/// A selector that started on path 0 at t = 0.
fn started() -> Selector {
    let mut sel = Selector::new(Policy::default());
    let d = sel.observe(s(0), &[view(0, 10.0), view(1, 10.0)]);
    assert_eq!(
        d,
        Decision::Move {
            to: 0,
            reason: Reason::Start
        }
    );
    sel
}

#[test]
fn dwell_blocks_a_move_for_30_seconds_after_the_last_switch() {
    let mut sel = started();
    for t in [1, 11, 21, 29] {
        let d = sel.observe(s(t), &[view(0, 10.0), view(1, 100.0)]);
        assert_eq!(d, Decision::Hold, "at {t} s");
    }
}

#[test]
fn two_confirmations_are_not_enough_and_three_are() {
    let mut sel = started();
    let paths = [view(0, 10.0), view(1, 13.0)];
    assert_eq!(sel.observe(s(40), &paths), Decision::Hold);
    assert_eq!(sel.observe(s(50), &paths), Decision::Hold);
    assert_eq!(
        sel.observe(s(60), &paths),
        Decision::Move {
            to: 1,
            reason: Reason::Faster
        }
    );
    assert_eq!(sel.current(), Some(1));
}

#[test]
fn measurements_closer_than_10_seconds_do_not_count_as_confirmations() {
    let mut sel = started();
    let paths = [view(0, 10.0), view(1, 13.0)];
    for t in [40, 42, 44, 46, 48] {
        assert_eq!(sel.observe(s(t), &paths), Decision::Hold, "at {t} s");
    }
    assert_eq!(sel.observe(s(50), &paths), Decision::Hold);
    assert!(matches!(sel.observe(s(60), &paths), Decision::Move { .. }));
}

#[test]
fn a_measurement_that_does_not_beat_the_threshold_resets_the_count() {
    let mut sel = started();
    let fast = [view(0, 10.0), view(1, 13.0)];
    let same = [view(0, 10.0), view(1, 12.0)];
    assert_eq!(sel.observe(s(40), &fast), Decision::Hold);
    assert_eq!(sel.observe(s(50), &fast), Decision::Hold);
    assert_eq!(sel.observe(s(60), &same), Decision::Hold);
    assert_eq!(sel.observe(s(70), &fast), Decision::Hold);
    assert_eq!(sel.observe(s(80), &fast), Decision::Hold);
    assert!(matches!(sel.observe(s(90), &fast), Decision::Move { .. }));
}

#[test]
fn a_slower_path_is_taken_only_when_the_current_one_died() {
    let mut sel = started();
    let slow = [view(0, 100.0), view(1, 10.0)];
    for t in (40..400).step_by(10) {
        assert_eq!(sel.observe(s(t), &slow), Decision::Hold, "at {t} s");
    }
    let dead = [
        PathView {
            id: 0,
            rate: 0.0,
            alive: false,
        },
        view(1, 10.0),
    ];
    // A death overrides the dwell: the session cannot wait.
    assert_eq!(
        sel.observe(s(401), &dead),
        Decision::Move {
            to: 1,
            reason: Reason::CurrentLost
        }
    );
}

#[test]
fn a_slower_path_is_taken_when_the_current_one_fell_below_the_floor() {
    let policy = Policy {
        floor: 50.0,
        ..Policy::default()
    };
    let mut sel = Selector::new(policy);
    sel.observe(s(0), &[view(0, 100.0), view(1, 30.0)]);
    assert_eq!(sel.current(), Some(0));
    let d = sel.observe(s(5), &[view(0, 20.0), view(1, 30.0)]);
    assert_eq!(
        d,
        Decision::Move {
            to: 1,
            reason: Reason::CurrentLost
        }
    );
}

#[test]
fn the_old_path_stays_until_the_new_one_carried_acknowledged_traffic() {
    let mut sel = started();
    let paths = [view(0, 10.0), view(1, 13.0)];
    sel.observe(s(40), &paths);
    sel.observe(s(50), &paths);
    assert!(matches!(sel.observe(s(60), &paths), Decision::Move { .. }));
    // Traffic acknowledged on an unrelated path says nothing about the new one.
    assert_eq!(sel.carried_acked(0), None);
    assert_eq!(sel.carried_acked(1), Some(0));
    // The handover happens once.
    assert_eq!(sel.carried_acked(1), None);
}

#[test]
fn background_probes_stay_within_two_percent_of_the_traffic() {
    let policy = Policy::default();
    assert!(probe_allowed(&policy, 0, 1_000_000));
    assert!(probe_allowed(&policy, 19_999, 1_000_000));
    assert!(!probe_allowed(&policy, 20_000, 1_000_000));
    assert!(!probe_allowed(&policy, 1, 0));
}
