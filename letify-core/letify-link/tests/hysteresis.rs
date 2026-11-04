//! Pins SPEC.md letify-link, Paths and the path manager item 3 and item 6: dwell,
//! confirmation, no downgrade, handover, probe budget and the floor. Time is injected, so no test waits.

use std::time::Duration;

use letify_link::hysteresis::{
    Decision, FloorWatch, PathView, Policy, Reason, Selector, probe_allowed, probe_may_start,
};

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

#[test]
fn a_probe_may_start_while_the_spend_is_under_the_budget_or_the_minimum() {
    let policy = Policy::default();
    // 2% of 1 MB is 20 kB, above the 16 KiB minimum.
    assert!(probe_may_start(&policy, 19_999, 1_000_000, 16 << 10));
    assert!(!probe_may_start(&policy, 20_000, 1_000_000, 16 << 10));
    // An idle session still gets one probe: the minimum is the budget.
    assert!(probe_may_start(&policy, 0, 0, 16 << 10));
    assert!(!probe_may_start(&policy, 16 << 10, 0, 16 << 10));
}

fn watch() -> FloorWatch {
    FloorWatch::new(100.0, &Policy::default())
}

#[test]
fn a_subflow_below_the_floor_is_closed_after_three_spaced_measurements() {
    let mut w = watch();
    let paths = [view(0, 50.0), view(1, 500.0)];
    assert!(w.observe(s(0), &paths).is_empty());
    assert!(w.observe(s(10), &paths).is_empty());
    assert_eq!(w.observe(s(20), &paths), vec![0]);
}

#[test]
fn measurements_closer_than_the_spacing_do_not_count_toward_the_floor_rule() {
    let mut w = watch();
    let paths = [view(0, 50.0), view(1, 500.0)];
    for t in 0..20 {
        assert!(w.observe(s(t), &paths).is_empty());
    }
    assert_eq!(w.observe(s(20), &paths), vec![0]);
}

#[test]
fn a_measurement_at_or_above_the_floor_resets_the_floor_count() {
    let mut w = watch();
    let low = [view(0, 50.0), view(1, 500.0)];
    let ok = [view(0, 100.0), view(1, 500.0)];
    assert!(w.observe(s(0), &low).is_empty());
    assert!(w.observe(s(10), &low).is_empty());
    assert!(w.observe(s(20), &ok).is_empty());
    assert!(w.observe(s(30), &low).is_empty());
    assert!(w.observe(s(40), &low).is_empty());
    assert_eq!(w.observe(s(50), &low), vec![0]);
}

#[test]
fn a_sole_subflow_is_never_closed_for_being_slow() {
    let mut w = watch();
    for t in 0..10 {
        assert!(w.observe(s(t * 10), &[view(0, 1.0)]).is_empty());
    }
    // A dead neighbour is not a subflow above the floor.
    let dead = PathView {
        id: 1,
        rate: 0.0,
        alive: false,
    };
    for t in 10..20 {
        assert!(w.observe(s(t * 10), &[view(0, 1.0), dead]).is_empty());
    }
}

#[test]
fn no_subflow_is_closed_while_every_subflow_is_below_the_floor() {
    let mut w = watch();
    let paths = [view(0, 50.0), view(1, 60.0)];
    for t in 0..10 {
        assert!(w.observe(s(t * 10), &paths).is_empty());
    }
}
