//! The opt-in diagnostic trace of the sidecar: one JSON line per session per interval, with
//! the per-subflow counters and the session-wide queues that decide a transfer's rate. It owns
//! the setting, the line format and the sampling task. It does not own what is measured, which
//! the session reports, and it changes no behaviour of the session. SPEC.md letify-link,
//! Sidecar interface item 10.

use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use tokio::task::AbortHandle;

use crate::paths::{PathKind, PathKinds};
use crate::session::{FlowSnapshot, Session, SubflowId, SubflowStats, TraceEvent, WeakSession};

/// The environment variable that names the trace file.
pub const ENV_TRACE: &str = "LETIFY_LINK_TRACE";
/// The environment variable that sets the sampling interval in milliseconds.
pub const ENV_TRACE_MS: &str = "LETIFY_LINK_TRACE_MS";
/// The trace file a binary falls back to when it was built with `LETIFY_LINK_TRACE_DEFAULT`.
const BUILT_IN_PATH: Option<&str> = option_env!("LETIFY_LINK_TRACE_DEFAULT");
/// The interval a binary falls back to when it was built with `LETIFY_LINK_TRACE_MS_DEFAULT`.
const BUILT_IN_MS: Option<&str> = option_env!("LETIFY_LINK_TRACE_MS_DEFAULT");
const DEFAULT_MS: u64 = 1000;
const LEAST_MS: u64 = 50;

/// Where the trace goes and how often it is sampled.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceConfig {
    /// The file the lines are appended to.
    pub path: PathBuf,
    /// The time between two lines of one session.
    pub interval: Duration,
}

impl TraceConfig {
    /// The setting from the environment and the build, or `None` when tracing is off.
    pub fn from_env() -> Option<TraceConfig> {
        TraceConfig::parse(
            std::env::var(ENV_TRACE).ok().as_deref(),
            std::env::var(ENV_TRACE_MS).ok().as_deref().or(BUILT_IN_MS),
            BUILT_IN_PATH,
        )
    }

    /// The setting from the three raw values: the variable, the interval and the build default.
    /// An empty path is off. An interval that is not a number is the default, and one below
    /// 50 ms is 50 ms.
    pub fn parse(
        path: Option<&str>,
        ms: Option<&str>,
        built_in: Option<&str>,
    ) -> Option<TraceConfig> {
        let path = path.or(built_in).filter(|p| !p.is_empty())?;
        let ms = ms
            .and_then(|m| m.trim().parse::<u64>().ok())
            .unwrap_or(DEFAULT_MS)
            .max(LEAST_MS);
        Some(TraceConfig {
            path: path.into(),
            interval: Duration::from_millis(ms),
        })
    }
}

/// User plus system CPU time of this process in milliseconds, where `/proc/self/stat` exists.
fn cpu_ms() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/self/stat").ok()?;
    // The command name may hold spaces, so count fields after its closing parenthesis.
    let rest = &text[text.rfind(')')? + 1..];
    let mut fields = rest.split_whitespace();
    let utime: u64 = fields.nth(11)?.parse().ok()?;
    let stime: u64 = fields.next()?.parse().ok()?;
    // A clock tick is 10 ms on every Linux the sidecar runs on.
    Some((utime + stime) * 10)
}

fn unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

fn ms(d: Option<Duration>) -> Value {
    d.map_or(Value::Null, |d| json!(d.as_secs_f64() * 1000.0))
}

/// The kind of a subflow: the path manager's answer when there is one, otherwise UDP for a
/// subflow with a congestion gauge and TCP for one without.
fn kind_name(kinds: Option<&PathKinds>, id: SubflowId, gauged: bool) -> &'static str {
    match kinds.and_then(|k| k.of(id)) {
        Some(PathKind::Tcp) => "tcp",
        Some(PathKind::Udp) => "udp",
        Some(PathKind::Relay) => "relay",
        None if gauged => "udp",
        None => "tcp",
    }
}

/// The line for one event: a subflow that closed or the session failing.
pub fn event_line(t_ms: u64, session: u64, event: &TraceEvent, kinds: Option<&PathKinds>) -> Value {
    let mut v = match event {
        TraceEvent::SubflowClosed { id, gauged, reason } => json!({
            "event": "subflow_closed", "subflow": id, "reason": reason,
            "kind": kind_name(kinds, *id, *gauged), "gauged": gauged,
        }),
        TraceEvent::Failed(why) => json!({"event": "session_failed", "reason": why}),
    };
    v["t_ms"] = json!(t_ms);
    v["unix_ms"] = json!(unix_ms());
    v["session"] = json!(session);
    v
}

/// One line of the trace.
pub fn line(
    t_ms: u64,
    session: u64,
    cpu: Option<u64>,
    flow: &FlowSnapshot,
    subflows: &[SubflowStats],
    kinds: Option<&PathKinds>,
) -> Value {
    let subs: Vec<Value> = subflows
        .iter()
        .map(|s| {
            let mut v = json!({
                "id": s.id, "sent": s.sent_bytes, "received": s.received_bytes,
                "acked": s.acked_bytes, "inflight": s.inflight_bytes, "window": s.window,
                "bulk_cap": s.bulk_cap, "srtt_ms": ms(s.srtt),
                "delivery_rate": s.delivery_rate,
                "kind": kind_name(kinds, s.id, s.gauged), "gauged": s.gauged,
                "window_limit": s.window_limit, "min_rtt_ms": ms(s.min_rtt),
            });
            if let Some(c) = &s.congestion {
                v["cwnd"] = json!(c.window);
                v["lost"] = json!(c.lost_packets);
                v["sent_packets"] = json!(c.sent_packets);
            }
            v
        })
        .collect();
    let mut v = json!({
        "t_ms": t_ms, "unix_ms": unix_ms(), "session": session,
        "unacked_bytes": flow.unacked_bytes, "max_unacked": flow.max_unacked,
        "reinjected": flow.reinjected, "reorder_bytes": flow.reorder_bytes,
        "reorder_peak": flow.reorder_peak, "pending_frames": flow.pending_frames,
        "queued_bytes": flow.queued_bytes, "min_send_credit": flow.min_send_credit,
        "subflows": subs,
    });
    if let Some(c) = cpu {
        v["cpu_ms"] = json!(c);
    }
    v
}

/// Sample `session` into the trace file until the session is gone. The task ends when the
/// last handle to the session is dropped, and the returned handle aborts it earlier.
pub fn spawn(
    config: TraceConfig,
    id: u64,
    session: WeakSession,
    start: Instant,
    kinds: Option<PathKinds>,
) -> AbortHandle {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(config.interval);
        loop {
            tick.tick().await;
            let Some(session) = session.upgrade() else {
                return;
            };
            if !write_line(&config, id, &session, start, kinds.as_ref()) {
                return;
            }
        }
    })
    .abort_handle()
}

fn write_line(
    config: &TraceConfig,
    id: u64,
    session: &Session,
    start: Instant,
    kinds: Option<&PathKinds>,
) -> bool {
    let t_ms = start.elapsed().as_millis() as u64;
    let mut out = String::new();
    for ev in session.take_trace_events() {
        out.push_str(&format!("{}\n", event_line(t_ms, id, &ev, kinds)));
    }
    let value = line(
        t_ms,
        id,
        cpu_ms(),
        &session.flow_snapshot(),
        &session.subflow_stats(),
        kinds,
    );
    out.push_str(&format!("{value}\n"));
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&config.path)
        .and_then(|mut f| f.write_all(out.as_bytes()))
        .is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_trace_is_off_without_a_path() {
        assert_eq!(TraceConfig::parse(None, Some("100"), None), None);
        assert_eq!(TraceConfig::parse(Some(""), None, Some("/x")), None);
    }

    #[test]
    fn the_variable_wins_over_the_built_in_path() {
        let c = TraceConfig::parse(Some("/a"), None, Some("/b")).unwrap();
        assert_eq!(c.path, PathBuf::from("/a"));
        let c = TraceConfig::parse(None, None, Some("/b")).unwrap();
        assert_eq!(c.path, PathBuf::from("/b"));
    }

    #[test]
    fn the_interval_defaults_to_one_second_and_is_at_least_fifty_milliseconds() {
        let at = |ms| TraceConfig::parse(Some("/a"), ms, None).unwrap().interval;
        assert_eq!(at(None), Duration::from_secs(1));
        assert_eq!(at(Some("soon")), Duration::from_secs(1));
        assert_eq!(at(Some("5")), Duration::from_millis(50));
        assert_eq!(at(Some("250")), Duration::from_millis(250));
    }

    #[test]
    fn a_line_names_the_session_queues_and_every_subflow() {
        let flow = FlowSnapshot {
            unacked_bytes: 7,
            max_unacked: 8,
            reinjected: 3,
            reorder_bytes: 5,
            reorder_peak: 6,
            pending_frames: 1,
            queued_bytes: 2,
            min_send_credit: Some(9),
        };
        let v = line(1500, 4, Some(30), &flow, &[], None);
        assert_eq!(v["t_ms"], 1500);
        assert_eq!(v["session"], 4);
        assert_eq!(v["cpu_ms"], 30);
        assert_eq!(v["reinjected"], 3);
        assert_eq!(v["min_send_credit"], 9);
        assert_eq!(v["subflows"].as_array().unwrap().len(), 0);
    }
}
