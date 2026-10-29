//! The path manager for transports the caller can dial (TCP today). It owns candidate dialing,
//! starting the session on the first candidate that completes the handshake, adding the others
//! as subflows, redialing when a subflow dies, and measuring each subflow from its traffic. It
//! does not own the switching rules (see `hysteresis`), the session or any listener.
//!
//! With TCP only, every candidate that connects is a subflow and the session spreads frames
//! over all of them. The switching rules then only decide which path is reported as preferred.
//! The manager also measures an idle subflow with probes within the probe budget, closes a
//! subflow that stays below the caller's floor while another is above it, and reports path
//! events without loss for the lifecycle kinds.

use std::collections::{HashSet, VecDeque};
use std::future::Future;
use std::io;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::mpsc;
use tokio::task::AbortHandle;
use tokio::time::Instant;

use crate::config::Config;
use crate::error::{Error, Result};
use crate::hysteresis::{Decision, FloorWatch, PathView, Policy, Selector, probe_may_start};
use crate::identity::{Identity, NodeId};
use crate::session::{Session, SubflowId, Transport, WeakSession};

/// A transport behind a pointer, so candidates of different kinds share one type.
pub type BoxTransport = Box<dyn Transport>;

type Dial = Pin<Box<dyn Future<Output = io::Result<BoxTransport>> + Send>>;

/// One way to reach the peer: a name and an async function that returns a new transport.
#[derive(Clone)]
pub struct Candidate {
    name: String,
    dial: Arc<dyn Fn() -> Dial + Send + Sync>,
}

impl Candidate {
    /// A candidate that calls `dial` for each new transport.
    pub fn new<F, Fut, T>(name: impl Into<String>, dial: F) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = io::Result<T>> + Send + 'static,
        T: Transport,
    {
        Candidate {
            name: name.into(),
            dial: Arc::new(move || {
                let fut = dial();
                Box::pin(async move { fut.await.map(|t| Box::new(t) as BoxTransport) })
            }),
        }
    }

    /// A TCP connection to `addr`, tuned by [`crate::tcp::tune`].
    pub fn tcp(name: impl Into<String>, addr: SocketAddr) -> Self {
        Self::new(name, move || crate::tcp::connect(addr))
    }

    /// The candidate's name, as path events report it.
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Settings of the path manager.
#[derive(Clone, Debug)]
pub struct ManagerConfig {
    /// How often subflows are measured. Default 1 s.
    pub measure_interval: Duration,
    /// First wait before redialing a candidate. Default 250 ms.
    pub redial_initial: Duration,
    /// The wait doubles on each failure up to this. Default 10 s.
    pub redial_max: Duration,
    /// Time limit for one dial. Default 10 s.
    pub dial_timeout: Duration,
    /// Time limit for the first handshake over any candidate. Default 60 s.
    pub start_timeout: Duration,
    /// The switching rules.
    pub policy: Policy,
    /// Unread `Measured` events kept for a caller that reads slowly; later ones are dropped.
    /// `Added`, `Lost` and `Preferred` events are never dropped. Default 256.
    pub event_capacity: usize,
    /// A subflow below this rate (bytes per second) for the confirmation rule of `policy`,
    /// while another subflow is at or above it, is closed. A sole subflow is never closed for
    /// this. Default none.
    pub floor: Option<f64>,
    /// How long a candidate closed for being below the floor is left alone before it is
    /// dialed again. Default 60 s.
    pub floor_backoff: Duration,
    /// The sliding window of the probe budget. Default 30 s.
    pub probe_window: Duration,
    /// Size of one probe record in bytes, and the least budget a window gets. Default 16 KiB.
    pub probe_min_bytes: usize,
    /// A subflow that carried no data for this long is probed. Zero turns probes off.
    /// Default 5 s.
    pub probe_idle: Duration,
}

impl Default for ManagerConfig {
    fn default() -> Self {
        ManagerConfig {
            measure_interval: Duration::from_secs(1),
            redial_initial: Duration::from_millis(250),
            redial_max: Duration::from_secs(10),
            dial_timeout: Duration::from_secs(10),
            start_timeout: Duration::from_secs(60),
            policy: Policy::default(),
            event_capacity: 256,
            floor: None,
            floor_backoff: Duration::from_secs(60),
            probe_window: Duration::from_secs(30),
            probe_min_bytes: 16 << 10,
            probe_idle: Duration::from_secs(5),
        }
    }
}

/// What happened to a path.
#[derive(Clone, Debug)]
pub enum PathEvent {
    /// A candidate became a subflow of the session.
    Added {
        /// The candidate's name.
        name: String,
    },
    /// The subflow of a candidate died or was closed. The manager redials it.
    Lost {
        /// The candidate's name.
        name: String,
        /// Why: "below the floor" when the manager closed it, otherwise "the subflow died".
        reason: String,
    },
    /// A measurement of a live subflow.
    Measured {
        /// The candidate's name.
        name: String,
        /// Bytes per second in both directions together, smoothed.
        throughput: f64,
        /// Round trip of the latest answered keepalive.
        rtt: Option<Duration>,
    },
    /// The switching rules now prefer this path.
    Preferred {
        /// The previous preferred path, if any.
        from: Option<String>,
        /// The new preferred path.
        to: String,
    },
}

struct Table {
    names: Vec<String>,
    subflow: Vec<Option<SubflowId>>,
    /// Set when the manager closed the candidate's subflow for being below the floor.
    floor_closed: Vec<bool>,
}

/// The sending side of the event queue. One queue keeps the order of events. Lifecycle events
/// always enter it. A `Measured` event enters only while fewer than `cap` are unread.
#[derive(Clone)]
struct Events {
    tx: mpsc::UnboundedSender<PathEvent>,
    unread_measured: Arc<AtomicUsize>,
    cap: usize,
}

impl Events {
    fn emit(&self, ev: PathEvent) {
        if matches!(ev, PathEvent::Measured { .. }) {
            if self.unread_measured.load(Ordering::Relaxed) >= self.cap {
                return;
            }
            self.unread_measured.fetch_add(1, Ordering::Relaxed);
        }
        let _ = self.tx.send(ev);
    }
}

fn emit(events: &Events, ev: PathEvent) {
    events.emit(ev);
}

/// A session started and kept alive over a set of candidate paths.
pub struct PathManager {
    session: Session,
    events: mpsc::UnboundedReceiver<PathEvent>,
    unread_measured: Arc<AtomicUsize>,
    probe_spent: Arc<AtomicU64>,
    tasks: Vec<AbortHandle>,
}

impl Drop for PathManager {
    fn drop(&mut self) {
        self.tasks.iter().for_each(AbortHandle::abort);
    }
}

enum Begin {
    Dial,
    Transport(BoxTransport),
    Attached(SubflowId),
}

impl PathManager {
    /// Dial every candidate at once and start the session on the first one that completes the
    /// handshake. The rest keep dialing and join the session as subflows.
    pub async fn start(
        identity: Identity,
        remote: NodeId,
        candidates: Vec<Candidate>,
        link: Config,
        cfg: ManagerConfig,
    ) -> Result<PathManager> {
        if candidates.is_empty() {
            return Err(Error::Subflow("no candidate paths".into()));
        }
        let n = candidates.len();
        let (ready_tx, mut ready_rx) = mpsc::channel::<(usize, BoxTransport)>(n * 2);
        let mut dialers: Vec<Option<AbortHandle>> = (0..n).map(|_| None).collect();
        let spawn_dialer = |i: usize, dialers: &mut Vec<Option<AbortHandle>>| {
            let (cand, tx, cfg) = (candidates[i].clone(), ready_tx.clone(), cfg.clone());
            let h = tokio::spawn(async move {
                let mut wait = cfg.redial_initial;
                loop {
                    match tokio::time::timeout(cfg.dial_timeout, (cand.dial)()).await {
                        Ok(Ok(t)) => {
                            let _ = tx.send((i, t)).await;
                            return;
                        }
                        _ => {
                            tokio::time::sleep(wait).await;
                            wait = (wait * 2).min(cfg.redial_max);
                        }
                    }
                }
            });
            dialers[i] = Some(h.abort_handle());
        };
        for i in 0..n {
            spawn_dialer(i, &mut dialers);
        }

        let deadline = Instant::now() + cfg.start_timeout;
        let mut consumed: HashSet<usize> = HashSet::new();
        let (session, winner) = loop {
            let got = tokio::time::timeout_at(deadline, ready_rx.recv()).await;
            let Ok(Some((i, transport))) = got else {
                dialers.iter().flatten().for_each(AbortHandle::abort);
                return Err(Error::Handshake(
                    "no candidate completed the handshake in time".into(),
                ));
            };
            consumed.insert(i);
            match Session::connect(transport, &identity, remote, link.clone()).await {
                Ok(s) => break (s, i),
                Err(e @ Error::VersionMismatch { .. }) => {
                    dialers.iter().flatten().for_each(AbortHandle::abort);
                    return Err(e);
                }
                Err(_) => {
                    consumed.remove(&i);
                    spawn_dialer(i, &mut dialers);
                }
            }
        };
        dialers.iter().flatten().for_each(AbortHandle::abort);

        let mut begin: Vec<Begin> = (0..n).map(|_| Begin::Dial).collect();
        let first = session
            .subflow_stats()
            .first()
            .map(|s| s.id)
            .ok_or_else(|| Error::ConnectionLost("the first subflow is already gone".into()))?;
        begin[winner] = Begin::Attached(first);
        while let Ok((i, t)) = ready_rx.try_recv() {
            if i != winner {
                begin[i] = Begin::Transport(t);
            }
        }

        let table = Arc::new(Mutex::new(Table {
            names: candidates.iter().map(|c| c.name.clone()).collect(),
            subflow: vec![None; n],
            floor_closed: vec![false; n],
        }));
        let (tx, events) = mpsc::unbounded_channel();
        let unread_measured = Arc::new(AtomicUsize::new(0));
        let events_tx = Events {
            tx,
            unread_measured: unread_measured.clone(),
            cap: cfg.event_capacity.max(1),
        };
        let probe_spent = Arc::new(AtomicU64::new(0));
        let mut tasks = Vec::new();
        for (i, (cand, begin)) in candidates.into_iter().zip(begin).enumerate() {
            let task = tokio::spawn(candidate_loop(
                i,
                cand,
                session.downgrade(),
                table.clone(),
                events_tx.clone(),
                begin,
                cfg.clone(),
            ));
            tasks.push(task.abort_handle());
        }
        let measure = tokio::spawn(measure_loop(
            session.downgrade(),
            table,
            events_tx,
            probe_spent.clone(),
            cfg,
        ));
        tasks.push(measure.abort_handle());
        Ok(PathManager {
            session,
            events,
            unread_measured,
            probe_spent,
            tasks,
        })
    }

    /// The session. Dropping every handle to it, the manager's included, closes it.
    pub fn session(&self) -> &Session {
        &self.session
    }

    /// The next path event, or `None` once the session is over and the events are read.
    pub async fn next_event(&mut self) -> Option<PathEvent> {
        let ev = self.events.recv().await?;
        if matches!(ev, PathEvent::Measured { .. }) {
            self.unread_measured.fetch_sub(1, Ordering::Relaxed);
        }
        Some(ev)
    }

    /// Bytes of probe records this manager has started so far.
    pub fn probe_bytes_spent(&self) -> u64 {
        self.probe_spent.load(Ordering::Relaxed)
    }
}

/// Keep one subflow per candidate: dial, add, wait for its death, redial with backoff.
async fn candidate_loop(
    idx: usize,
    cand: Candidate,
    weak: WeakSession,
    table: Arc<Mutex<Table>>,
    events: Events,
    begin: Begin,
    cfg: ManagerConfig,
) {
    let mut wait = cfg.redial_initial;
    let (mut attached, mut transport) = match begin {
        Begin::Dial => (None, None),
        Begin::Transport(t) => (None, Some(t)),
        Begin::Attached(id) => (Some(id), None),
    };
    loop {
        let Some(session) = weak.upgrade() else {
            return;
        };
        if session.is_closed() {
            return;
        }
        let id = match attached.take() {
            Some(id) => id,
            None => {
                drop(session);
                let t = match transport.take() {
                    Some(t) => t,
                    None => match tokio::time::timeout(cfg.dial_timeout, (cand.dial)()).await {
                        Ok(Ok(t)) => t,
                        _ => {
                            tokio::time::sleep(wait).await;
                            wait = (wait * 2).min(cfg.redial_max);
                            continue;
                        }
                    },
                };
                let Some(session) = weak.upgrade() else {
                    return;
                };
                match session.add_subflow(t).await {
                    Ok(id) => id,
                    Err(Error::ConnectionLost(_) | Error::Closed) => return,
                    Err(_) => {
                        drop(session);
                        tokio::time::sleep(wait).await;
                        wait = (wait * 2).min(cfg.redial_max);
                        continue;
                    }
                }
            }
        };
        let mut changes = match weak.upgrade() {
            Some(s) => s.subflow_changes(),
            None => return,
        };
        {
            let mut t = table.lock().unwrap_or_else(|e| e.into_inner());
            t.subflow[idx] = Some(id);
            t.floor_closed[idx] = false;
        }
        emit(
            &events,
            PathEvent::Added {
                name: cand.name.clone(),
            },
        );
        let up_since = Instant::now();
        loop {
            let alive = match weak.upgrade() {
                Some(s) => !s.is_closed() && s.subflow_stats().iter().any(|st| st.id == id),
                None => return,
            };
            if !alive {
                break;
            }
            if changes.changed().await.is_err() {
                return;
            }
        }
        let by_floor = {
            let mut t = table.lock().unwrap_or_else(|e| e.into_inner());
            t.subflow[idx] = None;
            std::mem::take(&mut t.floor_closed[idx])
        };
        emit(
            &events,
            PathEvent::Lost {
                name: cand.name.clone(),
                reason: if by_floor {
                    "below the floor"
                } else {
                    "the subflow died"
                }
                .into(),
            },
        );
        if by_floor {
            tokio::time::sleep(cfg.floor_backoff).await;
            wait = cfg.redial_initial;
        } else if up_since.elapsed() > Duration::from_secs(1) {
            wait = cfg.redial_initial;
        } else {
            tokio::time::sleep(wait).await;
            wait = (wait * 2).min(cfg.redial_max);
        }
    }
}

/// What the measure loop remembers about one candidate's subflow between two ticks.
#[derive(Clone, Copy)]
struct Seen {
    id: SubflowId,
    /// Bytes in both directions, probes included.
    total: u64,
    acked: u64,
    /// Bytes in both directions without probes.
    data: u64,
    probe_acked: u64,
}

/// Drop the entries of a sliding window that are older than `window` and sum the rest.
fn window_sum(log: &mut VecDeque<(Instant, u64)>, now: Instant, window: Duration) -> u64 {
    while log
        .front()
        .is_some_and(|(at, _)| now.duration_since(*at) > window)
    {
        log.pop_front();
    }
    log.iter().map(|(_, bytes)| bytes).sum()
}

/// Measure each subflow from its traffic, probe the idle ones, close the ones below the floor
/// and feed the switching rules.
async fn measure_loop(
    weak: WeakSession,
    table: Arc<Mutex<Table>>,
    events: Events,
    probe_spent: Arc<AtomicU64>,
    cfg: ManagerConfig,
) {
    let n = table.lock().unwrap_or_else(|e| e.into_inner()).names.len();
    let mut policy = cfg.policy.clone();
    if let Some(floor) = cfg.floor {
        policy.floor = floor;
    }
    let mut selector = Selector::new(policy.clone());
    let mut floor_watch = cfg.floor.map(|f| FloorWatch::new(f, &policy));
    let origin = Instant::now();
    let mut seen: Vec<Option<Seen>> = vec![None; n];
    let mut rate = vec![0.0f64; n];
    // The rate a probe gave an idle subflow, kept until the subflow carries data again.
    let mut probed: Vec<Option<f64>> = vec![None; n];
    let mut last_data = vec![origin; n];
    let mut last_probe: Vec<Option<Instant>> = vec![None; n];
    let mut carried_log: VecDeque<(Instant, u64)> = VecDeque::new();
    let mut probe_log: VecDeque<(Instant, u64)> = VecDeque::new();
    let mut last_at = origin;
    loop {
        tokio::time::sleep(cfg.measure_interval).await;
        let Some(session) = weak.upgrade() else {
            return;
        };
        if session.is_closed() {
            return;
        }
        let stats = session.subflow_stats();
        let now = Instant::now();
        let dt = now.duration_since(last_at).as_secs_f64().max(1e-6);
        last_at = now;
        let (names, subflow) = {
            let t = table.lock().unwrap_or_else(|e| e.into_inner());
            (t.names.clone(), t.subflow.clone())
        };
        let mut views = Vec::with_capacity(n);
        let mut carried_now = 0u64;
        for i in 0..n {
            let stat = subflow[i].and_then(|id| stats.iter().find(|s| s.id == id));
            let Some(stat) = stat else {
                seen[i] = None;
                rate[i] = 0.0;
                probed[i] = None;
                views.push(PathView {
                    id: i,
                    rate: 0.0,
                    alive: false,
                });
                continue;
            };
            let total = stat.sent_bytes + stat.received_bytes;
            let data = total.saturating_sub(stat.probe_sent_bytes + stat.probe_received_bytes);
            match seen[i] {
                Some(prev) if prev.id == stat.id => {
                    let instant = total.saturating_sub(prev.total) as f64 / dt;
                    rate[i] = 0.5 * rate[i] + 0.5 * instant;
                    let data_delta = data.saturating_sub(prev.data);
                    carried_now += data_delta;
                    if data_delta > 1024 {
                        last_data[i] = now;
                        probed[i] = None;
                    }
                    if stat.probe_acked_bytes > prev.probe_acked {
                        probed[i] = stat.probe_rate;
                    }
                }
                _ => {
                    // A subflow seen for the first time counts as active now.
                    last_data[i] = now;
                    probed[i] = None;
                }
            }
            let shown = rate[i].max(probed[i].unwrap_or(0.0));
            if seen[i].is_some_and(|p| p.id == stat.id) {
                emit(
                    &events,
                    PathEvent::Measured {
                        name: names[i].clone(),
                        throughput: shown,
                        rtt: stat.rtt,
                    },
                );
            }
            if stat.acked_bytes > seen[i].filter(|p| p.id == stat.id).map_or(0, |p| p.acked) {
                selector.carried_acked(i);
            }
            seen[i] = Some(Seen {
                id: stat.id,
                total,
                acked: stat.acked_bytes,
                data,
                probe_acked: stat.probe_acked_bytes,
            });
            views.push(PathView {
                id: i,
                rate: shown,
                alive: true,
            });
        }
        carried_log.push_back((now, carried_now));

        // Probe one idle subflow when the window's budget allows.
        if !cfg.probe_idle.is_zero() {
            let carried = window_sum(&mut carried_log, now, cfg.probe_window);
            let spent = window_sum(&mut probe_log, now, cfg.probe_window);
            if probe_may_start(&policy, spent, carried, cfg.probe_min_bytes as u64) {
                let pick = (0..n)
                    .filter(|&i| {
                        subflow[i].is_some()
                            && seen[i].is_some()
                            && now.duration_since(last_data[i]) >= cfg.probe_idle
                            && last_probe[i]
                                .is_none_or(|at| now.duration_since(at) >= cfg.probe_idle)
                    })
                    .min_by_key(|&i| last_probe[i]);
                if let Some(i) = pick
                    && let Some(id) = subflow[i]
                    && session.send_probe(id, cfg.probe_min_bytes).is_ok()
                {
                    last_probe[i] = Some(now);
                    probe_log.push_back((now, cfg.probe_min_bytes as u64));
                    probe_spent.fetch_add(cfg.probe_min_bytes as u64, Ordering::Relaxed);
                }
            }
        }

        if let Some(watch) = floor_watch.as_mut() {
            for i in watch.observe(now.duration_since(origin), &views) {
                if let Some(id) = subflow[i] {
                    table.lock().unwrap_or_else(|e| e.into_inner()).floor_closed[i] = true;
                    session.close_subflow(id);
                }
            }
        }
        drop(session);

        let before = selector.current();
        if let Decision::Move { to, .. } = selector.observe(now.duration_since(origin), &views) {
            emit(
                &events,
                PathEvent::Preferred {
                    from: before.map(|b| names[b].clone()),
                    to: names[to].clone(),
                },
            );
        }
    }
}
