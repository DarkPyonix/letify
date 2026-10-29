//! The session: sequencing, acknowledgement, retransmission, subflow scheduling and
//! resumption. It owns the shared state and the tasks that move records over subflows. It does
//! not own stream reads and writes (see `stream`) or path discovery.
//!
//! Scheduling: pull based. Every subflow has a writer task that takes the next pending frame
//! when its previous write completed and its in-flight bytes are below its window. In-flight
//! bytes are freed by the subflow ack (the bytes the receiver counted on that subflow), so a
//! faster subflow takes proportionally more frames without a rate estimate. The session ack
//! alone frees the retransmit buffer. A frame that blocks delivery behind a slow or stalled
//! subflow is reinjected on another subflow, and the slow one's window is halved. The receiver
//! reorders by sequence number in a buffer bounded in bytes.
//!
//! Stream scheduling (SPEC.md letify-link, Session item 7): each stream has its own send queue
//! of frames without a sequence number. A writer picks retransmissions first, then control
//! frames, then interactive streams and then bulk streams, each class by deficit round robin
//! over bytes, and assigns the sequence number when it takes the frame. Bulk bytes in flight on
//! a subflow are capped to about two bandwidth delay products so interactive frames do not wait
//! behind a deep queue.

use std::any::Any;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::Waker;
use std::time::{Duration, Instant};

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::sync::{Notify, watch};
use tokio::task::AbortHandle;

use crate::config::{Config, PowerState};
use crate::error::{Error, Result};
use crate::frame::{self, Frame, HDR, Kind, RECORD_HDR, TAG};
use crate::handshake::{self, Established};
use crate::identity::{Identity, NodeId, PeerPolicy};
use crate::keys::Keys;
use crate::replay::MAX_UNACKED_FRAMES;
use crate::stream::Stream;

/// A reliable, ordered byte transport that a session can use as a subflow: a TCP connection,
/// a pipe, or anything else that implements `AsyncRead` and `AsyncWrite`.
pub trait Transport: AsyncRead + AsyncWrite + Send + Unpin + Any {}
impl<T: AsyncRead + AsyncWrite + Send + Unpin + 'static> Transport for T {}

/// What a transport that runs its own congestion control reports about it.
#[derive(Clone, Copy, Debug)]
pub struct CongestionStats {
    /// The congestion window in bytes: the most the transport puts in flight.
    pub window: usize,
    /// The transport's smoothed round trip, which a lost packet's recovery does not inflate.
    pub rtt: Duration,
    /// The lowest round trip the transport has seen.
    pub min_rtt: Duration,
    /// Packets the transport has declared lost so far.
    pub lost_packets: u64,
    /// Packets the transport has sent so far.
    pub sent_packets: u64,
}

/// Reads the congestion statistics of one subflow's transport.
pub(crate) type Gauge = Arc<dyn Fn() -> CongestionStats + Send + Sync>;

/// The gauge of `transport` when it is a UDP path, directly or behind a [`crate::BoxTransport`].
pub(crate) fn gauge_of<T: Transport>(transport: &T) -> Option<Gauge> {
    let any: &dyn Any = transport;
    if let Some(path) = any.downcast_ref::<crate::udp::UdpPath>() {
        return Some(path.gauge());
    }
    let boxed = any.downcast_ref::<crate::BoxTransport>()?;
    let inner: &dyn Any = &**boxed;
    inner
        .downcast_ref::<crate::udp::UdpPath>()
        .map(crate::udp::UdpPath::gauge)
}

const SUBFLOW_MAGIC: &[u8; 4] = b"LLNS";

pub(crate) struct Sent {
    bytes: Arc<Vec<u8>>,
    /// The frame belongs to a bulk stream and counts toward the bulk cap of its subflow.
    bulk: bool,
    /// The subflow that carries the latest copy.
    on: Option<u64>,
    /// When the latest copy was taken by a writer.
    at: Option<Instant>,
    /// A subflow that must not take this frame again, because it was reinjected away from it.
    avoid: Option<u64>,
}

/// A subflow that has not answered yet is assumed to have this round trip, and no round trip
/// counts as shorter.
const RTT_FLOOR: Duration = Duration::from_millis(20);
/// Reinjection waits for this many round trips.
const REINJECT_K: u32 = 2;
/// How often a session with several subflows checks for a blocking frame.
const TICK: Duration = Duration::from_millis(25);
/// The window of a subflow is halved at most this often.
const PENALTY_GAP: Duration = Duration::from_millis(100);
/// A penalised window never falls below this: two full frames.
const MIN_WINDOW: usize = 2 * (HDR + frame::CHUNK);
/// Most frames one reinjection moves.
const REINJECT_MAX: usize = 256;
/// Bytes of frames one stream may queue before its writer waits.
const QUEUE_LIMIT: usize = 256 << 10;
/// What a stream receives each turn of the deficit round robin: one full frame.
const QUANTUM: usize = HDR + frame::CHUNK;
/// The bulk cap of a subflow is never below this: two full frames.
const MIN_BULK: usize = 2 * (HDR + frame::CHUNK);
/// A subflow starts with this bulk cap: four full frames.
const START_BULK: usize = 4 * (HDR + frame::CHUNK);
/// A delivery rate sample covers at least this long, and at least four smoothed round trips.
const RATE_SAMPLE_FLOOR: Duration = Duration::from_millis(150);
/// The bulk cap grows by this factor while a larger cap still raises the delivery rate.
const CAP_GROWTH: f64 = 1.5;
/// A rate this much above the previous sample counts as a response to a larger cap.
const CAP_RESPONSE: f64 = 1.25;
/// A held cap is tried at this fraction of itself after `CAP_HOLD` samples.
const CAP_SHRINK: f64 = 0.75;
/// The bulk cap never exceeds this many times rate x lowest round trip, whatever probing says.
const CAP_BOUND: f64 = 8.0;
/// The rate bound uses the best of this many latest delivery rate samples.
const RATE_KEEP: usize = 8;
/// The lowest round trip of a subflow is taken over this span.
const MIN_RTT_SPAN: Duration = Duration::from_secs(10);
const CAP_HOLD: u32 = 6;
/// A subflow whose transport lost a packet this recently is recovering from loss.
const LOSS_HOLD: Duration = Duration::from_secs(1);
/// The loss ratio of a transport is taken over intervals of this length.
const LOSS_SPAN: Duration = Duration::from_millis(250);
/// A path that loses more than this share of its packets is overflowing a queue. A larger cap
/// would only add delay there, so the cap stays the probed one.
const HEAVY_LOSS: f64 = 0.10;
/// While a path recovers from loss, the cap holds at most this long a flight at the delivery
/// rate, so a small message waits behind at most this much bulk.
const LOSS_DELAY: Duration = Duration::from_millis(150);
/// A transport's congestion window is read at most this often.
const GAUGE_EVERY: Duration = Duration::from_millis(10);
/// While both classes have frames queued, this many interactive frames are followed by one
/// bulk frame, so bulk gets one stream frame in ten at least.
const GUARD_RUN: u32 = 9;

/// How the bulk cap of a subflow is moving.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CapMode {
    /// Raising the cap while the delivery rate responds.
    Grow,
    /// The rate did not respond to a larger cap; counting samples until a trial shrink.
    Hold(u32),
    /// Trying a smaller cap; the larger one is kept in `cap_saved`.
    Shrink,
}

/// Identifies a subflow within its session. Ids are never reused.
pub type SubflowId = u64;

/// Identifies a session: the first 16 bytes of its handshake hash.
pub type SessionId = [u8; 16];

/// What the session measured on one subflow.
#[derive(Clone, Debug)]
pub struct SubflowStats {
    /// The id that [`Session::add_subflow`] returned.
    pub id: SubflowId,
    /// Bytes written to the subflow, record headers included.
    pub sent_bytes: u64,
    /// Bytes read from the subflow, record headers included.
    pub received_bytes: u64,
    /// Bytes of frames sent on this subflow that the peer acknowledged.
    pub acked_bytes: u64,
    /// Round trip of the latest keepalive answered on this subflow, if any.
    pub rtt: Option<Duration>,
    /// Bytes of probe records written to this subflow.
    pub probe_sent_bytes: u64,
    /// Bytes of probe records read from this subflow.
    pub probe_received_bytes: u64,
    /// The subflow's current window in bytes: the configured window, less any penalty.
    pub window: usize,
    /// The smoothed time from writing a frame to the subflow ack that covers it, if any.
    pub srtt: Option<Duration>,
    /// Bytes of probe records this side sent that the peer answered.
    pub probe_acked_bytes: u64,
    /// The latest probe's size divided by the time to its answer, in bytes per second. A lower
    /// bound of the subflow's rate, because one small probe does not fill the path.
    pub probe_rate: Option<f64>,
    /// The most bulk bytes the session lets this subflow carry in flight at this moment.
    pub bulk_cap: usize,
    /// The latest delivery rate sample of the subflow in bytes per second, if any.
    pub delivery_rate: Option<f64>,
    /// What the subflow's transport reports about its congestion control, for a transport that
    /// reports it (a UDP path), as read at most every 10 ms.
    pub congestion: Option<CongestionStats>,
}

const NO_RTT: u64 = u64::MAX;

/// The handles a subflow shares with its two tasks.
struct SubShared {
    sent: Arc<AtomicU64>,
    received: Arc<AtomicU64>,
    rtt_us: Arc<AtomicU64>,
    notify: Arc<Notify>,
    aborts: Vec<AbortHandle>,
}

pub(crate) struct Sub {
    /// Bytes of sequenced frames written to this subflow.
    sent_seq: u64,
    /// The count the peer's subflow ack returned: bytes of sequenced frames it received here.
    acked_seq: u64,
    /// Bytes of sequenced frames received on this subflow, duplicates included.
    recv_seq: u64,
    /// A subflow ack is owed on this subflow.
    ack_dirty: bool,
    window: usize,
    last_penalty: Option<Instant>,
    srtt: Option<Duration>,
    /// For each frame written: the value of `sent_seq` after it and when, to time its ack.
    marks: VecDeque<(u64, Instant, usize)>,
    /// Bytes of bulk frames written to this subflow and not yet covered by its subflow ack.
    bulk_inflight: usize,
    /// Start of the current delivery rate sample and the bytes acknowledged since.
    rate_from: Instant,
    rate_bytes: u64,
    /// The latest delivery rate samples in bytes per second.
    rates: VecDeque<f64>,
    /// The lowest write to subflow ack time, and when it was taken.
    min_rtt: Option<(Duration, Instant)>,
    /// The bulk cap, see `adjust_cap`, the rate it was last judged by and its mode.
    cap: usize,
    cap_ref: f64,
    cap_saved: usize,
    cap_mode: CapMode,
    /// The keepalive interval in milliseconds that the peer knows through this subflow.
    told_keepalive: u64,
    /// A bulk frame had to wait for the cap since the last sample.
    capped: bool,
    /// Reads the transport's congestion window, and the latest reading with its time.
    gauge: Option<Gauge>,
    gauged: Option<(CongestionStats, Instant)>,
    /// When the transport last reported more lost packets than before.
    loss_at: Option<Instant>,
    /// The start of the current loss interval with the counts then, and the share of packets
    /// the transport lost in the last interval.
    loss_base: Option<(Instant, u64, u64)>,
    loss_ratio: f64,
    sent: Arc<AtomicU64>,
    received: Arc<AtomicU64>,
    acked: u64,
    rtt_us: Arc<AtomicU64>,
    /// A Ping stamp waiting to be answered by this subflow's writer.
    pong: Option<u64>,
    /// The size of a probe record this subflow's writer should send.
    probe_request: Option<usize>,
    /// The size of a probe record received and not yet answered.
    probe_answer: Option<u64>,
    /// The probe in flight: when it was sent and how big it was.
    probe_pending: Option<(Instant, usize)>,
    probe_sent: u64,
    probe_received: u64,
    probe_acked: u64,
    probe_rate: Option<f64>,
    notify: Arc<Notify>,
    aborts: Vec<AbortHandle>,
}

impl Sub {
    fn new(window: usize, gauge: Option<Gauge>, shared: SubShared) -> Sub {
        Sub {
            sent_seq: 0,
            acked_seq: 0,
            recv_seq: 0,
            ack_dirty: false,
            window,
            last_penalty: None,
            srtt: None,
            marks: VecDeque::new(),
            bulk_inflight: 0,
            rate_from: Instant::now(),
            rate_bytes: 0,
            rates: VecDeque::new(),
            min_rtt: None,
            cap: START_BULK,
            cap_ref: 0.0,
            cap_saved: START_BULK,
            cap_mode: CapMode::Grow,
            told_keepalive: 0,
            capped: false,
            gauge,
            gauged: None,
            loss_at: None,
            loss_base: None,
            loss_ratio: 0.0,
            sent: shared.sent,
            received: shared.received,
            acked: 0,
            rtt_us: shared.rtt_us,
            pong: None,
            probe_request: None,
            probe_answer: None,
            probe_pending: None,
            probe_sent: 0,
            probe_received: 0,
            probe_acked: 0,
            probe_rate: None,
            notify: shared.notify,
            aborts: shared.aborts,
        }
    }

    /// The most bulk bytes this subflow may have in flight: the smallest cap found by probing
    /// that keeps the delivery rate, between two frames and half the window. See SPEC.md
    /// letify-link, Session item 7.
    fn bulk_cap(&self) -> usize {
        let best = self.rates.iter().copied().fold(0.0, f64::max);
        let rtt = self.min_rtt.map_or(RTT_FLOOR, |(r, _)| r).max(RTT_FLOOR);
        let bound = (CAP_BOUND * best * rtt.as_secs_f64()) as usize;
        let top = (self.window / 2).max(QUANTUM);
        let probed = self.cap.min(bound).max(MIN_BULK).min(top);
        match self.gauged {
            // A path whose transport recovers losses by itself holds back the acknowledgements
            // for about one round trip of rate while it does, on top of what it has in flight.
            Some((stats, at))
                if self
                    .loss_at
                    .is_some_and(|l| at.saturating_duration_since(l) <= LOSS_HOLD)
                    && self.loss_ratio <= HEAVY_LOSS =>
            {
                let stall = (best * rtt.as_secs_f64()) as usize;
                let flight = stats.window.saturating_add(stall);
                let allowed = (best * LOSS_DELAY.as_secs_f64()) as usize;
                probed
                    .max(flight.min(allowed))
                    .min(self.window.max(QUANTUM))
            }
            _ => probed,
        }
    }

    /// Read the transport's congestion window if the last reading is older than `GAUGE_EVERY`.
    fn read_gauge(&mut self, now: Instant) {
        let Some(gauge) = &self.gauge else {
            return;
        };
        if self
            .gauged
            .is_none_or(|(_, at)| now.saturating_duration_since(at) >= GAUGE_EVERY)
        {
            let stats = gauge();
            if self
                .gauged
                .is_some_and(|(old, _)| stats.lost_packets > old.lost_packets)
            {
                self.loss_at = Some(now);
            }
            match self.loss_base {
                Some((from, lost, sent)) if now.saturating_duration_since(from) >= LOSS_SPAN => {
                    let sent_now = stats.sent_packets.saturating_sub(sent);
                    let lost_now = stats.lost_packets.saturating_sub(lost);
                    self.loss_ratio = if sent_now == 0 {
                        0.0
                    } else {
                        lost_now as f64 / sent_now as f64
                    };
                    self.loss_base = Some((now, stats.lost_packets, stats.sent_packets));
                }
                Some(_) => {}
                None => self.loss_base = Some((now, stats.lost_packets, stats.sent_packets)),
            }
            self.gauged = Some((stats, now));
        }
    }

    /// Judge the cap by one delivery rate sample `rate` (bytes per second). Only a sample in
    /// which a bulk frame waited for the cap says anything about it. The cap grows by half
    /// while the rate responds to it. Once the rate stops responding the cap is held, and after
    /// `CAP_HOLD` samples a smaller cap is tried and kept if the rate stays within 10%.
    fn adjust_cap(&mut self, rate: f64) {
        self.rates.push_back(rate);
        if self.rates.len() > RATE_KEEP {
            self.rates.pop_front();
        }
        if !std::mem::take(&mut self.capped) {
            return;
        }
        match self.cap_mode {
            CapMode::Grow => {
                if rate > self.cap_ref * CAP_RESPONSE {
                    self.cap = (self.cap as f64 * CAP_GROWTH) as usize;
                    self.cap_ref = rate;
                } else {
                    self.cap_ref = self.cap_ref.max(rate);
                    self.cap_mode = CapMode::Hold(0);
                }
            }
            CapMode::Hold(n) if n + 1 < CAP_HOLD => {
                self.cap_ref = self.cap_ref.max(rate);
                self.cap_mode = CapMode::Hold(n + 1);
            }
            CapMode::Hold(_) => {
                self.cap_ref = rate;
                self.cap_saved = self.cap;
                self.cap = ((self.cap as f64 * CAP_SHRINK) as usize).max(MIN_BULK);
                self.cap_mode = CapMode::Shrink;
            }
            CapMode::Shrink => {
                if rate < self.cap_ref * 0.9 {
                    // The smaller cap cost rate: go back and grow from there if need be.
                    self.cap = self.cap_saved;
                    self.cap_ref = rate;
                    self.cap_mode = CapMode::Grow;
                } else {
                    self.cap_ref = rate;
                    self.cap_mode = CapMode::Hold(CAP_HOLD / 2);
                }
            }
        }
    }
}

#[derive(Default)]
pub(crate) struct StreamState {
    pub send_credit: u64,
    pub recv_buf: VecDeque<(Vec<u8>, usize)>,
    pub unreported: u64,
    pub remote_fin: bool,
    pub fin_sent: bool,
    pub dropped: bool,
    pub read_waker: Option<Waker>,
    pub write_waker: Option<Waker>,
    /// Encoded frames (sequence number 0) waiting for a subflow writer.
    pub queue: VecDeque<Vec<u8>>,
    pub queued: usize,
    pub bulk: bool,
    deficit: usize,
    in_ring: bool,
}

/// The next fresh frame a writer would take.
enum Fresh {
    Ctl,
    Stream(u32),
    Close,
}

pub(crate) struct State {
    window: u32,
    peer_window: u32,
    max_unacked: usize,
    subflow_window: usize,
    next_seq: u64,
    unacked: BTreeMap<u64, Sent>,
    pub unacked_bytes: usize,
    pending: BTreeSet<u64>,
    /// Control frames (credit) that go before any stream frame.
    ctl: VecDeque<Vec<u8>>,
    /// Streams with queued frames, by class, in turn order.
    ring_interactive: VecDeque<u32>,
    ring_bulk: VecDeque<u32>,
    /// The Close frame is owed once every queue is empty.
    close_wanted: bool,
    /// Interactive stream frames taken since the last bulk one while bulk had frames queued.
    bulk_wait: u32,
    peer_next: u64,
    recv_next: u64,
    reorder: BTreeMap<u64, (Frame, usize)>,
    reorder_bytes: usize,
    max_reorder: usize,
    pub reorder_peak: usize,
    pub reinjected: u64,
    pub streams: HashMap<u32, StreamState>,
    accept_queue: VecDeque<u32>,
    next_stream_id: u32,
    subflows: HashMap<u64, Sub>,
    next_sub_id: u64,
    epoch: u64,
    pub failed: Option<String>,
    pub closing: bool,
    pub remote_closed: bool,
    /// Why the session was closed, as the closing side gave it.
    close_reason: Option<String>,
}

impl State {
    fn wake_subflows(&self) {
        for sub in self.subflows.values() {
            sub.notify.notify_one();
        }
    }

    fn wake_writers(&mut self) {
        for s in self.streams.values_mut() {
            if let Some(w) = s.write_waker.take() {
                w.wake();
            }
        }
    }

    fn wake_all_streams(&mut self) {
        for s in self.streams.values_mut() {
            if let Some(w) = s.write_waker.take() {
                w.wake();
            }
            if let Some(w) = s.read_waker.take() {
                w.wake();
            }
        }
    }

    /// Room for a data frame of `len` payload bytes, by the byte bound on the unacked buffer.
    pub fn has_room(&self, len: usize) -> bool {
        self.unacked_bytes == 0
            || (self.unacked_bytes + HDR + len <= self.max_unacked
                && self.unacked.len() < MAX_UNACKED_FRAMES)
    }

    /// Whether every queue of fresh frames and the retransmission set are empty.
    pub fn out_empty(&self) -> bool {
        self.pending.is_empty()
            && self.ctl.is_empty()
            && self.ring_interactive.is_empty()
            && self.ring_bulk.is_empty()
            && !self.close_wanted
    }

    /// Room in the stream's own queue for a frame of `len` payload bytes.
    pub fn queue_room(&self, stream: u32, len: usize) -> bool {
        self.streams
            .get(&stream)
            .is_none_or(|s| s.queued == 0 || s.queued + HDR + len <= QUEUE_LIMIT)
    }

    /// Queue a frame of `stream`, encoded with sequence number 0. The writer that takes it
    /// assigns the real number.
    pub fn enqueue_stream(&mut self, stream: u32, bytes: Vec<u8>) {
        let Some(s) = self.streams.get_mut(&stream) else {
            return;
        };
        s.queued += bytes.len();
        s.queue.push_back(bytes);
        if !s.in_ring {
            s.in_ring = true;
            s.deficit = 0;
            if s.bulk {
                self.ring_bulk.push_back(stream);
            } else {
                self.ring_interactive.push_back(stream);
            }
        }
        self.wake_subflows();
    }

    pub fn enqueue_fin(&mut self, stream: u32) {
        self.enqueue_stream(stream, frame::encode_fin(0, stream));
    }

    /// Ask for the Close frame, which goes after every queued frame.
    pub fn want_close(&mut self) {
        self.close_wanted = true;
        self.wake_subflows();
    }

    /// Change the class of a stream's sends.
    pub fn set_bulk(&mut self, stream: u32, bulk: bool) {
        let Some(s) = self.streams.get_mut(&stream) else {
            return;
        };
        if s.bulk == bulk {
            return;
        }
        s.bulk = bulk;
        if s.in_ring {
            s.deficit = 0;
            self.ring_interactive.retain(|i| *i != stream);
            self.ring_bulk.retain(|i| *i != stream);
            if bulk {
                self.ring_bulk.push_back(stream);
            } else {
                self.ring_interactive.push_back(stream);
            }
        }
        self.wake_subflows();
    }

    /// Account for bytes the reader consumed or discarded, and grant credit when enough
    /// accumulated.
    pub fn consumed(&mut self, stream: u32, n: usize) {
        let threshold = (self.window as u64 / 2).max(1);
        let Some(s) = self.streams.get_mut(&stream) else {
            return;
        };
        s.unreported += n as u64;
        if s.unreported >= threshold {
            let grant = std::mem::take(&mut s.unreported);
            self.ctl.push_back(frame::encode_credit(0, stream, grant));
            self.wake_subflows();
        }
    }

    /// The next fresh frame, without taking it, and its size. Deficit round robin: the stream
    /// at the front of its ring is served while its deficit covers its next frame, and
    /// otherwise gets a quantum and moves to the back. `None` when nothing may be sent now.
    ///
    /// Every `GUARD_RUN` interactive frames taken while bulk has frames queued, bulk goes first
    /// (the starvation guard). `fits` says whether the writer's subflow can take a frame now,
    /// so a bulk frame that does not fit never holds an interactive one back.
    fn peek_fresh(
        &mut self,
        fits: &mut dyn FnMut(usize, bool) -> bool,
    ) -> Option<(Fresh, usize, bool)> {
        let count_ok = |st: &State, reserve: usize| st.unacked.len() + reserve < MAX_UNACKED_FRAMES;
        if let Some(f) = self.ctl.front() {
            return count_ok(self, 0).then_some((Fresh::Ctl, f.len(), false));
        }
        let guard_turn = self.bulk_wait >= GUARD_RUN
            && !self.ring_bulk.is_empty()
            && !self.ring_interactive.is_empty();
        let order = if guard_turn {
            [true, false]
        } else {
            [false, true]
        };
        for bulk in order {
            // Bulk frames leave an eighth of the retransmit buffer to interactive ones.
            let (bytes_cap, count_reserve) = if bulk {
                (
                    self.max_unacked - self.max_unacked / 8,
                    MAX_UNACKED_FRAMES / 8,
                )
            } else {
                (self.max_unacked, 0)
            };
            loop {
                let ring = if bulk {
                    &mut self.ring_bulk
                } else {
                    &mut self.ring_interactive
                };
                let Some(&id) = ring.front() else {
                    break;
                };
                let Some(s) = self.streams.get_mut(&id) else {
                    ring.pop_front();
                    continue;
                };
                let Some(front) = s.queue.front() else {
                    s.in_ring = false;
                    ring.pop_front();
                    continue;
                };
                let len = front.len();
                if s.deficit < len {
                    s.deficit += QUANTUM;
                    ring.rotate_left(1);
                    continue;
                }
                let ok = (self.unacked_bytes == 0 || self.unacked_bytes + len <= bytes_cap)
                    && count_ok(self, count_reserve);
                if guard_turn && bulk {
                    if ok && fits(len, true) {
                        return Some((Fresh::Stream(id), len, true));
                    }
                    break;
                }
                return ok.then_some((Fresh::Stream(id), len, bulk));
            }
        }
        (self.close_wanted && count_ok(self, 0)).then_some((Fresh::Close, frame::HDR, false))
    }

    /// Take the frame that `peek_fresh` returned, give it the next sequence number, and keep
    /// it for retransmission.
    fn take_fresh(&mut self, which: Fresh, bulk: bool, id: u64, now: Instant) -> Arc<Vec<u8>> {
        let mut bytes = match which {
            Fresh::Ctl => self.ctl.pop_front().expect("peeked"),
            Fresh::Close => frame::encode_close(0, self.close_reason.as_deref()),
            Fresh::Stream(stream) => {
                let s = self.streams.get_mut(&stream).expect("peeked");
                let bytes = s.queue.pop_front().expect("peeked");
                s.queued -= bytes.len();
                s.deficit -= bytes.len();
                if let Some(w) = s.write_waker.take() {
                    w.wake();
                }
                if s.queue.is_empty() {
                    s.in_ring = false;
                    s.deficit = 0;
                    let ring = if s.bulk {
                        &mut self.ring_bulk
                    } else {
                        &mut self.ring_interactive
                    };
                    ring.retain(|i| *i != stream);
                    self.forget_if_done(stream);
                }
                if bulk || self.ring_bulk.is_empty() {
                    self.bulk_wait = 0;
                } else {
                    self.bulk_wait += 1;
                }
                bytes
            }
        };
        if matches!(which, Fresh::Close) {
            self.close_wanted = false;
        }
        let seq = self.next_seq;
        self.next_seq += 1;
        bytes[1..9].copy_from_slice(&seq.to_be_bytes());
        self.unacked_bytes += bytes.len();
        let bytes = Arc::new(bytes);
        self.unacked.insert(
            seq,
            Sent {
                bytes: bytes.clone(),
                bulk,
                on: Some(id),
                at: Some(now),
                avoid: None,
            },
        );
        bytes
    }

    pub fn forget_if_done(&mut self, stream: u32) {
        if let Some(s) = self.streams.get(&stream)
            && s.dropped
            && s.remote_fin
            && s.fin_sent
            && s.queue.is_empty()
        {
            self.streams.remove(&stream);
        }
    }

    fn apply_ack(&mut self, next: u64) {
        if next <= self.peer_next {
            return;
        }
        self.peer_next = next;
        let keep = self.unacked.split_off(&next);
        let done = std::mem::replace(&mut self.unacked, keep);
        for (_, sent) in done {
            self.unacked_bytes -= sent.bytes.len();
            if let Some(sub) = sent.on.and_then(|id| self.subflows.get_mut(&id)) {
                sub.acked += sent.bytes.len() as u64;
            }
        }
        self.pending = self.pending.split_off(&next);
        self.wake_writers();
        self.wake_subflows();
    }

    /// The subflow ack: the peer received `n` bytes of sequenced frames on subflow `id`.
    fn apply_sub_ack(&mut self, id: u64, n: u64) {
        let cap = self.subflow_window;
        let Some(sub) = self.subflows.get_mut(&id) else {
            return;
        };
        let n = n.min(sub.sent_seq);
        if n <= sub.acked_seq {
            return;
        }
        let delta = n - sub.acked_seq;
        sub.acked_seq = n;
        let now = Instant::now();
        let mut covered = None;
        while sub.marks.front().is_some_and(|(after, _, _)| *after <= n) {
            if let Some((_, at, bulk)) = sub.marks.pop_front() {
                sub.bulk_inflight = sub.bulk_inflight.saturating_sub(bulk);
                covered = Some((at, bulk));
            }
        }
        if let Some((at, _)) = covered {
            let sample = now.saturating_duration_since(at);
            if sub
                .min_rtt
                .is_none_or(|(low, when)| sample <= low || now - when > MIN_RTT_SPAN)
            {
                sub.min_rtt = Some((sample, now));
            }

            sub.srtt = Some(match sub.srtt {
                None => sample,
                Some(old) => old.mul_f64(7.0 / 8.0) + sample.mul_f64(1.0 / 8.0),
            });
        }
        sub.rate_bytes += delta;
        let span = (sub.srtt.unwrap_or_default() * 4).max(RATE_SAMPLE_FLOOR);
        let since = now.saturating_duration_since(sub.rate_from);
        if since >= span {
            let rate = sub.rate_bytes as f64 / since.as_secs_f64();
            sub.adjust_cap(rate);
            sub.rate_from = now;
            sub.rate_bytes = 0;
        }
        // A penalised window regains an eighth of what the subflow got acknowledged.
        sub.window = (sub.window + (delta / 8) as usize).min(cap);
        sub.notify.notify_one();
    }

    /// Reinject the frame that blocks delivery when it sits on a stalled or slower subflow.
    /// See SPEC.md letify-link, Session item 4.
    fn maybe_reinject(&mut self, now: Instant) {
        if self.subflows.len() < 2 {
            return;
        }
        let Some((&head, sent)) = self.unacked.iter().next() else {
            return;
        };
        if self.pending.contains(&head) {
            return;
        }
        let (Some(on), Some(at)) = (sent.on, sent.at) else {
            return;
        };
        let Some(owner) = self.subflows.get(&on) else {
            return;
        };
        let age = now.saturating_duration_since(at);
        let owner_rtt = owner.srtt.unwrap_or(RTT_FLOOR).max(RTT_FLOOR);
        let mut threshold = None;
        if age > owner_rtt * REINJECT_K {
            threshold = Some(owner_rtt * REINJECT_K);
        } else if !self.has_room(frame::CHUNK)
            || self
                .streams
                .values()
                .any(|s| s.write_waker.is_some() && s.queued + HDR + frame::CHUNK <= QUEUE_LIMIT)
        {
            let fastest = self
                .subflows
                .iter()
                .filter(|(id, s)| {
                    **id != on
                        && s.srtt.unwrap_or(RTT_FLOOR) < owner.srtt.unwrap_or(RTT_FLOOR)
                        && (s.sent_seq - s.acked_seq) as usize + HDR + frame::CHUNK <= s.window
                })
                .map(|(_, s)| s.srtt.unwrap_or(RTT_FLOOR).max(RTT_FLOOR))
                .min();
            if let Some(rtt) = fastest
                && age > rtt * REINJECT_K
            {
                threshold = Some(rtt * REINJECT_K);
            }
        }
        let Some(threshold) = threshold else {
            return;
        };
        let moved: Vec<u64> = self
            .unacked
            .iter()
            .filter(|(q, s)| {
                s.on == Some(on)
                    && !self.pending.contains(q)
                    && s.at
                        .is_some_and(|t| now.saturating_duration_since(t) > threshold)
            })
            .map(|(q, _)| *q)
            .take(REINJECT_MAX)
            .collect();
        for seq in &moved {
            if let Some(s) = self.unacked.get_mut(seq) {
                s.avoid = Some(on);
            }
            self.pending.insert(*seq);
        }
        self.reinjected += moved.len() as u64;
        if let Some(sub) = self.subflows.get_mut(&on)
            && sub
                .last_penalty
                .is_none_or(|t| now.saturating_duration_since(t) >= PENALTY_GAP)
        {
            sub.window = (sub.window / 2).max(MIN_WINDOW);
            sub.last_penalty = Some(now);
        }
        self.wake_subflows();
    }

    /// Handle a record read from subflow `id`. `len` is the size of its plaintext. An error
    /// ends the session.
    fn handle(
        &mut self,
        id: u64,
        len: usize,
        frame: Frame,
        accept: &Notify,
    ) -> std::result::Result<(), String> {
        match frame {
            Frame::Ack { next, sub } => {
                self.apply_ack(next);
                self.apply_sub_ack(id, sub);
            }
            Frame::Hello { next } => self.apply_ack(next),
            // Keepalives are answered by the subflow's reader, which knows the subflow.
            Frame::Ping { .. }
            | Frame::Pong { .. }
            | Frame::Probe
            | Frame::ProbeAck { .. }
            | Frame::Keepalive { .. } => {}
            Frame::Seq { seq, .. } => {
                if let Some(sub) = self.subflows.get_mut(&id) {
                    sub.recv_seq += len as u64;
                    sub.ack_dirty = true;
                }
                if seq >= self.recv_next && !self.reorder.contains_key(&seq) {
                    if seq != self.recv_next && self.reorder_bytes + len > self.max_reorder {
                        return Err(format!(
                            "the peer exceeded the receive buffer bound of {} bytes",
                            self.max_reorder
                        ));
                    }
                    self.reorder_bytes += len;
                    self.reorder.insert(seq, (frame, len));
                    while let Some((f, l)) = self.reorder.remove(&self.recv_next) {
                        self.recv_next += 1;
                        self.reorder_bytes -= l;
                        self.deliver(f, accept);
                    }
                    self.reorder_peak = self.reorder_peak.max(self.reorder_bytes);
                }
                self.wake_subflows();
            }
        }
        Ok(())
    }

    fn deliver(&mut self, frame: Frame, accept: &Notify) {
        let Frame::Seq { stream, kind, .. } = frame else {
            return;
        };
        match kind {
            Kind::Open => {
                if !self.streams.contains_key(&stream) {
                    let s = StreamState {
                        send_credit: self.peer_window as u64,
                        ..Default::default()
                    };
                    self.streams.insert(stream, s);
                    self.accept_queue.push_back(stream);
                    accept.notify_waiters();
                }
            }
            Kind::Data(buf) => {
                let len = buf.len() - HDR;
                let Some(s) = self.streams.get_mut(&stream) else {
                    return;
                };
                if s.dropped {
                    self.consumed(stream, len);
                    return;
                }
                s.recv_buf.push_back((buf, HDR));
                if let Some(w) = s.read_waker.take() {
                    w.wake();
                }
            }
            Kind::Fin => {
                if let Some(s) = self.streams.get_mut(&stream) {
                    s.remote_fin = true;
                    if let Some(w) = s.read_waker.take() {
                        w.wake();
                    }
                }
                self.forget_if_done(stream);
            }
            Kind::Credit(n) => {
                if let Some(s) = self.streams.get_mut(&stream) {
                    s.send_credit += n;
                    if let Some(w) = s.write_waker.take() {
                        w.wake();
                    }
                }
            }
            Kind::Close(reason) => {
                self.remote_closed = true;
                if self.close_reason.is_none() {
                    self.close_reason = reason;
                }
                self.closing = true;
                self.wake_all_streams();
                accept.notify_waiters();
                self.wake_subflows();
            }
        }
    }
}

pub(crate) struct Shared {
    pub state: Mutex<State>,
    accept: Notify,
    /// The keys of both directions, shared by every subflow and kept across resumption.
    keys: Keys,
    dropped: AtomicU64,
    cfg: Config,
    peer: NodeId,
    version: u16,
    session_id: SessionId,
    /// Counts every change in the set of subflows, so a path manager can wait for one.
    changes: watch::Sender<u64>,
    /// Origin of the keepalive stamps.
    origin: Instant,
    /// This side's keepalive interval in milliseconds (0 is off) and the peer's, as the
    /// handshake or a later Keepalive frame advertised it. A subflow with no record for 3 x
    /// the larger one is dead; with both off only a transport error ends a subflow.
    keepalive_ms: AtomicU64,
    peer_keepalive_ms: AtomicU64,
    /// The interval this side's handshake advertised. A subflow tells the peer when the
    /// current one differs.
    handshake_keepalive_ms: u64,
    /// Wakes the readers when either interval changed, so they recompute their dead time.
    keepalive_changed: Notify,
    /// Counts the calls of `resume_now`, so a path manager can redial at once.
    hints: watch::Sender<u64>,
}

/// An interval in the milliseconds the wire carries: a nonzero interval is at least 1.
fn interval_ms(interval: Duration) -> u64 {
    match interval.as_millis().min(u32::MAX as u128) as u64 {
        0 if !interval.is_zero() => 1,
        ms => ms,
    }
}

/// Closes the session when the last handle goes away.
pub(crate) struct Guard {
    pub shared: Arc<Shared>,
    /// The task that closes the session when its peer is revoked, see `Session::watch_policy`.
    watcher: Mutex<Option<AbortHandle>>,
}

impl Drop for Guard {
    fn drop(&mut self) {
        if let Some(task) = self
            .watcher
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
        {
            task.abort();
        }
        let mut st = self.shared.lock();
        if !st.closing && st.failed.is_none() {
            st.closing = true;
            st.want_close();
        }
    }
}

impl Shared {
    pub fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn fail(&self, st: &mut State, why: String) {
        if st.failed.is_some() {
            return;
        }
        st.failed = Some(why);
        for (_, sub) in st.subflows.drain() {
            sub.aborts.iter().for_each(AbortHandle::abort);
        }
        self.bump();
        st.wake_all_streams();
        self.accept.notify_waiters();
    }

    /// This side's keepalive interval, zero for off.
    fn keepalive(&self) -> Duration {
        Duration::from_millis(self.keepalive_ms.load(Ordering::Relaxed))
    }

    /// How long a subflow may stay silent before it is dead, or `None` when neither side
    /// sends keepalives.
    fn dead_after(&self) -> Option<Duration> {
        let longer = self
            .keepalive_ms
            .load(Ordering::Relaxed)
            .max(self.peer_keepalive_ms.load(Ordering::Relaxed));
        (longer > 0).then(|| Duration::from_millis(longer) * 3)
    }

    fn bump(&self) {
        self.changes.send_modify(|n| *n += 1);
    }

    fn stamp(&self) -> u64 {
        self.origin.elapsed().as_micros() as u64
    }

    fn subflow_died(self: &Arc<Self>, id: u64) {
        let mut st = self.lock();
        let Some(sub) = st.subflows.remove(&id) else {
            return;
        };
        self.bump();
        sub.aborts.iter().for_each(AbortHandle::abort);
        // Everything sent on the dead subflow and not acknowledged goes out again.
        let lost: Vec<u64> = st
            .unacked
            .iter()
            .filter(|(_, s)| s.on == Some(id))
            .map(|(q, _)| *q)
            .collect();
        for seq in lost {
            if let Some(s) = st.unacked.get_mut(&seq) {
                s.on = None;
            }
            st.pending.insert(seq);
        }
        if st.subflows.is_empty() && !st.closing && st.failed.is_none() {
            st.epoch += 1;
            let epoch = st.epoch;
            let window = self.cfg.resume_window;
            let me = self.clone();
            drop(st);
            tokio::spawn(async move {
                tokio::time::sleep(window).await;
                let mut st = me.lock();
                if st.epoch == epoch && st.subflows.is_empty() && !st.closing {
                    let why = format!("no subflow for {window:?}, the resume window ended");
                    me.fail(&mut st, why);
                }
            });
        } else {
            st.wake_subflows();
        }
    }

    /// Encrypt `plain` into a record: length, epoch, nonce, ciphertext.
    fn seal(&self, plain: &[u8]) -> Result<Vec<u8>> {
        self.keys.seal(plain)
    }

    /// Decrypt a record. `None` means the record was dropped without an error: its nonce was
    /// already accepted, it is older than the replay window, or its epoch is not kept. It is
    /// counted.
    fn open(&self, epoch: u16, nonce: u64, ct: &[u8]) -> Result<Option<Vec<u8>>> {
        let opened = self.keys.open(epoch, nonce, ct)?;
        if opened.is_none() {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
        Ok(opened)
    }

    fn start_subflow<T: Transport>(
        self: &Arc<Self>,
        t: T,
        gauge: Option<Gauge>,
    ) -> Result<SubflowId> {
        let (rd, wr) = tokio::io::split(t);
        let notify = Arc::new(Notify::new());
        let sent = Arc::new(AtomicU64::new(0));
        let received = Arc::new(AtomicU64::new(0));
        let rtt_us = Arc::new(AtomicU64::new(NO_RTT));
        let mut st = self.lock();
        if let Some(why) = &st.failed {
            return Err(Error::ConnectionLost(why.clone()));
        }
        let id = st.next_sub_id;
        st.next_sub_id += 1;
        let reader = tokio::spawn(reader_task(
            self.clone(),
            id,
            rd,
            received.clone(),
            rtt_us.clone(),
        ));
        let writer = tokio::spawn(writer_task(
            self.clone(),
            id,
            wr,
            notify.clone(),
            sent.clone(),
        ));
        let aborts = vec![reader.abort_handle(), writer.abort_handle()];
        st.subflows.insert(
            id,
            Sub::new(
                self.cfg.subflow_window,
                gauge,
                SubShared {
                    sent,
                    received,
                    rtt_us,
                    notify,
                    aborts,
                },
            ),
        );
        if let Some(sub) = st.subflows.get_mut(&id) {
            sub.told_keepalive = self.handshake_keepalive_ms;
        }
        // A new subflow cancels any running resume timer.
        st.epoch += 1;
        self.bump();
        Ok(id)
    }
}

/// Read one record: its epoch, nonce and ciphertext.
async fn read_record<R: AsyncRead + Unpin>(rd: &mut R) -> std::io::Result<(u16, u64, Vec<u8>)> {
    let mut head = [0u8; RECORD_HDR];
    rd.read_exact(&mut head).await?;
    let len = u16::from_be_bytes([head[0], head[1]]) as usize;
    let epoch = u16::from_be_bytes([head[2], head[3]]);
    let nonce = u64::from_be_bytes(head[4..].try_into().unwrap());
    let mut ct = vec![0u8; len];
    rd.read_exact(&mut ct).await?;
    Ok((epoch, nonce, ct))
}

async fn reader_task<R: AsyncRead + Unpin>(
    shared: Arc<Shared>,
    id: u64,
    rd: R,
    received: Arc<AtomicU64>,
    rtt_us: Arc<AtomicU64>,
) {
    let mut rd = BufReader::with_capacity(128 * 1024, rd);
    let result: Result<()> = async {
        let mut last = tokio::time::Instant::now();
        loop {
            let read = read_record(&mut rd);
            tokio::pin!(read);
            // The dead time follows the intervals, which can change while a record is awaited.
            let (epoch, nonce, ct) = loop {
                let changed = shared.keepalive_changed.notified();
                tokio::pin!(changed);
                changed.as_mut().enable();
                let deadline = shared.dead_after().map(|limit| last + limit);
                tokio::select! {
                    record = &mut read => break record?,
                    _ = async {
                        match deadline {
                            Some(at) => tokio::time::sleep_until(at).await,
                            None => std::future::pending().await,
                        }
                    } => {
                        let now = tokio::time::Instant::now();
                        if shared.dead_after().is_some_and(|limit| last + limit <= now) {
                            return Err(Error::ConnectionLost(
                                "no record for 3 keepalive intervals".into(),
                            ));
                        }
                    }
                    // A new interval starts the silence over: a side that had keepalives off
                    // has heard nothing for as long as it liked.
                    _ = &mut changed => last = tokio::time::Instant::now(),
                }
            };
            last = tokio::time::Instant::now();
            received.fetch_add((RECORD_HDR + ct.len()) as u64, Ordering::Relaxed);
            let Some(plain) = shared.open(epoch, nonce, &ct)? else {
                continue;
            };
            match frame::decode(plain)? {
                Frame::Ping { stamp } => {
                    if let Some(sub) = shared.lock().subflows.get_mut(&id) {
                        sub.pong = Some(stamp);
                        sub.notify.notify_one();
                    }
                }
                Frame::Keepalive { ms } => {
                    shared.peer_keepalive_ms.store(ms, Ordering::Relaxed);
                    shared.keepalive_changed.notify_waiters();
                }
                Frame::Pong { stamp } => {
                    let rtt = shared.stamp().saturating_sub(stamp);
                    rtt_us.store(rtt, Ordering::Relaxed);
                }
                Frame::Probe => {
                    // The filler is discarded here. Only its size is answered.
                    let bytes = (RECORD_HDR + ct.len()) as u64;
                    if let Some(sub) = shared.lock().subflows.get_mut(&id) {
                        sub.probe_received += bytes;
                        sub.probe_answer = Some(bytes);
                        sub.notify.notify_one();
                    }
                }
                Frame::ProbeAck { bytes } => {
                    if let Some(sub) = shared.lock().subflows.get_mut(&id)
                        && let Some((at, size)) = sub.probe_pending.take()
                        && size as u64 == bytes
                    {
                        sub.probe_acked += bytes;
                        let secs = at.elapsed().as_secs_f64().max(1e-6);
                        sub.probe_rate = Some(bytes as f64 / secs);
                    }
                }
                frame => {
                    let len = ct.len() - TAG;
                    let mut st = shared.lock();
                    if let Err(why) = st.handle(id, len, frame, &shared.accept) {
                        shared.fail(&mut st, why.clone());
                        return Err(Error::ConnectionLost(why));
                    }
                }
            }
        }
    }
    .await;
    let _ = result;
    shared.subflow_died(id);
}

enum Work {
    Keepalive(u64),
    Pong(u64),
    ProbeAck(u64),
    Probe(usize),
    Ack(u64, u64),
    Frame(Arc<Vec<u8>>),
    Idle,
    Done,
    Gone,
}

async fn writer_task<W: AsyncWrite + Unpin>(
    shared: Arc<Shared>,
    id: u64,
    mut wr: W,
    notify: Arc<Notify>,
    sent: Arc<AtomicU64>,
) {
    let mut last_write = tokio::time::Instant::now();
    let result: Result<()> = async {
        loop {
            let work = {
                let mut st = shared.lock();
                let pong = st.subflows.get_mut(&id).and_then(|s| s.pong.take());
                let answer = if pong.is_some() {
                    None
                } else {
                    st.subflows.get_mut(&id).and_then(|s| s.probe_answer.take())
                };
                if let Some(stamp) = pong {
                    Work::Pong(stamp)
                } else if let Some(bytes) = answer {
                    Work::ProbeAck(bytes)
                } else if !st.subflows.contains_key(&id) {
                    // The subflow was declared dead while this task ran. Taking a frame now
                    // would assign it to a subflow that no longer exists and lose it.
                    Work::Gone
                } else if st.failed.is_some() {
                    Work::Done
                } else if let Some(ms) = {
                    let now_ms = shared.keepalive_ms.load(Ordering::Relaxed);
                    st.subflows.get_mut(&id).and_then(|s| {
                        (s.told_keepalive != now_ms).then(|| {
                            s.told_keepalive = now_ms;
                            now_ms
                        })
                    })
                } {
                    Work::Keepalive(ms)
                } else if let Some(owed) = {
                    let next = st.recv_next;
                    st.subflows.get_mut(&id).and_then(|s| {
                        std::mem::take(&mut s.ack_dirty).then_some((next, s.recv_seq))
                    })
                } {
                    Work::Ack(owed.0, owed.1)
                } else if let Some(size) = st.subflows.get_mut(&id).and_then(|s| {
                    let size = s.probe_request.take()?;
                    s.probe_pending = Some((Instant::now(), size));
                    s.probe_sent += size as u64;
                    Some(size)
                }) {
                    Work::Probe(size)
                } else {
                    let now = Instant::now();
                    st.maybe_reinject(now);
                    if let Some(s) = st.subflows.get_mut(&id) {
                        s.read_gauge(now);
                    }
                    let (window, inflight, bulk_cap, bulk_inflight) =
                        st.subflows.get(&id).map_or((0, 0, 0, 0), |s| {
                            (
                                s.window,
                                (s.sent_seq - s.acked_seq) as usize,
                                s.bulk_cap(),
                                s.bulk_inflight,
                            )
                        });
                    // A frame reinjected away from this subflow is skipped, unless it is the
                    // only subflow left.
                    let sole = st.subflows.len() == 1;
                    let retransmit = st
                        .pending
                        .iter()
                        .copied()
                        .find(|q| sole || st.unacked[q].avoid != Some(id));
                    let mut hit_cap = false;
                    let mut fits = |len: usize, bulk: bool| {
                        let in_window = inflight == 0 || inflight + len <= window;
                        let in_cap = !bulk || bulk_inflight == 0 || bulk_inflight + len <= bulk_cap;
                        hit_cap |= in_window && !in_cap;
                        in_window && in_cap
                    };
                    // `Some(frame, bulk)` when this writer takes a frame now.
                    let taken = if let Some(seq) = retransmit {
                        let (len, bulk) = (st.unacked[&seq].bytes.len(), st.unacked[&seq].bulk);
                        if fits(len, bulk) {
                            st.pending.remove(&seq);
                            let entry = st.unacked.get_mut(&seq).expect("pending is unacked");
                            entry.on = Some(id);
                            entry.at = Some(now);
                            Some((entry.bytes.clone(), bulk))
                        } else {
                            None
                        }
                    } else if let Some((which, len, bulk)) = st.peek_fresh(&mut fits) {
                        fits(len, bulk).then(|| (st.take_fresh(which, bulk, id, now), bulk))
                    } else {
                        None
                    };
                    if hit_cap && let Some(s) = st.subflows.get_mut(&id) {
                        s.capped = true;
                    }
                    match taken {
                        Some((bytes, bulk)) => {
                            let len = bytes.len();
                            if let Some(s) = st.subflows.get_mut(&id) {
                                s.sent_seq += len as u64;
                                s.marks
                                    .push_back((s.sent_seq, now, if bulk { len } else { 0 }));
                                if bulk {
                                    s.bulk_inflight += len;
                                }
                            }
                            Work::Frame(bytes)
                        }
                        None if st.closing && st.out_empty() => Work::Done,
                        None => Work::Idle,
                    }
                }
            };
            match work {
                Work::Keepalive(ms) => {
                    let record = shared.seal(&frame::encode_keepalive(ms))?;
                    wr.write_all(&record).await?;
                    wr.flush().await?;
                    last_write = tokio::time::Instant::now();
                }
                Work::Pong(stamp) => {
                    let record = shared.seal(&frame::encode_pong(stamp))?;
                    wr.write_all(&record).await?;
                    wr.flush().await?;
                    // An answer is not activity of this side: each side still pings on its
                    // own schedule, so both get a round trip estimate.
                }
                Work::ProbeAck(bytes) => {
                    let record = shared.seal(&frame::encode_probe_ack(bytes))?;
                    wr.write_all(&record).await?;
                    wr.flush().await?;
                }
                Work::Probe(size) => {
                    let record = shared.seal(&frame::encode_probe(size))?;
                    wr.write_all(&record).await?;
                    wr.flush().await?;
                    sent.fetch_add(record.len() as u64, Ordering::Relaxed);
                    last_write = tokio::time::Instant::now();
                }
                Work::Ack(next, sub) => {
                    let record = shared.seal(&frame::encode_ack(next, sub))?;
                    wr.write_all(&record).await?;
                    last_write = tokio::time::Instant::now();
                }
                Work::Frame(bytes) => {
                    let record = shared.seal(&bytes)?;
                    wr.write_all(&record).await?;
                    sent.fetch_add(record.len() as u64, Ordering::Relaxed);
                    last_write = tokio::time::Instant::now();
                }
                Work::Idle => {
                    wr.flush().await?;
                    let keepalive = shared.keepalive();
                    // With several subflows and frames in flight, wake every tick to look for a
                    // frame that blocks delivery.
                    let ticking = {
                        let st = shared.lock();
                        st.subflows.len() > 1 && !st.unacked.is_empty()
                    };
                    let now = tokio::time::Instant::now();
                    let ka_due = (!keepalive.is_zero()).then(|| last_write + keepalive);
                    let wake = match (ka_due, ticking) {
                        (Some(d), true) => Some(d.min(now + TICK)),
                        (Some(d), false) => Some(d),
                        (None, true) => Some(now + TICK),
                        (None, false) => None,
                    };
                    let timed_out = match wake {
                        None => {
                            notify.notified().await;
                            false
                        }
                        Some(at) => tokio::time::timeout_at(at, notify.notified())
                            .await
                            .is_err(),
                    };
                    if timed_out && ka_due.is_some_and(|d| tokio::time::Instant::now() >= d) {
                        // Idle for a whole interval: keep the path and its NAT mapping alive.
                        let record = shared.seal(&frame::encode_ping(shared.stamp()))?;
                        wr.write_all(&record).await?;
                        wr.flush().await?;
                        last_write = tokio::time::Instant::now();
                    }
                }
                Work::Gone => return Ok(()),
                Work::Done => {
                    wr.flush().await?;
                    wr.shutdown().await?;
                    return Ok(());
                }
            }
        }
    }
    .await;
    if result.is_err() {
        shared.subflow_died(id);
    }
}

/// How a stream's frames are scheduled against those of other streams.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Priority {
    /// Served before any bulk frame. For control frames and small messages.
    #[default]
    Interactive,
    /// Served only while no interactive stream has a frame queued, and limited in flight.
    Bulk,
}

/// An encrypted, multiplexed session between two nodes that survives the loss of its transport.
pub struct Session {
    guard: Arc<Guard>,
}

impl Session {
    /// Connect as the initiator over `transport`, to the node that holds `remote`.
    pub async fn connect<T: Transport>(
        mut transport: T,
        identity: &Identity,
        remote: NodeId,
        config: Config,
    ) -> Result<Session> {
        let gauge = gauge_of(&transport);
        let est = handshake::initiate(&mut transport, identity, remote, &config).await?;
        Self::establish(transport, est, config, true, gauge)
    }

    /// Accept as the responder over `transport`. An initiator the policy does not accept is
    /// refused with `Error::UnknownPeer`.
    pub async fn accept<T: Transport>(
        transport: T,
        identity: &Identity,
        policy: PeerPolicy,
        config: Config,
    ) -> Result<Session> {
        let gauge = gauge_of(&transport);
        Self::accept_gauged(transport, identity, policy, config, gauge).await
    }

    /// `accept` for a transport that the caller wrapped, with the gauge of the transport inside.
    pub(crate) async fn accept_gauged<T: Transport>(
        mut transport: T,
        identity: &Identity,
        policy: PeerPolicy,
        config: Config,
        gauge: Option<Gauge>,
    ) -> Result<Session> {
        let est = handshake::respond(&mut transport, identity, &policy, &config).await?;
        let session = Self::establish(transport, est, config, false, gauge)?;
        session.watch_policy(policy);
        Ok(session)
    }

    fn establish<T: Transport>(
        transport: T,
        est: Established,
        cfg: Config,
        initiator: bool,
        gauge: Option<Gauge>,
    ) -> Result<Session> {
        let own_keepalive_ms = interval_ms(cfg.keepalive_interval);
        let state = State {
            window: cfg.stream_window,
            peer_window: est.peer_window,
            max_unacked: cfg.max_unacked_bytes.min(est.peer_reorder),
            subflow_window: cfg.subflow_window,
            next_seq: 0,
            unacked: BTreeMap::new(),
            unacked_bytes: 0,
            pending: BTreeSet::new(),
            ctl: VecDeque::new(),
            ring_interactive: VecDeque::new(),
            ring_bulk: VecDeque::new(),
            close_wanted: false,
            bulk_wait: 0,
            peer_next: 0,
            recv_next: 0,
            reorder: BTreeMap::new(),
            reorder_bytes: 0,
            max_reorder: (cfg.reorder_kib() as usize) << 10,
            reorder_peak: 0,
            reinjected: 0,
            streams: HashMap::new(),
            accept_queue: VecDeque::new(),
            next_stream_id: if initiator { 1 } else { 2 },
            subflows: HashMap::new(),
            next_sub_id: 0,
            epoch: 0,
            failed: None,
            closing: false,
            remote_closed: false,
            close_reason: None,
        };
        let shared = Arc::new(Shared {
            state: Mutex::new(state),
            accept: Notify::new(),
            keys: Keys::new(
                est.crypto,
                est.rekey_secret,
                initiator,
                cfg.rekey_after_records,
                cfg.rekey_after,
            ),
            dropped: AtomicU64::new(0),
            cfg,
            peer: est.peer,
            version: est.version,
            session_id: est.session_id,
            changes: watch::channel(0).0,
            origin: Instant::now(),
            keepalive_ms: AtomicU64::new(own_keepalive_ms),
            peer_keepalive_ms: AtomicU64::new(interval_ms(est.peer_keepalive)),
            handshake_keepalive_ms: own_keepalive_ms,
            keepalive_changed: Notify::new(),
            hints: watch::channel(0).0,
        });
        shared.start_subflow(transport, gauge)?;
        Ok(Session {
            guard: Arc::new(Guard {
                shared,
                watcher: Mutex::new(None),
            }),
        })
    }

    /// Close the session with the reason `revoked` as soon as `policy` stops accepting the
    /// peer. SPEC.md letify-link, Session item 9.
    fn watch_policy(&self, policy: PeerPolicy) {
        let peer = self.shared().peer;
        let weak = self.downgrade();
        let mut changes = policy.changes();
        let task = tokio::spawn(async move {
            loop {
                if !policy.accepts(&peer) {
                    if let Some(session) = weak.upgrade() {
                        session.begin_close(Some("revoked"));
                    }
                    return;
                }
                if changes.changed().await.is_err() {
                    return;
                }
            }
        });
        *self.guard.watcher.lock().unwrap_or_else(|e| e.into_inner()) = Some(task.abort_handle());
    }

    pub(crate) fn from_guard(guard: Arc<Guard>) -> Session {
        Session { guard }
    }

    fn shared(&self) -> &Arc<Shared> {
        &self.guard.shared
    }

    /// The node id of the other side, authenticated by the handshake.
    pub fn peer(&self) -> NodeId {
        self.shared().peer
    }

    /// The protocol version the two sides agreed on.
    pub fn version(&self) -> u16 {
        self.shared().version
    }

    /// Open a stream. The peer sees it from `accept_stream`.
    pub async fn open_stream(&self) -> Result<Stream> {
        self.open_stream_with(Priority::Interactive).await
    }

    /// Open a stream whose sends have the given class.
    pub async fn open_stream_with(&self, priority: Priority) -> Result<Stream> {
        let shared = self.shared();
        let mut st = shared.lock();
        if let Some(why) = &st.failed {
            return Err(Error::ConnectionLost(why.clone()));
        }
        if st.closing {
            return Err(Error::Closed);
        }
        let id = st.next_stream_id;
        st.next_stream_id += 2;
        let s = StreamState {
            send_credit: st.peer_window as u64,
            bulk: priority == Priority::Bulk,
            ..Default::default()
        };
        st.streams.insert(id, s);
        st.enqueue_stream(id, frame::encode_open(0, id));
        Ok(Stream::new(self.guard.clone(), id))
    }

    /// Wait for a stream that the peer opened.
    pub async fn accept_stream(&self) -> Result<Stream> {
        let shared = self.shared();
        loop {
            let notified = shared.accept.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            {
                let mut st = shared.lock();
                if let Some(id) = st.accept_queue.pop_front() {
                    return Ok(Stream::new(self.guard.clone(), id));
                }
                if let Some(why) = &st.failed {
                    return Err(Error::ConnectionLost(why.clone()));
                }
                if st.remote_closed || st.closing {
                    return Err(Error::Closed);
                }
            }
            notified.await;
        }
    }

    /// Add a subflow. Both sides call this with the two ends of one new transport. The two
    /// sides first name the highest sequence they received, so a session whose last subflow
    /// died resumes on this one and retransmits what the peer lacks. A listener that routes
    /// transports by session id uses [`crate::Acceptor`] instead on its side.
    pub async fn add_subflow<T: Transport>(&self, transport: T) -> Result<SubflowId> {
        self.add_subflow_inner(transport, false).await
    }

    /// Add a subflow whose peer preamble (magic and session id) the caller already read and
    /// checked, as the acceptor does.
    pub(crate) async fn add_subflow_routed<T: Transport>(&self, transport: T) -> Result<SubflowId> {
        self.add_subflow_inner(transport, true).await
    }

    async fn add_subflow_inner<T: Transport>(
        &self,
        mut transport: T,
        preamble_read: bool,
    ) -> Result<SubflowId> {
        let shared = self.shared().clone();
        let gauge = gauge_of(&transport);
        {
            let st = shared.lock();
            if let Some(why) = &st.failed {
                return Err(Error::ConnectionLost(why.clone()));
            }
            if st.closing {
                return Err(Error::Closed);
            }
        }
        let limit = shared.cfg.handshake_timeout;
        let peer_next = tokio::time::timeout(limit, async {
            let next = shared.lock().recv_next;
            let mut out = SUBFLOW_MAGIC.to_vec();
            out.extend_from_slice(&shared.session_id);
            out.extend_from_slice(&shared.seal(&frame::encode_hello(next))?);
            transport.write_all(&out).await?;
            transport.flush().await?;
            if !preamble_read {
                let mut head = [0u8; 20];
                transport.read_exact(&mut head).await?;
                if &head[..4] != SUBFLOW_MAGIC || head[4..] != shared.session_id {
                    return Err(Error::Subflow(
                        "the subflow belongs to another session".into(),
                    ));
                }
            }
            let mut rec = [0u8; RECORD_HDR];
            transport.read_exact(&mut rec).await?;
            let len = u16::from_be_bytes([rec[0], rec[1]]) as usize;
            let epoch = u16::from_be_bytes([rec[2], rec[3]]);
            let nonce = u64::from_be_bytes(rec[4..].try_into().unwrap());
            let mut ct = vec![0u8; len];
            transport.read_exact(&mut ct).await?;
            let plain = shared
                .open(epoch, nonce, &ct)
                .map_err(|e| Error::Subflow(e.to_string()))?
                .ok_or_else(|| Error::Subflow("the hello was replayed".into()))?;
            match frame::decode(plain)? {
                Frame::Hello { next } => Ok(next),
                _ => Err(Error::Subflow("the first record is not a hello".into())),
            }
        })
        .await
        .map_err(|_| Error::Subflow("timed out".into()))??;
        shared.lock().apply_ack(peer_next);
        shared.start_subflow(transport, gauge)
    }

    /// Close the session. Frames already queued are written first; the peer's streams then end
    /// as they would after a clean close.
    pub async fn close(&self) {
        self.close_inner(None).await;
    }

    /// Close the session as [`Session::close`] does, and tell the peer why. The reason is UTF-8
    /// and is cut to 256 bytes. Both sides can read it with [`Session::close_reason`].
    pub async fn close_with_reason(&self, reason: &str) {
        self.close_inner(Some(reason)).await;
    }

    /// Mark the session as closing with `reason`, without waiting for the Close frame to leave.
    pub(crate) fn begin_close(&self, reason: Option<&str>) {
        let mut st = self.shared().lock();
        if !st.closing && st.failed.is_none() {
            st.closing = true;
            st.close_reason = reason.map(|r| frame::cut_reason(r).to_string());
            st.want_close();
        }
    }

    /// Why the session was closed, if the side that closed it gave a reason.
    pub fn close_reason(&self) -> Option<String> {
        self.shared().lock().close_reason.clone()
    }

    async fn close_inner(&self, reason: Option<&str>) {
        let shared = self.shared();
        self.begin_close(reason);
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        loop {
            {
                let st = shared.lock();
                if st.out_empty() || st.subflows.is_empty() || st.failed.is_some() {
                    return;
                }
            }
            if tokio::time::Instant::now() >= deadline {
                return;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    }

    /// The id of this session, shared by both sides.
    pub fn session_id(&self) -> SessionId {
        self.shared().session_id
    }

    /// A handle that does not keep the session open.
    pub fn downgrade(&self) -> WeakSession {
        WeakSession {
            guard: Arc::downgrade(&self.guard),
        }
    }

    /// True once the session was closed or failed. It accepts no subflow then.
    pub fn is_closed(&self) -> bool {
        let st = self.shared().lock();
        st.closing || st.failed.is_some()
    }

    /// The epoch this side seals its records in. It starts at 0 and rises when the sending
    /// keys rotate, see SPEC.md letify-link, Session item 8.
    pub fn send_epoch(&self) -> u16 {
        self.shared().keys.send_epoch()
    }

    /// The newest epoch of the peer's keys that a record opened in.
    pub fn receive_epoch(&self) -> u16 {
        self.shared().keys.receive_epoch()
    }

    /// Change this side's keepalive interval at once. Zero turns this side's keepalives off.
    /// Every subflow tells the peer, which then computes its dead time from the new value, so a
    /// longer interval does not get the subflows declared dead. SPEC.md letify-link, Mobile and
    /// power states item 2.
    pub fn set_keepalive_interval(&self, interval: Duration) {
        let shared = self.shared();
        shared
            .keepalive_ms
            .store(interval_ms(interval), Ordering::Relaxed);
        shared.keepalive_changed.notify_waiters();
        shared.lock().wake_subflows();
    }

    /// This side's keepalive interval now, zero when off.
    pub fn keepalive_interval(&self) -> Duration {
        self.shared().keepalive()
    }

    /// Set the keepalive interval of a power state from the session's [`Config`].
    pub fn set_power_state(&self, state: PowerState) {
        self.set_keepalive_interval(self.shared().cfg.keepalive_for(state));
    }

    /// Drop every subflow and tell the path manager to dial again at once. A client calls it
    /// when the application returns to the foreground, because a session that was suspended
    /// cannot tell which of its subflows still reach the peer. What the peer has not
    /// acknowledged goes out again on the first new subflow. SPEC.md letify-link, Mobile and
    /// power states item 4.
    pub fn resume_now(&self) {
        let shared = self.shared();
        let ids: Vec<SubflowId> = shared.lock().subflows.keys().copied().collect();
        for id in ids {
            shared.subflow_died(id);
        }
        shared.hints.send_modify(|n| *n += 1);
    }

    /// A counter that changes on every [`Session::resume_now`], for a path manager that waits
    /// to redial.
    pub fn resume_hints(&self) -> watch::Receiver<u64> {
        self.shared().hints.subscribe()
    }

    /// A counter that changes whenever a subflow is added or dies.
    pub fn subflow_changes(&self) -> watch::Receiver<u64> {
        self.shared().changes.subscribe()
    }

    /// What was measured on each live subflow, ordered by the subflow's age.
    pub fn subflow_stats(&self) -> Vec<SubflowStats> {
        let st = self.shared().lock();
        let mut ids: Vec<_> = st.subflows.keys().copied().collect();
        ids.sort_unstable();
        ids.iter()
            .map(|id| {
                let sub = &st.subflows[id];
                let rtt = sub.rtt_us.load(Ordering::Relaxed);
                SubflowStats {
                    id: *id,
                    sent_bytes: sub.sent.load(Ordering::Relaxed),
                    received_bytes: sub.received.load(Ordering::Relaxed),
                    acked_bytes: sub.acked,
                    rtt: (rtt != NO_RTT).then(|| Duration::from_micros(rtt)),
                    probe_sent_bytes: sub.probe_sent,
                    probe_received_bytes: sub.probe_received,
                    window: sub.window,
                    srtt: sub.srtt,
                    probe_acked_bytes: sub.probe_acked,
                    probe_rate: sub.probe_rate,
                    bulk_cap: sub.bulk_cap(),
                    delivery_rate: sub.rates.back().copied(),
                    congestion: sub.gauged.map(|(g, _)| g),
                }
            })
            .collect()
    }

    /// Ask the writer of subflow `id` to send a probe whose record is `record_bytes` long (at
    /// least 64, at most 60000; other values are clamped). The peer discards the filler and
    /// answers, and [`SubflowStats::probe_rate`] then holds the measured rate. One probe at a
    /// time is in flight per subflow.
    pub fn send_probe(&self, id: SubflowId, record_bytes: usize) -> Result<()> {
        let mut st = self.shared().lock();
        let sub = st
            .subflows
            .get_mut(&id)
            .ok_or_else(|| Error::Subflow("no such subflow".into()))?;
        if sub.probe_request.is_some() || sub.probe_pending.is_some() {
            return Err(Error::Subflow("a probe is already in flight".into()));
        }
        sub.probe_request = Some(record_bytes.clamp(64, 60_000));
        sub.notify.notify_one();
        Ok(())
    }

    /// Close subflow `id` without ending the session. The frames it carried that the peer has
    /// not acknowledged go out again on another subflow. Does nothing if the subflow is gone.
    pub fn close_subflow(&self, id: SubflowId) {
        self.shared().subflow_died(id);
    }

    /// The number of subflows that are alive now.
    pub fn subflow_count(&self) -> usize {
        self.shared().lock().subflows.len()
    }

    /// Bytes written to each live subflow so far, ordered by the subflow's age.
    pub fn subflow_sent_bytes(&self) -> Vec<u64> {
        let st = self.shared().lock();
        let mut ids: Vec<_> = st.subflows.keys().copied().collect();
        ids.sort_unstable();
        ids.iter()
            .map(|id| st.subflows[id].sent.load(Ordering::Relaxed))
            .collect()
    }

    /// Records dropped because their nonce was already accepted or older than the replay
    /// window ([`crate::REPLAY_WINDOW`]).
    pub fn dropped_records(&self) -> u64 {
        self.shared().dropped.load(Ordering::Relaxed)
    }

    /// The most bytes the reorder buffer held at once: frames that arrived ahead of a gap.
    pub fn reorder_peak_bytes(&self) -> usize {
        self.shared().lock().reorder_peak
    }

    /// Frames sent again on another subflow because they blocked delivery.
    pub fn reinjected_frames(&self) -> u64 {
        self.shared().lock().reinjected
    }

    /// Bytes of frames sent and not yet acknowledged.
    pub fn unacked_bytes(&self) -> usize {
        self.shared().lock().unacked_bytes
    }
}

/// A reference to a session that does not keep it open.
#[derive(Clone)]
pub struct WeakSession {
    guard: std::sync::Weak<Guard>,
}

impl WeakSession {
    /// The session, unless every handle to it was dropped.
    pub fn upgrade(&self) -> Option<Session> {
        self.guard.upgrade().map(Session::from_guard)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sub(gauge: Option<Gauge>) -> Sub {
        Sub::new(
            2 << 20,
            gauge,
            SubShared {
                sent: Arc::new(AtomicU64::new(0)),
                received: Arc::new(AtomicU64::new(0)),
                rtt_us: Arc::new(AtomicU64::new(NO_RTT)),
                notify: Arc::new(Notify::new()),
                aborts: Vec::new(),
            },
        )
    }

    /// A subflow that carried 10 MB/s with a lowest round trip of 50 ms.
    fn busy(gauge: Option<Gauge>) -> Sub {
        let mut s = sub(gauge);
        s.rates.push_back(10e6);
        s.min_rtt = Some((Duration::from_millis(50), Instant::now()));
        s
    }

    fn reading(window: usize, lost: u64) -> Gauge {
        Arc::new(move || CongestionStats {
            window,
            rtt: Duration::from_millis(50),
            min_rtt: Duration::from_millis(50),
            lost_packets: lost,
            sent_packets: 0,
        })
    }

    #[test]
    fn a_subflow_without_loss_keeps_the_probed_cap_whatever_its_congestion_window() {
        let mut s = busy(Some(reading(1900 << 10, 0)));
        let probed = busy(None).bulk_cap();
        s.read_gauge(Instant::now());
        assert_eq!(s.bulk_cap(), probed);
        assert!(s.bulk_cap() < 1900 << 10);
    }

    #[test]
    fn a_subflow_recovering_from_loss_sizes_its_bulk_cap_from_the_window_and_one_stall() {
        let lost = Arc::new(AtomicU64::new(0));
        let gauge: Gauge = {
            let lost = lost.clone();
            Arc::new(move || CongestionStats {
                window: 600 << 10,
                rtt: Duration::from_millis(50),
                min_rtt: Duration::from_millis(50),
                lost_packets: lost.load(Ordering::Relaxed),
                sent_packets: 0,
            })
        };
        let mut s = busy(Some(gauge));
        let t0 = Instant::now();
        s.read_gauge(t0);
        lost.store(3, Ordering::Relaxed);
        s.read_gauge(t0 + GAUGE_EVERY);
        // The window and 10 MB/s x 50 ms of rate: 600 KiB + 500 kB.
        let want = (600 << 10) + 500_000;
        assert_eq!(s.bulk_cap(), want);
        assert!(s.bulk_cap() > busy(None).bulk_cap());
    }

    #[test]
    fn the_cap_of_a_subflow_recovering_from_loss_holds_at_most_150_ms_at_the_delivery_rate() {
        let mut s = busy(Some(reading(8 << 20, 0)));
        let now = Instant::now();
        s.gauged = Some((
            CongestionStats {
                window: 8 << 20,
                rtt: Duration::from_millis(50),
                min_rtt: Duration::from_millis(50),
                lost_packets: 9,
                sent_packets: 1000,
            },
            now,
        ));
        s.loss_at = Some(now);
        // 10 MB/s for 150 ms.
        assert_eq!(s.bulk_cap(), 1_500_000);
    }

    #[test]
    fn a_path_that_loses_more_than_a_tenth_of_its_packets_keeps_the_probed_cap() {
        let mut s = busy(Some(reading(600 << 10, 0)));
        let now = Instant::now();
        s.gauged = Some((
            CongestionStats {
                window: 600 << 10,
                rtt: Duration::from_millis(50),
                min_rtt: Duration::from_millis(50),
                lost_packets: 9,
                sent_packets: 1000,
            },
            now,
        ));
        s.loss_at = Some(now);
        s.loss_ratio = 0.25;
        assert_eq!(s.bulk_cap(), busy(None).bulk_cap());
    }

    #[test]
    fn the_loss_ratio_is_the_share_of_packets_lost_in_the_last_interval() {
        let counts = Arc::new((AtomicU64::new(0), AtomicU64::new(0)));
        let gauge: Gauge = {
            let counts = counts.clone();
            Arc::new(move || CongestionStats {
                window: 1 << 20,
                rtt: Duration::from_millis(50),
                min_rtt: Duration::from_millis(50),
                lost_packets: counts.0.load(Ordering::Relaxed),
                sent_packets: counts.1.load(Ordering::Relaxed),
            })
        };
        let mut s = sub(Some(gauge));
        let t0 = Instant::now();
        s.read_gauge(t0);
        counts.0.store(30, Ordering::Relaxed);
        counts.1.store(1000, Ordering::Relaxed);
        s.read_gauge(t0 + LOSS_SPAN);
        assert!((s.loss_ratio - 0.03).abs() < 1e-9, "{}", s.loss_ratio);
    }

    #[test]
    fn the_cap_of_a_subflow_recovering_from_loss_stays_inside_the_subflow_window() {
        let mut s = busy(Some(reading(8 << 20, 0)));
        // 20 MB/s for 150 ms is 3 MB, more than the 2 MiB window.
        s.rates.push_back(20e6);
        s.gauged = Some((
            CongestionStats {
                window: 8 << 20,
                rtt: Duration::from_millis(50),
                min_rtt: Duration::from_millis(50),
                lost_packets: 9,
                sent_packets: 0,
            },
            Instant::now(),
        ));
        s.loss_at = Some(Instant::now());
        assert_eq!(s.bulk_cap(), 2 << 20);
    }

    #[test]
    fn a_loss_forgotten_after_a_second_returns_the_cap_to_the_probed_one() {
        let mut s = busy(Some(reading(600 << 10, 0)));
        let t0 = Instant::now();
        s.gauged = Some((
            CongestionStats {
                window: 600 << 10,
                rtt: Duration::from_millis(50),
                min_rtt: Duration::from_millis(50),
                lost_packets: 3,
                sent_packets: 0,
            },
            t0 + Duration::from_secs(2),
        ));
        s.loss_at = Some(t0);
        assert_eq!(s.bulk_cap(), busy(None).bulk_cap());
    }

    #[test]
    fn the_congestion_window_is_read_at_most_every_ten_milliseconds() {
        let reads = Arc::new(AtomicU64::new(0));
        let gauge: Gauge = {
            let reads = reads.clone();
            Arc::new(move || {
                reads.fetch_add(1, Ordering::Relaxed);
                CongestionStats {
                    window: 1 << 20,
                    rtt: Duration::from_millis(50),
                    min_rtt: Duration::from_millis(50),
                    lost_packets: 0,
                    sent_packets: 0,
                }
            })
        };
        let mut s = sub(Some(gauge));
        let t0 = Instant::now();
        s.read_gauge(t0);
        s.read_gauge(t0 + Duration::from_millis(5));
        assert_eq!(reads.load(Ordering::Relaxed), 1);
        s.read_gauge(t0 + GAUGE_EVERY);
        assert_eq!(reads.load(Ordering::Relaxed), 2);
    }
}
