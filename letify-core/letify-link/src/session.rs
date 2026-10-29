//! The session: sequencing, acknowledgement, retransmission, subflow scheduling and
//! resumption. It owns the shared state and the tasks that move records over subflows. It does
//! not own stream reads and writes (see `stream`) or path discovery.
//!
//! Scheduling: pull based. Every subflow has a writer task that takes the next pending frame
//! when its previous write completed and its in-flight bytes are below `subflow_window`. A
//! faster subflow completes writes sooner and so takes proportionally more frames, without a
//! rate estimate. The receiver reorders by sequence number.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::Waker;
use std::time::Duration;

use snow::StatelessTransportState;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::sync::Notify;
use tokio::task::AbortHandle;

use crate::config::Config;
use crate::error::{Error, Result};
use crate::frame::{self, Frame, HDR, Kind, RECORD_HDR, TAG};
use crate::handshake::{self, Established};
use crate::identity::{Identity, NodeId, PeerPolicy};
use crate::stream::Stream;

/// A reliable, ordered byte transport that a session can use as a subflow: a TCP connection,
/// a pipe, or anything else that implements `AsyncRead` and `AsyncWrite`.
pub trait Transport: AsyncRead + AsyncWrite + Send + Unpin + 'static {}
impl<T: AsyncRead + AsyncWrite + Send + Unpin + 'static> Transport for T {}

const SUBFLOW_MAGIC: &[u8; 4] = b"LLNS";

pub(crate) struct Sent {
    bytes: Arc<Vec<u8>>,
    on: Option<u64>,
}

pub(crate) struct Sub {
    inflight: usize,
    sent: Arc<AtomicU64>,
    notify: Arc<Notify>,
    aborts: Vec<AbortHandle>,
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
    peer_next: u64,
    recv_next: u64,
    reorder: BTreeMap<u64, Frame>,
    ack_dirty: bool,
    pub streams: HashMap<u32, StreamState>,
    accept_queue: VecDeque<u32>,
    next_stream_id: u32,
    subflows: HashMap<u64, Sub>,
    next_sub_id: u64,
    epoch: u64,
    pub failed: Option<String>,
    pub closing: bool,
    pub remote_closed: bool,
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
        self.unacked_bytes == 0 || self.unacked_bytes + HDR + len <= self.max_unacked
    }

    /// Queue a sequenced frame built by `build(seq)`.
    pub fn enqueue(&mut self, build: impl FnOnce(u64) -> Vec<u8>) {
        let seq = self.next_seq;
        self.next_seq += 1;
        let bytes = build(seq);
        self.unacked_bytes += bytes.len();
        self.unacked.insert(
            seq,
            Sent {
                bytes: Arc::new(bytes),
                on: None,
            },
        );
        self.pending.insert(seq);
        self.wake_subflows();
    }

    pub fn enqueue_fin(&mut self, stream: u32) {
        self.enqueue(|seq| frame::encode_fin(seq, stream));
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
            self.enqueue(|seq| frame::encode_credit(seq, stream, grant));
        }
    }

    pub fn forget_if_done(&mut self, stream: u32) {
        if let Some(s) = self.streams.get(&stream)
            && s.dropped
            && s.remote_fin
            && s.fin_sent
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
                sub.inflight = sub.inflight.saturating_sub(sent.bytes.len());
            }
        }
        self.pending = self.pending.split_off(&next);
        self.wake_writers();
        self.wake_subflows();
    }

    fn handle(&mut self, frame: Frame, accept: &Notify) {
        match frame {
            Frame::Ack { next } | Frame::Hello { next } => self.apply_ack(next),
            Frame::Seq { seq, .. } if seq < self.recv_next => {
                self.ack_dirty = true;
                self.wake_subflows();
            }
            f @ Frame::Seq { .. } => {
                if let Frame::Seq { seq, .. } = &f {
                    self.reorder.entry(*seq).or_insert(f);
                }
                while let Some(f) = self.reorder.remove(&self.recv_next) {
                    self.recv_next += 1;
                    self.deliver(f, accept);
                }
                self.ack_dirty = true;
                self.wake_subflows();
            }
        }
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
            Kind::Close => {
                self.remote_closed = true;
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
    crypto: Arc<StatelessTransportState>,
    nonce: AtomicU64,
    cfg: Config,
    peer: NodeId,
    version: u16,
    session_id: [u8; 16],
}

/// Closes the session when the last handle goes away.
pub(crate) struct Guard {
    pub shared: Arc<Shared>,
}

impl Drop for Guard {
    fn drop(&mut self) {
        let mut st = self.shared.lock();
        if !st.closing && st.failed.is_none() {
            st.closing = true;
            st.enqueue(frame::encode_close);
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
        st.wake_all_streams();
        self.accept.notify_waiters();
    }

    fn subflow_died(self: &Arc<Self>, id: u64) {
        let mut st = self.lock();
        let Some(sub) = st.subflows.remove(&id) else {
            return;
        };
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

    fn next_nonce(&self) -> u64 {
        self.nonce.fetch_add(1, Ordering::Relaxed)
    }

    /// Encrypt `plain` into a record: length, nonce, ciphertext.
    fn seal(&self, plain: &[u8]) -> Result<Vec<u8>> {
        let nonce = self.next_nonce();
        let mut out = vec![0u8; RECORD_HDR + plain.len() + TAG];
        let n = self
            .crypto
            .write_message(nonce, plain, &mut out[RECORD_HDR..])
            .map_err(|e| Error::ConnectionLost(format!("encrypt: {e}")))?;
        out.truncate(RECORD_HDR + n);
        out[..2].copy_from_slice(&(n as u16).to_be_bytes());
        out[2..RECORD_HDR].copy_from_slice(&nonce.to_be_bytes());
        Ok(out)
    }

    fn open(&self, nonce: u64, ct: &[u8]) -> Result<Vec<u8>> {
        let mut out = vec![0u8; ct.len()];
        let n = self
            .crypto
            .read_message(nonce, ct, &mut out)
            .map_err(|e| Error::ConnectionLost(format!("decrypt: {e}")))?;
        out.truncate(n);
        Ok(out)
    }

    fn start_subflow<T: Transport>(self: &Arc<Self>, t: T) -> Result<()> {
        let (rd, wr) = tokio::io::split(t);
        let notify = Arc::new(Notify::new());
        let sent = Arc::new(AtomicU64::new(0));
        let mut st = self.lock();
        if let Some(why) = &st.failed {
            return Err(Error::ConnectionLost(why.clone()));
        }
        let id = st.next_sub_id;
        st.next_sub_id += 1;
        let reader = tokio::spawn(reader_task(self.clone(), id, rd));
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
            Sub {
                inflight: 0,
                sent,
                notify,
                aborts,
            },
        );
        // A new subflow cancels any running resume timer.
        st.epoch += 1;
        Ok(())
    }
}

async fn reader_task<R: AsyncRead + Unpin>(shared: Arc<Shared>, id: u64, rd: R) {
    let mut rd = BufReader::with_capacity(128 * 1024, rd);
    let result: Result<()> = async {
        loop {
            let mut head = [0u8; RECORD_HDR];
            rd.read_exact(&mut head).await?;
            let len = u16::from_be_bytes([head[0], head[1]]) as usize;
            let nonce = u64::from_be_bytes(head[2..].try_into().unwrap());
            let mut ct = vec![0u8; len];
            rd.read_exact(&mut ct).await?;
            let frame = frame::decode(shared.open(nonce, &ct)?)?;
            shared.lock().handle(frame, &shared.accept);
        }
    }
    .await;
    let _ = result;
    shared.subflow_died(id);
}

enum Work {
    Ack(u64),
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
    let result: Result<()> = async {
        loop {
            let work = {
                let mut st = shared.lock();
                if !st.subflows.contains_key(&id) {
                    // The subflow was declared dead while this task ran. Taking a frame now
                    // would assign it to a subflow that no longer exists and lose it.
                    Work::Gone
                } else if st.failed.is_some() {
                    Work::Done
                } else if st.ack_dirty {
                    st.ack_dirty = false;
                    Work::Ack(st.recv_next)
                } else {
                    let window = st.subflow_window;
                    let inflight = st.subflows.get(&id).map_or(0, |s| s.inflight);
                    let next = st.pending.iter().next().copied();
                    match next {
                        Some(seq) => {
                            let len = st.unacked[&seq].bytes.len();
                            if inflight == 0 || inflight + len <= window {
                                st.pending.remove(&seq);
                                let entry = st.unacked.get_mut(&seq).expect("pending is unacked");
                                entry.on = Some(id);
                                let bytes = entry.bytes.clone();
                                if let Some(s) = st.subflows.get_mut(&id) {
                                    s.inflight += len;
                                }
                                Work::Frame(bytes)
                            } else {
                                Work::Idle
                            }
                        }
                        None if st.closing => Work::Done,
                        None => Work::Idle,
                    }
                }
            };
            match work {
                Work::Ack(next) => {
                    let record = shared.seal(&frame::encode_ack(next))?;
                    wr.write_all(&record).await?;
                }
                Work::Frame(bytes) => {
                    let record = shared.seal(&bytes)?;
                    wr.write_all(&record).await?;
                    sent.fetch_add(record.len() as u64, Ordering::Relaxed);
                }
                Work::Idle => {
                    wr.flush().await?;
                    notify.notified().await;
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
        let est = handshake::initiate(&mut transport, identity, remote, &config).await?;
        Self::establish(transport, est, config, true)
    }

    /// Accept as the responder over `transport`. An initiator the policy does not accept is
    /// refused with `Error::UnknownPeer`.
    pub async fn accept<T: Transport>(
        mut transport: T,
        identity: &Identity,
        policy: PeerPolicy,
        config: Config,
    ) -> Result<Session> {
        let est = handshake::respond(&mut transport, identity, &policy, &config).await?;
        Self::establish(transport, est, config, false)
    }

    fn establish<T: Transport>(
        transport: T,
        est: Established,
        cfg: Config,
        initiator: bool,
    ) -> Result<Session> {
        let state = State {
            window: cfg.stream_window,
            peer_window: est.peer_window,
            max_unacked: cfg.max_unacked_bytes,
            subflow_window: cfg.subflow_window,
            next_seq: 0,
            unacked: BTreeMap::new(),
            unacked_bytes: 0,
            pending: BTreeSet::new(),
            peer_next: 0,
            recv_next: 0,
            reorder: BTreeMap::new(),
            ack_dirty: false,
            streams: HashMap::new(),
            accept_queue: VecDeque::new(),
            next_stream_id: if initiator { 1 } else { 2 },
            subflows: HashMap::new(),
            next_sub_id: 0,
            epoch: 0,
            failed: None,
            closing: false,
            remote_closed: false,
        };
        let shared = Arc::new(Shared {
            state: Mutex::new(state),
            accept: Notify::new(),
            crypto: est.crypto,
            nonce: AtomicU64::new(0),
            cfg,
            peer: est.peer,
            version: est.version,
            session_id: est.session_id,
        });
        shared.start_subflow(transport)?;
        Ok(Session {
            guard: Arc::new(Guard { shared }),
        })
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
            ..Default::default()
        };
        st.streams.insert(id, s);
        st.enqueue(|seq| frame::encode_open(seq, id));
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
    /// died resumes on this one and retransmits what the peer lacks.
    pub async fn add_subflow<T: Transport>(&self, mut transport: T) -> Result<()> {
        let shared = self.shared().clone();
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
            let mut head = [0u8; 20];
            transport.read_exact(&mut head).await?;
            if &head[..4] != SUBFLOW_MAGIC || head[4..] != shared.session_id {
                return Err(Error::Subflow(
                    "the subflow belongs to another session".into(),
                ));
            }
            let mut rec = [0u8; RECORD_HDR];
            transport.read_exact(&mut rec).await?;
            let len = u16::from_be_bytes([rec[0], rec[1]]) as usize;
            let nonce = u64::from_be_bytes(rec[2..].try_into().unwrap());
            let mut ct = vec![0u8; len];
            transport.read_exact(&mut ct).await?;
            match frame::decode(
                shared
                    .open(nonce, &ct)
                    .map_err(|e| Error::Subflow(e.to_string()))?,
            )? {
                Frame::Hello { next } => Ok(next),
                _ => Err(Error::Subflow("the first record is not a hello".into())),
            }
        })
        .await
        .map_err(|_| Error::Subflow("timed out".into()))??;
        shared.lock().apply_ack(peer_next);
        shared.start_subflow(transport)
    }

    /// Close the session. Frames already queued are written first; the peer's streams then end
    /// as they would after a clean close.
    pub async fn close(&self) {
        let shared = self.shared();
        {
            let mut st = shared.lock();
            if !st.closing && st.failed.is_none() {
                st.closing = true;
                st.enqueue(frame::encode_close);
            }
        }
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        loop {
            {
                let st = shared.lock();
                if st.pending.is_empty() || st.subflows.is_empty() || st.failed.is_some() {
                    return;
                }
            }
            if tokio::time::Instant::now() >= deadline {
                return;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
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

    /// Bytes of frames sent and not yet acknowledged.
    pub fn unacked_bytes(&self) -> usize {
        self.shared().lock().unacked_bytes
    }
}
