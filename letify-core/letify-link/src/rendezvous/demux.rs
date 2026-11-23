//! The rendezvous demultiplexer: one [`Rendezvous`] shared by many concurrent punches. It owns
//! the pump that reads the shared Rendezvous, the per-peer queues and their bounds. It does not
//! decode messages, authenticate senders or run punches. SPEC.md "Rendezvous demultiplexer".
//!
//! ```no_run
//! use std::sync::Arc;
//! use letify_link::driver::{Endpoint, PunchOptions, punch_session};
//! use letify_link::rendezvous::{Demux, MemoryHub};
//! use letify_link::{Acceptor, Config, Identity, PeerPolicy};
//!
//! # async fn run() -> std::io::Result<()> {
//! let identity = Identity::generate();
//! let hub = MemoryHub::new();
//! let acceptor = Arc::new(Acceptor::new(
//!     identity.clone(),
//!     PeerPolicy::from_fn(|_| true),
//!     Config::default(),
//! ));
//! let demux = Demux::new(hub.endpoint(identity.node_id()));
//! loop {
//!     let (peer, rdv) = demux.accept().await?;
//!     let acceptor = acceptor.clone();
//!     tokio::spawn(async move {
//!         let opts = PunchOptions::default();
//!         let _ = punch_session(&rdv, peer, Endpoint::Responder { acceptor }, &opts).await;
//!     });
//! }
//! # }
//! ```

use std::collections::{HashMap, VecDeque};
use std::io;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use tokio::sync::Notify;
use tokio::task::JoinHandle;
use tokio::time::Instant;

use super::Rendezvous;
use crate::identity::NodeId;

/// The most messages queued for one peer. At the limit the oldest message is dropped.
pub const PEER_QUEUE: usize = 64;
/// The most peers waiting in [`Demux::accept`] at once.
pub const MAX_PENDING_PEERS: usize = 64;
/// How long an unclaimed peer waits, the time to live of the signalling service.
pub const PENDING_TTL: Duration = Duration::from_secs(60);
/// The first wait before the pump calls `recv` again after a transient error.
pub const PUMP_RETRY_FIRST: Duration = Duration::from_secs(1);
/// The longest wait between two retries of the pump.
pub const PUMP_RETRY_MAX: Duration = Duration::from_secs(30);
/// The pump ends after this many errors in a row.
pub const PUMP_MAX_RETRIES: u32 = 8;

#[derive(Default)]
struct Entry {
    items: VecDeque<Vec<u8>>,
    dropped: u64,
    handles: usize,
    notify: Arc<Notify>,
}

impl Entry {
    fn push(&mut self, msg: Vec<u8>) -> bool {
        let evicted = self.items.len() >= PEER_QUEUE;
        if evicted {
            self.items.pop_front();
            self.dropped += 1;
        }
        self.items.push_back(msg);
        evicted
    }
}

#[derive(Default)]
struct State {
    live: HashMap<NodeId, Entry>,
    pending: VecDeque<(NodeId, Entry, Instant)>,
    end: Option<(io::ErrorKind, String)>,
    dropped: u64,
    refused: u64,
}

impl State {
    fn expire(&mut self) {
        let now = Instant::now();
        while let Some((_, _, first)) = self.pending.front() {
            if now.duration_since(*first) < PENDING_TTL {
                break;
            }
            self.pending.pop_front();
        }
    }

    fn end_error(&self) -> Option<io::Error> {
        self.end
            .as_ref()
            .map(|(kind, text)| io::Error::new(*kind, text.clone()))
    }
}

struct Shared {
    state: Mutex<State>,
    accept_notify: Notify,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// End the pump: queued messages stay readable, then every recv fails with `kind`.
    fn finish(&self, kind: io::ErrorKind, text: String) {
        let mut st = self.lock();
        if st.end.is_none() {
            st.end = Some((kind, text));
        }
        for entry in st.live.values() {
            entry.notify.notify_waiters();
        }
        drop(st);
        self.accept_notify.notify_waiters();
    }

    fn route(&self, from: NodeId, msg: Vec<u8>) {
        let mut st = self.lock();
        st.expire();
        if let Some(entry) = st.live.get_mut(&from) {
            let evicted = entry.push(msg);
            let notify = entry.notify.clone();
            st.dropped += u64::from(evicted);
            drop(st);
            notify.notify_waiters();
        } else if let Some((_, entry, _)) = st.pending.iter_mut().find(|(n, _, _)| *n == from) {
            let evicted = entry.push(msg);
            st.dropped += u64::from(evicted);
        } else if st.pending.len() < MAX_PENDING_PEERS {
            let mut entry = Entry::default();
            entry.push(msg);
            st.pending.push_back((from, entry, Instant::now()));
            drop(st);
            self.accept_notify.notify_waiters();
        } else {
            st.refused += 1;
        }
    }
}

fn is_closed(kind: io::ErrorKind) -> bool {
    matches!(
        kind,
        io::ErrorKind::BrokenPipe | io::ErrorKind::NotConnected | io::ErrorKind::UnexpectedEof
    )
}

async fn pump<R: Rendezvous>(rdv: Arc<R>, shared: Arc<Shared>) {
    let mut wait = PUMP_RETRY_FIRST;
    let mut failures = 0u32;
    loop {
        match rdv.recv().await {
            Ok((from, msg)) => {
                failures = 0;
                wait = PUMP_RETRY_FIRST;
                shared.route(from, msg);
            }
            Err(e) => {
                failures += 1;
                if is_closed(e.kind()) || failures >= PUMP_MAX_RETRIES {
                    shared.finish(e.kind(), e.to_string());
                    return;
                }
                tokio::time::sleep(wait).await;
                wait = (wait * 2).min(PUMP_RETRY_MAX);
            }
        }
    }
}

/// Shares one [`Rendezvous`] between many concurrent punches by routing each message to the
/// queue of its sender. Dropping it stops the pump.
pub struct Demux<R: Rendezvous + 'static> {
    rdv: Arc<R>,
    shared: Arc<Shared>,
    pump: JoinHandle<()>,
}

impl<R: Rendezvous + 'static> Demux<R> {
    /// Take `rdv` and start the pump. Needs a running tokio runtime.
    pub fn new(rdv: R) -> Self {
        let rdv = Arc::new(rdv);
        let shared = Arc::new(Shared {
            state: Mutex::new(State::default()),
            accept_notify: Notify::new(),
        });
        let pump = tokio::spawn(pump(rdv.clone(), shared.clone()));
        Demux { rdv, shared, pump }
    }

    /// The handle of `node`. A pending peer of that node is claimed with its messages, and a
    /// node that already has a handle gets another on the same queue.
    pub fn peer(&self, node: NodeId) -> PeerRendezvous<R> {
        let mut st = self.shared.lock();
        st.expire();
        if !st.live.contains_key(&node) {
            let entry = match st.pending.iter().position(|(n, _, _)| *n == node) {
                Some(i) => st.pending.remove(i).map(|(_, e, _)| e).unwrap_or_default(),
                None => Entry::default(),
            };
            st.live.insert(node, entry);
        }
        self.handle(&mut st, node)
    }

    /// Wait for a sender that has no handle and return its handle, with the messages already
    /// queued for it. Fails with the pump's error once the pump has ended and no peer waits.
    pub async fn accept(&self) -> io::Result<(NodeId, PeerRendezvous<R>)> {
        loop {
            let notified = self.shared.accept_notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            {
                let mut st = self.shared.lock();
                st.expire();
                if let Some((node, entry, _)) = st.pending.pop_front() {
                    st.live.insert(node, entry);
                    let handle = self.handle(&mut st, node);
                    return Ok((node, handle));
                }
                if let Some(e) = st.end_error() {
                    return Err(e);
                }
            }
            notified.await;
        }
    }

    fn handle(&self, st: &mut State, node: NodeId) -> PeerRendezvous<R> {
        let entry = st.live.get_mut(&node).expect("entry was just inserted");
        entry.handles += 1;
        PeerRendezvous {
            node,
            rdv: self.rdv.clone(),
            shared: self.shared.clone(),
            notify: entry.notify.clone(),
        }
    }

    /// Messages dropped so far because a queue was full, over all peers.
    pub fn dropped(&self) -> u64 {
        self.shared.lock().dropped
    }

    /// Messages discarded so far because [`MAX_PENDING_PEERS`] peers were already waiting.
    pub fn refused(&self) -> u64 {
        self.shared.lock().refused
    }
}

impl<R: Rendezvous + 'static> Drop for Demux<R> {
    fn drop(&mut self) {
        self.pump.abort();
        self.shared
            .finish(io::ErrorKind::BrokenPipe, "the demux was dropped".into());
    }
}

/// The view of a [`Demux`] for one peer: `recv` yields only that peer's messages and `send`
/// passes through. It is a [`Rendezvous`], so it is the `rdv` of `driver::punch`.
pub struct PeerRendezvous<R: Rendezvous + 'static> {
    node: NodeId,
    rdv: Arc<R>,
    shared: Arc<Shared>,
    notify: Arc<Notify>,
}

impl<R: Rendezvous + 'static> PeerRendezvous<R> {
    /// The node whose messages this handle yields.
    pub fn node(&self) -> NodeId {
        self.node
    }

    /// Messages of this peer dropped so far because its queue was full.
    pub fn dropped(&self) -> u64 {
        self.shared
            .lock()
            .live
            .get(&self.node)
            .map_or(0, |e| e.dropped)
    }
}

impl<R: Rendezvous + 'static> Drop for PeerRendezvous<R> {
    fn drop(&mut self) {
        let mut st = self.shared.lock();
        if let Some(entry) = st.live.get_mut(&self.node) {
            entry.handles -= 1;
            if entry.handles == 0 {
                st.live.remove(&self.node);
            }
        }
    }
}

impl<R: Rendezvous + 'static> Rendezvous for PeerRendezvous<R> {
    async fn send(&self, to: NodeId, msg: Vec<u8>) -> io::Result<()> {
        self.rdv.send(to, msg).await
    }

    async fn recv(&self) -> io::Result<(NodeId, Vec<u8>)> {
        loop {
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            {
                let mut st = self.shared.lock();
                let end = st.end_error();
                if let Some(entry) = st.live.get_mut(&self.node)
                    && let Some(msg) = entry.items.pop_front()
                {
                    return Ok((self.node, msg));
                }
                if let Some(e) = end {
                    return Err(e);
                }
            }
            notified.await;
        }
    }
}

impl<R: Rendezvous + 'static> std::fmt::Debug for PeerRendezvous<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PeerRendezvous")
            .field("node", &self.node)
            .finish()
    }
}
