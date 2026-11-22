//! Pins SPEC.md "Rendezvous demultiplexer": concurrent punches over one Rendezvous, per-peer
//! queues, bounds, pending peers, pump errors and drop. Public API only.

use std::collections::VecDeque;
use std::io;
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

use letify_link::driver::{Endpoint, PunchOptions, punch_session};
use letify_link::rendezvous::{
    Demux, MAX_PENDING_PEERS, MemoryHub, PEER_QUEUE, PENDING_TTL, Rendezvous,
};
use letify_link::{Acceptor, Config, Identity, NodeId, PeerPolicy};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

fn quick() -> PunchOptions {
    PunchOptions {
        tcp_sockets: 2,
        udp_sockets: 0,
        lead: Duration::from_millis(300),
        window: Duration::from_secs(8),
        ..PunchOptions::default()
    }
}

fn node() -> (Identity, NodeId) {
    let id = Identity::generate();
    let n = id.node_id();
    (id, n)
}

async fn within<T>(f: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(20), f)
        .await
        .expect("timed out")
}

#[tokio::test]
async fn concurrent_punches_behind_one_demux_all_succeed_and_exchange_bytes() {
    let (server, server_id) = node();
    let hub = MemoryHub::new();
    let demux = Demux::new(hub.endpoint(server_id));
    let acceptor = Arc::new(Acceptor::new(
        server,
        PeerPolicy::from_fn(|_| true),
        Config::default(),
    ));

    let n = 3usize;
    let opts = quick();
    let responder = {
        let opts = opts.clone();
        tokio::spawn(async move {
            let mut tasks = Vec::new();
            for _ in 0..n {
                let (peer, rdv) = demux.accept().await.unwrap();
                let (acceptor, opts) = (acceptor.clone(), opts.clone());
                tasks.push(tokio::spawn(async move {
                    let got = punch_session(&rdv, peer, Endpoint::Responder { acceptor }, &opts)
                        .await
                        .expect("responder punch");
                    let mut stream = got.session.accept_stream().await.unwrap();
                    let mut buf = [0u8; 1];
                    stream.read_exact(&mut buf).await.unwrap();
                    stream.write_all(&[buf[0] + 100]).await.unwrap();
                    stream.flush().await.unwrap();
                    (peer, buf[0], got)
                }));
            }
            let mut out = Vec::new();
            for t in tasks {
                out.push(t.await.unwrap());
            }
            out
        })
    };

    let mut initiators = Vec::new();
    let mut ids = Vec::new();
    for i in 0..n {
        let (id, nid) = node();
        ids.push(nid);
        let rdv = hub.endpoint(nid);
        let opts = opts.clone();
        initiators.push(tokio::spawn(async move {
            let ep = Endpoint::Initiator {
                identity: &id,
                config: Config::default(),
            };
            let got = punch_session(&rdv, server_id, ep, &opts)
                .await
                .expect("initiator punch");
            let mut stream = got.session.open_stream().await.unwrap();
            stream.write_all(&[i as u8]).await.unwrap();
            stream.flush().await.unwrap();
            let mut back = [0u8; 1];
            stream.read_exact(&mut back).await.unwrap();
            assert_eq!(back[0], i as u8 + 100);
            got
        }));
    }
    let mut keep = Vec::new();
    for t in initiators {
        keep.push(within(t).await.unwrap());
    }
    let served = within(responder).await.unwrap();
    assert_eq!(served.len(), n);
    for (peer, byte, _) in &served {
        let i = ids.iter().position(|x| x == peer).unwrap();
        assert_eq!(*byte as usize, i);
    }
}

#[tokio::test(start_paused = true)]
async fn a_handle_receives_only_its_own_peer_and_send_reaches_any_node() {
    let hub = MemoryHub::new();
    let (_, s) = node();
    let (_, a) = node();
    let (_, b) = node();
    let (ra, rb) = (hub.endpoint(a), hub.endpoint(b));
    let demux = Demux::new(hub.endpoint(s));
    let (ha, hb) = (demux.peer(a), demux.peer(b));

    ra.send(s, b"from a 1".to_vec()).await.unwrap();
    rb.send(s, b"from b 1".to_vec()).await.unwrap();
    ra.send(s, b"from a 2".to_vec()).await.unwrap();

    assert_eq!(hb.recv().await.unwrap(), (b, b"from b 1".to_vec()));
    assert_eq!(ha.recv().await.unwrap(), (a, b"from a 1".to_vec()));
    assert_eq!(ha.recv().await.unwrap(), (a, b"from a 2".to_vec()));

    // A handle sends to any node, not only its own.
    ha.send(b, b"to b".to_vec()).await.unwrap();
    assert_eq!(rb.recv().await.unwrap(), (s, b"to b".to_vec()));
}

#[tokio::test(start_paused = true)]
async fn accept_yields_a_new_peer_with_its_first_messages_already_queued() {
    let hub = MemoryHub::new();
    let (_, s) = node();
    let (_, a) = node();
    let ra = hub.endpoint(a);
    let demux = Demux::new(hub.endpoint(s));

    ra.send(s, b"one".to_vec()).await.unwrap();
    ra.send(s, b"two".to_vec()).await.unwrap();
    let (peer, h) = demux.accept().await.unwrap();
    assert_eq!(peer, a);
    assert_eq!(h.recv().await.unwrap().1, b"one");
    assert_eq!(h.recv().await.unwrap().1, b"two");

    // The peer now has a live handle: later messages go to it, not to accept.
    ra.send(s, b"three".to_vec()).await.unwrap();
    assert_eq!(h.recv().await.unwrap().1, b"three");
    let none = tokio::time::timeout(Duration::from_secs(1), demux.accept()).await;
    assert!(none.is_err(), "no second accept for a live peer");
}

#[tokio::test(start_paused = true)]
async fn peer_claims_a_pending_peer_so_accept_does_not_yield_it_again() {
    let hub = MemoryHub::new();
    let (_, s) = node();
    let (_, a) = node();
    let ra = hub.endpoint(a);
    let demux = Demux::new(hub.endpoint(s));
    ra.send(s, b"early".to_vec()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(10)).await;
    let h = demux.peer(a);
    assert_eq!(h.recv().await.unwrap().1, b"early");
    let none = tokio::time::timeout(Duration::from_secs(1), demux.accept()).await;
    assert!(none.is_err());
}

#[tokio::test(start_paused = true)]
async fn a_full_peer_queue_drops_the_oldest_and_counts_it() {
    let hub = MemoryHub::new();
    let (_, s) = node();
    let (_, a) = node();
    let (_, b) = node();
    let (ra, rb) = (hub.endpoint(a), hub.endpoint(b));
    let demux = Demux::new(hub.endpoint(s));
    let (ha, hb) = (demux.peer(a), demux.peer(b));

    let total = PEER_QUEUE + 10;
    for i in 0..total {
        ra.send(s, (i as u32).to_be_bytes().to_vec()).await.unwrap();
    }
    rb.send(s, b"quiet".to_vec()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(10)).await;

    assert_eq!(ha.dropped(), 10);
    assert_eq!(hb.dropped(), 0);
    assert_eq!(demux.dropped(), 10);
    // The oldest 10 are gone and the rest are in order.
    let first = ha.recv().await.unwrap().1;
    assert_eq!(first, 10u32.to_be_bytes().to_vec());
    assert_eq!(hb.recv().await.unwrap().1, b"quiet");
}

#[tokio::test(start_paused = true)]
async fn pending_peers_are_capped_and_expire_after_the_ttl() {
    let hub = MemoryHub::new();
    let (_, s) = node();
    let demux = Demux::new(hub.endpoint(s));

    let mut senders = Vec::new();
    for _ in 0..MAX_PENDING_PEERS + 3 {
        let (_, n) = node();
        let r = hub.endpoint(n);
        r.send(s, b"hi".to_vec()).await.unwrap();
        senders.push((n, r));
    }
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert_eq!(demux.refused(), 3);

    // A pending sender keeps queueing even when the pending list is full.
    senders[0].1.send(s, b"again".to_vec()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert_eq!(demux.refused(), 3);
    let (first, h) = demux.accept().await.unwrap();
    assert_eq!(first, senders[0].0);
    assert_eq!(h.recv().await.unwrap().1, b"hi");
    assert_eq!(h.recv().await.unwrap().1, b"again");

    // After the TTL the rest are gone, and a new message opens a fresh pending peer.
    tokio::time::sleep(PENDING_TTL + Duration::from_secs(1)).await;
    let last = &senders[5];
    last.1.send(s, b"late".to_vec()).await.unwrap();
    let (peer, h) = demux.accept().await.unwrap();
    assert_eq!(peer, last.0);
    assert_eq!(h.recv().await.unwrap().1, b"late");
    let none = tokio::time::timeout(Duration::from_secs(1), demux.accept()).await;
    assert!(none.is_err(), "expired pending peers are not accepted");
}

#[tokio::test(start_paused = true)]
async fn a_dropped_handle_discards_its_backlog_and_the_peer_is_new_again() {
    let hub = MemoryHub::new();
    let (_, s) = node();
    let (_, a) = node();
    let ra = hub.endpoint(a);
    let demux = Demux::new(hub.endpoint(s));
    let h = demux.peer(a);
    ra.send(s, b"old".to_vec()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(10)).await;
    drop(h);

    let h2 = demux.peer(a);
    assert!(
        tokio::time::timeout(Duration::from_secs(1), h2.recv())
            .await
            .is_err(),
        "a new handle starts empty"
    );
    drop(h2);
    ra.send(s, b"retry".to_vec()).await.unwrap();
    let (peer, h3) = demux.accept().await.unwrap();
    assert_eq!(peer, a);
    assert_eq!(h3.recv().await.unwrap().1, b"retry");
}

/// A Rendezvous that replays scripted recv results, then waits forever.
struct Scripted {
    script: Mutex<VecDeque<io::Result<(NodeId, Vec<u8>)>>>,
    calls: Mutex<usize>,
    alive: Arc<()>,
}

impl Scripted {
    fn new(script: Vec<io::Result<(NodeId, Vec<u8>)>>) -> (Self, Weak<()>) {
        let alive = Arc::new(());
        let weak = Arc::downgrade(&alive);
        (
            Scripted {
                script: Mutex::new(script.into()),
                calls: Mutex::new(0),
                alive,
            },
            weak,
        )
    }
}

impl Rendezvous for Scripted {
    async fn send(&self, _to: NodeId, _msg: Vec<u8>) -> io::Result<()> {
        let _ = &self.alive;
        Ok(())
    }
    async fn recv(&self) -> io::Result<(NodeId, Vec<u8>)> {
        *self.calls.lock().unwrap() += 1;
        let next = self.script.lock().unwrap().pop_front();
        match next {
            Some(r) => r,
            None => std::future::pending().await,
        }
    }
}

fn err(kind: io::ErrorKind, text: &str) -> io::Result<(NodeId, Vec<u8>)> {
    Err(io::Error::new(kind, text.to_string()))
}

#[tokio::test(start_paused = true)]
async fn a_transient_recv_error_is_retried_and_routing_continues() {
    let (_, a) = node();
    let (rdv, _) = Scripted::new(vec![
        err(io::ErrorKind::TimedOut, "blip"),
        err(io::ErrorKind::TimedOut, "blip"),
        Ok((a, b"after".to_vec())),
    ]);
    let demux = Demux::new(rdv);
    let h = demux.peer(a);
    assert_eq!(h.recv().await.unwrap().1, b"after");
}

#[tokio::test(start_paused = true)]
async fn a_closed_rendezvous_ends_every_handle_after_its_queue_drains() {
    let (_, a) = node();
    let (_, b) = node();
    let (rdv, _) = Scripted::new(vec![
        Ok((a, b"last".to_vec())),
        err(io::ErrorKind::BrokenPipe, "hub closed"),
    ]);
    let demux = Demux::new(rdv);
    let (ha, hb) = (demux.peer(a), demux.peer(b));
    assert_eq!(ha.recv().await.unwrap().1, b"last");
    let e = ha.recv().await.unwrap_err();
    assert_eq!(e.kind(), io::ErrorKind::BrokenPipe);
    assert!(e.to_string().contains("hub closed"));
    assert_eq!(
        hb.recv().await.unwrap_err().kind(),
        io::ErrorKind::BrokenPipe
    );
    assert_eq!(
        demux.accept().await.unwrap_err().kind(),
        io::ErrorKind::BrokenPipe
    );
}

#[tokio::test(start_paused = true)]
async fn eight_transient_errors_in_a_row_end_the_pump_with_the_last_error() {
    let script = (0..8)
        .map(|i| err(io::ErrorKind::TimedOut, &format!("down {i}")))
        .collect();
    let (rdv, _) = Scripted::new(script);
    let demux = Demux::new(rdv);
    let (_, a) = node();
    let h = demux.peer(a);
    let e = h.recv().await.unwrap_err();
    assert_eq!(e.kind(), io::ErrorKind::TimedOut);
    assert!(e.to_string().contains("down 7"));
}

#[tokio::test(start_paused = true)]
async fn dropping_the_demux_stops_the_pump_and_ends_handle_recv() {
    let (_, a) = node();
    let (rdv, alive) = Scripted::new(vec![Ok((a, b"queued".to_vec()))]);
    let demux = Demux::new(rdv);
    let h = demux.peer(a);
    assert_eq!(h.recv().await.unwrap().1, b"queued");
    assert!(alive.upgrade().is_some());
    drop(demux);
    // The pump task owned the only other reference. The handle still holds the Rendezvous.
    let e = h.recv().await.unwrap_err();
    assert_eq!(e.kind(), io::ErrorKind::BrokenPipe);
    drop(h);
    tokio::task::yield_now().await;
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert!(
        alive.upgrade().is_none(),
        "the pump released the Rendezvous"
    );
}
