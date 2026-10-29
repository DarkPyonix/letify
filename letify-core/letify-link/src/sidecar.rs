//! The sidecar: the control protocol that lets a process which cannot link this crate (the
//! pure-Python letify package) use it. It owns the control channel, the framing and the JSON
//! messages, the local stream sockets and the tables of sessions and listeners. It does not own
//! the session, the path manager or the punch, which it only calls, and it never lets a path
//! reach the controlling process. Spec "Sidecar interface".

use std::collections::HashMap;
use std::io;
use std::net::SocketAddr;
#[cfg(unix)]
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{Value, json};
use tokio::io::{
    AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, ReadHalf, WriteHalf,
    copy_bidirectional_with_sizes,
};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tokio::task::AbortHandle;

use crate::acceptor::{Accepted, Acceptor};
use crate::config::Config;
use crate::error::Error;
use crate::identity::{Identity, NodeId, PeerPolicy};
use crate::paths::{Candidate, ManagerConfig, PathEvent, PathManager};
use crate::punch::{self, Plan};
use crate::rendezvous::{Message, Proto, Punch, Rendezvous, recv_message, send_message};
use crate::session::{Priority, Session};

/// The longest control message accepted, in bytes.
pub const MAX_CONTROL_MESSAGE: usize = 1 << 20;
/// How long a local stream socket waits for the controlling process to connect.
pub const STREAM_CONNECT_WAIT: Duration = Duration::from_secs(30);
const DEFAULT_LEAD_MS: u64 = 2000;
const DEFAULT_WINDOW_MS: u64 = 20_000;

/// Where the sidecar listens and what its stream sockets are.
#[derive(Clone, Debug)]
pub struct ServeOptions {
    /// A Unix socket path, or the word `tcp` for a loopback TCP port. Windows has only `tcp`.
    pub control: String,
    /// Directory of the per-stream Unix sockets; the control socket's directory by default.
    pub stream_dir: Option<PathBuf>,
}

/// A byte pipe to the controlling process.
trait Local: AsyncRead + AsyncWrite + Send + Unpin {}
impl<T: AsyncRead + AsyncWrite + Send + Unpin> Local for T {}
type LocalConn = Box<dyn Local>;

/// Read the node secret from `path` (64 hex characters), or create the file with mode 0600.
pub fn load_or_create_key(path: &str) -> Result<Identity, String> {
    match std::fs::read_to_string(path) {
        Ok(text) => {
            let text = text.trim();
            if text.len() != 64 || !text.is_ascii() {
                return Err(format!("{path} does not hold 64 hex characters"));
            }
            let mut secret = [0u8; 32];
            for (i, b) in secret.iter_mut().enumerate() {
                *b = u8::from_str_radix(&text[2 * i..2 * i + 2], 16)
                    .map_err(|_| format!("{path} does not hold 64 hex characters"))?;
            }
            Ok(Identity::from_secret_bytes(secret))
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            let identity = Identity::generate();
            let hex: String = identity
                .secret_bytes()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            write_private(path, &hex).map_err(|e| format!("cannot write {path}: {e}"))?;
            Ok(identity)
        }
        Err(e) => Err(format!("cannot read {path}: {e}")),
    }
}

fn write_private(path: &str, text: &str) -> io::Result<()> {
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    io::Write::write_all(&mut opts.open(path)?, text.as_bytes())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex(text: &str) -> Option<Vec<u8>> {
    if !text.len().is_multiple_of(2) || !text.is_ascii() {
        return None;
    }
    (0..text.len() / 2)
        .map(|i| u8::from_str_radix(&text[2 * i..2 * i + 2], 16).ok())
        .collect()
}

/// A rendezvous whose transport is the control channel: `send` becomes a `rendezvous_out`
/// event and `recv` returns what a `rendezvous_message` request supplied.
pub struct ControlRendezvous {
    events: mpsc::UnboundedSender<Value>,
    feed: mpsc::UnboundedSender<(NodeId, Vec<u8>)>,
    inbox: tokio::sync::Mutex<mpsc::UnboundedReceiver<(NodeId, Vec<u8>)>>,
}

impl ControlRendezvous {
    fn new(events: mpsc::UnboundedSender<Value>) -> Self {
        let (feed, inbox) = mpsc::unbounded_channel();
        ControlRendezvous {
            events,
            feed,
            inbox: tokio::sync::Mutex::new(inbox),
        }
    }

    fn deliver(&self, from: NodeId, msg: Vec<u8>) {
        let _ = self.feed.send((from, msg));
    }
}

impl Rendezvous for ControlRendezvous {
    async fn send(&self, to: NodeId, msg: Vec<u8>) -> io::Result<()> {
        if msg.len() > crate::rendezvous::MAX_MESSAGE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "message over 4 KiB",
            ));
        }
        self.events
            .send(json!({"event": "rendezvous_out", "to": to.to_string(), "message": hex(&msg)}))
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "control channel closed"))
    }

    async fn recv(&self) -> io::Result<(NodeId, Vec<u8>)> {
        self.inbox
            .lock()
            .await
            .recv()
            .await
            .ok_or_else(|| io::Error::new(io::ErrorKind::BrokenPipe, "control channel closed"))
    }
}

/// A request failed. `kind` is `usage`, `infrastructure` or `closed`.
struct OpError {
    kind: &'static str,
    msg: String,
}

fn usage(msg: impl Into<String>) -> OpError {
    OpError {
        kind: "usage",
        msg: msg.into(),
    }
}

fn infra(msg: impl Into<String>) -> OpError {
    OpError {
        kind: "infrastructure",
        msg: msg.into(),
    }
}

impl From<Error> for OpError {
    fn from(e: Error) -> Self {
        let kind = if matches!(e, Error::Closed) {
            "closed"
        } else {
            "infrastructure"
        };
        OpError {
            kind,
            msg: e.to_string(),
        }
    }
}

type OpResult = Result<Value, OpError>;

fn field<'a>(req: &'a Value, name: &str) -> Result<&'a Value, OpError> {
    req.get(name)
        .ok_or_else(|| usage(format!("missing field {name}")))
}

fn str_field<'a>(req: &'a Value, name: &str) -> Result<&'a str, OpError> {
    field(req, name)?
        .as_str()
        .ok_or_else(|| usage(format!("{name} must be a string")))
}

fn u64_field(req: &Value, name: &str) -> Result<u64, OpError> {
    field(req, name)?
        .as_u64()
        .ok_or_else(|| usage(format!("{name} must be a non-negative integer")))
}

fn node_id(text: &str) -> Result<NodeId, OpError> {
    text.parse()
        .map_err(|_| usage(format!("not a node id: {text:?}")))
}

fn opt_u64(req: &Value, name: &str, default: u64) -> Result<u64, OpError> {
    match req.get(name) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => v
            .as_u64()
            .ok_or_else(|| usage(format!("{name} must be a non-negative integer"))),
    }
}

fn link_config(req: &Value) -> Result<Config, OpError> {
    let mut cfg = Config::default();
    if let Some(c) = req.get("config").filter(|c| !c.is_null()) {
        if let Some(ms) = c.get("keepalive_ms") {
            cfg.keepalive_interval = Duration::from_millis(
                ms.as_u64()
                    .ok_or_else(|| usage("keepalive_ms must be an integer"))?,
            );
        }
        if let Some(s) = c.get("resume_window_s") {
            cfg.resume_window = Duration::from_secs(
                s.as_u64()
                    .ok_or_else(|| usage("resume_window_s must be an integer"))?,
            );
        }
        if let Some(ms) = c.get("handshake_timeout_ms") {
            cfg.handshake_timeout = Duration::from_millis(
                ms.as_u64()
                    .ok_or_else(|| usage("handshake_timeout_ms must be an integer"))?,
            );
        }
    }
    Ok(cfg)
}

fn manager_config(req: &Value) -> Result<ManagerConfig, OpError> {
    let mut cfg = ManagerConfig::default();
    if let Some(ms) = req.get("config").and_then(|c| c.get("start_timeout_ms")) {
        cfg.start_timeout = Duration::from_millis(
            ms.as_u64()
                .ok_or_else(|| usage("start_timeout_ms must be an integer"))?,
        );
    }
    Ok(cfg)
}

struct SessionEntry {
    session: Session,
    tasks: Vec<AbortHandle>,
}

struct ListenerEntry {
    acceptor: Arc<Acceptor>,
    task: Option<AbortHandle>,
}

/// A local socket that waits for the controlling process, and the name to hand it.
struct LocalListener {
    inner: LocalListenerKind,
    address: String,
}

enum LocalListenerKind {
    Tcp(TcpListener),
    #[cfg(unix)]
    Unix(tokio::net::UnixListener, PathBuf),
}

impl Drop for LocalListener {
    fn drop(&mut self) {
        #[cfg(unix)]
        if let LocalListenerKind::Unix(_, path) = &self.inner {
            let _ = std::fs::remove_file(path);
        }
    }
}

impl LocalListener {
    async fn accept(&self) -> io::Result<LocalConn> {
        match &self.inner {
            LocalListenerKind::Tcp(l) => {
                let (s, _) = l.accept().await?;
                s.set_nodelay(true)?;
                Ok(Box::new(s))
            }
            #[cfg(unix)]
            LocalListenerKind::Unix(l, _) => Ok(Box::new(l.accept().await?.0)),
        }
    }
}

struct Shared {
    identity: Identity,
    stream_dir: Option<PathBuf>,
    out: mpsc::UnboundedSender<Value>,
    rendezvous: ControlRendezvous,
    next_session: AtomicU64,
    next_listener: AtomicU64,
    next_stream: AtomicU64,
    next_punch: AtomicU64,
    sessions: Mutex<HashMap<u64, SessionEntry>>,
    listeners: Mutex<HashMap<u64, ListenerEntry>>,
    punched: Mutex<HashMap<String, TcpStream>>,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl Shared {
    fn emit(&self, v: Value) {
        let _ = self.out.send(v);
    }

    async fn stream_listener(&self) -> io::Result<LocalListener> {
        let n = self.next_stream.fetch_add(1, Ordering::Relaxed) + 1;
        #[cfg(unix)]
        if let Some(dir) = &self.stream_dir {
            let path = dir.join(format!("s{n}"));
            let _ = std::fs::remove_file(&path);
            let l = tokio::net::UnixListener::bind(&path)?;
            restrict(&path)?;
            return Ok(LocalListener {
                address: format!("unix:{}", path.display()),
                inner: LocalListenerKind::Unix(l, path),
            });
        }
        let l = TcpListener::bind("127.0.0.1:0").await?;
        Ok(LocalListener {
            address: format!("tcp:{}", l.local_addr()?),
            inner: LocalListenerKind::Tcp(l),
        })
    }

    /// Bind a local socket for `stream`, bridge it once the controlling process connects and
    /// return the socket's address.
    async fn bridge(&self, stream: crate::stream::Stream) -> io::Result<String> {
        let listener = self.stream_listener().await?;
        let address = listener.address.clone();
        tokio::spawn(async move {
            let mut stream = stream;
            match tokio::time::timeout(STREAM_CONNECT_WAIT, listener.accept()).await {
                Ok(Ok(mut conn)) => {
                    drop(listener);
                    let _ =
                        copy_bidirectional_with_sizes(&mut conn, &mut stream, 64 << 10, 64 << 10)
                            .await;
                    let _ = stream.shutdown().await;
                }
                _ => {
                    let _ = stream.shutdown().await;
                }
            }
        });
        Ok(address)
    }
}

#[cfg(unix)]
fn restrict(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
}

/// Register a session, report its paths and incoming streams, and return its number.
fn register(
    shared: &Arc<Shared>,
    session: Session,
    manager: Option<PathManager>,
    listener: Option<u64>,
) -> u64 {
    let id = shared.next_session.fetch_add(1, Ordering::Relaxed) + 1;
    let mut tasks = Vec::new();
    let peer = session.peer();
    if let Some(mut manager) = manager {
        let sh = shared.clone();
        tasks.push(
            tokio::spawn(async move {
                while let Some(ev) = manager.next_event().await {
                    sh.emit(path_event(id, ev));
                }
            })
            .abort_handle(),
        );
    }
    let (sh, accepting) = (shared.clone(), handle_of(&session));
    tasks.push(
        tokio::spawn(async move {
            let why = loop {
                match accepting.accept_stream().await {
                    Ok(stream) => {
                        let sid = stream.id();
                        match sh.bridge(stream).await {
                            Ok(socket) => sh.emit(json!({
                                "event": "stream_incoming", "session": id,
                                "stream": sid, "socket": socket,
                            })),
                            Err(e) => eprintln!("letify-link: cannot bind a stream socket: {e}"),
                        }
                    }
                    Err(e) => break e.to_string(),
                }
            };
            let gone = lock(&sh.sessions).remove(&id);
            if let Some(entry) = gone {
                entry.tasks.iter().for_each(AbortHandle::abort);
                sh.emit(json!({"event": "session_closed", "session": id, "reason": why}));
            }
        })
        .abort_handle(),
    );
    lock(&shared.sessions).insert(id, SessionEntry { session, tasks });
    if let Some(l) = listener {
        shared.emit(json!({
            "event": "session_incoming", "listener": l, "session": id, "peer": peer.to_string(),
        }));
    }
    id
}

/// Another handle to the same session. `Session` is not `Clone`, a weak reference upgrades to
/// a new strong handle.
fn handle_of(session: &Session) -> Session {
    session
        .downgrade()
        .upgrade()
        .expect("a handle to a live session upgrades")
}

fn path_event(session: u64, ev: PathEvent) -> Value {
    match ev {
        PathEvent::Added { name } => {
            json!({"event": "path_added", "session": session, "name": name})
        }
        PathEvent::Lost { name, reason } => {
            json!({"event": "path_lost", "session": session, "name": name, "reason": reason})
        }
        PathEvent::Measured {
            name,
            throughput,
            rtt,
        } => json!({
            "event": "path_probe", "session": session, "name": name,
            "throughput": throughput, "rtt_ms": rtt.map(|d| d.as_secs_f64() * 1000.0),
        }),
        PathEvent::Preferred { from, to } => {
            json!({"event": "path_preferred", "session": session, "from": from, "to": to})
        }
    }
}

fn session_of(shared: &Shared, req: &Value) -> Result<Session, OpError> {
    let id = u64_field(req, "session")?;
    lock(&shared.sessions)
        .get(&id)
        .map(|e| handle_of(&e.session))
        .ok_or_else(|| usage(format!("no session {id}")))
}

async fn handle(shared: &Arc<Shared>, req: &Value) -> OpResult {
    let op = str_field(req, "op")?;
    match op {
        "listen" => op_listen(shared, req).await,
        "open_session" => op_open_session(shared, req).await,
        "punch" => op_punch(shared, req).await,
        "rendezvous_message" => {
            let from = node_id(str_field(req, "from")?)?;
            let msg =
                unhex(str_field(req, "message")?).ok_or_else(|| usage("message is not hex"))?;
            shared.rendezvous.deliver(from, msg);
            Ok(json!({}))
        }
        "paths" => {
            let session = session_of(shared, req)?;
            let ms = |d: Option<Duration>| d.map(|d| d.as_secs_f64() * 1000.0);
            let paths: Vec<Value> = session
                .subflow_stats()
                .into_iter()
                .map(|s| {
                    json!({
                        "id": s.id, "sent_bytes": s.sent_bytes,
                        "received_bytes": s.received_bytes, "acked_bytes": s.acked_bytes,
                        "rtt_ms": ms(s.rtt), "srtt_ms": ms(s.srtt),
                        "window": s.window, "probe_rate": s.probe_rate,
                    })
                })
                .collect();
            Ok(json!({"paths": paths}))
        }
        "open_stream" => {
            let session = session_of(shared, req)?;
            let priority = match req.get("priority").and_then(Value::as_str) {
                None | Some("interactive") => Priority::Interactive,
                Some("bulk") => Priority::Bulk,
                Some(other) => {
                    return Err(usage(format!(
                        "priority must be interactive or bulk, not {other}"
                    )));
                }
            };
            let stream = session.open_stream_with(priority).await?;
            let sid = stream.id();
            let socket = shared
                .bridge(stream)
                .await
                .map_err(|e| infra(format!("cannot bind a stream socket: {e}")))?;
            Ok(json!({"stream": sid, "socket": socket}))
        }
        "close" => op_close(shared, req).await,
        other => Err(usage(format!("unknown op {other}"))),
    }
}

async fn op_listen(shared: &Arc<Shared>, req: &Value) -> OpResult {
    let allow: Vec<NodeId> = field(req, "allow")?
        .as_array()
        .ok_or_else(|| usage("allow must be a list of node ids"))?
        .iter()
        .map(|v| node_id(v.as_str().unwrap_or("")))
        .collect::<Result<_, _>>()?;
    let acceptor = Arc::new(Acceptor::new(
        shared.identity.clone(),
        PeerPolicy::allow(allow),
        link_config(req)?,
    ));
    let id = shared.next_listener.fetch_add(1, Ordering::Relaxed) + 1;
    let mut reply = json!({"listener": id});
    let mut task = None;
    if let Some(bind) = req.get("bind").and_then(Value::as_str) {
        let listener = TcpListener::bind(bind)
            .await
            .map_err(|e| infra(format!("cannot listen on {bind}: {e}")))?;
        reply["addr"] = json!(
            listener
                .local_addr()
                .map_err(|e| infra(e.to_string()))?
                .to_string()
        );
        let (sh, acc) = (shared.clone(), acceptor.clone());
        task = Some(
            tokio::spawn(async move {
                while let Ok((stream, _)) = listener.accept().await {
                    let (sh, acc) = (sh.clone(), acc.clone());
                    tokio::spawn(async move {
                        if let Ok(Accepted::Session(s)) = acc.accept_tcp(stream).await {
                            register(&sh, s, None, Some(id));
                        }
                    });
                }
            })
            .abort_handle(),
        );
    }
    lock(&shared.listeners).insert(id, ListenerEntry { acceptor, task });
    Ok(reply)
}

async fn op_open_session(shared: &Arc<Shared>, req: &Value) -> OpResult {
    let peer = node_id(str_field(req, "peer")?)?;
    let list = field(req, "candidates")?
        .as_array()
        .ok_or_else(|| usage("candidates must be a list"))?;
    let mut candidates = Vec::new();
    for (i, c) in list.iter().enumerate() {
        let name = c
            .get("name")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| format!("path{i}"));
        if let Some(addr) = c.get("tcp").and_then(Value::as_str) {
            let addr: SocketAddr = addr
                .parse()
                .map_err(|_| usage(format!("not an address: {addr:?}")))?;
            candidates.push(Candidate::tcp(name, addr));
        } else if let Some(key) = c.get("punched").and_then(Value::as_str) {
            let stream = lock(&shared.punched)
                .remove(key)
                .ok_or_else(|| usage(format!("no unused punched socket {key}")))?;
            let slot = Arc::new(Mutex::new(Some(stream)));
            candidates.push(Candidate::new(name, move || {
                let slot = slot.clone();
                async move {
                    lock(&slot).take().ok_or_else(|| {
                        io::Error::new(io::ErrorKind::ConnectionAborted, "punched socket used")
                    })
                }
            }));
        } else {
            return Err(usage("a candidate needs tcp or punched"));
        }
    }
    let manager = PathManager::start(
        shared.identity.clone(),
        peer,
        candidates,
        link_config(req)?,
        manager_config(req)?,
    )
    .await?;
    let session = handle_of(manager.session());
    let id = register(shared, session, Some(manager), None);
    Ok(json!({"session": id, "peer": peer.to_string()}))
}

async fn op_punch(shared: &Arc<Shared>, req: &Value) -> OpResult {
    let peer = node_id(str_field(req, "peer")?)?;
    let initiator = match str_field(req, "role")? {
        "initiator" => true,
        "responder" => false,
        other => {
            return Err(usage(format!(
                "role must be initiator or responder, not {other}"
            )));
        }
    };
    let n = (opt_u64(req, "sockets", punch::PUNCH_SOCKETS as u64)? as usize)
        .clamp(1, punch::MAX_PUNCH_SOCKETS);
    let ip: std::net::IpAddr = match req.get("public_ip").and_then(Value::as_str) {
        Some(text) => text
            .parse()
            .map_err(|_| usage(format!("not an address: {text:?}")))?,
        None => [127, 0, 0, 1].into(),
    };
    let lead = opt_u64(req, "lead_ms", DEFAULT_LEAD_MS)?;
    let window = opt_u64(req, "window_ms", DEFAULT_WINDOW_MS)?;
    let acceptor = if initiator {
        None
    } else {
        let l = u64_field(req, "listener")?;
        let found = lock(&shared.listeners).get(&l).map(|e| e.acceptor.clone());
        Some((l, found.ok_or_else(|| usage(format!("no listener {l}")))?))
    };

    // Hold the bound sockets until the punch ends, so no other process takes the ports.
    let mut held = Vec::new();
    let mut locals = Vec::new();
    for _ in 0..n {
        let sock = punch::reusable_socket("0.0.0.0:0".parse().unwrap())
            .map_err(|e| infra(format!("cannot bind a punch socket: {e}")))?;
        locals.push(sock.local_addr().map_err(|e| infra(e.to_string()))?);
        held.push(sock);
    }
    let mappings: Vec<SocketAddr> = locals
        .iter()
        .map(|a| SocketAddr::new(ip, a.port()))
        .collect();
    let rdv = &shared.rendezvous;
    let timed_out = || infra("the punch window closed before the peer answered");

    let (plan, theirs, token) = if initiator {
        let start = punch::now_ms() + lead;
        let offer = Punch {
            proto: Proto::Tcp,
            token: random_token(),
            start_at_ms: start,
            until_ms: start + window,
            mappings,
        };
        send_message(rdv, peer, &Message::PunchOffer(offer.clone()))
            .await
            .map_err(|e| infra(e.to_string()))?;
        let wait = Duration::from_millis(lead + window);
        let answer = tokio::time::timeout(wait, async {
            loop {
                match recv_message(rdv).await {
                    Ok((from, Message::PunchAnswer(a)))
                        if from == peer && a.token == offer.token =>
                    {
                        break Ok(a);
                    }
                    Ok(_) => continue,
                    Err(e) if e.kind() == io::ErrorKind::InvalidData => continue,
                    Err(e) => break Err(e),
                }
            }
        })
        .await
        .map_err(|_| timed_out())?
        .map_err(|e| infra(e.to_string()))?;
        (
            Plan::of(&offer).with_peer_until(answer.until_ms),
            answer.mappings,
            offer.token,
        )
    } else {
        let wait = Duration::from_millis(lead + window + 60_000);
        let offer = tokio::time::timeout(wait, async {
            loop {
                match recv_message(rdv).await {
                    Ok((from, Message::PunchOffer(o))) if from == peer && o.proto == Proto::Tcp => {
                        break Ok(o);
                    }
                    Ok(_) => continue,
                    Err(e) if e.kind() == io::ErrorKind::InvalidData => continue,
                    Err(e) => break Err(e),
                }
            }
        })
        .await
        .map_err(|_| timed_out())?
        .map_err(|e| infra(e.to_string()))?;
        let answer = Punch {
            proto: Proto::Tcp,
            token: offer.token,
            start_at_ms: offer.start_at_ms,
            until_ms: offer.until_ms,
            mappings,
        };
        send_message(rdv, peer, &Message::PunchAnswer(answer))
            .await
            .map_err(|e| infra(e.to_string()))?;
        (Plan::of(&offer), offer.mappings.clone(), offer.token)
    };

    let streams = punch::punch_tcp_many(&locals, &theirs, &token, initiator, plan)
        .await
        .map_err(|e| infra(format!("punch failed: {e}")))?;
    drop(held);
    match acceptor {
        None => {
            let batch = shared.next_punch.fetch_add(1, Ordering::Relaxed) + 1;
            let mut names = Vec::new();
            for (i, s) in streams.into_iter().enumerate() {
                let name = format!("punched-{batch}-{i}");
                lock(&shared.punched).insert(name.clone(), s);
                names.push(name);
            }
            Ok(json!({"punched": names}))
        }
        Some((l, acceptor)) => {
            let count = streams.len();
            for s in streams {
                let (sh, acc) = (shared.clone(), acceptor.clone());
                tokio::spawn(async move {
                    if let Ok(Accepted::Session(sess)) = acc.accept(s).await {
                        register(&sh, sess, None, Some(l));
                    }
                });
            }
            Ok(json!({"sockets": count}))
        }
    }
}

fn random_token() -> [u8; 16] {
    let mut t = [0u8; 16];
    getrandom::getrandom(&mut t).expect("the operating system gave no random bytes");
    t
}

async fn op_close(shared: &Arc<Shared>, req: &Value) -> OpResult {
    if req.get("session").is_some() {
        let id = u64_field(req, "session")?;
        let entry = lock(&shared.sessions)
            .remove(&id)
            .ok_or_else(|| usage(format!("no session {id}")))?;
        entry.session.close().await;
        entry.tasks.iter().for_each(AbortHandle::abort);
        shared.emit(json!({"event": "session_closed", "session": id, "reason": "closed locally"}));
        Ok(json!({}))
    } else if req.get("listener").is_some() {
        let id = u64_field(req, "listener")?;
        let entry = lock(&shared.listeners)
            .remove(&id)
            .ok_or_else(|| usage(format!("no listener {id}")))?;
        if let Some(t) = entry.task {
            t.abort();
        }
        Ok(json!({}))
    } else {
        Err(usage("close needs session or listener"))
    }
}

async fn read_frame<R: AsyncRead + Unpin>(rd: &mut R) -> io::Result<Value> {
    let mut len = [0u8; 4];
    rd.read_exact(&mut len).await?;
    let len = u32::from_be_bytes(len) as usize;
    if len > MAX_CONTROL_MESSAGE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "control message over 1 MiB",
        ));
    }
    let mut body = vec![0u8; len];
    rd.read_exact(&mut body).await?;
    serde_json::from_slice(&body).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

async fn write_frames(mut wr: WriteHalf<LocalConn>, mut rx: mpsc::UnboundedReceiver<Value>) {
    while let Some(v) = rx.recv().await {
        let Ok(body) = serde_json::to_vec(&v) else {
            continue;
        };
        let mut frame = (body.len() as u32).to_be_bytes().to_vec();
        frame.extend(body);
        if wr.write_all(&frame).await.is_err() {
            break;
        }
    }
}

async fn control_loop(shared: Arc<Shared>, mut rd: ReadHalf<LocalConn>) {
    loop {
        let req = match read_frame(&mut rd).await {
            Ok(v) => v,
            Err(_) => return,
        };
        let sh = shared.clone();
        tokio::spawn(async move {
            let id = req.get("id").cloned().unwrap_or(Value::Null);
            let reply = match handle(&sh, &req).await {
                Ok(mut v) => {
                    v["id"] = id;
                    v["ok"] = json!(true);
                    v
                }
                Err(e) => json!({"id": id, "ok": false, "error": e.msg, "kind": e.kind}),
            };
            sh.emit(reply);
        });
    }
}

/// Bind the control channel, print the ready line and serve until the channel closes.
pub async fn serve(opts: ServeOptions, identity: Identity) -> io::Result<()> {
    let node = identity.node_id();
    let (conn, stream_dir): (LocalConn, Option<PathBuf>) = {
        #[cfg(unix)]
        let unix = opts.control != "tcp";
        #[cfg(not(unix))]
        let unix = false;
        if unix {
            #[cfg(unix)]
            {
                let path = PathBuf::from(&opts.control);
                let _ = std::fs::remove_file(&path);
                let listener = tokio::net::UnixListener::bind(&path)?;
                restrict(&path)?;
                announce(&node, &format!("unix:{}", path.display()), None);
                let (conn, _) = listener.accept().await?;
                drop(listener);
                let dir = opts
                    .stream_dir
                    .clone()
                    .or_else(|| path.parent().map(Path::to_path_buf))
                    .unwrap_or_else(|| PathBuf::from("."));
                (Box::new(conn), Some(dir))
            }
            #[cfg(not(unix))]
            unreachable!()
        } else {
            let listener = TcpListener::bind("127.0.0.1:0").await?;
            let text = format!("tcp:{}", listener.local_addr()?);
            let token = new_token()?;
            announce(&node, &text, Some(&token));
            // Any local process can connect to a loopback port, so the first message of a
            // connection must carry the token. A connection that fails that is closed and the
            // listener stays open for the right one.
            let conn = loop {
                let (mut conn, _) = listener.accept().await?;
                conn.set_nodelay(true)?;
                if tokio::time::timeout(AUTH_TIMEOUT, authenticate(&mut conn, &token))
                    .await
                    .unwrap_or(false)
                {
                    break conn;
                }
            };
            drop(listener);
            (Box::new(conn), None)
        }
    };
    let (rd, wr) = tokio::io::split(conn);
    let (out, rx) = mpsc::unbounded_channel();
    let writer = tokio::spawn(write_frames(wr, rx));
    let shared = Arc::new(Shared {
        identity,
        stream_dir,
        rendezvous: ControlRendezvous::new(out.clone()),
        out,
        next_session: AtomicU64::new(0),
        next_listener: AtomicU64::new(0),
        next_stream: AtomicU64::new(0),
        next_punch: AtomicU64::new(0),
        sessions: Mutex::new(HashMap::new()),
        listeners: Mutex::new(HashMap::new()),
        punched: Mutex::new(HashMap::new()),
    });
    control_loop(shared, rd).await;
    writer.abort();
    Ok(())
}

/// Time a loopback connection has to send its `auth` message.
const AUTH_TIMEOUT: Duration = Duration::from_secs(5);

/// 32 random bytes from the operating system as 64 lowercase hex characters.
fn new_token() -> io::Result<String> {
    let mut bytes = [0u8; 32];
    getrandom::getrandom(&mut bytes).map_err(|e| io::Error::other(e.to_string()))?;
    Ok(hex(&bytes))
}

/// Compare without stopping at the first difference.
fn same_token(a: &str, b: &str) -> bool {
    a.len() == b.len()
        && a.bytes()
            .zip(b.bytes())
            .fold(0u8, |diff, (x, y)| diff | (x ^ y))
            == 0
}

/// Read the first message of a loopback connection. It must be `auth` with the right token;
/// the reply is sent here, before the connection becomes the control channel.
async fn authenticate(conn: &mut TcpStream, token: &str) -> bool {
    let Ok(req) = read_frame(conn).await else {
        return false;
    };
    let given = req.get("token").and_then(Value::as_str).unwrap_or("");
    if req.get("op").and_then(Value::as_str) != Some("auth") || !same_token(given, token) {
        return false;
    }
    let reply = json!({"id": req.get("id").cloned().unwrap_or(Value::Null), "ok": true});
    let Ok(body) = serde_json::to_vec(&reply) else {
        return false;
    };
    let mut frame = (body.len() as u32).to_be_bytes().to_vec();
    frame.extend(body);
    conn.write_all(&frame).await.is_ok()
}

fn announce(node: &NodeId, control: &str, token: Option<&str>) {
    let mut line = json!({"event": "ready", "node_id": node.to_string(), "control": control});
    if let Some(token) = token {
        line["token"] = json!(token);
    }
    println!("{line}");
    let _ = io::Write::flush(&mut io::stdout());
}
