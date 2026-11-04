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
use tokio::net::{TcpListener, TcpStream, UdpSocket};
use tokio::sync::mpsc;
use tokio::task::AbortHandle;
use tokio::task::JoinSet;

use crate::acceptor::{Accepted, Acceptor};
use crate::config::Config;
use crate::error::Error;
use crate::identity::{Identity, NodeId, PeerPolicy};
use crate::observe::AddressObserver;
use crate::paths::{
    BoxTransport, Candidate, ManagerConfig, PathEvent, PathKind, PathKinds, PathManager,
};
use crate::plan;
use crate::policing::{self, Verdict};
use crate::punch::{self, Plan};
use crate::rendezvous::{Message, Proto, Punch, PunchExt, Rendezvous, recv_message, send_message};
use crate::session::{Priority, Session};
use crate::stun::{self, NatType, StunObserver, classify_with};
use crate::udp::UdpPath;

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
    #[cfg(unix)]
    check_key_file(path)?;
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

/// Refuse an existing key file that group or others can reach, or that another user owns.
#[cfg(unix)]
fn check_key_file(path: &str) -> Result<(), String> {
    use std::os::unix::fs::MetadataExt;
    let Ok(meta) = std::fs::metadata(path) else {
        // A missing file is created later; any other error surfaces when it is read.
        return Ok(());
    };
    if meta.mode() & 0o077 != 0 {
        return Err(format!(
            "{path} can be read by group or others (mode {:03o}); run chmod 600 {path}",
            meta.mode() & 0o777
        ));
    }
    // SAFETY: geteuid takes no arguments and cannot fail.
    if meta.uid() != unsafe { libc::geteuid() } {
        return Err(format!(
            "{path} is not owned by the user running letify-link; run chown to that user and chmod 600 {path}"
        ));
    }
    Ok(())
}

/// Refuse a directory in which any user could replace a socket: one that everyone can write to
/// and that does not have the sticky bit, as `/tmp` has. Group write is allowed, because the
/// default umask 002 with a private group per user gives every new directory that mode.
#[cfg(unix)]
fn check_socket_dir(dir: &Path) -> io::Result<()> {
    use std::os::unix::fs::MetadataExt;
    let dir = if dir.as_os_str().is_empty() {
        Path::new(".")
    } else {
        dir
    };
    let mode = std::fs::metadata(dir)?.mode();
    if mode & 0o002 != 0 && mode & 0o1000 == 0 {
        return Err(io::Error::other(format!(
            "{} can be written by every user and has no sticky bit; use a private directory",
            dir.display()
        )));
    }
    Ok(())
}

/// A new directory with mode 0700 inside `parent`, named from random bytes. A socket bound
/// inside it cannot be reached by anyone else while its own mode is still the default.
#[cfg(unix)]
fn private_dir(parent: &Path, tag: &str) -> io::Result<PathBuf> {
    use std::os::unix::fs::DirBuilderExt;
    let parent = if parent.as_os_str().is_empty() {
        Path::new(".")
    } else {
        parent
    };
    let dir = parent.join(format!(".letify-{tag}-{}", hex(&new_secret()?[..8])));
    std::fs::DirBuilder::new().mode(0o700).create(&dir)?;
    Ok(dir)
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
    /// Structured facts about the failure, sent as the `detail` field of the error reply.
    detail: Option<Value>,
}

fn usage(msg: impl Into<String>) -> OpError {
    OpError {
        kind: "usage",
        msg: msg.into(),
        detail: None,
    }
}

fn infra(msg: impl Into<String>) -> OpError {
    OpError {
        kind: "infrastructure",
        msg: msg.into(),
        detail: None,
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
            detail: None,
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
    /// Which kind each subflow runs on, known only for a session this side started.
    kinds: Option<PathKinds>,
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
    /// The bytes a client must send first. Only a TCP socket has one; a Unix socket is
    /// limited to its owner by its mode.
    secret: Option<[u8; 32]>,
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
    /// Wait for the controlling process. A TCP connection counts only after it has sent the
    /// stream secret; the others are closed and the wait goes on. Connections are checked
    /// concurrently, so a connection that stays silent delays no other.
    async fn accept(&self) -> io::Result<LocalConn> {
        match &self.inner {
            LocalListenerKind::Tcp(l) => {
                let secret = self.secret.unwrap_or([0; 32]);
                let (tx, mut rx) = mpsc::unbounded_channel();
                loop {
                    tokio::select! {
                        accepted = l.accept() => {
                            let (s, _) = accepted?;
                            s.set_nodelay(true)?;
                            let tx = tx.clone();
                            tokio::spawn(async move {
                                let mut s = s;
                                let mut given = [0u8; 32];
                                let read = tokio::time::timeout(
                                    AUTH_TIMEOUT,
                                    s.read_exact(&mut given),
                                )
                                .await;
                                if matches!(read, Ok(Ok(_))) && same_bytes(&given, &secret) {
                                    let _ = tx.send(s);
                                }
                            });
                        }
                        Some(s) = rx.recv() => return Ok(Box::new(s)),
                    }
                }
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
    punched: Mutex<HashMap<String, Punched>>,
    /// When the sidecar started, the zero of the diagnostic trace.
    started: std::time::Instant,
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
                secret: None,
            });
        }
        let l = TcpListener::bind("127.0.0.1:0").await?;
        Ok(LocalListener {
            address: format!("tcp:{}", l.local_addr()?),
            inner: LocalListenerKind::Tcp(l),
            secret: Some(new_secret()?),
        })
    }

    /// Bind a local socket for `stream`, bridge it once the controlling process connects and
    /// return the socket's address.
    async fn bridge(&self, stream: crate::stream::Stream) -> io::Result<(String, Option<String>)> {
        let listener = self.stream_listener().await?;
        let address = listener.address.clone();
        let secret = listener.secret.map(|s| hex(&s));
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
        Ok((address, secret))
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
    // The protocol reports the id as presented, which is the id the Python side knows.
    let peer = session.peer_presented();
    let kinds = manager.as_ref().map(PathManager::kinds);
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
                            Ok((socket, secret)) => {
                                let mut ev = json!({
                                    "event": "stream_incoming", "session": id,
                                    "stream": sid, "socket": socket,
                                });
                                if let Some(secret) = secret {
                                    ev["secret"] = json!(secret);
                                }
                                sh.emit(ev);
                            }
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
    if let Some(config) = crate::trace::TraceConfig::from_env() {
        tasks.push(crate::trace::spawn(
            config,
            id,
            session.downgrade(),
            shared.started,
            kinds.clone(),
        ));
    }
    lock(&shared.sessions).insert(
        id,
        SessionEntry {
            session,
            kinds,
            tasks,
        },
    );
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
            let kinds = lock(&shared.sessions)
                .get(&u64_field(req, "session")?)
                .and_then(|e| e.kinds.clone());
            let kind_of = |id| match kinds.as_ref().and_then(|k| k.of(id)) {
                Some(PathKind::Tcp) => json!("tcp"),
                Some(PathKind::Udp) => json!("udp"),
                Some(PathKind::Relay) => json!("relay"),
                None => Value::Null,
            };
            let ms = |d: Option<Duration>| d.map(|d| d.as_secs_f64() * 1000.0);
            let paths: Vec<Value> = session
                .subflow_stats()
                .into_iter()
                .map(|s| {
                    json!({
                        "id": s.id, "kind": kind_of(s.id), "sent_bytes": s.sent_bytes,
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
            let (socket, secret) = shared
                .bridge(stream)
                .await
                .map_err(|e| infra(format!("cannot bind a stream socket: {e}")))?;
            let mut reply = json!({"stream": sid, "socket": socket});
            if let Some(secret) = secret {
                reply["secret"] = json!(secret);
            }
            Ok(reply)
        }
        "close" => op_close(shared, req).await,
        "sign" => {
            let bytes = signed_bytes(req)?;
            Ok(json!({"signature": hex(&shared.identity.sign(&bytes))}))
        }
        "verify" => {
            let node = node_id(str_field(req, "node")?)?;
            let bytes = signed_bytes(req)?;
            let signature: [u8; 64] = unhex(str_field(req, "signature")?)
                .and_then(|s| s.try_into().ok())
                .ok_or_else(|| usage("signature must be 128 hex characters"))?;
            Ok(json!({"valid": node.verify(&bytes, &signature)}))
        }
        other => Err(usage(format!("unknown op {other}"))),
    }
}

/// The domain prefix every signature of the `sign` and `verify` ops must start with. It keeps
/// the node key from signing a handshake, an address record or anything else it is used for.
const SIGN_DOMAIN_PREFIX: &str = "letify-signal ";
/// The longest domain, in characters.
const SIGN_DOMAIN_MAX: usize = 64;
/// The longest message to sign, in bytes.
const SIGN_MESSAGE_MAX: usize = 8192;

/// The bytes a signature covers: the domain, a newline, then the message.
fn signed_bytes(req: &Value) -> Result<Vec<u8>, OpError> {
    let domain = str_field(req, "domain")?;
    let printable = domain.bytes().all(|b| (0x20..0x7f).contains(&b));
    if !printable || !domain.starts_with(SIGN_DOMAIN_PREFIX) || domain.len() > SIGN_DOMAIN_MAX {
        return Err(usage(format!(
            "domain must be printable ASCII of at most {SIGN_DOMAIN_MAX} characters and start with {SIGN_DOMAIN_PREFIX:?}"
        )));
    }
    let message = unhex(str_field(req, "message")?).ok_or_else(|| usage("message is not hex"))?;
    if message.len() > SIGN_MESSAGE_MAX {
        return Err(usage(format!(
            "message is over {SIGN_MESSAGE_MAX} bytes: {}",
            message.len()
        )));
    }
    let mut bytes = Vec::with_capacity(domain.len() + 1 + message.len());
    bytes.extend_from_slice(domain.as_bytes());
    bytes.push(b'\n');
    bytes.extend_from_slice(&message);
    Ok(bytes)
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
            let path = lock(&shared.punched)
                .remove(key)
                .ok_or_else(|| usage(format!("no unused punched socket {key}")))?;
            let (udp, transport): (bool, BoxTransport) = match path {
                Punched::Tcp(s) => (false, Box::new(s)),
                Punched::Udp(p) => (true, Box::new(p)),
            };
            let slot = Arc::new(Mutex::new(Some(transport)));
            let dial = move || {
                let slot = slot.clone();
                async move {
                    lock(&slot).take().ok_or_else(|| {
                        io::Error::new(io::ErrorKind::ConnectionAborted, "punched socket used")
                    })
                }
            };
            candidates.push(if udp {
                Candidate::udp(name, dial)
            } else {
                Candidate::new(name, dial)
            });
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

/// A path that a punch made and has not yet been given to a session.
enum Punched {
    Tcp(TcpStream),
    Udp(UdpPath),
}

/// How long each address observer has to answer one query.
const OBSERVE_WAIT: Duration = Duration::from_millis(1500);
/// How long the initiator waits for the second answer after the first, and the responder for
/// the second offer. Both messages leave together, so they arrive together.
const SECOND_MESSAGE_GRACE: Duration = Duration::from_millis(400);
/// The default length of a policing burst.
const DEFAULT_BURST_MS: u64 = 500;

fn nat_name(nat: NatType) -> String {
    match nat {
        NatType::Unknown => "unknown".into(),
        NatType::EndpointIndependent => "endpoint_independent".into(),
        NatType::Symmetric { step } => format!("symmetric:{step}"),
    }
}

/// The STUN servers named by the `observers` field: none, the default list (or the servers of
/// `LETIFY_LINK_STUN`), or the listed `host:port` names.
async fn observer_servers(req: &Value) -> Result<Option<Vec<SocketAddr>>, OpError> {
    let names: Vec<String> = match req.get("observers") {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::String(word)) if word == "default" => match std::env::var("LETIFY_LINK_STUN") {
            Ok(list) if !list.trim().is_empty() => list
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect(),
            _ => stun::DEFAULT_SERVERS
                .iter()
                .map(|s| s.to_string())
                .collect(),
        },
        Some(Value::Array(list)) => list
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| usage("observers must be host:port strings"))
            })
            .collect::<Result<_, _>>()?,
        Some(_) => {
            return Err(usage(
                "observers must be \"default\" or a list of host:port",
            ));
        }
    };
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let mut found = Vec::new();
    for name in &refs {
        match name.parse::<SocketAddr>() {
            Ok(addr) => found.push(addr),
            Err(_) => found.extend(stun::resolve(&[name]).await),
        }
    }
    Ok(Some(found))
}

/// What one punch offers: this side's sockets, mappings and NAT.
struct Own {
    nat: NatType,
    tcp_maps: Vec<SocketAddr>,
    udp_maps: Vec<SocketAddr>,
}

/// What the other side offered, as far as the exchange got.
struct There {
    nat: NatType,
    tcp: usize,
    udp: usize,
}

/// The facts of one punch that explain how it went. Spec "NAT traversal" item 8.
struct Report {
    /// This side's NAT, or why there is none.
    here: Result<NatType, String>,
    tcp: usize,
    tcp_predicted: usize,
    udp: usize,
    /// `None` until the peer has answered.
    there: Option<There>,
    /// The UDP plan, once both NATs are known.
    plan: Option<String>,
}

impl Report {
    fn new(own: &Own, bound: &[u16]) -> Report {
        let predicted = own
            .tcp_maps
            .iter()
            .zip(bound)
            .filter(|(m, port)| m.port() != **port)
            .count();
        Report {
            here: Ok(own.nat),
            tcp: own.tcp_maps.len(),
            tcp_predicted: predicted,
            udp: own.udp_maps.len(),
            there: None,
            plan: None,
        }
    }

    fn text(&self) -> String {
        let here = match &self.here {
            Ok(nat) => format!(
                "NAT {}, offered tcp {} ({} predicted) udp {}",
                nat_name(*nat),
                self.tcp,
                self.tcp_predicted,
                self.udp
            ),
            Err(why) => format!("NAT unknown ({why}), nothing offered"),
        };
        let there = match &self.there {
            Some(t) => format!(
                "NAT {}, offered tcp {} udp {}",
                nat_name(t.nat),
                t.tcp,
                t.udp
            ),
            None => "no answer".to_string(),
        };
        let mut out = format!("here: {here}; there: {there}");
        if let Some(plan) = &self.plan {
            out.push_str("; udp plan: ");
            out.push_str(plan);
        }
        out
    }

    /// `e` with the report appended to its message and its facts in `detail`.
    fn explain(&self, mut e: OpError) -> OpError {
        let text = self.text();
        e.msg = format!("{}; {text}", e.msg);
        let mut detail = json!({
            "nat": self.here.as_ref().ok().map(|n| nat_name(*n)),
            "peer_nat": self.there.as_ref().map(|t| nat_name(t.nat)),
            "report": text,
        });
        if let Err(why) = &self.here {
            detail["nat_error"] = json!(why);
        }
        e.detail = Some(detail);
        e
    }
}

/// Ask the observers about the first UDP socket and work out the mappings to offer. Without
/// observers the mappings are the bound addresses on `public_ip`.
async fn own_mappings(
    observers: Option<&[SocketAddr]>,
    public_ip: std::net::IpAddr,
    tcp_ports: &[u16],
    udp: &[UdpSocket],
    probe: &UdpSocket,
) -> Result<Own, OpError> {
    let Some(servers) = observers else {
        return Ok(Own {
            nat: NatType::Unknown,
            tcp_maps: plan::tcp_mappings(public_ip, tcp_ports, NatType::Unknown, None),
            udp_maps: udp
                .iter()
                .map(|s| {
                    s.local_addr()
                        .map(|a| SocketAddr::new(public_ip, a.port()))
                        .map_err(|e| infra(e.to_string()))
                })
                .collect::<Result<_, _>>()?,
        });
    };
    let observers: Vec<StunObserver> = servers
        .iter()
        .map(|s| StunObserver::new(*s, OBSERVE_WAIT))
        .collect();
    let refs: Vec<&dyn AddressObserver> = observers
        .iter()
        .map(|o| o as &dyn AddressObserver)
        .collect();
    let class = classify_with(probe, &refs)
        .await
        .map_err(|e| infra(format!("cannot classify the NAT: {e}")))?;
    let last = class.last_mapped();
    let ip = class.mappings[0].mapped.ip();
    let mut udp_maps = Vec::new();
    for (i, sock) in udp.iter().enumerate() {
        if i == 0 {
            udp_maps.extend(last);
        } else {
            let seen = refs[0]
                .observe(sock)
                .await
                .map_err(|e| infra(format!("cannot observe a UDP socket: {e}")))?;
            udp_maps.push(seen.mapped);
        }
    }
    Ok(Own {
        nat: class.nat,
        tcp_maps: plan::tcp_mappings(ip, tcp_ports, class.nat, last),
        udp_maps,
    })
}

/// One side's view of an exchanged offer or answer.
type Terms = (Punch, PunchExt);

fn terms_of(msg: Message) -> Option<(bool, Terms)> {
    let plain = PunchExt {
        nat: NatType::Unknown,
        police: false,
    };
    match msg {
        Message::PunchOffer(p) => Some((true, (p, plain))),
        Message::PunchOfferExt(p, e) => Some((true, (p, e))),
        Message::PunchAnswer(p) => Some((false, (p, plain))),
        Message::PunchAnswerExt(p, e) => Some((false, (p, e))),
        _ => None,
    }
}

/// What the initiator got back for the offers it sent.
struct Answers {
    tcp: Option<Terms>,
    udp: Option<Terms>,
}

async fn await_answers(
    rdv: &ControlRendezvous,
    peer: NodeId,
    tcp: Option<&Punch>,
    udp: Option<&Punch>,
    wait: Duration,
) -> Result<Answers, OpError> {
    let overall = tokio::time::Instant::now() + wait;
    let mut deadline = overall;
    let mut got = Answers {
        tcp: None,
        udp: None,
    };
    while (tcp.is_some() && got.tcp.is_none()) || (udp.is_some() && got.udp.is_none()) {
        let next = match tokio::time::timeout_at(deadline, recv_message(rdv)).await {
            Err(_) => break,
            Ok(Ok(next)) => next,
            Ok(Err(e)) if e.kind() == io::ErrorKind::InvalidData => continue,
            Ok(Err(e)) => return Err(infra(e.to_string())),
        };
        let (from, msg) = next;
        let Some((false, (answer, ext))) = terms_of(msg) else {
            continue;
        };
        if from != peer {
            continue;
        }
        match (answer.proto, tcp, udp) {
            (Proto::Tcp, Some(o), _) if o.token == answer.token => got.tcp = Some((answer, ext)),
            (Proto::Udp, _, Some(o)) if o.token == answer.token => got.udp = Some((answer, ext)),
            _ => continue,
        }
        deadline = overall.min(tokio::time::Instant::now() + SECOND_MESSAGE_GRACE);
    }
    if got.tcp.is_none() && got.udp.is_none() {
        return Err(infra("the punch window closed before the peer answered"));
    }
    Ok(got)
}

/// What the responder got from the initiator.
struct Offers {
    tcp: Option<Terms>,
    udp: Option<Terms>,
}

async fn await_offers(
    rdv: &ControlRendezvous,
    peer: NodeId,
    want_tcp: bool,
    want_udp: bool,
    wait: Duration,
) -> Result<Offers, OpError> {
    let overall = tokio::time::Instant::now() + wait;
    let mut deadline = overall;
    let mut got = Offers {
        tcp: None,
        udp: None,
    };
    let mut first = false;
    loop {
        let next = match tokio::time::timeout_at(deadline, recv_message(rdv)).await {
            Err(_) => break,
            Ok(Ok(next)) => next,
            Ok(Err(e)) if e.kind() == io::ErrorKind::InvalidData => continue,
            Ok(Err(e)) => return Err(infra(e.to_string())),
        };
        let (from, msg) = next;
        let Some((true, terms)) = terms_of(msg) else {
            continue;
        };
        if from != peer {
            continue;
        }
        match terms.0.proto {
            Proto::Tcp if want_tcp && got.tcp.is_none() => got.tcp = Some(terms),
            Proto::Udp if want_udp && got.udp.is_none() => got.udp = Some(terms),
            _ => continue,
        }
        if !first {
            first = true;
            deadline = overall.min(tokio::time::Instant::now() + SECOND_MESSAGE_GRACE);
        }
        if (!want_tcp || got.tcp.is_some()) && (!want_udp || got.udp.is_some()) {
            break;
        }
    }
    if got.tcp.is_none() && got.udp.is_none() {
        return Err(infra("the punch window closed before the peer answered"));
    }
    Ok(got)
}

/// The result of checking one punched path on the initiator.
enum Vetted {
    Usable(String, Punched),
    Policed(String, &'static str, policing::Rates),
    Failed(String, String),
}

async fn vet<S>(
    name: String,
    proto: &'static str,
    mut stream: S,
    police: Option<(u64, Duration)>,
    wrap: impl FnOnce(S) -> Punched,
) -> Vetted
where
    S: AsyncRead + AsyncWrite + Unpin + Send,
{
    if let Some((floor, burst)) = police {
        match policing::check(&mut stream, floor, burst).await {
            Ok(Verdict::Usable(_)) => {}
            Ok(Verdict::Policed(rates)) => return Vetted::Policed(name, proto, rates),
            Err(e) => return Vetted::Failed(name, e.to_string()),
        }
    }
    Vetted::Usable(name, wrap(stream))
}

/// Check a punched path as the responder, then hand it to the acceptor.
async fn accept_punched<T: crate::session::Transport>(
    shared: Arc<Shared>,
    acceptor: Arc<Acceptor>,
    listener: u64,
    mut transport: T,
    police: bool,
) {
    if police && policing::serve(&mut transport).await.is_err() {
        return;
    }
    if let Ok(Accepted::Session(session)) = acceptor.accept(transport).await {
        register(&shared, session, None, Some(listener));
    }
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
    let tcp_n = (opt_u64(req, "sockets", punch::PUNCH_SOCKETS as u64)? as usize)
        .min(punch::MAX_PUNCH_SOCKETS);
    let udp_n = (opt_u64(req, "udp", 0)? as usize).min(punch::MAX_PUNCH_SOCKETS);
    if tcp_n + udp_n == 0 {
        return Err(usage("a punch needs sockets or udp above 0"));
    }
    let ip: std::net::IpAddr = match req.get("public_ip").and_then(Value::as_str) {
        Some(text) => text
            .parse()
            .map_err(|_| usage(format!("not an address: {text:?}")))?,
        None => [127, 0, 0, 1].into(),
    };
    let police = match req.get("police") {
        None | Some(Value::Null) => None,
        Some(p) => Some((
            u64_field(p, "floor_bps")?,
            Duration::from_millis(opt_u64(p, "burst_ms", DEFAULT_BURST_MS)?),
        )),
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
    let observers = observer_servers(req).await?;

    // Hold the bound sockets until the punch ends, so no other process takes the ports.
    let mut held = Vec::new();
    let mut locals = Vec::new();
    for _ in 0..tcp_n {
        let sock = punch::reusable_socket("0.0.0.0:0".parse().unwrap())
            .map_err(|e| infra(format!("cannot bind a punch socket: {e}")))?;
        locals.push(sock.local_addr().map_err(|e| infra(e.to_string()))?);
        held.push(sock);
    }
    let mut udp_socks = Vec::new();
    for _ in 0..udp_n {
        udp_socks.push(
            UdpSocket::bind("0.0.0.0:0")
                .await
                .map_err(|e| infra(format!("cannot bind a UDP socket: {e}")))?,
        );
    }
    // The NAT is classified from the first UDP socket, or from one made for it.
    let spare;
    let probe = match udp_socks.first() {
        Some(s) => s,
        None => {
            spare = UdpSocket::bind("0.0.0.0:0")
                .await
                .map_err(|e| infra(format!("cannot bind a UDP socket: {e}")))?;
            &spare
        }
    };
    let ports: Vec<u16> = locals.iter().map(|a| a.port()).collect();
    let own = match own_mappings(observers.as_deref(), ip, &ports, &udp_socks, probe).await {
        Ok(own) => own,
        Err(e) => {
            let report = Report {
                here: Err(e.msg.clone()),
                tcp: 0,
                tcp_predicted: 0,
                udp: 0,
                there: None,
                plan: None,
            };
            return Err(report.explain(e));
        }
    };
    let mut report = Report::new(&own, &ports);
    let rdv = &shared.rendezvous;
    let ext = |police: bool| PunchExt {
        nat: own.nat,
        police,
    };

    // The terms of each protocol: (theirs, token, plan, peer nat).
    let mut tcp_terms: Option<(Vec<SocketAddr>, [u8; 16], Plan)> = None;
    let mut udp_terms: Option<(Vec<SocketAddr>, [u8; 16], Plan)> = None;
    let peer_nat;
    let policed_by_offer;
    if initiator {
        let start = punch::now_ms() + lead;
        let make = |proto: Proto, mappings: &[SocketAddr]| Punch {
            proto,
            token: random_token(),
            start_at_ms: start,
            until_ms: start + window,
            mappings: mappings.to_vec(),
        };
        let tcp_offer = (tcp_n > 0).then(|| make(Proto::Tcp, &own.tcp_maps));
        let udp_offer = (udp_n > 0).then(|| make(Proto::Udp, &own.udp_maps));
        for offer in tcp_offer.iter().chain(&udp_offer) {
            send_message(
                rdv,
                peer,
                &Message::PunchOfferExt(offer.clone(), ext(police.is_some())),
            )
            .await
            .map_err(|e| infra(e.to_string()))?;
        }
        let answers = await_answers(
            rdv,
            peer,
            tcp_offer.as_ref(),
            udp_offer.as_ref(),
            Duration::from_millis(lead + window),
        )
        .await
        .map_err(|e| report.explain(e))?;
        peer_nat = answers
            .tcp
            .iter()
            .chain(&answers.udp)
            .next()
            .map_or(NatType::Unknown, |(_, e)| e.nat);
        policed_by_offer = police.is_some();
        if let (Some(o), Some((a, _))) = (&tcp_offer, &answers.tcp) {
            tcp_terms = Some((
                a.mappings.clone(),
                o.token,
                Plan::of(o).with_peer_until(a.until_ms),
            ));
        }
        if let (Some(o), Some((a, _))) = (&udp_offer, &answers.udp) {
            udp_terms = Some((
                a.mappings.clone(),
                o.token,
                Plan::of(o).with_peer_until(a.until_ms),
            ));
        }
    } else {
        let offers = await_offers(
            rdv,
            peer,
            tcp_n > 0,
            udp_n > 0,
            Duration::from_millis(lead + window + 60_000),
        )
        .await
        .map_err(|e| report.explain(e))?;
        peer_nat = offers
            .tcp
            .iter()
            .chain(&offers.udp)
            .next()
            .map_or(NatType::Unknown, |(_, e)| e.nat);
        policed_by_offer = offers.tcp.iter().chain(&offers.udp).any(|(_, e)| e.police);
        for (terms, maps, slot) in [
            (&offers.tcp, &own.tcp_maps, &mut tcp_terms),
            (&offers.udp, &own.udp_maps, &mut udp_terms),
        ] {
            let Some((o, e)) = terms else { continue };
            let answer = Punch {
                proto: o.proto,
                token: o.token,
                start_at_ms: o.start_at_ms,
                until_ms: o.until_ms,
                mappings: maps.clone(),
            };
            send_message(rdv, peer, &Message::PunchAnswerExt(answer, ext(e.police)))
                .await
                .map_err(|e| infra(e.to_string()))?;
            *slot = Some((o.mappings.clone(), o.token, Plan::of(o)));
        }
    }

    report.there = Some(There {
        nat: peer_nat,
        tcp: tcp_terms.as_ref().map_or(0, |t| t.0.len()),
        udp: udp_terms.as_ref().map_or(0, |t| t.0.len()),
    });
    // The UDP plan depends on both NATs, so it is made after the exchange.
    let udp_job = match &udp_terms {
        Some((theirs, token, plan)) => {
            let p = plan::udp_plan(own.nat, peer_nat, udp_n, theirs);
            let have = udp_socks.len();
            if p.sockets > have {
                udp_socks.extend(
                    punch::bind_birthday_sockets("0.0.0.0".parse().unwrap(), p.sockets - have)
                        .await
                        .map_err(|e| infra(format!("cannot bind UDP sockets: {e}")))?,
                );
            }
            udp_socks.truncate(p.sockets);
            report.plan = Some(match &p.dests {
                plan::Destinations::Fixed(v) => {
                    let extra = v.len().saturating_sub(theirs.len());
                    let kind = if extra == 0 {
                        "fixed".to_string()
                    } else {
                        format!("fixed, {extra} predicted")
                    };
                    format!(
                        "{} socket(s) to {} destination(s) ({kind})",
                        p.sockets,
                        v.len()
                    )
                }
                plan::Destinations::Birthday { count, .. } => {
                    format!(
                        "{} socket(s) to {count} destination(s) (birthday)",
                        p.sockets
                    )
                }
            });
            let dests = match p.dests {
                plan::Destinations::Fixed(v) => v,
                plan::Destinations::Birthday { ip, count } => {
                    punch::birthday_destinations(ip, count)
                }
            };
            Some((dests, *token, *plan))
        }
        None => None,
    };
    let tcp_fut = async {
        match &tcp_terms {
            Some((theirs, token, plan)) => {
                punch::punch_tcp_many(&locals, theirs, token, initiator, *plan)
                    .await
                    .map(Some)
            }
            None => Ok(None),
        }
    };
    let udp_fut = async {
        match udp_job {
            Some((dests, token, plan)) => punch::punch_udp_all(
                std::mem::take(&mut udp_socks),
                &dests,
                &token,
                plan,
                punch::SPRAY_PER_SECOND,
            )
            .await
            .map(|(socks, index, from)| Some((socks, index, from, plan))),
            None => Ok(None),
        }
    };
    let (tcp_res, udp_res) = tokio::join!(tcp_fut, udp_fut);
    drop(held);
    let (tcp_streams, udp_found) = match (tcp_res, udp_res) {
        (Err(t), Err(u)) => {
            return Err(report.explain(infra(format!("punch failed: tcp: {t}; udp: {u}"))));
        }
        (Err(t), Ok(None)) => return Err(report.explain(infra(format!("punch failed: {t}")))),
        (Ok(None), Err(u)) => return Err(report.explain(infra(format!("punch failed: {u}")))),
        (t, u) => (t.ok().flatten().unwrap_or_default(), u.ok().flatten()),
    };
    let mapped = json!({
        "tcp": own.tcp_maps.iter().map(|a| a.to_string()).collect::<Vec<_>>(),
        "udp": own.udp_maps.iter().map(|a| a.to_string()).collect::<Vec<_>>(),
    });
    let summary = |mut reply: Value, udp_failed: &[String]| {
        reply["nat"] = json!(nat_name(own.nat));
        reply["peer_nat"] = json!(nat_name(peer_nat));
        reply["mapped"] = mapped.clone();
        reply["report"] = json!(format!("{}{}", report.text(), udp_clause(udp_failed)));
        reply
    };
    match acceptor {
        None => {
            let batch = shared.next_punch.fetch_add(1, Ordering::Relaxed) + 1;
            let mut set = JoinSet::new();
            for (i, s) in tcp_streams.into_iter().enumerate() {
                let name = format!("punched-{batch}-{i}");
                set.spawn(vet(name, "tcp", s, police, Punched::Tcp));
            }
            if let Some((mut socks, index, from, _)) = udp_found {
                let socket = socks.swap_remove(index);
                drop(socks);
                let name = format!("punched-{batch}-u");
                set.spawn(async move {
                    let std = match socket.into_std() {
                        Ok(s) => s,
                        Err(e) => return Vetted::Failed(name, e.to_string()),
                    };
                    match UdpPath::connect(std, from).await {
                        Ok(path) => vet(name, "udp", path, police, Punched::Udp).await,
                        Err(e) => Vetted::Failed(name, e.to_string()),
                    }
                });
            }
            let mut names = Vec::new();
            let mut policed = Vec::new();
            let mut failed = Vec::new();
            while let Some(done) = set.join_next().await {
                match done {
                    Ok(Vetted::Usable(name, path)) => {
                        lock(&shared.punched).insert(name.clone(), path);
                        names.push(name);
                    }
                    Ok(Vetted::Policed(name, proto, rates)) => {
                        shared.emit(json!({
                            "event": "path_policed", "peer": peer.to_string(), "name": name,
                            "proto": proto, "up_bps": rates.up_bps, "down_bps": rates.down_bps,
                        }));
                        policed.push(name);
                    }
                    Ok(Vetted::Failed(name, why)) => failed.push(format!("{name}: {why}")),
                    Err(e) => failed.push(e.to_string()),
                }
            }
            names.sort();
            policed.sort();
            if names.is_empty() {
                return Err(infra(format!(
                    "every punched path was dropped: {} policed ({}), {} failed its check ({})",
                    policed.len(),
                    policed.join(", "),
                    failed.len(),
                    failed.join("; ")
                )));
            }
            let udp_failed: Vec<String> = failed
                .iter()
                .filter(|f| f.starts_with("punched-") && f.contains("-u: "))
                .cloned()
                .collect();
            Ok(summary(
                json!({"punched": names, "policed": policed}),
                &udp_failed,
            ))
        }
        Some((l, acceptor)) => {
            let count = tcp_streams.len() + usize::from(udp_found.is_some());
            for s in tcp_streams {
                tokio::spawn(accept_punched(
                    shared.clone(),
                    acceptor.clone(),
                    l,
                    s,
                    policed_by_offer,
                ));
            }
            if let Some((socks, _, _, plan)) = udp_found {
                let (sh, acc) = (shared.clone(), acceptor.clone());
                tokio::spawn(async move {
                    if let Ok(path) = punch::accept_udp(socks, plan).await {
                        accept_punched(sh, acc, l, path, policed_by_offer).await;
                    }
                });
            }
            Ok(summary(json!({"sockets": count}), &[]))
        }
    }
}

/// The report clause for UDP paths that failed their QUIC connect, from the `name: reason`
/// entries of the failed vets. Empty when none failed. Spec "NAT traversal" item 8.
fn udp_clause(failed: &[String]) -> String {
    match failed.first() {
        Some(first) => {
            let why = first.split_once(": ").map_or(first.as_str(), |(_, w)| w);
            format!("; udp: did not open: {why}")
        }
        None => String::new(),
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
                Err(e) => {
                    let mut reply = json!({"id": id, "ok": false, "error": e.msg, "kind": e.kind});
                    if let Some(detail) = e.detail {
                        reply["detail"] = detail;
                    }
                    reply
                }
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
                let parent = path.parent().map(Path::to_path_buf).unwrap_or_default();
                check_socket_dir(&parent)?;
                if let Some(dir) = &opts.stream_dir {
                    check_socket_dir(dir)?;
                }
                // Bind inside a directory only this user can enter, then move the socket to
                // its name, so it is never reachable with the default mode.
                let hidden = private_dir(&parent, "c")?;
                let bound = hidden.join("c");
                let listener = tokio::net::UnixListener::bind(&bound)?;
                restrict(&bound)?;
                let _ = std::fs::remove_file(&path);
                std::fs::rename(&bound, &path)?;
                let _ = std::fs::remove_dir(&hidden);
                announce(&node, &format!("unix:{}", path.display()), None);
                let (conn, _) = listener.accept().await?;
                drop(listener);
                let dir = opts.stream_dir.clone().unwrap_or_else(|| parent.clone());
                (Box::new(conn), Some(private_dir(&dir, "s")?))
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
            // Connections are checked at the same time, so a silent one delays no other.
            let (won_tx, mut won_rx) = mpsc::unbounded_channel();
            let conn = loop {
                tokio::select! {
                    accepted = listener.accept() => {
                        let (mut conn, _) = accepted?;
                        conn.set_nodelay(true)?;
                        let (token, won) = (token.clone(), won_tx.clone());
                        tokio::spawn(async move {
                            if tokio::time::timeout(AUTH_TIMEOUT, authenticate(&mut conn, &token))
                                .await
                                .unwrap_or(false)
                            {
                                let _ = won.send(conn);
                            }
                        });
                    }
                    Some(conn) = won_rx.recv() => break conn,
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
        started: std::time::Instant::now(),
    });
    control_loop(shared.clone(), rd).await;
    writer.abort();
    #[cfg(unix)]
    if let Some(dir) = &shared.stream_dir {
        let _ = std::fs::remove_dir(dir);
    }
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

/// 32 random bytes from the operating system.
fn new_secret() -> io::Result<[u8; 32]> {
    let mut bytes = [0u8; 32];
    getrandom::getrandom(&mut bytes).map_err(|e| io::Error::other(e.to_string()))?;
    Ok(bytes)
}

/// Compare without stopping at the first difference.
fn same_bytes(a: &[u8; 32], b: &[u8; 32]) -> bool {
    a.iter().zip(b).fold(0u8, |diff, (x, y)| diff | (x ^ y)) == 0
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

#[cfg(test)]
mod udp_clause_tests {
    use super::udp_clause;

    #[test]
    fn a_failed_udp_vet_is_named_in_the_report() {
        let failed = vec!["punched-3-u: the UDP path did not open".to_string()];
        assert_eq!(
            udp_clause(&failed),
            "; udp: did not open: the UDP path did not open"
        );
    }

    #[test]
    fn no_failed_udp_vet_adds_nothing_to_the_report() {
        assert_eq!(udp_clause(&[]), "");
    }
}
