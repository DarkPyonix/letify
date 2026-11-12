//! The sidecar control protocol, driven over its Unix socket (or loopback port) between two
//! `letify-link serve` processes on loopback. Spec "Sidecar interface".
#![cfg(unix)]

mod common;

use std::collections::VecDeque;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpStream, UnixStream};
use tokio::process::{Child, Command};
use tokio::sync::mpsc;

const BIN: &str = env!("CARGO_BIN_EXE_letify-link");

type Frames = mpsc::UnboundedSender<Vec<u8>>;

struct Sidecar {
    child: Child,
    node_id: String,
    control: String,
    next_id: u64,
    out: Frames,
    inbox: mpsc::UnboundedReceiver<Value>,
    stash: VecDeque<Value>,
    forward: Option<Frames>,
    /// Every `rendezvous_out` message this sidecar emitted, decoded from hex.
    sent: Vec<Vec<u8>>,
    _dir: tempdir::Dir,
}

mod tempdir {
    use std::path::{Path, PathBuf};

    /// A short directory under /tmp (Unix socket paths are short) removed on drop.
    pub struct Dir(PathBuf);
    impl Dir {
        pub fn new() -> Dir {
            let mut b = [0u8; 6];
            getrandom_fill(&mut b);
            let name: String = b.iter().map(|x| format!("{x:02x}")).collect();
            let p = std::env::temp_dir().join(format!("ll{name}"));
            std::fs::create_dir_all(&p).unwrap();
            Dir(p)
        }
        pub fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn getrandom_fill(b: &mut [u8]) {
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        // The clock on macOS ticks in microseconds, so threads that start together read the
        // same time. A process wide counter keeps their directories apart.
        static COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        let mut x = ((t as u64)
            ^ ((std::process::id() as u64) << 32)
            ^ n.wrapping_mul(0x9e37_79b9_7f4a_7c15))
            | 1;
        for v in b {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *v = x as u8;
        }
    }
}

fn frame(v: &Value) -> Vec<u8> {
    let body = serde_json::to_vec(v).unwrap();
    let mut out = (body.len() as u32).to_be_bytes().to_vec();
    out.extend(body);
    out
}

/// Send the `auth` message of the loopback control channel and check the reply.
async fn authenticate(tcp: &mut TcpStream, token: &str) {
    tcp.write_all(&frame(&json!({"op": "auth", "token": token, "id": 0})))
        .await
        .unwrap();
    let reply = read_frame(tcp).await.expect("an auth reply");
    assert_eq!(reply["ok"], true, "{reply}");
}

/// One control frame, or `None` when the connection ends first.
async fn read_frame(tcp: &mut TcpStream) -> Option<Value> {
    let mut len = [0u8; 4];
    tcp.read_exact(&mut len).await.ok()?;
    let mut body = vec![0u8; u32::from_be_bytes(len) as usize];
    tcp.read_exact(&mut body).await.ok()?;
    serde_json::from_slice(&body).ok()
}

impl Sidecar {
    async fn start(control_arg: Option<&str>) -> Sidecar {
        Self::start_with(control_arg, &[]).await
    }

    async fn start_with(control_arg: Option<&str>, env: &[(&str, String)]) -> Sidecar {
        let dir = tempdir::Dir::new();
        let control_path = dir.path().join("c");
        let key = dir.path().join("key");
        let mut child = Command::new(BIN)
            .args([
                "serve",
                "--control",
                control_arg.unwrap_or(control_path.to_str().unwrap()),
                "--key",
                key.to_str().unwrap(),
            ])
            .envs(env.iter().map(|(k, v)| (*k, v.as_str())))
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();
        let ready: Value = serde_json::from_str(
            &tokio::time::timeout(Duration::from_secs(10), lines.next_line())
                .await
                .expect("the sidecar printed no ready line")
                .unwrap()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(ready["event"], "ready");
        let control = ready["control"].as_str().unwrap().to_string();
        let node_id = ready["node_id"].as_str().unwrap().to_string();
        let (rd, mut wr): (
            Box<dyn tokio::io::AsyncRead + Send + Unpin>,
            Box<dyn tokio::io::AsyncWrite + Send + Unpin>,
        ) = if let Some(path) = control.strip_prefix("unix:") {
            let (r, w) = UnixStream::connect(path).await.unwrap().into_split();
            (Box::new(r), Box::new(w))
        } else {
            let addr = control.strip_prefix("tcp:").unwrap();
            let mut tcp = TcpStream::connect(addr).await.unwrap();
            authenticate(&mut tcp, ready["token"].as_str().unwrap()).await;
            let (r, w) = tcp.into_split();
            (Box::new(r), Box::new(w))
        };
        let (out, mut out_rx) = mpsc::unbounded_channel::<Vec<u8>>();
        tokio::spawn(async move {
            while let Some(f) = out_rx.recv().await {
                if wr.write_all(&f).await.is_err() {
                    break;
                }
            }
        });
        let (tx, inbox) = mpsc::unbounded_channel();
        tokio::spawn(async move {
            let mut rd = rd;
            loop {
                let mut len = [0u8; 4];
                if rd.read_exact(&mut len).await.is_err() {
                    break;
                }
                let mut body = vec![0u8; u32::from_be_bytes(len) as usize];
                if rd.read_exact(&mut body).await.is_err() {
                    break;
                }
                let _ = tx.send(serde_json::from_slice(&body).unwrap());
            }
        });
        // Keep the stdout pipe open so the sidecar never gets SIGPIPE.
        tokio::spawn(async move { while let Ok(Some(_)) = lines.next_line().await {} });
        Sidecar {
            child,
            node_id,
            control,
            next_id: 0,
            out,
            inbox,
            stash: VecDeque::new(),
            forward: None,
            sent: Vec::new(),
            _dir: dir,
        }
    }

    fn writer(&self) -> Frames {
        self.out.clone()
    }

    /// Send the `rendezvous_out` events of this sidecar to `other` as `rendezvous_message`.
    fn forward_rendezvous_to(&mut self, other: Frames) {
        self.forward = Some(other);
    }

    fn send(&mut self, mut req: Value) -> u64 {
        self.next_id += 1;
        req["id"] = json!(self.next_id);
        self.out.send(frame(&req)).unwrap();
        self.next_id
    }

    async fn next_frame(&mut self) -> Value {
        let v = tokio::time::timeout(Duration::from_secs(60), self.inbox.recv())
            .await
            .expect("timed out waiting for the sidecar")
            .expect("the control channel closed");
        if v["event"] == "rendezvous_out" {
            self.sent.push(unhex(v["message"].as_str().unwrap()));
        }
        if v["event"] == "rendezvous_out"
            && let Some(other) = &self.forward
        {
            let msg = json!({
                "op": "rendezvous_message",
                "id": 900_000,
                "from": self.node_id,
                "message": v["message"],
            });
            let _ = other.send(frame(&msg));
        }
        v
    }

    async fn reply(&mut self, id: u64) -> Value {
        loop {
            let v = self.next_frame().await;
            if v["id"] == json!(id) {
                return v;
            }
            if v.get("event").is_some() {
                self.stash.push_back(v);
            }
        }
    }

    async fn call(&mut self, req: Value) -> Value {
        let id = self.send(req);
        self.reply(id).await
    }

    async fn ok(&mut self, req: Value) -> Value {
        let r = self.call(req).await;
        assert_eq!(r["ok"], true, "request failed: {r}");
        r
    }

    async fn event(&mut self, name: &str) -> Value {
        if let Some(i) = self.stash.iter().position(|v| v["event"] == name) {
            return self.stash.remove(i).unwrap();
        }
        loop {
            let v = self.next_frame().await;
            if v["event"] == name {
                return v;
            }
            if v.get("event").is_some() {
                self.stash.push_back(v);
            }
        }
    }
}

/// Connect to the stream socket named by an `open_stream` reply or a `stream_incoming` event,
/// sending its secret first when it has one.
async fn local_connect(reply: &Value) -> Box<dyn Conn> {
    let socket = reply["socket"].as_str().unwrap();
    if let Some(p) = socket.strip_prefix("unix:") {
        Box::new(UnixStream::connect(PathBuf::from(p)).await.unwrap())
    } else {
        let mut tcp = TcpStream::connect(socket.strip_prefix("tcp:").unwrap())
            .await
            .unwrap();
        if let Some(secret) = reply["secret"].as_str() {
            tcp.write_all(&unhex(secret)).await.unwrap();
        }
        Box::new(tcp)
    }
}

fn unhex(text: &str) -> Vec<u8> {
    (0..text.len() / 2)
        .map(|i| u8::from_str_radix(&text[2 * i..2 * i + 2], 16).unwrap())
        .collect()
}

trait Conn: tokio::io::AsyncRead + tokio::io::AsyncWrite + Send + Unpin {}
impl<T: tokio::io::AsyncRead + tokio::io::AsyncWrite + Send + Unpin> Conn for T {}

/// Send `data` one way and read the same amount the other way.
async fn exchange(a: Box<dyn Conn>, b: Box<dyn Conn>, up: Vec<u8>, down: Vec<u8>) {
    let (mut ar, mut aw) = tokio::io::split(a);
    let (mut br, mut bw) = tokio::io::split(b);
    let (up2, down2) = (up.clone(), down.clone());
    let t1 = tokio::spawn(async move {
        aw.write_all(&up2).await.unwrap();
        aw.shutdown().await.unwrap();
    });
    let t2 = tokio::spawn(async move {
        bw.write_all(&down2).await.unwrap();
        bw.shutdown().await.unwrap();
    });
    let (mut got_b, mut got_a) = (Vec::new(), Vec::new());
    br.read_to_end(&mut got_b).await.unwrap();
    ar.read_to_end(&mut got_a).await.unwrap();
    t1.await.unwrap();
    t2.await.unwrap();
    assert!(got_b == up, "bytes from a to b differ");
    assert!(got_a == down, "bytes from b to a differ");
}

async fn pair_over_tcp(control_arg: Option<&str>) -> (Sidecar, Sidecar, u64, u64) {
    let mut a = Sidecar::start(control_arg).await;
    let mut b = Sidecar::start(control_arg).await;
    let l = b
        .ok(json!({"op": "listen", "allow": [a.node_id], "bind": "127.0.0.1:0"}))
        .await;
    let addr = l["addr"].as_str().unwrap().to_string();
    let s = a
        .ok(json!({
            "op": "open_session",
            "peer": b.node_id,
            "candidates": [{"name": "lan", "tcp": addr}],
        }))
        .await;
    assert_eq!(s["peer"], json!(b.node_id));
    let incoming = b.event("session_incoming").await;
    assert_eq!(incoming["peer"], json!(a.node_id));
    (
        a,
        b,
        s["session"].as_u64().unwrap(),
        incoming["session"].as_u64().unwrap(),
    )
}

#[tokio::test]
async fn a_stream_carries_bytes_both_ways_between_two_sidecars() {
    let (mut a, mut b, sa, sb) = pair_over_tcp(None).await;
    let st = a.ok(json!({"op": "open_stream", "session": sa})).await;
    let mut local_a = local_connect(&st).await;
    local_a.write_all(b"x").await.unwrap();
    let inc = b.event("stream_incoming").await;
    assert_eq!(inc["session"], json!(sb));
    let mut local_b = local_connect(&inc).await;
    let mut first = [0u8; 1];
    local_b.read_exact(&mut first).await.unwrap();
    assert_eq!(&first, b"x");
    exchange(
        local_a,
        local_b,
        common::pattern(3 << 20, 1),
        common::pattern(3 << 20, 2),
    )
    .await;
    let paths = a.ok(json!({"op": "paths", "session": sa})).await;
    assert_eq!(paths["paths"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn the_loopback_control_channel_works_like_the_unix_socket() {
    let (mut a, _b, sa, _sb) = pair_over_tcp(Some("tcp")).await;
    assert!(a.control.starts_with("tcp:127.0.0.1:"));
    let st = a.ok(json!({"op": "open_stream", "session": sa})).await;
    assert!(st["socket"].as_str().unwrap().starts_with("tcp:127.0.0.1:"));
}

#[tokio::test]
async fn a_bad_request_gets_a_usage_error_and_the_channel_stays_open() {
    let mut a = Sidecar::start(None).await;
    let r = a.call(json!({"op": "no_such_op"})).await;
    assert_eq!(r["ok"], false);
    assert_eq!(r["kind"], "usage");
    let r = a.call(json!({"op": "open_stream", "session": 99})).await;
    assert_eq!(r["kind"], "usage");
    let r = a
        .call(json!({"op": "open_session", "peer": "zz", "candidates": []}))
        .await;
    assert_eq!(r["kind"], "usage");
    a.ok(json!({"op": "listen", "allow": []})).await;
}

#[tokio::test]
async fn a_peer_outside_the_allow_list_is_refused() {
    let mut a = Sidecar::start(None).await;
    let mut b = Sidecar::start(None).await;
    let stranger = letify_link::Identity::generate().node_id().to_string();
    let l = b
        .ok(json!({"op": "listen", "allow": [stranger], "bind": "127.0.0.1:0"}))
        .await;
    let r = a
        .call(json!({
            "op": "open_session",
            "peer": b.node_id,
            "candidates": [{"name": "lan", "tcp": l["addr"]}],
            "config": {"handshake_timeout_ms": 1500, "start_timeout_ms": 3000},
        }))
        .await;
    assert_eq!(r["ok"], false, "{r}");
    assert_eq!(r["kind"], "infrastructure");
}

#[tokio::test]
async fn closing_a_session_ends_it_on_both_sides() {
    let (mut a, mut b, sa, sb) = pair_over_tcp(None).await;
    let r = a.ok(json!({"op": "close", "session": sa})).await;
    assert_eq!(r["ok"], true);
    let ev = b.event("session_closed").await;
    assert_eq!(ev["session"], json!(sb));
    let r = a.call(json!({"op": "open_stream", "session": sa})).await;
    assert_eq!(r["kind"], "usage");
}

#[tokio::test]
async fn the_sidecar_exits_when_the_control_channel_closes() {
    let mut a = Sidecar::start(None).await;
    a.ok(json!({"op": "listen", "allow": []})).await;
    let Sidecar { mut child, out, .. } = a;
    drop(out);
    // Dropping the sender ends the writer task, whose write half then shuts the socket down.
    let status = tokio::time::timeout(Duration::from_secs(10), child.wait()).await;
    assert!(status.is_ok(), "the sidecar did not exit");
}

#[tokio::test]
async fn a_punch_through_the_control_rendezvous_gives_a_session() {
    let mut a = Sidecar::start(None).await;
    let mut b = Sidecar::start(None).await;
    a.forward_rendezvous_to(b.writer());
    b.forward_rendezvous_to(a.writer());
    let l = b.ok(json!({"op": "listen", "allow": [a.node_id]})).await;
    let (a_id, b_id) = (a.node_id.clone(), b.node_id.clone());
    let (ra, rb) = tokio::join!(
        a.call(json!({
            "op": "punch", "peer": b_id, "role": "initiator", "sockets": 2,
            "lead_ms": 300, "window_ms": 10000,
        })),
        b.call(json!({
            "op": "punch", "peer": a_id, "role": "responder", "sockets": 2,
            "listener": l["listener"], "lead_ms": 300, "window_ms": 10000,
        })),
    );
    assert_eq!(ra["ok"], true, "{ra}");
    assert_eq!(rb["ok"], true, "{rb}");
    let names = ra["punched"].as_array().unwrap().clone();
    assert!(!names.is_empty());
    let cands: Vec<Value> = names
        .iter()
        .map(|n| json!({"name": n, "punched": n}))
        .collect();
    let s = a
        .ok(json!({"op": "open_session", "peer": b.node_id, "candidates": cands}))
        .await;
    let inc = b.event("session_incoming").await;
    assert_eq!(inc["peer"], json!(a.node_id));
    let st = a
        .ok(json!({"op": "open_stream", "session": s["session"]}))
        .await;
    let mut conn = local_connect(&st).await;
    conn.write_all(b"hi").await.unwrap();
    let ev = b.event("stream_incoming").await;
    let mut other = local_connect(&ev).await;
    let mut buf = [0u8; 2];
    other.read_exact(&mut buf).await.unwrap();
    assert_eq!(&buf, b"hi");
}

/// Start a sidecar on the loopback channel and return it with its ready line, without
/// connecting.
async fn start_unconnected(control: &str) -> (Child, Value, tempdir::Dir) {
    let dir = tempdir::Dir::new();
    let key = dir.path().join("key");
    let mut child = Command::new(BIN)
        .args([
            "serve",
            "--control",
            control,
            "--key",
            key.to_str().unwrap(),
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();
    let line = tokio::time::timeout(Duration::from_secs(10), lines.next_line())
        .await
        .expect("the sidecar printed no ready line")
        .unwrap()
        .unwrap();
    tokio::spawn(async move { while let Ok(Some(_)) = lines.next_line().await {} });
    (child, serde_json::from_str(&line).unwrap(), dir)
}

#[tokio::test]
async fn the_loopback_control_channel_requires_the_token_in_its_first_message() {
    let (_child, ready, _dir) = start_unconnected("tcp").await;
    let token = ready["token"].as_str().expect("a token in the ready line");
    assert_eq!(token.len(), 64);
    assert!(
        token
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    );
    let addr = ready["control"]
        .as_str()
        .unwrap()
        .strip_prefix("tcp:")
        .unwrap()
        .to_string();

    // A first message that is not `auth` closes the connection.
    let mut early = TcpStream::connect(&addr).await.unwrap();
    early
        .write_all(&frame(&json!({"op": "no_such_op", "id": 1})))
        .await
        .unwrap();
    assert!(
        read_frame(&mut early).await.is_none(),
        "answered without auth"
    );

    // So does an `auth` with the wrong token, even one that differs in the last character.
    let mut wrong = format!("{}0", &token[..63]);
    if wrong == token {
        wrong = format!("{}1", &token[..63]);
    }
    let mut bad = TcpStream::connect(&addr).await.unwrap();
    bad.write_all(&frame(&json!({"op": "auth", "token": wrong, "id": 1})))
        .await
        .unwrap();
    assert!(
        read_frame(&mut bad).await.is_none(),
        "accepted a wrong token"
    );

    // The sidecar kept listening, and the right token opens the channel.
    let mut good = TcpStream::connect(&addr).await.unwrap();
    authenticate(&mut good, token).await;
    good.write_all(&frame(&json!({"op": "no_such_op", "id": 2})))
        .await
        .unwrap();
    let reply = read_frame(&mut good).await.unwrap();
    assert_eq!(reply["kind"], "usage");

    // The listener closed with the first authenticated connection.
    assert!(
        TcpStream::connect(&addr).await.is_err(),
        "the control listener is still open"
    );
}

#[tokio::test]
async fn a_unix_control_socket_has_no_token() {
    let dir = tempdir::Dir::new();
    let path = dir.path().join("c");
    let (_child, ready, _d) = start_unconnected(path.to_str().unwrap()).await;
    assert!(ready.get("token").is_none(), "{ready}");
    assert!(ready["control"].as_str().unwrap().starts_with("unix:"));
}

#[tokio::test]
async fn open_stream_takes_a_priority_and_refuses_an_unknown_one() {
    let (mut a, _b, sa, _sb) = pair_over_tcp(None).await;
    let bulk = a
        .call(json!({"op": "open_stream", "session": sa, "priority": "bulk"}))
        .await;
    assert_eq!(bulk["ok"], true, "{bulk}");
    let interactive = a
        .call(json!({"op": "open_stream", "session": sa, "priority": "interactive"}))
        .await;
    assert_eq!(interactive["ok"], true, "{interactive}");
    let bad = a
        .call(json!({"op": "open_stream", "session": sa, "priority": "urgent"}))
        .await;
    assert_eq!(bad["ok"], false);
    assert_eq!(bad["kind"], "usage");
}

#[tokio::test]
async fn a_tcp_stream_socket_needs_its_secret_and_ignores_a_stray_connection() {
    let (mut a, mut b, sa, sb) = pair_over_tcp(Some("tcp")).await;
    let st = a.ok(json!({"op": "open_stream", "session": sa})).await;
    let secret = st["secret"].as_str().expect("a secret for a tcp socket");
    assert_eq!(secret.len(), 64);
    let addr = st["socket"].as_str().unwrap().strip_prefix("tcp:").unwrap();

    // A local process that sends nothing is not served, and does not block the right one.
    let _silent = TcpStream::connect(addr).await.unwrap();
    // One that sends 32 wrong bytes is closed.
    let mut wrong = TcpStream::connect(addr).await.unwrap();
    let mut bytes = unhex(secret);
    bytes[31] ^= 1;
    wrong.write_all(&bytes).await.unwrap();
    let mut buf = [0u8; 1];
    let n = tokio::time::timeout(Duration::from_secs(5), wrong.read(&mut buf))
        .await
        .expect("the wrong secret was not closed");
    assert!(matches!(n, Ok(0) | Err(_)), "a wrong secret got data");

    // The right secret still gets the stream.
    let mut local_a = local_connect(&st).await;
    local_a.write_all(b"x").await.unwrap();
    let inc = b.event("stream_incoming").await;
    assert_eq!(inc["session"], json!(sb));
    assert_eq!(inc["secret"].as_str().unwrap().len(), 64);
    assert_ne!(inc["secret"], st["secret"]);
    let mut local_b = local_connect(&inc).await;
    let mut first = [0u8; 1];
    local_b.read_exact(&mut first).await.unwrap();
    assert_eq!(&first, b"x");
}

#[tokio::test]
async fn a_silent_connection_does_not_delay_the_one_that_authenticates() {
    let (_child, ready, _dir) = start_unconnected("tcp").await;
    let token = ready["token"].as_str().unwrap();
    let addr = ready["control"]
        .as_str()
        .unwrap()
        .strip_prefix("tcp:")
        .unwrap()
        .to_string();
    // The silent connection comes first and holds the 5 s authentication window.
    let _silent = TcpStream::connect(&addr).await.unwrap();
    let mut good = TcpStream::connect(&addr).await.unwrap();
    tokio::time::timeout(Duration::from_secs(2), authenticate(&mut good, token))
        .await
        .expect("the silent connection delayed the right one");
}

/// Run `letify-link serve` until it exits and return whether it succeeded and what it printed
/// on standard error. A sidecar that starts instead is killed and reported as a success.
async fn serve_outcome(control: &str, key: &str, extra: &[&str]) -> (bool, String) {
    let mut child = Command::new(BIN)
        .args(["serve", "--control", control, "--key", key])
        .args(extra)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let mut err = child.stderr.take().unwrap();
    let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();
    tokio::select! {
        line = lines.next_line() => match line {
            Ok(Some(_)) => (true, String::new()),
            _ => {
                let status = child.wait().await.unwrap();
                let mut text = String::new();
                let _ = err.read_to_string(&mut text).await;
                (status.success(), text)
            }
        },
        status = child.wait() => {
            let mut text = String::new();
            let _ = err.read_to_string(&mut text).await;
            (status.unwrap().success(), text)
        }
        _ = tokio::time::sleep(Duration::from_secs(10)) => panic!("neither ready nor exited"),
    }
}

fn mode(path: &std::path::Path) -> u32 {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path).unwrap().permissions().mode() & 0o7777
}

fn set_mode(path: &std::path::Path, bits: u32) {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(bits)).unwrap();
}

#[tokio::test]
async fn a_control_directory_that_others_can_write_is_refused() {
    let dir = tempdir::Dir::new();
    let key = dir.path().join("key");
    set_mode(dir.path(), 0o777);
    let control = dir.path().join("c");
    let (ok, err) = serve_outcome(control.to_str().unwrap(), key.to_str().unwrap(), &[]).await;
    assert!(!ok, "started in a world writable directory");
    assert!(err.contains(dir.path().to_str().unwrap()), "{err}");
    // With the sticky bit, as /tmp has, it is accepted.
    set_mode(dir.path(), 0o1777);
    let (ok, _) = serve_outcome(control.to_str().unwrap(), key.to_str().unwrap(), &[]).await;
    assert!(ok);
}

#[tokio::test]
async fn a_stream_directory_that_others_can_write_is_refused() {
    let dir = tempdir::Dir::new();
    let streams = dir.path().join("streams");
    std::fs::create_dir(&streams).unwrap();
    set_mode(&streams, 0o757);
    let key = dir.path().join("key");
    let control = dir.path().join("c");
    let (ok, err) = serve_outcome(
        control.to_str().unwrap(),
        key.to_str().unwrap(),
        &["--stream-dir", streams.to_str().unwrap()],
    )
    .await;
    assert!(!ok);
    assert!(err.contains("streams"), "{err}");
}

#[tokio::test]
async fn the_control_socket_is_private_and_so_is_the_directory_of_a_stream_socket() {
    let (mut a, _b, sa, _sb) = pair_over_tcp(None).await;
    let control = a.control.strip_prefix("unix:").unwrap().to_string();
    assert_eq!(mode(std::path::Path::new(&control)), 0o600);
    let st = a.ok(json!({"op": "open_stream", "session": sa})).await;
    let socket = st["socket"]
        .as_str()
        .unwrap()
        .strip_prefix("unix:")
        .unwrap();
    let socket = std::path::Path::new(socket);
    assert_eq!(mode(socket), 0o600);
    assert_eq!(mode(socket.parent().unwrap()), 0o700);
    // The private directory of the control socket is gone; only the one of the streams stays.
    let beside: Vec<_> = std::fs::read_dir(std::path::Path::new(&control).parent().unwrap())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    let private = beside.iter().filter(|n| n.starts_with(".letify-")).count();
    assert_eq!(private, 1, "{beside:?}");
}

#[tokio::test]
async fn a_key_file_that_others_can_read_is_refused() {
    let dir = tempdir::Dir::new();
    let key = dir.path().join("key");
    std::fs::write(&key, "ab".repeat(32)).unwrap();
    let control = dir.path().join("c");
    set_mode(&key, 0o644);
    let (ok, err) = serve_outcome(control.to_str().unwrap(), key.to_str().unwrap(), &[]).await;
    assert!(!ok, "loaded a world readable key");
    assert!(err.contains("chmod 600"), "{err}");
    set_mode(&key, 0o600);
    let (ok, _) = serve_outcome(control.to_str().unwrap(), key.to_str().unwrap(), &[]).await;
    assert!(ok);
}

// -- NAT-aware punch: observers, UDP paths, policing ------------------------------------

use letify_link::rendezvous::{Message, Proto, Punch, PunchExt};
use letify_link::stun::NatType;
use tokio::net::UdpSocket;

/// A STUN server on loopback. `report` turns the address the client was seen at into the
/// address the fake NAT claims to have given it.
async fn fake_stun(report: impl Fn(SocketAddr) -> SocketAddr + Send + 'static) -> SocketAddr {
    let socket = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let addr = socket.local_addr().unwrap();
    tokio::spawn(async move {
        let mut buf = [0u8; 1500];
        while let Ok((n, from)) = socket.recv_from(&mut buf).await {
            if n < 20 {
                continue;
            }
            let txid = &buf[8..20];
            let SocketAddr::V4(mapped) = report(from) else {
                continue;
            };
            let cookie = 0x2112_A442u32.to_be_bytes();
            let mut attr = vec![0, 1];
            attr.extend((mapped.port() ^ 0x2112).to_be_bytes());
            attr.extend(mapped.ip().octets().iter().zip(cookie).map(|(a, c)| a ^ c));
            let mut out = Vec::new();
            out.extend(0x0101u16.to_be_bytes());
            out.extend((attr.len() as u16 + 4).to_be_bytes());
            out.extend(cookie);
            out.extend(txid);
            out.extend(0x0020u16.to_be_bytes());
            out.extend((attr.len() as u16).to_be_bytes());
            out.extend(attr);
            let _ = socket.send_to(&out, from).await;
        }
    });
    addr
}

/// Three servers that report the client's own address: a NAT that keeps ports.
async fn open_nat() -> Vec<SocketAddr> {
    let mut out = Vec::new();
    for _ in 0..3 {
        out.push(fake_stun(|from| from).await);
    }
    out
}

fn join_addrs(servers: &[SocketAddr]) -> String {
    servers
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn offers_of(sidecar: &Sidecar) -> Vec<(Punch, PunchExt)> {
    sidecar
        .sent
        .iter()
        .filter_map(|m| match Message::decode(m) {
            Ok(Message::PunchOfferExt(p, e)) => Some((p, e)),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn the_offer_carries_the_mappings_the_observers_reported() {
    let rewrite = |from: SocketAddr| SocketAddr::new("203.0.113.9".parse().unwrap(), from.port());
    let mut servers = Vec::new();
    for _ in 0..3 {
        servers.push(fake_stun(rewrite).await);
    }
    let mut a = Sidecar::start(None).await;
    let peer = Sidecar::start(None).await.node_id;
    let reply = a
        .call(json!({
            "op": "punch", "peer": peer, "role": "initiator", "sockets": 2, "udp": 1,
            "observers": servers.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            "lead_ms": 100, "window_ms": 400,
        }))
        .await;
    // Nobody answers, so the punch ends, but the offers were sent first.
    assert_eq!(reply["ok"], false, "{reply}");
    assert_eq!(reply["kind"], "infrastructure");
    let offers = offers_of(&a);
    let tcp = offers.iter().find(|(p, _)| p.proto == Proto::Tcp).unwrap();
    let udp = offers.iter().find(|(p, _)| p.proto == Proto::Udp).unwrap();
    assert_eq!(tcp.0.mappings.len(), 2);
    assert_eq!(udp.0.mappings.len(), 1);
    for m in tcp.0.mappings.iter().chain(&udp.0.mappings) {
        assert_eq!(m.ip().to_string(), "203.0.113.9", "{m}");
    }
    assert_eq!(tcp.1.nat, NatType::EndpointIndependent);
    assert_eq!(udp.1.nat, NatType::EndpointIndependent);
    assert!(!tcp.1.police);
    assert_eq!(tcp.0.start_at_ms, udp.0.start_at_ms);
    assert_ne!(tcp.0.token, udp.0.token);
}

#[tokio::test]
async fn a_symmetric_classification_is_offered_with_its_step_and_predicted_tcp_ports() {
    let mut servers = Vec::new();
    for k in 0..3u16 {
        servers.push(
            fake_stun(move |_| SocketAddr::new("203.0.113.9".parse().unwrap(), 10_000 + 2 * k))
                .await,
        );
    }
    let mut a = Sidecar::start(None).await;
    let peer = Sidecar::start(None).await.node_id;
    let _ = a
        .call(json!({
            "op": "punch", "peer": peer, "role": "initiator", "sockets": 2, "udp": 1,
            "observers": servers.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            "lead_ms": 100, "window_ms": 400,
        }))
        .await;
    let offers = offers_of(&a);
    let tcp = offers.iter().find(|(p, _)| p.proto == Proto::Tcp).unwrap();
    let udp = offers.iter().find(|(p, _)| p.proto == Proto::Udp).unwrap();
    assert_eq!(tcp.1.nat, NatType::Symmetric { step: 2 });
    // The last observation was port 10004; the NAT is expected to hand out 10006, 10008 next.
    let ports: Vec<u16> = tcp.0.mappings.iter().map(|m| m.port()).collect();
    assert_eq!(ports, vec![10_006, 10_008]);
    assert_eq!(udp.0.mappings[0].port(), 10_004);
}

#[tokio::test]
async fn fewer_than_three_answering_observers_fail_the_punch_with_the_count() {
    let one = fake_stun(|from| from).await;
    let mut a = Sidecar::start(None).await;
    let peer = Sidecar::start(None).await.node_id;
    let r = a
        .call(json!({
            "op": "punch", "peer": peer, "role": "initiator", "udp": 1,
            "observers": [one.to_string()], "lead_ms": 100, "window_ms": 400,
        }))
        .await;
    assert_eq!(r["ok"], false, "{r}");
    assert_eq!(r["kind"], "infrastructure");
    assert!(
        r["error"]
            .as_str()
            .unwrap()
            .contains("1 of 1 address observers"),
        "{r}"
    );
    assert!(offers_of(&a).is_empty());
}

/// Two sidecars with a punch between them and the observers of an open NAT. Returns both and
/// the two replies.
async fn punch_pair(
    tcp: u64,
    udp: u64,
    police: Option<Value>,
) -> (Sidecar, Sidecar, Value, Value, Value) {
    punch_pair_timed(tcp, udp, police, 300, None).await
}

/// `punch_pair` with the initiator's `lead_ms` and the responder's `answer_margin_ms`.
async fn punch_pair_timed(
    tcp: u64,
    udp: u64,
    police: Option<Value>,
    lead_ms: u64,
    answer_margin_ms: Option<u64>,
) -> (Sidecar, Sidecar, Value, Value, Value) {
    let servers = open_nat().await;
    let env = [("LETIFY_LINK_STUN", join_addrs(&servers))];
    let mut a = Sidecar::start_with(None, &env).await;
    let mut b = Sidecar::start_with(None, &env).await;
    a.forward_rendezvous_to(b.writer());
    b.forward_rendezvous_to(a.writer());
    let l = b.ok(json!({"op": "listen", "allow": [a.node_id]})).await;
    let (a_id, b_id) = (a.node_id.clone(), b.node_id.clone());
    let mut init = json!({
        "op": "punch", "peer": b_id, "role": "initiator", "sockets": tcp, "udp": udp,
        "observers": "default", "lead_ms": lead_ms, "window_ms": 15000,
    });
    if let Some(p) = police {
        init["police"] = p;
    }
    let mut resp = json!({
        "op": "punch", "peer": a_id, "role": "responder", "sockets": tcp, "udp": udp,
        "observers": "default", "listener": l["listener"], "lead_ms": lead_ms,
        "window_ms": 15000,
    });
    if let Some(margin) = answer_margin_ms {
        resp["answer_margin_ms"] = json!(margin);
    }
    let (ra, rb) = tokio::join!(a.call(init), b.call(resp));
    (a, b, ra, rb, l)
}

fn answers_of(sidecar: &Sidecar) -> Vec<Punch> {
    sidecar
        .sent
        .iter()
        .filter_map(|m| match Message::decode(m) {
            Ok(Message::PunchAnswerExt(p, _)) => Some(p),
            _ => None,
        })
        .collect()
}

async fn carries_a_stream(a: &mut Sidecar, b: &mut Sidecar, names: &[Value]) {
    let cands: Vec<Value> = names
        .iter()
        .map(|n| json!({"name": n, "punched": n}))
        .collect();
    let s = a
        .ok(json!({"op": "open_session", "peer": b.node_id, "candidates": cands}))
        .await;
    b.event("session_incoming").await;
    let st = a
        .ok(json!({"op": "open_stream", "session": s["session"]}))
        .await;
    let mut conn = local_connect(&st).await;
    conn.write_all(b"over udp").await.unwrap();
    let ev = b.event("stream_incoming").await;
    let mut other = local_connect(&ev).await;
    let mut buf = [0u8; 8];
    other.read_exact(&mut buf).await.unwrap();
    assert_eq!(&buf, b"over udp");
}

#[tokio::test]
async fn a_punched_udp_path_carries_a_session_between_two_sidecars() {
    let (mut a, mut b, ra, rb, _l) = punch_pair(0, 1, None).await;
    assert_eq!(ra["ok"], true, "{ra}");
    assert_eq!(rb["ok"], true, "{rb}");
    assert_eq!(ra["nat"], "endpoint_independent", "{ra}");
    assert_eq!(ra["peer_nat"], "endpoint_independent", "{ra}");
    assert_eq!(ra["mapped"]["udp"].as_array().unwrap().len(), 1);
    assert_eq!(ra["mapped"]["tcp"].as_array().unwrap().len(), 0);
    let names = ra["punched"].as_array().unwrap().clone();
    assert_eq!(names.len(), 1, "{ra}");
    carries_a_stream(&mut a, &mut b, &names).await;
}

#[tokio::test]
async fn tcp_and_udp_paths_from_one_punch_join_one_session() {
    let (mut a, mut b, ra, rb, _l) = punch_pair(2, 1, None).await;
    assert_eq!(ra["ok"], true, "{ra}");
    assert_eq!(rb["ok"], true, "{rb}");
    let names = ra["punched"].as_array().unwrap().clone();
    assert_eq!(names.len(), 3, "{ra}");
    assert_eq!(rb["sockets"], 3, "{rb}");
    carries_a_stream(&mut a, &mut b, &names).await;
}

#[tokio::test]
async fn policed_paths_are_reported_and_not_used() {
    // A floor no loopback path can reach makes every bulk burst count as policed.
    let police = json!({"floor_bps": 1u64 << 50, "burst_ms": 200});
    let (mut a, _b, ra, _rb, _l) = punch_pair(2, 1, Some(police)).await;
    assert_eq!(ra["ok"], false, "{ra}");
    assert_eq!(ra["kind"], "infrastructure");
    assert!(ra["error"].as_str().unwrap().contains("policed"), "{ra}");
    let ev = a.event("path_policed").await;
    assert!(ev["name"].as_str().unwrap().starts_with("punched-"), "{ev}");
    assert!(ev["up_bps"].as_f64().is_some());
    // The dropped sockets are gone: a candidate that names one is refused.
    let r = a
        .call(json!({
            "op": "open_session", "peer": a.node_id.clone(),
            "candidates": [{"name": "x", "punched": ev["name"]}],
        }))
        .await;
    assert_eq!(r["kind"], "usage", "{r}");
}

#[tokio::test]
async fn a_path_above_the_policing_floor_is_kept() {
    let police = json!({"floor_bps": 1024, "burst_ms": 200});
    let (mut a, mut b, ra, rb, _l) = punch_pair(1, 1, Some(police)).await;
    assert_eq!(ra["ok"], true, "{ra}");
    assert_eq!(rb["ok"], true, "{rb}");
    assert_eq!(ra["policed"].as_array().unwrap().len(), 0);
    let names = ra["punched"].as_array().unwrap().clone();
    assert_eq!(names.len(), 2, "{ra}");
    carries_a_stream(&mut a, &mut b, &names).await;
}

#[tokio::test]
async fn a_path_far_below_the_best_is_left_out_of_the_punch_when_asked() {
    // A ratio above 1 leaves only the best path of the set, so the rule is observable on
    // loopback, where every path is fast. The best path is never left out.
    let police = json!({"floor_bps": 1024, "burst_ms": 200, "shaped_ratio": 2.0});
    let (mut a, mut b, ra, rb, _l) = punch_pair(3, 0, Some(police)).await;
    assert_eq!(ra["ok"], true, "{ra}");
    assert_eq!(rb["ok"], true, "{rb}");
    assert_eq!(ra["punched"].as_array().unwrap().len(), 1, "{ra}");
    assert_eq!(ra["policed"].as_array().unwrap().len(), 2, "{ra}");
    let ev = a.event("path_policed").await;
    assert_eq!(ev["reason"], "shaped", "{ev}");
    let names = ra["punched"].as_array().unwrap().clone();
    carries_a_stream(&mut a, &mut b, &names).await;
}

#[tokio::test]
async fn every_path_is_kept_when_no_shaped_ratio_is_given() {
    let police = json!({"floor_bps": 1024, "burst_ms": 200});
    let (_a, _b, ra, _rb, _l) = punch_pair(3, 0, Some(police)).await;
    assert_eq!(ra["punched"].as_array().unwrap().len(), 3, "{ra}");
}

// -- A failed punch explains itself (spec "NAT traversal" item 8) -----------------------

/// Three servers that claim every client holds the TEST-NET-3 address `203.0.113.9`, with the
/// port the client used, so a punch dials an address nobody answers on.
async fn blackhole_nat() -> Vec<SocketAddr> {
    let rewrite = |from: SocketAddr| SocketAddr::new("203.0.113.9".parse().unwrap(), from.port());
    let mut out = Vec::new();
    for _ in 0..3 {
        out.push(fake_stun(rewrite).await);
    }
    out
}

#[tokio::test]
async fn a_punch_nobody_answers_reports_this_sides_nat_and_offers() {
    let mut servers = Vec::new();
    for k in 0..3u16 {
        servers.push(
            fake_stun(move |_| SocketAddr::new("203.0.113.9".parse().unwrap(), 10_000 + 2 * k))
                .await,
        );
    }
    let mut a = Sidecar::start(None).await;
    let peer = Sidecar::start(None).await.node_id;
    let r = a
        .call(json!({
            "op": "punch", "peer": peer, "role": "initiator", "sockets": 2, "udp": 1,
            "observers": servers.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            "lead_ms": 100, "window_ms": 400,
        }))
        .await;
    assert_eq!(r["ok"], false, "{r}");
    let text = r["error"].as_str().unwrap();
    assert!(
        text.contains("here: NAT symmetric:2, offered tcp 2 (2 predicted) udp 1"),
        "{text}"
    );
    assert!(text.contains("there: no answer"), "{text}");
    assert_eq!(r["detail"]["nat"], "symmetric:2", "{r}");
    assert!(r["detail"]["peer_nat"].is_null(), "{r}");
    assert_eq!(
        r["detail"]["report"].as_str().unwrap(),
        &text[text.find("here:").unwrap()..]
    );
}

#[tokio::test]
async fn a_punch_whose_dials_fail_reports_both_nats_and_both_offers() {
    let servers = blackhole_nat().await;
    let env = [("LETIFY_LINK_STUN", join_addrs(&servers))];
    let mut a = Sidecar::start_with(None, &env).await;
    let mut b = Sidecar::start_with(None, &env).await;
    a.forward_rendezvous_to(b.writer());
    b.forward_rendezvous_to(a.writer());
    let l = b.ok(json!({"op": "listen", "allow": [a.node_id]})).await;
    let (a_id, b_id) = (a.node_id.clone(), b.node_id.clone());
    let (ra, _rb) = tokio::join!(
        a.call(json!({
            "op": "punch", "peer": b_id, "role": "initiator", "sockets": 2, "udp": 1,
            "observers": "default", "lead_ms": 300, "window_ms": 1500,
        })),
        b.call(json!({
            "op": "punch", "peer": a_id, "role": "responder", "sockets": 2, "udp": 1,
            "observers": "default", "listener": l["listener"], "lead_ms": 300,
            "window_ms": 1500,
        })),
    );
    assert_eq!(ra["ok"], false, "{ra}");
    let text = ra["error"].as_str().unwrap();
    assert!(text.starts_with("punch failed: "), "{text}");
    assert!(
        text.contains(
            "here: NAT endpoint_independent, offered tcp 2 (0 predicted) udp 1; \
             there: NAT endpoint_independent, offered tcp 2 udp 1; \
             udp plan: 1 socket(s) to 1 destination(s) (fixed)"
        ),
        "{text}"
    );
    assert_eq!(ra["detail"]["nat"], "endpoint_independent", "{ra}");
    assert_eq!(ra["detail"]["peer_nat"], "endpoint_independent", "{ra}");
}

#[tokio::test]
async fn a_punch_that_cannot_classify_says_why_in_the_detail() {
    let one = fake_stun(|from| from).await;
    let mut a = Sidecar::start(None).await;
    let peer = Sidecar::start(None).await.node_id;
    let r = a
        .call(json!({
            "op": "punch", "peer": peer, "role": "initiator", "udp": 1,
            "observers": [one.to_string()], "lead_ms": 100, "window_ms": 400,
        }))
        .await;
    assert_eq!(r["ok"], false, "{r}");
    assert!(r["detail"]["nat"].is_null(), "{r}");
    assert!(
        r["detail"]["nat_error"]
            .as_str()
            .unwrap()
            .contains("1 of 1 address observers"),
        "{r}"
    );
}

#[tokio::test]
async fn a_successful_punch_reply_carries_the_report() {
    let (_a, _b, ra, _rb, _l) = punch_pair(2, 1, None).await;
    let report = ra["report"].as_str().unwrap();
    assert!(report.starts_with("here: NAT endpoint_independent, offered tcp 2 (0 predicted) udp 1; there: NAT endpoint_independent, offered tcp 2 udp 1"), "{report}");
}

// -- Signing for the signalling service ------------------------------------------------------

const SIGNAL_DOMAIN: &str = "letify-signal v1";

fn hex_of(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[tokio::test]
async fn a_signature_by_the_sidecar_verifies_under_its_node_id_in_another_sidecar() {
    let mut a = Sidecar::start(None).await;
    let mut b = Sidecar::start(None).await;
    let message = hex_of(b"POST\n/send/x\n1700000000");
    let signed = a
        .ok(json!({"op": "sign", "domain": SIGNAL_DOMAIN, "message": message}))
        .await;
    let signature = signed["signature"].as_str().unwrap().to_string();
    assert_eq!(signature.len(), 128);
    let check = |node: &str, message: &str, signature: &str| {
        json!({"op": "verify", "node": node, "domain": SIGNAL_DOMAIN,
               "message": message, "signature": signature})
    };
    let good = b.ok(check(&a.node_id, &message, &signature)).await;
    assert_eq!(good["valid"], true);
    // Another message, another signer and another domain each fail to verify.
    let other = hex_of(b"POST\n/send/y\n1700000000");
    assert_eq!(
        b.ok(check(&a.node_id, &other, &signature)).await["valid"],
        false
    );
    assert_eq!(
        b.ok(check(&b.node_id.clone(), &message, &signature)).await["valid"],
        false
    );
    let moved = json!({"op": "verify", "node": a.node_id, "domain": "letify-signal line v1",
                       "message": message, "signature": signature});
    assert_eq!(b.ok(moved).await["valid"], false);
    // A flipped bit in the signature is refused as well.
    let mut flipped = signature.into_bytes();
    flipped[0] = if flipped[0] == b'0' { b'1' } else { b'0' };
    let flipped = String::from_utf8(flipped).unwrap();
    assert_eq!(
        b.ok(check(&a.node_id, &message, &flipped)).await["valid"],
        false
    );
}

#[tokio::test]
async fn a_signature_is_plain_ed25519_over_the_domain_a_newline_and_the_message() {
    use ed25519_dalek::{Signature, VerifyingKey};
    let mut a = Sidecar::start(None).await;
    let signed = a
        .ok(json!({"op": "sign", "domain": SIGNAL_DOMAIN, "message": hex_of(b"hello")}))
        .await;
    let signature = unhex(signed["signature"].as_str().unwrap());
    let node = unhex(&a.node_id);
    let key = VerifyingKey::from_bytes(node.as_slice().try_into().unwrap()).unwrap();
    let signature = Signature::from_bytes(signature.as_slice().try_into().unwrap());
    key.verify_strict(b"letify-signal v1\nhello", &signature)
        .expect("the signature is over domain, newline, message");
}

#[tokio::test]
async fn the_sidecar_signs_only_inside_the_signalling_domains() {
    let mut a = Sidecar::start(None).await;
    for domain in [
        "letify-link record v2",
        "",
        "letify-signal",
        "other",
        "letify-signal v1\nsmuggled",
    ] {
        let r = a
            .call(json!({"op": "sign", "domain": domain, "message": "00"}))
            .await;
        assert_eq!(r["ok"], false, "signed under {domain:?}");
        assert_eq!(r["kind"], "usage");
    }
    let long = format!("letify-signal {}", "x".repeat(60));
    let r = a
        .call(json!({"op": "sign", "domain": long, "message": "00"}))
        .await;
    assert_eq!(r["kind"], "usage");
    let big = "00".repeat(8193);
    let r = a
        .call(json!({"op": "sign", "domain": SIGNAL_DOMAIN, "message": big}))
        .await;
    assert_eq!(r["kind"], "usage");
    let r = a
        .call(json!({"op": "sign", "domain": SIGNAL_DOMAIN, "message": "zz"}))
        .await;
    assert_eq!(r["kind"], "usage");
}

#[tokio::test]
async fn verify_refuses_node_ids_that_are_not_canonical_keys_and_malformed_signatures() {
    let mut a = Sidecar::start(None).await;
    let signature = "00".repeat(64);
    // The identity point, a short id and a non-hex id are not node ids.
    let identity = format!("01{}", "00".repeat(31));
    for node in [identity, "ab".repeat(31), "zz".repeat(32)] {
        let r = a
            .call(
                json!({"op": "verify", "node": node, "domain": SIGNAL_DOMAIN,
                         "message": "00", "signature": signature}),
            )
            .await;
        assert_eq!(r["ok"], false, "accepted {node}");
        assert_eq!(r["kind"], "usage");
    }
    let r = a
        .call(
            json!({"op": "verify", "node": a.node_id, "domain": SIGNAL_DOMAIN,
                     "message": "00", "signature": "00"}),
        )
        .await;
    assert_eq!(r["kind"], "usage");
}

#[tokio::test]
async fn several_udp_sockets_still_give_the_two_sides_one_working_path() {
    // Each side keeps the first (socket, address) pair that answered it. With 4 sockets the two
    // sides can keep different pairs, and a QUIC connection over mismatched pairs never opens,
    // so the UDP path is silently lost. Spec: NAT traversal item 4 and Sidecar interface, punch.
    let mut lost = Vec::new();
    for round in 0..6 {
        let (_a, _b, ra, _rb, _l) = punch_pair(1, 4, None).await;
        let names = ra["punched"].as_array().cloned().unwrap_or_default();
        if !names
            .iter()
            .any(|n| n.as_str().is_some_and(|s| s.ends_with("-u")))
        {
            lost.push(round);
        }
    }
    assert!(
        lost.is_empty(),
        "the UDP path was lost in rounds {lost:?} of 6"
    );
}

// -- The responder names the start (spec "Sidecar interface" item 7, Answer start) ------------

#[tokio::test]
async fn a_responder_that_names_a_start_makes_the_punch_begin_before_the_offered_lead() {
    let begin = std::time::Instant::now();
    let (a, b, ra, rb, _l) = punch_pair_timed(2, 0, None, 8000, Some(300)).await;
    let took = begin.elapsed();
    assert_eq!(ra["ok"], true, "{ra}");
    assert_eq!(rb["ok"], true, "{rb}");
    assert_eq!(ra["punched"].as_array().unwrap().len(), 2, "{ra}");
    assert!(
        took < Duration::from_secs(5),
        "the punch took {took:?} with an 8 s lead"
    );
    let offer = &offers_of(&a)[0].0;
    let answer = &answers_of(&b)[0];
    assert!(
        answer.start_at_ms + 5000 < offer.start_at_ms,
        "start not moved earlier"
    );
    assert_eq!(
        answer.until_ms - answer.start_at_ms,
        offer.until_ms - offer.start_at_ms,
        "the window is the offered one"
    );
}

#[tokio::test]
async fn a_responder_with_no_margin_repeats_the_offered_start() {
    let (a, b, ra, rb, _l) = punch_pair_timed(1, 0, None, 700, None).await;
    assert_eq!(ra["ok"], true, "{ra}");
    assert_eq!(rb["ok"], true, "{rb}");
    let offer = &offers_of(&a)[0].0;
    let answer = &answers_of(&b)[0];
    assert_eq!(answer.start_at_ms, offer.start_at_ms);
    assert_eq!(answer.until_ms, offer.until_ms);
}

#[tokio::test]
async fn a_margin_later_than_the_offered_start_does_not_delay_the_punch() {
    let (a, b, ra, rb, _l) = punch_pair_timed(1, 0, None, 400, Some(30_000)).await;
    assert_eq!(ra["ok"], true, "{ra}");
    assert_eq!(rb["ok"], true, "{rb}");
    let offer = &offers_of(&a)[0].0;
    assert_eq!(answers_of(&b)[0].start_at_ms, offer.start_at_ms);
}
