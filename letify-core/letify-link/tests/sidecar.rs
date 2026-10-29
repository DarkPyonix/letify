//! The sidecar control protocol, driven over its Unix socket (or loopback port) between two
//! `letify-link serve` processes on loopback. Spec "Sidecar interface".
#![cfg(unix)]

mod common;

use std::collections::VecDeque;
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
        let mut x = (t as u64) ^ ((std::process::id() as u64) << 32) | 1;
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
