//! Pins SPEC.md Signalling service item 9 (the Rust client): the signed string and headers
//! match the Python client byte for byte, a message crosses a fake Worker that verifies
//! signatures the way auth.ts does, and the cursor, duplicate, error and connection rules hold.
//! The fake Worker is local. No test sends traffic to the production service.
#![cfg(feature = "signal")]

use std::collections::{HashMap, HashSet};
use std::io;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use base64::Engine as _;
use ed25519_dalek::{Signature, VerifyingKey};
use letify_link::rendezvous::{MAX_MESSAGE, Rendezvous};
use letify_link::signal::{SignalRendezvous, request_headers, signed_text};
use letify_link::{Identity, NodeId};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

// Produced by letify/transport/linksignal.py (SignalClient._headers) with the key seed
// 00 01 .. 1f, the clock 1700000000 and the nonce 00112233445566778899aabbccddeeff.
const VECTOR_NODE: &str = "03a107bff3ce10be1d70dd18e74bc09967e4d6309ba50d5f1ddc8664125531b8";
const VECTOR_NONCE: &str = "00112233445566778899aabbccddeeff";
const VECTOR_TO: &str = "abababababababababababababababababababababababababababababababab";
const VECTOR_SEND_SIG: &str = "0e960eb49f13dcf9cfbcd7e30c24ae423339a3515b1cf0bf9a477938161ca60708934b2fcb38ba9f2f902601632dd94d957f173ee51fcad8590c543b02c76703";
const VECTOR_RECV_SIG: &str = "74b6754a03102cd9b13afc8247e8a39d3aa0d3b1a20684d300b6fee6db9161c11b697226501d7660d7b344b9af5c4ba9f60d0698c6f9a1d1211f120c26fa1b0e";

fn vector_identity() -> Identity {
    let mut seed = [0u8; 32];
    for (i, b) in seed.iter_mut().enumerate() {
        *b = i as u8;
    }
    Identity::from_secret_bytes(seed)
}

fn header<'a>(headers: &'a [(String, String)], name: &str) -> &'a str {
    &headers.iter().find(|(k, _)| k == name).unwrap().1
}

#[test]
fn the_signed_text_is_the_domain_method_target_time_nonce_and_body_hash() {
    let text = signed_text("POST", "/send/x", 1700000000, VECTOR_NONCE, b"hello");
    let hash = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";
    assert_eq!(
        text,
        format!("letify-signal v1\nPOST\n/send/x\n1700000000\n{VECTOR_NONCE}\n{hash}")
    );
    let empty = signed_text("GET", "/y", 1, VECTOR_NONCE, b"");
    assert!(empty.ends_with("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"));
}

#[test]
fn the_headers_match_the_python_client_byte_for_byte() {
    let id = vector_identity();
    assert_eq!(id.node_id().to_string(), VECTOR_NODE);
    let target = format!("/send/{VECTOR_TO}");
    let h = request_headers(&id, "POST", &target, b"hello", 1700000000, VECTOR_NONCE);
    assert_eq!(header(&h, "X-Letify-Node"), VECTOR_NODE);
    assert_eq!(header(&h, "X-Letify-Time"), "1700000000");
    assert_eq!(header(&h, "X-Letify-Nonce"), VECTOR_NONCE);
    assert_eq!(header(&h, "X-Letify-Signature"), VECTOR_SEND_SIG);
    let target = format!("/recv/{VECTOR_NODE}?after=7&wait=25");
    let h = request_headers(&id, "GET", &target, b"", 1700000000, VECTOR_NONCE);
    assert_eq!(header(&h, "X-Letify-Signature"), VECTOR_RECV_SIG);
}

// ---- the fake Worker ----

#[derive(Default)]
struct World {
    next_seq: HashMap<String, u64>,
    boxes: HashMap<String, Vec<(u64, String, Vec<u8>)>>,
    nonces: HashSet<String>,
    agents: Vec<String>,
    accepts: usize,
    recv_afters: Vec<u64>,
    /// Redeliver every stored message regardless of `after`, as a service that lost an ack.
    redeliver: bool,
    chunked: bool,
    /// Answer the next send with 429 and this Retry-After.
    throttle: Option<u32>,
    /// Close the connection after each reply without saying so.
    silent_close: bool,
    sends: usize,
}

type Shared = Arc<Mutex<World>>;

struct Request {
    method: String,
    target: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

async fn read_request(s: &mut TcpStream) -> Option<Request> {
    let mut buf = Vec::new();
    let end = loop {
        if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break p;
        }
        let mut chunk = [0u8; 1024];
        let n = s.read(&mut chunk).await.ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&chunk[..n]);
    };
    let head = String::from_utf8(buf[..end].to_vec()).unwrap();
    let mut lines = head.split("\r\n");
    let mut first = lines.next().unwrap().split(' ');
    let method = first.next().unwrap().to_string();
    let target = first.next().unwrap().to_string();
    let headers: Vec<(String, String)> = lines
        .map(|l| {
            let (k, v) = l.split_once(": ").unwrap();
            (k.to_string(), v.to_string())
        })
        .collect();
    let len: usize = headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
        .map_or(0, |(_, v)| v.parse().unwrap());
    let mut body = buf[end + 4..].to_vec();
    while body.len() < len {
        let mut chunk = [0u8; 4096];
        let n = s.read(&mut chunk).await.ok()?;
        if n == 0 {
            return None;
        }
        body.extend_from_slice(&chunk[..n]);
    }
    Some(Request {
        method,
        target,
        headers,
        body,
    })
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

/// What auth.ts checks: the four headers, the canonical key, the window, the signature, the
/// replay of a nonce. Returns the signer or the refusal.
fn authenticate(world: &mut World, r: &Request) -> Result<String, (u16, &'static str)> {
    let get = |n: &str| {
        r.headers
            .iter()
            .find(|(k, _)| k == n)
            .map(|(_, v)| v.as_str())
    };
    let (Some(node), Some(time), Some(nonce), Some(sig)) = (
        get("X-Letify-Node"),
        get("X-Letify-Time"),
        get("X-Letify-Nonce"),
        get("X-Letify-Signature"),
    ) else {
        return Err((400, "bad_request"));
    };
    let id: NodeId = node.parse().map_err(|_| (400, "bad_key"))?;
    let t: u64 = time.parse().map_err(|_| (400, "bad_request"))?;
    if t.abs_diff(now()) > 60 {
        return Err((401, "stale"));
    }
    let text = format!(
        "letify-signal v1\n{}\n{}\n{time}\n{nonce}\n{}",
        r.method,
        r.target,
        hex(&Sha256::digest(&r.body))
    );
    let sig: Vec<u8> = (0..sig.len() / 2)
        .map(|i| u8::from_str_radix(&sig[2 * i..2 * i + 2], 16).unwrap())
        .collect();
    let sig = Signature::from_slice(&sig).map_err(|_| (400, "bad_request"))?;
    VerifyingKey::from_bytes(id.as_bytes())
        .unwrap()
        .verify_strict(text.as_bytes(), &sig)
        .map_err(|_| (401, "bad_signature"))?;
    if !world.nonces.insert(nonce.to_string()) {
        return Err((401, "replay"));
    }
    Ok(node.to_string())
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

async fn reply(s: &mut TcpStream, status: u16, extra: &str, body: &str, chunked: bool) {
    let mut out = format!("HTTP/1.1 {status} X\r\ncontent-type: application/json\r\n{extra}");
    if chunked {
        out += "transfer-encoding: chunked\r\n\r\n";
        let (a, b) = body.split_at(body.len() / 2);
        out += &format!("{:x}\r\n{a}\r\n{:x}\r\n{b}\r\n0\r\n\r\n", a.len(), b.len());
    } else {
        out += &format!("content-length: {}\r\n\r\n{body}", body.len());
    }
    s.write_all(out.as_bytes()).await.unwrap();
}

async fn serve_connection(mut s: TcpStream, world: Shared) {
    while let Some(r) = read_request(&mut s).await {
        let (outcome, chunked, silent) = {
            let mut w = world.lock().unwrap();
            if let Some(ua) = r.headers.iter().find(|(k, _)| k == "User-Agent") {
                w.agents.push(ua.1.clone());
            }
            (handle(&mut w, &r), w.chunked, w.silent_close)
        };
        let (status, extra, body) = match outcome {
            Handled::Reply(status, extra, body) => (status, extra, body),
            Handled::Wait => {
                // An empty long poll: the real service waits, the fake waits briefly.
                tokio::time::sleep(Duration::from_millis(150)).await;
                let node = r
                    .target
                    .split('?')
                    .next()
                    .unwrap()
                    .rsplit('/')
                    .next()
                    .unwrap();
                let w = world.lock().unwrap();
                let cursor = w.next_seq.get(node).copied().unwrap_or(0);
                (
                    200,
                    String::new(),
                    format!("{{\"messages\":[],\"cursor\":{cursor}}}"),
                )
            }
        };
        reply(&mut s, status, &extra, &body, chunked).await;
        if silent {
            return;
        }
    }
}

enum Handled {
    Reply(u16, String, String),
    Wait,
}

fn refuse(status: u16, code: &str) -> Handled {
    Handled::Reply(
        status,
        String::new(),
        format!("{{\"ok\":false,\"error\":\"{code}\"}}"),
    )
}

fn handle(w: &mut World, r: &Request) -> Handled {
    let signer = match authenticate(w, r) {
        Ok(s) => s,
        Err((status, code)) => return refuse(status, code),
    };
    let path = r.target.split('?').next().unwrap();
    if let Some(to) = path.strip_prefix("/send/") {
        w.sends += 1;
        if let Some(s) = w.throttle.take() {
            return Handled::Reply(
                429,
                format!("retry-after: {s}\r\n"),
                "{\"ok\":false,\"error\":\"rate_limited\"}".into(),
            );
        }
        if r.body.len() > 4096 {
            return refuse(413, "too_large");
        }
        let seq = w.next_seq.entry(to.to_string()).or_insert(0);
        *seq += 1;
        let seq = *seq;
        w.boxes
            .entry(to.to_string())
            .or_default()
            .push((seq, signer, r.body.clone()));
        return Handled::Reply(202, String::new(), format!("{{\"ok\":true,\"seq\":{seq}}}"));
    }
    if let Some(node) = path.strip_prefix("/recv/") {
        if node != signer {
            return refuse(403, "forbidden");
        }
        let query = r.target.split_once('?').unwrap().1;
        let param = |n: &str| -> u64 {
            query
                .split('&')
                .find_map(|p| p.strip_prefix(&format!("{n}=")))
                .unwrap()
                .parse()
                .unwrap()
        };
        let after = param("after");
        assert!(param("wait") <= 25);
        w.recv_afters.push(after);
        let redeliver = w.redeliver;
        let held = w.boxes.entry(node.to_string()).or_default();
        if !redeliver {
            held.retain(|(seq, _, _)| *seq > after);
        }
        if held.is_empty() {
            return Handled::Wait;
        }
        let cursor = w.next_seq[node];
        let items: Vec<String> = w.boxes[node]
            .iter()
            .map(|(seq, from, body)| {
                format!(
                    "{{\"seq\":{seq},\"from\":\"{from}\",\"at\":{},\"body\":\"{}\"}}",
                    now(),
                    base64::engine::general_purpose::STANDARD.encode(body)
                )
            })
            .collect();
        return Handled::Reply(
            200,
            String::new(),
            format!("{{\"messages\":[{}],\"cursor\":{cursor}}}", items.join(",")),
        );
    }
    refuse(404, "not_found")
}

async fn start() -> (String, Shared) {
    let world: Shared = Arc::default();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let shared = world.clone();
    tokio::spawn(async move {
        loop {
            let (s, _) = listener.accept().await.unwrap();
            shared.lock().unwrap().accepts += 1;
            tokio::spawn(serve_connection(s, shared.clone()));
        }
    });
    (url, world)
}

fn pair(url: &str) -> (SignalRendezvous, SignalRendezvous) {
    (
        SignalRendezvous::new(url, Identity::generate()),
        SignalRendezvous::new(url, Identity::generate()),
    )
}

async fn recv(r: &SignalRendezvous) -> io::Result<(NodeId, Vec<u8>)> {
    tokio::time::timeout(Duration::from_secs(10), r.recv())
        .await
        .expect("recv did not return in 10 s")
}

#[tokio::test]
async fn a_message_crosses_the_service_with_the_sender_the_service_stamped() {
    let (url, _) = start().await;
    let (a, b) = pair(&url);
    a.send(b.node_id(), b"\x01raw body".to_vec()).await.unwrap();
    let (from, body) = recv(&b).await.unwrap();
    assert_eq!(from, a.node_id());
    assert_eq!(body, b"\x01raw body");
}

#[tokio::test]
async fn the_user_agent_defaults_to_letify_link_and_an_embedder_can_name_itself() {
    let (url, world) = start().await;
    let (a, b) = pair(&url);
    a.send(b.node_id(), vec![1]).await.unwrap();
    let b = b.user_agent("ember/1.0");
    recv(&b).await.unwrap();
    let agents = world.lock().unwrap().agents.clone();
    assert_eq!(
        agents[0],
        format!("letify-link/{}", env!("CARGO_PKG_VERSION"))
    );
    assert_eq!(agents[1], "ember/1.0");
}

#[tokio::test]
async fn a_message_over_the_cap_is_refused_before_any_request() {
    let (url, world) = start().await;
    let (a, b) = pair(&url);
    let err = a
        .send(b.node_id(), vec![0; MAX_MESSAGE + 1])
        .await
        .unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    assert_eq!(world.lock().unwrap().sends, 0);
    a.send(b.node_id(), vec![0; MAX_MESSAGE]).await.unwrap();
}

#[tokio::test]
async fn recv_advances_after_and_returns_messages_in_order() {
    let (url, world) = start().await;
    let (a, b) = pair(&url);
    for i in 0..3u8 {
        a.send(b.node_id(), vec![i]).await.unwrap();
    }
    for i in 0..3u8 {
        assert_eq!(recv(&b).await.unwrap().1, vec![i]);
    }
    a.send(b.node_id(), vec![9]).await.unwrap();
    assert_eq!(recv(&b).await.unwrap().1, vec![9]);
    // One request got the three; the next acknowledged 3 and found the fourth.
    assert_eq!(world.lock().unwrap().recv_afters, vec![0, 3]);
}

#[tokio::test]
async fn an_empty_poll_is_repeated_with_the_same_after() {
    let (url, world) = start().await;
    let (a, b) = pair(&url);
    let waiting = tokio::spawn({
        let b = b.clone();
        async move { recv(&b).await }
    });
    tokio::time::sleep(Duration::from_millis(500)).await;
    a.send(b.node_id(), vec![5]).await.unwrap();
    assert_eq!(waiting.await.unwrap().unwrap().1, vec![5]);
    let afters = world.lock().unwrap().recv_afters.clone();
    assert!(
        afters.len() >= 3 && afters.iter().all(|a| *a == 0),
        "{afters:?}"
    );
}

#[tokio::test]
async fn a_message_the_service_sends_again_is_dropped() {
    let (url, world) = start().await;
    let (a, b) = pair(&url);
    world.lock().unwrap().redeliver = true;
    a.send(b.node_id(), vec![1]).await.unwrap();
    assert_eq!(recv(&b).await.unwrap().1, vec![1]);
    // The service redelivers seq 1 with the next poll; only seq 2 may come out.
    a.send(b.node_id(), vec![2]).await.unwrap();
    assert_eq!(recv(&b).await.unwrap().1, vec![2]);
}

#[tokio::test]
async fn a_throttled_send_fails_with_the_retry_after() {
    let (url, world) = start().await;
    let (a, b) = pair(&url);
    world.lock().unwrap().throttle = Some(7);
    let err = a.send(b.node_id(), vec![1]).await.unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::WouldBlock);
    assert_eq!(
        err.to_string(),
        "the signalling service refused the request: rate_limited (HTTP 429, retry in 7 s)"
    );
    a.send(b.node_id(), vec![1]).await.unwrap();
}

#[tokio::test]
async fn an_unreachable_service_is_reported_as_not_answering() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);
    let (a, b) = pair(&url);
    let err = a.send(b.node_id(), vec![1]).await.unwrap_err();
    assert!(
        err.to_string()
            .starts_with("the signalling service did not answer: "),
        "{err}"
    );
}

#[tokio::test]
async fn a_chunked_reply_is_read() {
    let (url, world) = start().await;
    let (a, b) = pair(&url);
    world.lock().unwrap().chunked = true;
    a.send(b.node_id(), vec![4]).await.unwrap();
    assert_eq!(recv(&b).await.unwrap().1, vec![4]);
}

#[tokio::test]
async fn requests_reuse_one_connection() {
    let (url, world) = start().await;
    let (a, b) = pair(&url);
    for _ in 0..3 {
        a.send(b.node_id(), vec![1]).await.unwrap();
    }
    assert_eq!(world.lock().unwrap().accepts, 1);
    drop(b);
}

#[tokio::test]
async fn a_connection_the_service_closed_while_idle_is_replaced_once() {
    let (url, world) = start().await;
    let (a, b) = pair(&url);
    world.lock().unwrap().silent_close = true;
    for _ in 0..3 {
        a.send(b.node_id(), vec![1]).await.unwrap();
    }
    assert_eq!(world.lock().unwrap().accepts, 3);
    assert_eq!(world.lock().unwrap().sends, 3);
}

#[tokio::test]
async fn an_unusable_url_fails_the_first_request_with_invalid_input() {
    let a = SignalRendezvous::new("ftp://example.org", Identity::generate());
    let err = a
        .send(Identity::generate().node_id(), vec![1])
        .await
        .unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
}
