//! The [`Rendezvous`] trait over the owner's signalling service (`letify-signal/`, served at
//! `https://letify.darkpyonix.dev`). It owns the signed HTTP requests, a minimal HTTP/1.1
//! client over rustls with one reused connection, and the long-poll loop with its cursor and
//! duplicate check. It does not own the service, the WebSocket of `/ws/<node>` (see the spec
//! for why), any envelope around a message, or the choice of what to do after a failure. Spec
//! "Signalling service" item 9.

use std::collections::VecDeque;
use std::io;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use base64::Engine as _;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::sync::Mutex;
use tokio_rustls::TlsConnector;
use tokio_rustls::rustls::pki_types::ServerName;
use tokio_rustls::rustls::{ClientConfig, RootCertStore};

use crate::identity::{Identity, NodeId};
use crate::rendezvous::{MAX_MESSAGE, Rendezvous};

/// The service the owner operates.
pub const DEFAULT_URL: &str = "https://letify.darkpyonix.dev";
/// The signature domain of a request.
const REQUEST_DOMAIN: &str = "letify-signal v1";
/// Seconds one long poll waits at the service (its limit is 25).
const POLL_S: u64 = 25;
/// The largest reply the client reads, in bytes. A full mailbox is 32 messages of 4096 bytes,
/// which is about 175 KB as base64 in JSON.
const MAX_REPLY: usize = 1 << 20;
/// The longest header block the client accepts, in bytes.
const MAX_HEAD: usize = 16 * 1024;

/// A request to the service, signed as the Python client signs it.
///
/// The text is `letify-signal v1\n<METHOD>\n<target>\n<time>\n<nonce>\n<sha256 hex of body>`,
/// where `target` is the path and query exactly as sent.
pub fn signed_text(method: &str, target: &str, time: u64, nonce: &str, body: &[u8]) -> String {
    format!(
        "{REQUEST_DOMAIN}\n{method}\n{target}\n{time}\n{nonce}\n{}",
        hex(&Sha256::digest(body))
    )
}

/// The four signed headers of a request, as `(name, value)` pairs: `X-Letify-Node`,
/// `X-Letify-Time`, `X-Letify-Nonce` and `X-Letify-Signature`. `nonce` is 32 lowercase hex
/// characters.
pub fn request_headers(
    identity: &Identity,
    method: &str,
    target: &str,
    body: &[u8],
    time: u64,
    nonce: &str,
) -> Vec<(String, String)> {
    let text = signed_text(method, target, time, nonce, body);
    let signature = identity.sign(text.as_bytes());
    vec![
        ("X-Letify-Node".into(), identity.node_id().to_string()),
        ("X-Letify-Time".into(), time.to_string()),
        ("X-Letify-Nonce".into(), nonce.into()),
        ("X-Letify-Signature".into(), hex(&signature)),
    ]
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(String::new(), |mut out, b| {
        let _ = write!(out, "{b:02x}");
        out
    })
}

trait Io: AsyncRead + AsyncWrite + Unpin + Send {}
impl<T: AsyncRead + AsyncWrite + Unpin + Send> Io for T {}

type Conn = BufReader<Box<dyn Io>>;

struct Target {
    tls: bool,
    host: String,
    port: u16,
}

fn parse_url(url: &str) -> Result<Target, String> {
    let url = url.trim_end_matches('/');
    let (tls, rest) = if let Some(r) = url.strip_prefix("https://") {
        (true, r)
    } else if let Some(r) = url.strip_prefix("http://") {
        (false, r)
    } else {
        return Err(format!(
            "the signalling service URL {url:?} is not http or https"
        ));
    };
    if rest.is_empty() || rest.contains('/') {
        return Err(format!(
            "the signalling service URL {url:?} must be a scheme and a host"
        ));
    }
    let (host, port) = match rest.rsplit_once(':') {
        Some((h, p)) if !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()) => (
            h,
            p.parse::<u16>()
                .map_err(|_| format!("the port in {url:?} is not valid"))?,
        ),
        _ => (rest, if tls { 443 } else { 80 }),
    };
    Ok(Target {
        tls,
        host: host.trim_matches(['[', ']']).to_string(),
        port,
    })
}

struct Reply {
    status: u16,
    retry_after: Option<String>,
    body: Vec<u8>,
}

struct RecvState {
    /// The highest `seq` returned to the caller. It is the `after` of the next poll.
    cursor: u64,
    queue: VecDeque<(NodeId, Vec<u8>)>,
}

struct Inner {
    target: Result<Target, String>,
    identity: Identity,
    own: String,
    user_agent: String,
    timeout: Duration,
    tls: std::sync::OnceLock<TlsConnector>,
    idle: StdMutex<Option<Conn>>,
    state: Mutex<RecvState>,
}

/// A [`Rendezvous`] over the signalling service.
///
/// ```no_run
/// # async fn demo(identity: letify_link::Identity) {
/// use letify_link::signal::{DEFAULT_URL, SignalRendezvous};
/// let rdv = SignalRendezvous::new(DEFAULT_URL, identity).user_agent("ember/1.0");
/// // letify_link::driver::punch(&rdv, ...) now exchanges its messages through the service.
/// # let _ = rdv;
/// # }
/// ```
///
/// A message is sent as the raw body of `POST /send/<to>` and received from
/// `GET /recv/<own id>`. There is no envelope: the sender the trait reports is the `from` the
/// service stamped, and the library's own messages authenticate themselves. Cloning shares the
/// connection and the cursor.
#[derive(Clone)]
pub struct SignalRendezvous {
    inner: Arc<Inner>,
}

impl SignalRendezvous {
    /// A client for the service at `base_url` (`https://` or `http://`, a host and an optional
    /// port) that signs with `identity`. An unusable URL fails the first request with
    /// `InvalidInput`. The User-Agent is `letify-link/<version>` until [`Self::user_agent`]
    /// names another.
    pub fn new(base_url: &str, identity: Identity) -> Self {
        SignalRendezvous {
            inner: Arc::new(Inner {
                target: parse_url(base_url),
                own: identity.node_id().to_string(),
                identity,
                user_agent: default_agent(),
                timeout: Duration::from_secs(10),
                tls: std::sync::OnceLock::new(),
                idle: StdMutex::new(None),
                state: Mutex::new(RecvState {
                    cursor: 0,
                    queue: VecDeque::new(),
                }),
            }),
        }
    }

    /// Send `text` as the User-Agent of every request. The service sits behind Cloudflare's
    /// Browser Integrity Check, which refuses the default agents of HTTP libraries, so an
    /// embedder names itself, for example `ember/1.0`. Call it before the first request.
    ///
    /// # Panics
    ///
    /// When the client was already cloned.
    pub fn user_agent(mut self, text: &str) -> Self {
        self.settings().user_agent = text.to_string();
        self
    }

    /// The time a request may take beyond the long poll's own wait. The default is 10 s. Call
    /// it before the first request.
    ///
    /// # Panics
    ///
    /// When the client was already cloned.
    pub fn request_timeout(mut self, timeout: Duration) -> Self {
        self.settings().timeout = timeout;
        self
    }

    fn settings(&mut self) -> &mut Inner {
        Arc::get_mut(&mut self.inner).expect("configure the client before cloning it")
    }

    /// The node id this client signs and reads as.
    pub fn node_id(&self) -> NodeId {
        self.inner.identity.node_id()
    }
}

fn default_agent() -> String {
    format!("letify-link/{}", env!("CARGO_PKG_VERSION"))
}

fn unanswered(reason: impl std::fmt::Display) -> io::Error {
    io::Error::other(format!("the signalling service did not answer: {reason}"))
}

/// A failure of one exchange: whether it happened before any reply byte arrived, which is the
/// only case a reused connection is retried.
struct Failure {
    error: io::Error,
    before_reply: bool,
}

impl Inner {
    fn connector(&self) -> &TlsConnector {
        self.tls.get_or_init(|| {
            let roots = RootCertStore {
                roots: webpki_roots::TLS_SERVER_ROOTS.to_vec(),
            };
            let config = ClientConfig::builder_with_provider(Arc::new(
                tokio_rustls::rustls::crypto::ring::default_provider(),
            ))
            .with_safe_default_protocol_versions()
            .expect("ring supports the default protocol versions")
            .with_root_certificates(roots)
            .with_no_client_auth();
            TlsConnector::from(Arc::new(config))
        })
    }

    async fn connect(&self, target: &Target) -> io::Result<Conn> {
        let tcp = TcpStream::connect((target.host.as_str(), target.port)).await?;
        tcp.set_nodelay(true)?;
        let io: Box<dyn Io> = if target.tls {
            let name = ServerName::try_from(target.host.clone())
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
            Box::new(self.connector().connect(name, tcp).await?)
        } else {
            Box::new(tcp)
        };
        Ok(BufReader::new(io))
    }

    async fn exchange_on(
        &self,
        conn: &mut Conn,
        target: &Target,
        method: &str,
        path: &str,
        body: &[u8],
    ) -> Result<(Reply, bool), Failure> {
        let time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let mut nonce = [0u8; 16];
        getrandom::getrandom(&mut nonce).map_err(|e| Failure {
            error: io::Error::other(e),
            before_reply: true,
        })?;
        let headers = request_headers(&self.identity, method, path, body, time, &hex(&nonce));
        let mut head = format!(
            "{method} {path} HTTP/1.1\r\nHost: {}\r\nUser-Agent: {}\r\nAccept: application/json\r\n",
            host_header(target),
            self.user_agent
        );
        for (name, value) in &headers {
            head += &format!("{name}: {value}\r\n");
        }
        if method == "POST" {
            head += &format!(
                "Content-Type: application/octet-stream\r\nContent-Length: {}\r\n",
                body.len()
            );
        }
        head += "\r\n";
        let early = |error: io::Error| Failure {
            error,
            before_reply: true,
        };
        let stream = conn.get_mut();
        stream.write_all(head.as_bytes()).await.map_err(early)?;
        stream.write_all(body).await.map_err(early)?;
        stream.flush().await.map_err(early)?;
        read_reply(conn).await
    }

    async fn request(
        &self,
        method: &str,
        path: &str,
        body: &[u8],
        wait: Duration,
    ) -> io::Result<Reply> {
        let target = match &self.target {
            Ok(t) => t,
            Err(e) => return Err(io::Error::new(io::ErrorKind::InvalidInput, e.clone())),
        };
        let limit = self.timeout + wait;
        let run = async {
            let reused = self.idle.lock().expect("idle lock").take();
            let mut attempts: Vec<(Conn, bool)> = Vec::new();
            if let Some(conn) = reused {
                attempts.push((conn, true));
            }
            loop {
                let (mut conn, was_reused) = match attempts.pop() {
                    Some(pair) => pair,
                    None => (self.connect(target).await.map_err(unanswered)?, false),
                };
                match self
                    .exchange_on(&mut conn, target, method, path, body)
                    .await
                {
                    Ok((reply, keep)) => {
                        if keep {
                            *self.idle.lock().expect("idle lock") = Some(conn);
                        }
                        return Ok(reply);
                    }
                    Err(f) if was_reused && f.before_reply => continue,
                    Err(f) => return Err(unanswered(f.error)),
                }
            }
        };
        tokio::time::timeout(limit, run)
            .await
            .map_err(|_| unanswered("the request timed out"))?
    }
}

fn host_header(t: &Target) -> String {
    let host = if t.host.contains(':') {
        format!("[{}]", t.host)
    } else {
        t.host.clone()
    };
    let default = if t.tls { 443 } else { 80 };
    if t.port == default {
        host
    } else {
        format!("{host}:{}", t.port)
    }
}

async fn read_reply(conn: &mut Conn) -> Result<(Reply, bool), Failure> {
    let mut first = true;
    let fail = |error: io::Error, first: bool| Failure {
        error,
        before_reply: first,
    };
    let mut line = String::new();
    let n = conn
        .read_line(&mut line)
        .await
        .map_err(|e| fail(e, first))?;
    if n == 0 {
        return Err(fail(io::ErrorKind::UnexpectedEof.into(), first));
    }
    first = false;
    let mut parts = line.split_whitespace();
    let version = parts.next().unwrap_or("");
    let status: u16 = parts.next().and_then(|s| s.parse().ok()).ok_or_else(|| {
        fail(
            io::Error::new(io::ErrorKind::InvalidData, "bad status line"),
            first,
        )
    })?;
    let mut length: Option<usize> = None;
    let mut chunked = false;
    let mut close = version != "HTTP/1.1";
    let mut retry_after = None;
    let mut head_bytes = line.len();
    loop {
        line.clear();
        let n = conn
            .read_line(&mut line)
            .await
            .map_err(|e| fail(e, first))?;
        head_bytes += n;
        if n == 0 || head_bytes > MAX_HEAD {
            return Err(fail(io::ErrorKind::InvalidData.into(), first));
        }
        let text = line.trim_end();
        if text.is_empty() {
            break;
        }
        let Some((name, value)) = text.split_once(':') else {
            continue;
        };
        let value = value.trim();
        match name.to_ascii_lowercase().as_str() {
            "content-length" => length = value.parse().ok(),
            "transfer-encoding" => chunked = value.eq_ignore_ascii_case("chunked"),
            "connection" => close |= value.eq_ignore_ascii_case("close"),
            "retry-after" => retry_after = Some(value.to_string()),
            _ => {}
        }
    }
    let too_big = || io::Error::new(io::ErrorKind::InvalidData, "the reply is too large");
    let body = if chunked {
        let mut body = Vec::new();
        loop {
            line.clear();
            conn.read_line(&mut line)
                .await
                .map_err(|e| fail(e, first))?;
            let size = usize::from_str_radix(line.trim().split(';').next().unwrap_or(""), 16)
                .map_err(|_| {
                    fail(
                        io::Error::new(io::ErrorKind::InvalidData, "bad chunk"),
                        first,
                    )
                })?;
            if size == 0 {
                // Trailers end with an empty line.
                loop {
                    line.clear();
                    let n = conn
                        .read_line(&mut line)
                        .await
                        .map_err(|e| fail(e, first))?;
                    if n == 0 || line.trim_end().is_empty() {
                        break;
                    }
                }
                break;
            }
            if body.len() + size > MAX_REPLY {
                return Err(fail(too_big(), first));
            }
            let start = body.len();
            body.resize(start + size + 2, 0);
            conn.read_exact(&mut body[start..])
                .await
                .map_err(|e| fail(e, first))?;
            body.truncate(start + size);
        }
        body
    } else if let Some(len) = length {
        if len > MAX_REPLY {
            return Err(fail(too_big(), first));
        }
        let mut body = vec![0u8; len];
        conn.read_exact(&mut body)
            .await
            .map_err(|e| fail(e, first))?;
        body
    } else {
        close = true;
        let mut body = Vec::new();
        (&mut *conn)
            .take(MAX_REPLY as u64 + 1)
            .read_to_end(&mut body)
            .await
            .map_err(|e| fail(e, first))?;
        if body.len() > MAX_REPLY {
            return Err(fail(too_big(), first));
        }
        body
    };
    Ok((
        Reply {
            status,
            retry_after,
            body,
        },
        !close,
    ))
}

/// The error for a reply of status 400 or above.
fn refusal(reply: &Reply) -> io::Error {
    let code = serde_json::from_slice::<serde_json::Value>(&reply.body)
        .ok()
        .and_then(|v| v.get("error").and_then(|e| e.as_str().map(str::to_string)))
        .unwrap_or_else(|| "unknown".into());
    let kind = match reply.status {
        429 => io::ErrorKind::WouldBlock,
        413 => io::ErrorKind::InvalidData,
        401 | 403 => io::ErrorKind::PermissionDenied,
        _ => io::ErrorKind::Other,
    };
    let retry = reply
        .retry_after
        .as_ref()
        .map_or(String::new(), |s| format!(", retry in {s} s"));
    io::Error::new(
        kind,
        format!(
            "the signalling service refused the request: {code} (HTTP {}{retry})",
            reply.status
        ),
    )
}

impl Rendezvous for SignalRendezvous {
    async fn send(&self, to: NodeId, msg: Vec<u8>) -> io::Result<()> {
        if msg.len() > MAX_MESSAGE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "a message of {} bytes is over the {MAX_MESSAGE} byte cap of the signalling service",
                    msg.len()
                ),
            ));
        }
        let path = format!("/send/{to}");
        let reply = self
            .inner
            .request("POST", &path, &msg, Duration::ZERO)
            .await?;
        if reply.status >= 400 {
            return Err(refusal(&reply));
        }
        Ok(())
    }

    async fn recv(&self) -> io::Result<(NodeId, Vec<u8>)> {
        let mut state = self.inner.state.lock().await;
        loop {
            if let Some(next) = state.queue.pop_front() {
                return Ok(next);
            }
            // `after` acknowledges everything the caller already has, which deletes it.
            let path = format!(
                "/recv/{}?after={}&wait={POLL_S}",
                self.inner.own, state.cursor
            );
            let reply = self
                .inner
                .request("GET", &path, b"", Duration::from_secs(POLL_S))
                .await?;
            if reply.status >= 400 {
                return Err(refusal(&reply));
            }
            let bad = |why: &str| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("the signalling service sent a bad reply: {why}"),
                )
            };
            let value: serde_json::Value =
                serde_json::from_slice(&reply.body).map_err(|e| bad(&e.to_string()))?;
            let messages = value
                .get("messages")
                .and_then(|m| m.as_array())
                .ok_or_else(|| bad("no messages list"))?;
            let mut items: Vec<(u64, NodeId, Vec<u8>)> = Vec::new();
            for m in messages {
                let seq = m
                    .get("seq")
                    .and_then(|s| s.as_u64())
                    .ok_or_else(|| bad("no seq"))?;
                let from = m.get("from").and_then(|s| s.as_str()).unwrap_or("");
                let Ok(from) = from.parse::<NodeId>() else {
                    continue;
                };
                let body = m.get("body").and_then(|s| s.as_str()).unwrap_or("");
                let Ok(body) = base64::engine::general_purpose::STANDARD.decode(body) else {
                    continue;
                };
                items.push((seq, from, body));
            }
            items.sort_by_key(|(seq, _, _)| *seq);
            for (seq, from, body) in items {
                if seq > state.cursor {
                    state.cursor = seq;
                    state.queue.push_back((from, body));
                }
            }
        }
    }
}
