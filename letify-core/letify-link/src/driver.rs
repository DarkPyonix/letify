//! The punch driver: a whole hole punch over a [`Rendezvous`], as one public call. It owns the
//! classification, the offer and answer exchange, the timing, the TCP and UDP attempts, the
//! policing check, the one-line report and the choice of the first session path. It does not
//! own the transport of the rendezvous messages, which the embedder supplies, nor the identity
//! policy, which the [`Config`] and the [`Acceptor`] carry. Spec "Punch driver".

use std::future::Future;
use std::io;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::{TcpStream, UdpSocket};
use tokio::sync::mpsc;
use tokio::task::JoinSet;

use crate::acceptor::{Accepted, Acceptor};
use crate::config::Config;
use crate::identity::{Identity, NodeId};
use crate::observe::AddressObserver;
use crate::paths::BoxTransport;
use crate::plan;
use crate::policing::{self, Verdict};
use crate::punch::{self, Plan};
use crate::rendezvous::{Message, Proto, Punch, PunchExt, Rendezvous, recv_message, send_message};
use crate::session::Session;
use crate::stun::{NatType, StunObserver, classify_with};
use crate::udp::UdpPath;

/// The default time between the offer and the first attempt, in milliseconds.
pub const DEFAULT_LEAD_MS: u64 = 2000;
/// The default length of the attempt window, in milliseconds.
pub const DEFAULT_WINDOW_MS: u64 = 20_000;
/// The default length of a policing burst, in milliseconds.
pub const DEFAULT_BURST_MS: u64 = 500;
/// How long each address observer has to answer one query.
const OBSERVE_WAIT: Duration = Duration::from_millis(1500);
/// How long the initiator waits for the second answer after the first, and the responder for
/// the second offer. Both messages leave together, so they arrive together.
const SECOND_MESSAGE_GRACE: Duration = Duration::from_millis(400);
/// How long the responder waits for the offer beyond `lead + window`.
const OFFER_PATIENCE: Duration = Duration::from_secs(60);

/// The name of a NAT type in a report and in the sidecar's replies.
pub fn nat_name(nat: NatType) -> String {
    match nat {
        NatType::Unknown => "unknown".into(),
        NatType::EndpointIndependent => "endpoint_independent".into(),
        NatType::Symmetric { step } => format!("symmetric:{step}"),
    }
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

    /// `e` with the report appended to its message and its facts attached.
    fn explain(&self, mut e: PunchError) -> PunchError {
        let text = self.text();
        e.message = format!("{}; {text}", e.message);
        e.facts = Some(Box::new(Facts {
            nat: self.here.as_ref().ok().map(|n| nat_name(*n)),
            peer_nat: self.there.as_ref().map(|t| nat_name(t.nat)),
            nat_error: self.here.as_ref().err().cloned(),
            report: text,
        }));
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
) -> Result<Own, PunchError> {
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

async fn await_answers<R: Rendezvous>(
    rdv: &R,
    peer: NodeId,
    tcp: Option<&Punch>,
    udp: Option<&Punch>,
    wait: Duration,
) -> Result<Answers, PunchError> {
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

async fn await_offers<R: Rendezvous>(
    rdv: &R,
    peer: NodeId,
    want_tcp: bool,
    want_udp: bool,
    wait: Duration,
) -> Result<Offers, PunchError> {
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

fn random_token() -> [u8; 16] {
    let mut t = [0u8; 16];
    getrandom::getrandom(&mut t).expect("the operating system gave no random bytes");
    t
}

/// The report clause for UDP paths that failed their QUIC connect, from the `name: reason`
/// entries of the failed vets. Empty when none failed. Spec "NAT traversal" item 8.
pub fn udp_clause(failed: &[String]) -> String {
    match failed.first() {
        Some(first) => {
            let why = first.split_once(": ").map_or(first.as_str(), |(_, w)| w);
            format!("; udp: did not open: {why}")
        }
        None => String::new(),
    }
}

/// Which side of the punch this node is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// Offers, then dials and chooses the path.
    Initiator,
    /// Waits for the offer, answers it and accepts what arrives.
    Responder,
}

/// The initiator's check that a punched path does not throttle bulk data, see [`policing`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Policing {
    /// The slowest rate, in bytes per second, a path may deliver in either direction.
    pub floor_bps: u64,
    /// How long the bulk transfer of the check lasts.
    pub burst: Duration,
}

impl Policing {
    /// A check with `floor_bps` and the default burst of 500 ms.
    pub fn new(floor_bps: u64) -> Policing {
        Policing {
            floor_bps,
            burst: Duration::from_millis(DEFAULT_BURST_MS),
        }
    }
}

/// What one punch does. The defaults are the sidecar's.
#[derive(Clone, Debug)]
pub struct PunchOptions {
    /// TCP sockets to punch with, at most [`punch::MAX_PUNCH_SOCKETS`]. Default
    /// [`punch::PUNCH_SOCKETS`].
    pub tcp_sockets: usize,
    /// Probing UDP sockets, at most [`punch::MAX_PUNCH_SOCKETS`]; a punch yields at most one UDP
    /// path. Default 0.
    pub udp_sockets: usize,
    /// The address reported as the mapping of every socket when `observers` is `None`. Default
    /// `127.0.0.1`.
    pub public_ip: IpAddr,
    /// The resolved address observers that classify the NAT, or `None` to offer the bound ports
    /// on `public_ip`. Default `None`.
    pub observers: Option<Vec<SocketAddr>>,
    /// The initiator's policing check. Default `None`.
    pub police: Option<Policing>,
    /// The time between the offer and the first attempt. Default 2 s.
    pub lead: Duration,
    /// The length of the attempt window. Default 20 s.
    pub window: Duration,
    /// Responder only: when set, the start this side names for its answers is now plus this
    /// margin, if that is earlier than the offered start. Default `None`.
    pub answer_margin: Option<Duration>,
}

impl Default for PunchOptions {
    fn default() -> Self {
        PunchOptions {
            tcp_sockets: punch::PUNCH_SOCKETS,
            udp_sockets: 0,
            public_ip: [127, 0, 0, 1].into(),
            observers: None,
            police: None,
            lead: Duration::from_millis(DEFAULT_LEAD_MS),
            window: Duration::from_millis(DEFAULT_WINDOW_MS),
            answer_margin: None,
        }
    }
}

/// What this node is for [`punch_session`], with what that role needs.
pub enum Endpoint<'a> {
    /// Offer, dial, and connect a session over the first usable path.
    Initiator {
        /// This node's identity.
        identity: &'a Identity,
        /// The session configuration.
        config: Config,
    },
    /// Wait for the offer and hand every punched path to the acceptor.
    Responder {
        /// Routes the punched paths to a new session or to a live one.
        acceptor: Arc<Acceptor>,
    },
}

/// Whether a failure is the caller's request or the network.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PunchErrorKind {
    /// The request cannot be run as given. Never retry it unchanged.
    Usage,
    /// The network or the peer did not allow the punch. It may be retried.
    Infrastructure,
}

#[derive(Clone, Debug)]
struct Facts {
    nat: Option<String>,
    peer_nat: Option<String>,
    nat_error: Option<String>,
    report: String,
}

/// A punch failed. Display prints the message, which ends with the one-line report when this
/// side's NAT was classified. Spec "NAT traversal" item 8.
#[derive(Clone, Debug)]
pub struct PunchError {
    kind: PunchErrorKind,
    message: String,
    facts: Option<Box<Facts>>,
}

impl PunchError {
    fn new(kind: PunchErrorKind, message: impl Into<String>) -> PunchError {
        PunchError {
            kind,
            message: message.into(),
            facts: None,
        }
    }

    /// Whether the request or the network failed.
    pub fn kind(&self) -> PunchErrorKind {
        self.kind
    }

    /// The whole message, with the report appended after `; ` when there is one.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// The one-line report alone, when this side's NAT was classified or its failure noted.
    pub fn report(&self) -> Option<&str> {
        self.facts.as_ref().map(|f| f.report.as_str())
    }

    /// This side's NAT type as the report names it, `None` when it could not be classified.
    pub fn nat(&self) -> Option<&str> {
        self.facts.as_ref().and_then(|f| f.nat.as_deref())
    }

    /// The other side's NAT type, `None` when it never answered.
    pub fn peer_nat(&self) -> Option<&str> {
        self.facts.as_ref().and_then(|f| f.peer_nat.as_deref())
    }

    /// Why this side's NAT could not be classified.
    pub fn nat_error(&self) -> Option<&str> {
        self.facts.as_ref().and_then(|f| f.nat_error.as_deref())
    }
}

impl std::fmt::Display for PunchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for PunchError {}

fn usage(msg: impl Into<String>) -> PunchError {
    PunchError::new(PunchErrorKind::Usage, msg)
}

fn infra(msg: impl Into<String>) -> PunchError {
    PunchError::new(PunchErrorKind::Infrastructure, msg)
}

/// What a punch learned about the two sides.
#[derive(Clone, Debug)]
pub struct PunchInfo {
    /// This side's NAT.
    pub nat: NatType,
    /// The other side's NAT, `Unknown` when it did not say.
    pub peer_nat: NatType,
    /// The TCP addresses this side offered.
    pub mapped_tcp: Vec<SocketAddr>,
    /// The UDP addresses this side offered.
    pub mapped_udp: Vec<SocketAddr>,
    /// The one-line report of "NAT traversal" item 8.
    pub report: String,
}

impl PunchInfo {
    fn explain(&self, mut e: PunchError) -> PunchError {
        e.message = format!("{}; {}", e.message, self.report);
        e.facts = Some(Box::new(Facts {
            nat: Some(nat_name(self.nat)),
            peer_nat: Some(nat_name(self.peer_nat)),
            nat_error: None,
            report: self.report.clone(),
        }));
        e
    }
}

/// One punched path.
pub enum PunchedPath {
    /// A TCP connection made by simultaneous open.
    Tcp(TcpStream),
    /// A QUIC connection over a punched UDP socket.
    Udp(UdpPath),
}

impl PunchedPath {
    /// Whether this is a UDP path.
    pub fn is_udp(&self) -> bool {
        matches!(self, PunchedPath::Udp(_))
    }

    /// The path as a boxed [`crate::Transport`], which keeps a UDP path's congestion gauge.
    pub fn into_transport(self) -> BoxTransport {
        match self {
            PunchedPath::Tcp(s) => Box::new(s),
            PunchedPath::Udp(p) => Box::new(p),
        }
    }
}

/// The UDP result of an attempt: every socket, the index of the one that heard the peer, the
/// peer address it heard, and the plan.
type UdpFound = (Vec<UdpSocket>, usize, SocketAddr, Plan);

/// The sockets of a punch that connected, before any session is made over them.
pub struct Punched {
    role: Role,
    tcp: Vec<TcpStream>,
    udp: Option<UdpFound>,
    police: Option<Policing>,
    policed_by_offer: bool,
    info: PunchInfo,
}

/// A path the policing check dropped.
#[derive(Debug)]
pub struct PolicedPath {
    /// The path name.
    pub name: String,
    /// `tcp` or `udp`.
    pub proto: &'static str,
    /// What the check measured.
    pub rates: policing::Rates,
}

/// A path that failed its check or its QUIC connect.
#[derive(Debug)]
pub struct FailedPath {
    /// The path name, empty when the vetting task itself failed.
    pub name: String,
    /// `tcp` or `udp`, empty when the vetting task itself failed.
    pub proto: &'static str,
    /// Why it failed.
    pub why: String,
}

impl FailedPath {
    /// `name: why`, or `why` alone when there is no name.
    pub fn describe(&self) -> String {
        if self.name.is_empty() {
            self.why.clone()
        } else {
            format!("{}: {}", self.name, self.why)
        }
    }
}

/// The initiator's paths after the policing check.
pub struct Vetting {
    /// The usable paths with their names, sorted by name.
    pub usable: Vec<(String, PunchedPath)>,
    /// The paths the check dropped, in the order they finished.
    pub policed: Vec<PolicedPath>,
    /// The paths that failed, in the order they finished.
    pub failed: Vec<FailedPath>,
    /// What the punch learned.
    pub info: PunchInfo,
}

impl Vetting {
    /// The report of the punch, with the clause for a UDP path that did not open.
    pub fn report(&self) -> String {
        let udp: Vec<String> = self
            .failed
            .iter()
            .filter(|f| f.proto == "udp")
            .map(FailedPath::describe)
            .collect();
        format!("{}{}", self.info.report, udp_clause(&udp))
    }

    /// The message for the case where no path is usable.
    pub fn dropped_message(&self) -> String {
        let mut policed: Vec<&str> = self.policed.iter().map(|p| p.name.as_str()).collect();
        policed.sort();
        let failed: Vec<String> = self.failed.iter().map(FailedPath::describe).collect();
        format!(
            "every punched path was dropped: {} policed ({}), {} failed its check ({})",
            policed.len(),
            policed.join(", "),
            failed.len(),
            failed.join("; ")
        )
    }
}

/// The result of checking one punched path on the initiator.
enum Vetted {
    Usable(String, PunchedPath),
    Policed(String, &'static str, policing::Rates),
    Failed(String, &'static str, String),
}

async fn vet<S>(
    name: String,
    proto: &'static str,
    mut stream: S,
    police: Option<Policing>,
    wrap: impl FnOnce(S) -> PunchedPath,
) -> Vetted
where
    S: AsyncRead + AsyncWrite + Unpin + Send,
{
    if let Some(p) = police {
        match policing::check(&mut stream, p.floor_bps, p.burst).await {
            Ok(Verdict::Usable(_)) => {}
            Ok(Verdict::Policed(rates)) => return Vetted::Policed(name, proto, rates),
            Err(e) => return Vetted::Failed(name, proto, e.to_string()),
        }
    }
    Vetted::Usable(name, wrap(stream))
}

/// Run the policing check as the responder. `None` when the check failed.
async fn serve_checked(mut path: PunchedPath, police: bool) -> Option<PunchedPath> {
    if police {
        let served = match &mut path {
            PunchedPath::Tcp(s) => policing::serve(s).await,
            PunchedPath::Udp(p) => policing::serve(p).await,
        };
        served.ok()?;
    }
    Some(path)
}

impl Punched {
    /// What the punch learned.
    pub fn info(&self) -> &PunchInfo {
        &self.info
    }

    /// The number of paths that connected.
    pub fn paths(&self) -> usize {
        self.tcp.len() + usize::from(self.udp.is_some())
    }

    /// Initiator: check every path, in parallel, and name the paths `<prefix>-<i>` for TCP and
    /// `<prefix>-u` for UDP. A usage error for a responder.
    pub async fn vet(self, prefix: &str) -> Result<Vetting, PunchError> {
        if self.role != Role::Initiator {
            return Err(usage("only the initiator vets punched paths"));
        }
        let police = self.police;
        let mut set = JoinSet::new();
        for (i, s) in self.tcp.into_iter().enumerate() {
            let name = format!("{prefix}-{i}");
            set.spawn(vet(name, "tcp", s, police, PunchedPath::Tcp));
        }
        if let Some((mut socks, index, from, _)) = self.udp {
            let socket = socks.swap_remove(index);
            drop(socks);
            let name = format!("{prefix}-u");
            set.spawn(async move {
                let std = match socket.into_std() {
                    Ok(s) => s,
                    Err(e) => return Vetted::Failed(name, "udp", e.to_string()),
                };
                match UdpPath::connect(std, from).await {
                    Ok(path) => vet(name, "udp", path, police, PunchedPath::Udp).await,
                    Err(e) => Vetted::Failed(name, "udp", e.to_string()),
                }
            });
        }
        let mut out = Vetting {
            usable: Vec::new(),
            policed: Vec::new(),
            failed: Vec::new(),
            info: self.info,
        };
        while let Some(done) = set.join_next().await {
            match done {
                Ok(Vetted::Usable(name, path)) => out.usable.push((name, path)),
                Ok(Vetted::Policed(name, proto, rates)) => {
                    out.policed.push(PolicedPath { name, proto, rates })
                }
                Ok(Vetted::Failed(name, proto, why)) => {
                    out.failed.push(FailedPath { name, proto, why })
                }
                Err(e) => out.failed.push(FailedPath {
                    name: String::new(),
                    proto: "",
                    why: e.to_string(),
                }),
            }
        }
        out.usable.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(out)
    }

    /// Responder: for each path, serve the policing check when the offer asked for it, then call
    /// `handler` with the path in a task of its own. Returns how many paths were punched. A
    /// usage error for an initiator. Must be called inside a tokio runtime.
    pub fn accept_with<F, Fut>(self, handler: F) -> Result<usize, PunchError>
    where
        F: Fn(PunchedPath) -> Fut + Clone + Send + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        if self.role != Role::Responder {
            return Err(usage("only the responder accepts punched paths"));
        }
        let count = self.paths();
        let police = self.policed_by_offer;
        for s in self.tcp {
            let h = handler.clone();
            tokio::spawn(async move {
                if let Some(path) = serve_checked(PunchedPath::Tcp(s), police).await {
                    h(path).await;
                }
            });
        }
        if let Some((socks, _, _, plan)) = self.udp {
            let h = handler.clone();
            tokio::spawn(async move {
                if let Ok(path) = punch::accept_udp(socks, plan).await
                    && let Some(path) = serve_checked(PunchedPath::Udp(path), police).await
                {
                    h(path).await;
                }
            });
        }
        Ok(count)
    }
}

/// An established session and what the punch learned.
pub struct PunchedSession {
    /// The session. Its first subflow is the first usable punched path.
    pub session: Session,
    /// What the punch learned, with the report line.
    pub info: PunchInfo,
}

/// Run a whole punch over `rdv` and return an established session.
///
/// The initiator offers, dials and connects the session over the first usable path, adding the
/// others as subflows. The responder answers and returns the first session its acceptor makes
/// from the punched paths. A failure carries the report of "NAT traversal" item 8. There is no
/// fallback to a relay: the caller decides what to do with the error. Dropping the future
/// cancels the punch.
///
/// ```no_run
/// use std::sync::Arc;
/// use letify_link::driver::{Endpoint, PunchOptions, punch_session};
/// use letify_link::rendezvous::Rendezvous;
/// use letify_link::{Acceptor, Config, Identity, NodeId};
///
/// async fn connect<R: Rendezvous>(
///     my_control_channel: &R,
///     me: &Identity,
///     peer: NodeId,
/// ) -> Result<(), Box<dyn std::error::Error>> {
///     let options = PunchOptions {
///         tcp_sockets: 4,
///         udp_sockets: 1,
///         observers: None,
///         ..PunchOptions::default()
///     };
///     let endpoint = Endpoint::Initiator { identity: me, config: Config::default() };
///     let got = punch_session(my_control_channel, peer, endpoint, &options).await?;
///     println!("{}", got.info.report);
///     let _stream = got.session.open_stream().await?;
///     Ok(())
/// }
///
/// async fn serve<R: Rendezvous>(
///     my_control_channel: &R,
///     acceptor: Arc<Acceptor>,
///     peer: NodeId,
/// ) -> Result<(), Box<dyn std::error::Error>> {
///     let endpoint = Endpoint::Responder { acceptor };
///     let got = punch_session(my_control_channel, peer, endpoint, &PunchOptions::default()).await?;
///     let _stream = got.session.accept_stream().await?;
///     Ok(())
/// }
/// ```
pub async fn punch_session<R: Rendezvous>(
    rdv: &R,
    peer: NodeId,
    endpoint: Endpoint<'_>,
    opts: &PunchOptions,
) -> Result<PunchedSession, PunchError> {
    match endpoint {
        Endpoint::Initiator { identity, config } => {
            let punched = punch(rdv, peer, Role::Initiator, opts).await?;
            let vetting = punched.vet("punch").await?;
            let info = PunchInfo {
                report: vetting.report(),
                ..vetting.info.clone()
            };
            if vetting.usable.is_empty() {
                return Err(info.explain(infra(vetting.dropped_message())));
            }
            let mut session: Option<Session> = None;
            let mut last = String::new();
            for (name, path) in vetting.usable {
                match &session {
                    None => {
                        let made = match path {
                            PunchedPath::Tcp(s) => {
                                Session::connect(s, identity, peer, config.clone()).await
                            }
                            PunchedPath::Udp(p) => {
                                Session::connect(p, identity, peer, config.clone()).await
                            }
                        };
                        match made {
                            Ok(s) => session = Some(s),
                            Err(e) => last = format!("{name}: {e}"),
                        }
                    }
                    Some(s) => {
                        // A path that cannot join is not worth failing a session that is up.
                        let _ = match path {
                            PunchedPath::Tcp(t) => s.add_subflow(t).await,
                            PunchedPath::Udp(p) => s.add_subflow(p).await,
                        };
                    }
                }
            }
            match session {
                Some(session) => Ok(PunchedSession { session, info }),
                None => Err(info.explain(infra(format!(
                    "the session did not open over a punched path: {last}"
                )))),
            }
        }
        Endpoint::Responder { acceptor } => {
            let punched = punch(rdv, peer, Role::Responder, opts).await?;
            let info = punched.info().clone();
            let (tx, mut rx) = mpsc::unbounded_channel();
            punched.accept_with(move |path| {
                let (acceptor, tx) = (acceptor.clone(), tx.clone());
                async move {
                    if let Ok(Accepted::Session(s)) = acceptor.accept(path.into_transport()).await {
                        let _ = tx.send(s);
                    }
                }
            })?;
            match tokio::time::timeout(opts.window, rx.recv()).await {
                Ok(Some(session)) => Ok(PunchedSession { session, info }),
                _ => Err(info.explain(infra("no session opened over the punched paths"))),
            }
        }
    }
}

/// Run a punch over `rdv` as `role` and return the sockets that connected, with no session
/// made over them. The same exchange, timing and attempts as [`punch_session`].
pub async fn punch<R: Rendezvous>(
    rdv: &R,
    peer: NodeId,
    role: Role,
    opts: &PunchOptions,
) -> Result<Punched, PunchError> {
    let initiator = role == Role::Initiator;
    let tcp_n = opts.tcp_sockets.min(punch::MAX_PUNCH_SOCKETS);
    let udp_n = opts.udp_sockets.min(punch::MAX_PUNCH_SOCKETS);
    if tcp_n + udp_n == 0 {
        return Err(usage("a punch needs sockets or udp above 0"));
    }
    let police = opts.police;
    let lead = opts.lead.as_millis() as u64;
    let window = opts.window.as_millis() as u64;

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
    let own = match own_mappings(
        opts.observers.as_deref(),
        opts.public_ip,
        &ports,
        &udp_socks,
        probe,
    )
    .await
    {
        Ok(own) => own,
        Err(e) => {
            let report = Report {
                here: Err(e.message.clone()),
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
            tcp_terms = Some((a.mappings.clone(), o.token, Plan::of(o).with_answer(a)));
        }
        if let (Some(o), Some((a, _))) = (&udp_offer, &answers.udp) {
            udp_terms = Some((a.mappings.clone(), o.token, Plan::of(o).with_answer(a)));
        }
    } else {
        let offers = await_offers(
            rdv,
            peer,
            tcp_n > 0,
            udp_n > 0,
            Duration::from_millis(lead + window) + OFFER_PATIENCE,
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
        // The start this side names, once, so TCP and UDP answers agree.
        let named_start = opts
            .answer_margin
            .map(|m| punch::now_ms() + m.as_millis() as u64);
        for (terms, maps, slot) in [
            (&offers.tcp, &own.tcp_maps, &mut tcp_terms),
            (&offers.udp, &own.udp_maps, &mut udp_terms),
        ] {
            let Some((o, e)) = terms else { continue };
            let mut answer = Punch {
                proto: o.proto,
                token: o.token,
                start_at_ms: o.start_at_ms,
                until_ms: o.until_ms,
                mappings: maps.clone(),
            };
            if let Some(start) = named_start.filter(|s| *s < o.start_at_ms) {
                answer.until_ms = start + (o.until_ms - o.start_at_ms);
                answer.start_at_ms = start;
            }
            let plan = Plan::of(&answer);
            send_message(rdv, peer, &Message::PunchAnswerExt(answer, ext(e.police)))
                .await
                .map_err(|e| infra(e.to_string()))?;
            *slot = Some((o.mappings.clone(), o.token, plan));
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
    Ok(Punched {
        role,
        tcp: tcp_streams,
        udp: udp_found,
        police,
        policed_by_offer,
        info: PunchInfo {
            nat: own.nat,
            peer_nat,
            mapped_tcp: own.tcp_maps.clone(),
            mapped_udp: own.udp_maps.clone(),
            report: report.text(),
        },
    })
}
