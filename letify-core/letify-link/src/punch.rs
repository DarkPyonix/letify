//! Hole punching: TCP simultaneous open over several sockets at once, UDP probing, port
//! prediction and the many-port birthday attempt. It owns sockets while they punch and the
//! arithmetic that chooses what to try. It does not own the exchange of mappings and timing
//! (the [`rendezvous`](crate::rendezvous)), the STUN queries ([`stun`](crate::stun)) or what a
//! punched socket carries afterwards (a subflow, a UDP path).

use std::collections::HashMap;
use std::io;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpSocket, TcpStream, UdpSocket};
use tokio::task::JoinSet;
use tokio::time::{Instant, sleep, sleep_until, timeout_at};

use crate::rendezvous::Punch;
use crate::stun::NatType;
use crate::tcp;

/// Sockets a TCP punch opens by default, as letify's Python punch does.
pub const PUNCH_SOCKETS: usize = 4;
/// The most sockets one TCP punch opens.
pub const MAX_PUNCH_SOCKETS: usize = 8;
/// Ports a symmetric NAT is predicted to use next, by default.
pub const PREDICT_WINDOW: usize = 16;
/// Ports of the many-port birthday attempt, on each side.
pub const BIRTHDAY_PORTS: usize = 256;
/// Datagrams per second one UDP punch sends in total. The burst of a many-port attempt is
/// spread over time at this rate so a router is not flooded.
pub const SPRAY_PER_SECOND: u32 = 500;
/// Pause between two TCP dials from one socket after a failed one.
const REDIAL: Duration = Duration::from_millis(200);
/// Sent first on a TCP punch connection by the initiator, followed by the pair token.
const HELLO: &[u8] = b"LETIFY-PUNCH1";
const UDP_PROBE: &[u8; 10] = b"LTFY-PROBE";
const UDP_ACK: &[u8; 10] = b"LTFY-ACK\0\0";
/// First port a prediction or a birthday attempt uses; below it are privileged ports.
const LOW_PORT: u16 = 1024;

/// When a punch starts and stops, in unix milliseconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    /// Both sides begin dialing or probing at this time.
    pub start_at_ms: u64,
    /// Both sides stop at this time.
    pub until_ms: u64,
}

impl Plan {
    /// The plan of an offer or answer as received.
    pub fn of(punch: &Punch) -> Plan {
        Plan {
            start_at_ms: punch.start_at_ms,
            until_ms: punch.until_ms,
        }
    }

    /// Merge the sender's plan with the peer's reported stop: both sides stop at the earlier
    /// `until`, so neither spends a window the other has already closed.
    pub fn with_peer_until(self, peer_until_ms: u64) -> Plan {
        Plan {
            until_ms: self.until_ms.min(peer_until_ms),
            ..self
        }
    }
}

/// Unix milliseconds now.
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// The token of socket `index` of a punch, so a hello cannot connect the wrong pair.
pub fn pair_token(token: &[u8; 16], index: u8) -> Vec<u8> {
    let mut out = token.to_vec();
    out.push(index);
    out
}

/// A TCP socket bound to `local` with `SO_REUSEADDR` and, where it exists, `SO_REUSEPORT`, so
/// one port can hold a listener and a dialer at the same time.
///
/// Windows has no `SO_REUSEPORT`. There `SO_REUSEADDR` alone lets sockets share a port, and
/// `SO_EXCLUSIVEADDRUSE` is left off because it would refuse the second bind. A crossing SYN may
/// then complete on the dialer or on the listener, and the punch takes whichever completes.
pub fn reusable_socket(local: SocketAddr) -> io::Result<TcpSocket> {
    let sock = if local.is_ipv4() {
        TcpSocket::new_v4()?
    } else {
        TcpSocket::new_v6()?
    };
    // On Windows this is the whole of the sharing, see the doc comment above.
    sock.set_reuseaddr(true)?;
    #[cfg(all(unix, not(any(target_os = "solaris", target_os = "illumos"))))]
    sock.set_reuseport(true)?;
    sock.bind(local)?;
    Ok(sock)
}

enum Event {
    Dialed(io::Result<TcpStream>),
    Hello(Option<TcpStream>),
}

fn deadline_of(plan: &Plan) -> io::Result<Instant> {
    let left = plan.until_ms.saturating_sub(now_ms());
    if left == 0 {
        return Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "the punch window had already closed, so the other side has stopped dialing",
        ));
    }
    Ok(Instant::now() + Duration::from_millis(left))
}

async fn wait_for_start(plan: &Plan) {
    let wait = plan.start_at_ms.saturating_sub(now_ms());
    if wait > 0 {
        sleep(Duration::from_millis(wait)).await;
    }
}

/// Connect to `peer` from `local` while listening on `local`, and agree on one connection.
///
/// The initiator takes the first connection that completes and writes the hello with `token`.
/// The other side keeps the connection that the hello arrives on. The socket is tuned with
/// [`tcp::tune`]. It fails with `TimedOut` when no connection is made before `plan.until_ms`.
pub async fn punch_tcp(
    local: SocketAddr,
    peer: SocketAddr,
    token: &[u8],
    initiator: bool,
    plan: Plan,
) -> io::Result<TcpStream> {
    wait_for_start(&plan).await;
    let deadline = deadline_of(&plan)?;
    let listener: TcpListener = reusable_socket(local)?.listen(8)?;
    let expected: Vec<u8> = [HELLO, token].concat();
    let mut tasks: JoinSet<Event> = JoinSet::new();
    let mut dialing = false;
    let mut attempts = 0usize;
    let mut retry_at = Instant::now();
    // How each dial ended, so the failure names the cause: a refused dial and an unanswered
    // one have different remedies.
    let mut outcomes: HashMap<String, usize> = HashMap::new();
    loop {
        if !dialing && Instant::now() >= retry_at {
            dialing = true;
            attempts += 1;
            tasks.spawn(async move {
                Event::Dialed(match reusable_socket(local) {
                    Ok(sock) => sock.connect(peer).await,
                    Err(e) => Err(e),
                })
            });
        }
        let mut candidate = None;
        tokio::select! {
            _ = sleep_until(deadline) => break,
            accepted = listener.accept() => {
                if let Ok((stream, _)) = accepted {
                    candidate = Some(stream);
                }
            }
            Some(done) = tasks.join_next() => match done {
                Ok(Event::Dialed(Ok(stream))) => {
                    dialing = false;
                    candidate = Some(stream);
                }
                Ok(Event::Dialed(Err(e))) => {
                    dialing = false;
                    *outcomes.entry(format!("{:?}", e.kind())).or_default() += 1;
                    retry_at = Instant::now() + REDIAL;
                }
                Ok(Event::Hello(Some(stream))) => {
                    tcp::tune(&stream)?;
                    return Ok(stream);
                }
                _ => {}
            },
            _ = sleep_until(retry_at), if !dialing => {}
        }
        if let Some(mut stream) = candidate {
            if initiator {
                stream.write_all(&expected).await?;
                tcp::tune(&stream)?;
                return Ok(stream);
            }
            let want = expected.clone();
            tasks.spawn(async move {
                let mut seen = vec![0u8; want.len()];
                let read = timeout_at(deadline, stream.read_exact(&mut seen)).await;
                Event::Hello((matches!(read, Ok(Ok(_))) && seen == want).then_some(stream))
            });
        }
    }
    let mut causes: Vec<_> = outcomes
        .into_iter()
        .map(|(k, v)| format!("{v} x {k}"))
        .collect();
    // A dial still in flight has no outcome yet. Windows holds a dial to a closed port for
    // seconds before it fails, so the window can close first.
    if dialing {
        causes.push(format!("{attempts} x still dialing"));
    }
    causes.sort();
    Err(io::Error::new(
        io::ErrorKind::TimedOut,
        format!(
            "no connection with {peer} before the punch window closed; dials ended with {}",
            if causes.is_empty() {
                "nothing (still unanswered)".to_string()
            } else {
                causes.join(", ")
            }
        ),
    ))
}

/// Punch pair `i` from `locals[i]` to `peers[i]` at once, with the token of pair `i`, and
/// return the connections that were made in pair order. It fails only when no pair connected,
/// with the last failure. The streams are ready to become subflows.
pub async fn punch_tcp_many(
    locals: &[SocketAddr],
    peers: &[SocketAddr],
    token: &[u8; 16],
    initiator: bool,
    plan: Plan,
) -> io::Result<Vec<TcpStream>> {
    let pairs = locals.len().min(peers.len()).min(MAX_PUNCH_SOCKETS);
    let mut set = JoinSet::new();
    for i in 0..pairs {
        let (local, peer, token) = (locals[i], peers[i], pair_token(token, i as u8));
        set.spawn(async move { (i, punch_tcp(local, peer, &token, initiator, plan).await) });
    }
    let mut made = Vec::new();
    let mut last = io::Error::new(io::ErrorKind::InvalidInput, "no socket pairs to punch");
    while let Some(done) = set.join_next().await {
        match done {
            Ok((i, Ok(stream))) => made.push((i, stream)),
            Ok((_, Err(e))) => last = e,
            Err(e) => last = io::Error::other(e),
        }
    }
    if made.is_empty() {
        return Err(last);
    }
    made.sort_by_key(|(i, _)| *i);
    Ok(made.into_iter().map(|(_, s)| s).collect())
}

/// Probe the peer from `sockets` until one datagram pair is exchanged, and return the socket
/// that did it with the peer address it heard from.
///
/// Every socket sends a probe with `token` to every address in `dests`, cycling through the
/// pairs at no more than `per_second` datagrams a second in total. A socket that receives a
/// probe answers it with an ack. The punch succeeds on the first ack, then sends two more acks
/// so the peer sees one too. The other sockets are dropped. A success means the path works in
/// both directions at that moment, and the session handshake over it is the final proof.
pub async fn punch_udp(
    sockets: Vec<UdpSocket>,
    dests: &[SocketAddr],
    token: &[u8; 16],
    plan: Plan,
    per_second: u32,
) -> io::Result<(UdpSocket, SocketAddr)> {
    if sockets.is_empty() || dests.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "nothing to punch with",
        ));
    }
    wait_for_start(&plan).await;
    let deadline = deadline_of(&plan)?;
    let sockets: Vec<Arc<UdpSocket>> = sockets.into_iter().map(Arc::new).collect();
    let (found_tx, mut found_rx) = tokio::sync::mpsc::unbounded_channel();
    let mut readers = JoinSet::new();
    for (i, socket) in sockets.iter().enumerate() {
        let (socket, found, token) = (socket.clone(), found_tx.clone(), *token);
        readers.spawn(async move {
            let mut buf = [0u8; 64];
            loop {
                let (n, from) = match socket.recv_from(&mut buf).await {
                    Ok(got) => got,
                    // Windows reports an ICMP port unreachable, drawn by an earlier probe to a
                    // closed port, as a failed receive on this socket. It says nothing about
                    // the datagrams still to come, so the reader goes on.
                    Err(e) if e.kind() == io::ErrorKind::ConnectionReset => continue,
                    Err(_) => return,
                };
                if n != 26 || buf[10..26] != token {
                    continue;
                }
                if buf[..10] == UDP_PROBE[..] {
                    let _ = socket.send_to(&[&UDP_ACK[..], &token].concat(), from).await;
                } else if buf[..10] == UDP_ACK[..] {
                    let _ = found.send((i, from));
                    return;
                }
            }
        });
    }
    drop(found_tx);
    let probe = [&UDP_PROBE[..], token].concat();
    let total = sockets.len() * dests.len();
    let tick = Duration::from_millis(50);
    let per_tick = (per_second as usize / 20).max(1);
    let mut cursor = 0usize;
    let hit = 'outer: loop {
        for _ in 0..per_tick.min(total) {
            let (s, d) = (cursor / dests.len(), cursor % dests.len());
            cursor = (cursor + 1) % total;
            let _ = sockets[s].send_to(&probe, dests[d]).await;
        }
        tokio::select! {
            _ = sleep_until(deadline) => break 'outer None,
            hit = found_rx.recv() => break 'outer hit,
            _ = sleep(tick) => {}
        }
    };
    let Some((index, from)) = hit else {
        readers.shutdown().await;
        return Err(io::Error::new(
            io::ErrorKind::TimedOut,
            format!(
                "no UDP answer from {} addresses before the punch window closed",
                dests.len()
            ),
        ));
    };
    let ack = [&UDP_ACK[..], token].concat();
    for _ in 0..2 {
        let _ = sockets[index].send_to(&ack, from).await;
    }
    readers.shutdown().await;
    let chosen = sockets
        .into_iter()
        .nth(index)
        .and_then(|arc| Arc::try_unwrap(arc).ok())
        .ok_or_else(|| io::Error::other("the punched socket is still shared"))?;
    Ok((chosen, from))
}

/// The ports a symmetric NAT is predicted to map to next: `last + step * k` for k in
/// `1..=window`, stopping at the end of the port range. A step of 0 means no pattern, and the
/// result is empty.
pub fn predict_ports(last: u16, step: i16, window: usize) -> Vec<u16> {
    if step == 0 {
        return Vec::new();
    }
    (1..=window as i64)
        .map(|k| last as i64 + step as i64 * k)
        .take_while(|p| (LOW_PORT as i64..=u16::MAX as i64).contains(p))
        .map(|p| p as u16)
        .collect()
}

/// The addresses to send to when the peer is symmetric: its last observed mapping advanced by
/// its step. A peer that is not symmetric, or whose step is 0, gives its single mapping.
pub fn predicted_destinations(last: SocketAddr, nat: NatType, window: usize) -> Vec<SocketAddr> {
    match nat {
        NatType::Symmetric { step } if step != 0 => predict_ports(last.port(), step, window)
            .into_iter()
            .map(|p| SocketAddr::new(last.ip(), p))
            .collect(),
        _ => vec![last],
    }
}

/// `count` distinct random ports from `1024..=65535`, none of them in `exclude`.
pub fn random_ports(count: usize, exclude: &[u16]) -> Vec<u16> {
    let count = count.min(BIRTHDAY_PORTS * 4);
    let mut out: Vec<u16> = Vec::with_capacity(count);
    let mut raw = [0u8; 2];
    while out.len() < count {
        getrandom::getrandom(&mut raw).expect("the operating system has a random source");
        let port = u16::from_be_bytes(raw);
        if port >= LOW_PORT && !out.contains(&port) && !exclude.contains(&port) {
            out.push(port);
        }
    }
    out
}

/// Destinations for the many-port attempt: `count` random ports on `ip`.
pub fn birthday_destinations(ip: IpAddr, count: usize) -> Vec<SocketAddr> {
    random_ports(count, &[])
        .into_iter()
        .map(|p| SocketAddr::new(ip, p))
        .collect()
}

/// Open `count` UDP sockets on `ip`, each on a port the system chooses, for the many-port
/// attempt. Each one makes its own mapping in a symmetric NAT.
pub async fn bind_birthday_sockets(ip: IpAddr, count: usize) -> io::Result<Vec<UdpSocket>> {
    let mut out = Vec::with_capacity(count);
    for _ in 0..count.min(BIRTHDAY_PORTS) {
        out.push(UdpSocket::bind(SocketAddr::new(ip, 0)).await?);
    }
    Ok(out)
}

/// The chance that at least one of `dests` random destination ports hits one of `mappings`
/// open mappings on the other side, when ports are uniform over the 64512 ports from 1024:
/// `1 - (1 - mappings / 64512)^dests`. With 256 and 256 it is about 0.64.
pub fn birthday_hit_probability(mappings: usize, dests: usize) -> f64 {
    let ports = (u16::MAX as usize - LOW_PORT as usize + 1) as f64;
    1.0 - (1.0 - mappings as f64 / ports).powi(dests as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_listener_and_a_dialer_can_share_one_local_port_on_every_platform() {
        let first = reusable_socket("127.0.0.1:0".parse().unwrap()).unwrap();
        let addr = first.local_addr().unwrap();
        let _listener = first.listen(8).unwrap();
        // The second socket binds the port the listener holds, as the punch does to dial out.
        let second = reusable_socket(addr).unwrap();
        assert_eq!(second.local_addr().unwrap(), addr);
    }

    fn plan_in(start: u64, window: u64) -> Plan {
        let now = now_ms();
        Plan {
            start_at_ms: now + start,
            until_ms: now + start + window,
        }
    }

    #[test]
    fn prediction_advances_the_last_port_by_the_step() {
        assert_eq!(predict_ports(40000, 2, 4), vec![40002, 40004, 40006, 40008]);
        assert_eq!(predict_ports(40000, -3, 2), vec![39997, 39994]);
    }

    #[test]
    fn prediction_stops_at_the_end_of_the_port_range_and_skips_a_zero_step() {
        assert_eq!(predict_ports(65530, 2, 5), vec![65532, 65534]);
        assert_eq!(predict_ports(1030, -3, 5), vec![1027, 1024]);
        assert!(predict_ports(40000, 0, 8).is_empty());
    }

    #[test]
    fn a_symmetric_peer_with_a_step_is_aimed_at_a_window_of_ports() {
        let last: SocketAddr = "203.0.113.7:50000".parse().unwrap();
        let dests = predicted_destinations(last, NatType::Symmetric { step: 1 }, 3);
        let ports: Vec<u16> = dests.iter().map(|a| a.port()).collect();
        assert_eq!(ports, vec![50001, 50002, 50003]);
        assert!(dests.iter().all(|a| a.ip() == last.ip()));
        assert_eq!(
            predicted_destinations(last, NatType::EndpointIndependent, 3),
            vec![last]
        );
        assert_eq!(
            predicted_destinations(last, NatType::Symmetric { step: 0 }, 3),
            vec![last]
        );
    }

    #[test]
    fn random_ports_are_distinct_unprivileged_and_avoid_the_excluded() {
        let ports = random_ports(BIRTHDAY_PORTS, &[5000]);
        assert_eq!(ports.len(), BIRTHDAY_PORTS);
        let mut sorted = ports.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), BIRTHDAY_PORTS);
        assert!(ports.iter().all(|p| *p >= 1024 && *p != 5000));
    }

    #[test]
    fn the_birthday_attempt_hits_about_two_times_in_three_with_256_by_256() {
        let p = birthday_hit_probability(256, 256);
        assert!((0.62..0.66).contains(&p), "{p}");
        assert!(birthday_hit_probability(1, 16) < 0.001);
    }

    #[test]
    fn pair_tokens_differ_by_the_index_byte() {
        let t = [1u8; 16];
        assert_eq!(pair_token(&t, 2).len(), 17);
        assert_ne!(pair_token(&t, 0), pair_token(&t, 1));
    }

    #[tokio::test]
    async fn a_udp_punch_that_nobody_answers_times_out() {
        let sock = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let silent = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let err = punch_udp(
            vec![sock],
            &[silent.local_addr().unwrap()],
            &[0; 16],
            plan_in(0, 300),
            SPRAY_PER_SECOND,
        )
        .await
        .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::TimedOut);
    }
}
