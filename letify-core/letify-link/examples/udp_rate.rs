//! Measures what one punched UDP pair delivers in each direction, for three arms: raw paced
//! datagrams (`raw`), a letify-link UDP path with its default handshake (`quic`), and the same
//! path presenting as HTTP/3 (`h3`). It is a measurement tool for the experiment branch
//! `feat/link-udp-h3`, not part of the product.
//!
//! The two ends are the same program. Each binds a UDP socket, learns its public mapping from
//! STUN and writes it to `<dir>/map.txt`, then waits for `<dir>/peer.txt`, which holds
//! `<peer ip:port> <start unix ms>`. Both punch from that start time, run the arm in phase 1
//! (the `init` role sends) and phase 2 (the `resp` role sends), and write `<dir>/result.json`.
//!
//! Run with `cargo run --release -p letify-link --example udp_rate -- <arm> <init|resp> <dir>`.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use letify_link::punch::{Plan, punch_udp, punch_udp_all};
use letify_link::stun;
use letify_link::udp::{UdpConfig, UdpListener, UdpPath};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UdpSocket;
use tokio::time::{Instant, sleep, sleep_until, timeout, timeout_at};

const TOKEN: [u8; 16] = *b"udp-rate-token-1";
const PUNCH_MS: u64 = 10_000;
const PING_MS: u64 = 3_000;
const WINDOW_S: u64 = 6;
const GAP_S: u64 = 2;
const PACKET: usize = 1200;

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

/// The instant at which the unix time `ms` occurs.
fn at(ms: u64) -> Instant {
    let now = now_ms();
    if ms > now {
        Instant::now() + Duration::from_millis(ms - now)
    } else {
        Instant::now()
    }
}

fn config(arm: &str) -> UdpConfig {
    match arm {
        "h3" => UdpConfig {
            alpn: b"h3".to_vec(),
            server_name: "cdnjs.cloudflare.com".into(),
            ..UdpConfig::default()
        },
        _ => UdpConfig::default(),
    }
}

/// Bytes received in each second of a phase.
struct Series {
    per_second: Vec<u64>,
}

impl Series {
    fn new() -> Self {
        Series {
            per_second: vec![0; WINDOW_S as usize + 1],
        }
    }

    fn add(&mut self, start: Instant, bytes: usize) {
        let s = Instant::now().saturating_duration_since(start).as_secs() as usize;
        if let Some(slot) = self.per_second.get_mut(s) {
            *slot += bytes as u64;
        }
    }

    fn json(&self) -> String {
        let list: Vec<String> = self.per_second.iter().map(|b| b.to_string()).collect();
        format!("[{}]", list.join(","))
    }
}

/// Receive datagrams tagged `RAW1` until `end`.
async fn raw_receive(socket: &UdpSocket, start: Instant, end: Instant) -> (Series, u64, u64) {
    let mut series = Series::new();
    let (mut packets, mut highest) = (0u64, 0u64);
    let mut buf = vec![0u8; 2048];
    while let Ok(Ok(n)) = timeout_at(end, socket.recv(&mut buf)).await {
        if n >= 12 && &buf[..4] == b"RAW1" {
            series.add(start, n);
            packets += 1;
            highest = highest.max(u64::from_le_bytes(buf[4..12].try_into().unwrap()) + 1);
        }
    }
    (series, packets, highest)
}

/// Send `PACKET` byte datagrams at `rate` bytes a second until `end`. Returns the count sent.
async fn raw_send(socket: &UdpSocket, rate: f64, start: Instant, end: Instant) -> u64 {
    let mut buf = vec![0u8; PACKET];
    buf[..4].copy_from_slice(b"RAW1");
    let mut sent = 0u64;
    while Instant::now() < end {
        let due = (start.elapsed().as_secs_f64() * rate / PACKET as f64) as u64;
        let mut burst = 0;
        while sent < due && burst < 64 {
            buf[4..12].copy_from_slice(&sent.to_le_bytes());
            let _ = socket.send(&buf).await;
            sent += 1;
            burst += 1;
        }
        sleep(Duration::from_millis(1)).await;
    }
    sent
}

async fn raw_arm(
    socket: UdpSocket,
    init: bool,
    peer: SocketAddr,
    ts: u64,
    rate: f64,
) -> Vec<String> {
    socket.connect(peer).await.unwrap();
    let mut out = Vec::new();
    // Round trip: the initiator pings, the responder echoes.
    let ping_end = at(ts + PING_MS);
    let mut rtts = Vec::new();
    sleep_until(at(ts)).await;
    let mut buf = vec![0u8; 2048];
    if init {
        let mut n = 0u64;
        while Instant::now() < ping_end {
            let sent = Instant::now();
            let mut msg = b"PING".to_vec();
            msg.extend_from_slice(&n.to_le_bytes());
            let _ = socket.send(&msg).await;
            if let Ok(Ok(k)) = timeout(Duration::from_millis(500), socket.recv(&mut buf)).await
                && &buf[..4] == b"PONG"
                && k == msg.len()
            {
                rtts.push(sent.elapsed().as_secs_f64() * 1000.0);
            }
            n += 1;
            sleep(Duration::from_millis(100)).await;
        }
    } else {
        while let Ok(Ok(n)) = timeout_at(ping_end, socket.recv(&mut buf)).await {
            if &buf[..4] == b"PING" {
                let mut reply = buf[..n].to_vec();
                reply[..4].copy_from_slice(b"PONG");
                let _ = socket.send(&reply).await;
            }
        }
    }
    rtts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    out.push(format!(
        "\"rtt_ms_median\":{:.1},\"rtt_samples\":{}",
        rtts.get(rtts.len() / 2).copied().unwrap_or(-1.0),
        rtts.len()
    ));
    let t1 = ts + PING_MS + 1000;
    let t2 = t1 + (WINDOW_S + GAP_S) * 1000;
    for (phase, start_ms) in [(1, t1), (2, t2)] {
        let start = at(start_ms);
        sleep_until(start).await;
        let end = at(start_ms + WINDOW_S * 1000);
        let sends = (phase == 1) == init;
        if sends {
            let sent = raw_send(&socket, rate, start, end).await;
            out.push(format!("\"phase{phase}_sent_packets\":{sent}"));
        } else {
            let tail = at(start_ms + WINDOW_S * 1000 + 800);
            let (series, packets, highest) = raw_receive(&socket, start, tail).await;
            out.push(format!(
                "\"phase{phase}_recv_bytes_per_second\":{},\"phase{phase}_recv_packets\":{packets},\"phase{phase}_highest_seq\":{highest}",
                series.json()
            ));
        }
    }
    out
}

async fn quic_arm(mut path: UdpPath, init: bool, ts: u64) -> Vec<String> {
    let mut out = Vec::new();
    let seen = path.handshake();
    out.push(format!(
        "\"alpn\":\"{}\",\"server_name\":\"{}\"",
        String::from_utf8_lossy(seen.alpn.as_deref().unwrap_or(b"")),
        seen.server_name.unwrap_or_default()
    ));
    let t1 = ts + PING_MS + 1000;
    let t2 = t1 + (WINDOW_S + GAP_S) * 1000;
    let chunk = vec![0x5au8; 256 << 10];
    for (phase, start_ms) in [(1, t1), (2, t2)] {
        let start = at(start_ms);
        sleep_until(start).await;
        let end = at(start_ms + WINDOW_S * 1000);
        if (phase == 1) == init {
            let mut sent = 0u64;
            while Instant::now() < end {
                match timeout_at(end, path.write_all(&chunk)).await {
                    Ok(Ok(())) => sent += chunk.len() as u64,
                    _ => break,
                }
            }
            out.push(format!("\"phase{phase}_written_bytes\":{sent}"));
        } else {
            let mut series = Series::new();
            let mut buf = vec![0u8; 1 << 20];
            let tail = at(start_ms + WINDOW_S * 1000 + 800);
            while let Ok(Ok(n)) = timeout_at(tail, path.read(&mut buf)).await {
                if n == 0 {
                    break;
                }
                series.add(start, n);
            }
            out.push(format!(
                "\"phase{phase}_recv_bytes_per_second\":{}",
                series.json()
            ));
        }
    }
    // Keep the path open until the peer's last phase has drained.
    sleep(Duration::from_millis(500)).await;
    out
}

async fn run(arm: &str, init: bool, dir: &Path, rate: f64) -> Result<Vec<String>, String> {
    let err = |e: std::io::Error| e.to_string();
    let socket = UdpSocket::bind("0.0.0.0:0").await.map_err(err)?;
    let servers = stun::resolve(&stun::DEFAULT_SERVERS).await;
    let server = *servers.first().ok_or("no STUN server resolved")?;
    let mapped = stun::binding(&socket, server, Duration::from_secs(3))
        .await
        .map_err(err)?;
    std::fs::write(dir.join("map.txt"), mapped.to_string()).map_err(err)?;
    let peer_file = dir.join("peer.txt");
    let waited = Instant::now();
    let text = loop {
        if let Ok(t) = std::fs::read_to_string(&peer_file)
            && t.trim().split(' ').count() == 2
        {
            break t;
        }
        if waited.elapsed() > Duration::from_secs(120) {
            return Err("no peer.txt".into());
        }
        sleep(Duration::from_millis(100)).await;
    };
    let mut parts = text.trim().split(' ');
    let peer: SocketAddr = parts.next().unwrap().parse().map_err(|_| "bad peer")?;
    let t0: u64 = parts.next().unwrap().parse().map_err(|_| "bad start")?;
    let plan = Plan {
        start_at_ms: t0,
        until_ms: t0 + PUNCH_MS,
    };
    let ts = t0 + PUNCH_MS + 1500;
    let cfg = config(arm);
    let mut out = vec![format!("\"mapped\":\"{mapped}\",\"peer\":\"{peer}\"")];
    let (socket, from) = if init {
        punch_udp(vec![socket], &[peer], &TOKEN, plan, 200)
            .await
            .map_err(err)?
    } else {
        let (mut all, i, from) = punch_udp_all(vec![socket], &[peer], &TOKEN, plan, 200)
            .await
            .map_err(err)?;
        (all.swap_remove(i), from)
    };
    if arm == "raw" {
        out.extend(raw_arm(socket, init, from, ts, rate).await);
    } else if init {
        let std = socket.into_std().map_err(err)?;
        let path = UdpPath::connect_with(std, from, &cfg).await.map_err(err)?;
        out.extend(quic_arm(path, true, ts).await);
    } else {
        let std = socket.into_std().map_err(err)?;
        let mut listener = UdpListener::new(std, Some(from), &cfg).map_err(err)?;
        let path = timeout(Duration::from_secs(20), listener.accept())
            .await
            .map_err(|_| "no QUIC handshake arrived".to_string())?
            .map_err(err)?;
        out.extend(quic_arm(path, false, ts).await);
    }
    Ok(out)
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: udp_rate <raw|quic|h3> <init|resp> <dir> [rate MiB/s]");
        std::process::exit(2);
    }
    let dir = PathBuf::from(&args[3]);
    let rate = args
        .get(4)
        .and_then(|r| r.parse::<f64>().ok())
        .unwrap_or(4.0)
        * 1048576.0;
    let body = match run(&args[1], args[2] == "init", &dir, rate).await {
        Ok(fields) => format!("{{\"arm\":\"{}\",{}}}", args[1], fields.join(",")),
        Err(e) => format!("{{\"arm\":\"{}\",\"error\":{:?}}}", args[1], e),
    };
    let _ = std::fs::write(dir.join("result.json"), body);
    std::process::exit(0);
}
