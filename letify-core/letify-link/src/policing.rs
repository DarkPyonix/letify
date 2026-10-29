//! Policing detection: a burst check on a freshly connected subflow, before the session uses
//! it. It owns the check protocol (a client side and a responder side) and the verdict. It does
//! not own the floor, which the caller supplies, nor the decision to reject the path.
//!
//! The client runs 30 round trips, then a bulk burst up and a bulk burst down, each for the
//! burst time. Bulk data moves in frames of [`CHUNK`] bytes with at most [`WINDOW`] bytes
//! unacknowledged, so a path that delivers little is measured within a bounded wait instead of
//! after its whole buffer drains. A stream that was reported as policed is not reusable.

use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, split};
use tokio::sync::Semaphore;
use tokio::time::{Instant, timeout};

/// Round trips measured before the bulk burst.
pub const ROUND_TRIPS: usize = 30;
/// Default time of each bulk burst.
pub const BURST: Duration = Duration::from_secs(1);
/// Bytes per bulk frame.
pub const CHUNK: usize = 16 * 1024;
/// Most bulk bytes sent without an acknowledgement.
pub const WINDOW: usize = 4 * CHUNK;
/// Longest wait for one round trip or one acknowledgement.
const STEP_TIMEOUT: Duration = Duration::from_secs(5);

const OP_PING: u8 = b'P';
const OP_UP: u8 = b'U';
const OP_DOWN: u8 = b'D';
const ACK: u8 = b'A';
const END: u8 = b'E';

/// What the check measured.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rates {
    /// Median round trip over the 30 pings.
    pub rtt: Duration,
    /// Delivered upload rate in bytes per second.
    pub up_bps: f64,
    /// Delivered download rate in bytes per second.
    pub down_bps: f64,
}

/// The verdict of the check.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Verdict {
    /// Neither direction fell under a twentieth of the floor. Whether the rates meet the floor
    /// is the caller's decision.
    Usable(Rates),
    /// The round trips answered but a direction delivered under a twentieth of the floor: the
    /// network let the flow open and drops its bulk packets.
    Policed(Rates),
}

impl Verdict {
    /// The measured rates.
    pub fn rates(&self) -> &Rates {
        match self {
            Verdict::Usable(r) | Verdict::Policed(r) => r,
        }
    }
}

/// Run the check as the client over `stream`; the peer runs [`serve`]. `floor_bps` is the
/// caller's minimum throughput in bytes per second and `burst` the time of each burst.
pub async fn check<S>(stream: S, floor_bps: u64, burst: Duration) -> io::Result<Verdict>
where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    let (mut r, mut w) = split(stream);
    let mut rtts = Vec::with_capacity(ROUND_TRIPS);
    for i in 0..ROUND_TRIPS as u64 {
        let began = Instant::now();
        w.write_all(&[&[OP_PING][..], &i.to_be_bytes()].concat())
            .await?;
        let mut echo = [0u8; 9];
        step(r.read_exact(&mut echo)).await?;
        if echo[0] != OP_PING || echo[1..] != i.to_be_bytes() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "bad echo in the policing check",
            ));
        }
        rtts.push(began.elapsed());
    }
    rtts.sort();
    let rtt = rtts[rtts.len() / 2];

    w.write_all(&[OP_UP]).await?;
    let up_bps = send_burst(&mut r, &mut w, burst).await?;

    w.write_all(&[OP_DOWN]).await?;
    w.write_all(&(burst.as_millis() as u32).to_be_bytes())
        .await?;
    let down_bps = receive_burst(&mut r, &mut w).await?;

    let rates = Rates {
        rtt,
        up_bps,
        down_bps,
    };
    let limit = floor_bps as f64 / 20.0;
    Ok(if up_bps < limit || down_bps < limit {
        Verdict::Policed(rates)
    } else {
        Verdict::Usable(rates)
    })
}

/// Answer the check as the responder until the client has finished it. Returns once the
/// download burst is done, or with an error when the peer leaves early.
pub async fn serve<S>(stream: S) -> io::Result<()>
where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    let (mut r, mut w) = split(stream);
    loop {
        match r.read_u8().await? {
            OP_PING => {
                let mut id = [0u8; 8];
                r.read_exact(&mut id).await?;
                w.write_all(&[&[OP_PING][..], &id].concat()).await?;
            }
            OP_UP => {
                receive_burst(&mut r, &mut w).await?;
            }
            OP_DOWN => {
                let ms = r.read_u32().await?;
                send_burst(&mut r, &mut w, Duration::from_millis(ms as u64)).await?;
                return Ok(());
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "unknown policing op",
                ));
            }
        }
    }
}

async fn step<T>(fut: impl std::future::Future<Output = io::Result<T>>) -> io::Result<T> {
    timeout(STEP_TIMEOUT, fut).await.unwrap_or_else(|_| {
        Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "policing check step timed out",
        ))
    })
}

/// Send frames for `burst`, then a zero frame, and return the delivered bytes per second: the
/// bytes the peer acknowledged divided by the time until it confirmed the end.
async fn send_burst<R, W>(r: &mut R, w: &mut W, burst: Duration) -> io::Result<f64>
where
    R: AsyncRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send,
{
    let began = Instant::now();
    let credits = Arc::new(Semaphore::new(WINDOW / CHUNK));
    let acked = Arc::new(AtomicU64::new(0));
    let frame = {
        let mut f = (CHUNK as u32).to_be_bytes().to_vec();
        f.resize(4 + CHUNK, 0xA5);
        f
    };
    let write_side = async {
        while began.elapsed() < burst {
            match timeout(burst, credits.acquire()).await {
                Ok(Ok(permit)) => permit.forget(),
                _ => break,
            }
            w.write_all(&frame).await?;
        }
        w.write_all(&0u32.to_be_bytes()).await
    };
    let read_side = async {
        loop {
            match step(r.read_u8()).await? {
                ACK => {
                    acked.fetch_add(CHUNK as u64, Ordering::SeqCst);
                    credits.add_permits(1);
                }
                END => return Ok::<_, io::Error>(()),
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "bad ack in the policing check",
                    ));
                }
            }
        }
    };
    let (written, read) = tokio::join!(write_side, read_side);
    written?;
    read?;
    Ok(acked.load(Ordering::SeqCst) as f64 / began.elapsed().as_secs_f64())
}

/// Receive frames until the zero frame, acknowledging each, then confirm the end. Returns the
/// received bytes per second from the call to the zero frame.
async fn receive_burst<R, W>(r: &mut R, w: &mut W) -> io::Result<f64>
where
    R: AsyncRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send,
{
    let began = Instant::now();
    let mut total = 0u64;
    let mut sink = vec![0u8; CHUNK];
    loop {
        let len = step(r.read_u32()).await? as usize;
        if len == 0 {
            break;
        }
        if len > CHUNK {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "oversized policing frame",
            ));
        }
        step(r.read_exact(&mut sink[..len])).await?;
        total += len as u64;
        w.write_all(&[ACK]).await?;
    }
    w.write_all(&[END]).await?;
    Ok(total as f64 / began.elapsed().as_secs_f64())
}
