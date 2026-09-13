//! The connection to the agent, and the batching that makes forwarding viable.
//!
//! Most driver calls do not need an answer. A launch, a copy to the device or an
//! allocation changes state and returns immediately, so the shim queues it and moves on.
//! A call whose result the host reads has to wait, and when one arrives the queue is
//! flushed first so the agent sees everything in order.
//!
//! That is the whole reason the round trip count is the number of host synchronizations
//! rather than the number of calls. With a 0.5 s step and three synchronizations at a
//! 150 ms round trip, efficiency is `0.5 / (0.5 + 0.45)`, about 53 percent; reduce the
//! synchronizations to one per optimizer step and it is about 96 percent. Batching is
//! what makes the first number possible at all, since a step issues thousands of calls.

use std::io::{self, BufReader, BufWriter, Write};
use std::net::TcpStream;
use std::sync::Mutex;
use std::time::Duration;

use letify_wire::{Reply, Request, decode_reply, encode_request, read_frame, write_frame};

/// How many queued requests to hold before flushing anyway, so a long stretch of
/// asynchronous work does not grow without bound.
const QUEUE_LIMIT: usize = 512;

/// A connection to one agent.
pub struct Client {
    writer: BufWriter<TcpStream>,
    reader: BufReader<TcpStream>,
    queued: usize,
    /// Counts for the report the Python side reads back.
    pub sent: u64,
    pub round_trips: u64,
}

impl Client {
    /// Connect to an agent and check that it speaks the same protocol.
    pub fn connect(address: &str) -> io::Result<Self> {
        let stream = TcpStream::connect(address)?;
        stream.set_nodelay(true)?;
        stream.set_read_timeout(Some(Duration::from_secs(600)))?;
        let reader = BufReader::new(stream.try_clone()?);
        let mut client = Client {
            writer: BufWriter::new(stream),
            reader,
            queued: 0,
            sent: 0,
            round_trips: 0,
        };
        match client.request(Request::Hello { version: letify_wire::PROTOCOL_VERSION })? {
            Reply::Ready { version, .. } if version == letify_wire::PROTOCOL_VERSION => Ok(client),
            Reply::Ready { version, .. } => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "the agent speaks protocol {version} and this library speaks {}. Install \
                     matching builds on both sides.",
                    letify_wire::PROTOCOL_VERSION
                ),
            )),
            other => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("the agent answered {other:?} instead of announcing itself"),
            )),
        }
    }

    /// Queue a request that needs no answer.
    pub fn send(&mut self, request: Request) -> io::Result<()> {
        write_frame(&mut self.writer, &encode_request(&request))?;
        self.sent += 1;
        self.queued += 1;
        if self.queued >= QUEUE_LIMIT {
            self.flush()?;
        }
        Ok(())
    }

    /// Push everything queued to the agent without waiting for a reply.
    pub fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()?;
        self.queued = 0;
        Ok(())
    }

    /// Send a request and wait for its answer, flushing the queue first so ordering
    /// holds.
    pub fn request(&mut self, request: Request) -> io::Result<Reply> {
        write_frame(&mut self.writer, &encode_request(&request))?;
        self.sent += 1;
        self.queued = 0;
        self.writer.flush()?;
        self.round_trips += 1;
        let frame = read_frame(&mut self.reader)?;
        decode_reply(&frame)
    }

    /// Either queue or wait, according to the request itself.
    ///
    /// Callers use this so the batching rule lives in one place rather than at every
    /// intercepted symbol.
    pub fn dispatch(&mut self, request: Request) -> io::Result<Option<Reply>> {
        if request.needs_reply() {
            Ok(Some(self.request(request)?))
        } else {
            self.send(request)?;
            Ok(None)
        }
    }
}

/// The process wide connection.
///
/// One core serves one process and one device, so a mutex around a single client is
/// enough and keeps the ordering guarantees simple.
pub fn client() -> &'static Mutex<Option<Client>> {
    static CLIENT: std::sync::OnceLock<Mutex<Option<Client>>> = std::sync::OnceLock::new();
    CLIENT.get_or_init(|| Mutex::new(None))
}

/// The address the agent is listening on, from the environment.
///
/// letify sets this when it starts the runtime, so a user never types it.
pub fn agent_address() -> String {
    std::env::var("LETIFY_AGENT").unwrap_or_else(|_| "127.0.0.1:7654".to_string())
}

#[cfg(test)]
mod tests {
    use letify_wire::Request;

    #[test]
    fn the_batching_rule_comes_from_the_request() {
        // dispatch has no policy of its own, which is what keeps every intercepted
        // symbol from having to decide.
        assert!(!Request::Free { handle: 1 }.needs_reply());
        assert!(Request::MemoryInfo.needs_reply());
    }

    #[test]
    fn the_address_has_a_default() {
        assert!(super::agent_address().contains(':'));
    }
}
