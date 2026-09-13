//! The connection to the agent, and the batching that makes forwarding viable.
//!
//! Most driver calls do not need an answer. A launch, a copy to the device or an
//! allocation changes state and returns immediately, so the local driver queues it and moves on.
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

use letify_wire::{
    HostCopy, Reply, Request, decode_reply, encode_request, read_copy_to_host, read_frame,
    write_copy_to_device, write_frame,
};

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

    /// Queue a copy to the device, writing the caller's bytes without copying them.
    ///
    /// Counted and flushed exactly like [`Client::send`].
    pub fn send_copy_to_device(&mut self, handle: u64, offset: u64, payload: &[u8]) -> io::Result<()> {
        write_copy_to_device(&mut self.writer, handle, offset, payload)?;
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

    /// Ask for a copy to the host and put the answer into the caller's `destination`.
    ///
    /// A round trip like [`Client::request`]: the queue is flushed first.
    pub fn request_copy_to_host(
        &mut self,
        handle: u64,
        offset: u64,
        destination: &mut [u8],
    ) -> io::Result<HostCopy> {
        let bytes = destination.len() as u64;
        write_frame(&mut self.writer, &encode_request(&Request::CopyToHost { handle, offset, bytes }))?;
        self.sent += 1;
        self.queued = 0;
        self.writer.flush()?;
        self.round_trips += 1;
        read_copy_to_host(&mut self.reader, destination)
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
    fn a_copy_to_the_device_reaches_the_agent_whole_over_tcp() {
        use std::io::BufReader;
        use std::net::TcpListener;

        use letify_wire::{
            Incoming, PROTOCOL_VERSION, Reply, decode_request, encode_reply, read_frame,
            read_incoming, write_frame,
        };

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap().to_string();
        let payload: Vec<u8> = (0..300_000usize).map(|index| (index % 251) as u8).collect();
        let agent = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let hello = decode_request(&read_frame(&mut reader).unwrap()).unwrap();
            assert!(matches!(hello, Request::Hello { .. }));
            let ready = Reply::Ready {
                version: PROTOCOL_VERSION,
                device_name: "loopback".into(),
                compute: (0, 0),
            };
            write_frame(&mut stream, &encode_reply(&ready)).unwrap();
            let mut staging = Vec::new();
            let incoming = read_incoming(&mut reader, &mut staging).unwrap();
            (incoming, staging)
        });

        let mut client = super::Client::connect(&address).unwrap();
        client.send_copy_to_device(5, 32, &payload).unwrap();
        client.flush().unwrap();
        let (incoming, staging) = agent.join().unwrap();

        assert_eq!(incoming, Incoming::CopyToDevice { handle: 5, offset: 32, bytes: payload.len() });
        assert_eq!(&staging[..payload.len()], &payload[..]);
        assert_eq!(client.sent, 2);
    }

    #[test]
    fn a_copy_to_the_host_fills_the_callers_buffer_over_tcp() {
        use std::io::BufReader;
        use std::net::TcpListener;

        use letify_wire::{
            HostCopy, Incoming, PROTOCOL_VERSION, Reply, encode_reply, read_incoming,
            write_frame, write_payload_reply,
        };

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap().to_string();
        let payload: Vec<u8> = (0..300_000usize).map(|index| (index % 251) as u8).collect();
        let staged = payload.clone();
        let agent = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut staging = Vec::new();
            let hello = read_incoming(&mut reader, &mut staging).unwrap();
            assert!(matches!(hello, Incoming::Request(Request::Hello { .. })));
            let ready = Reply::Ready {
                version: PROTOCOL_VERSION,
                device_name: "loopback".into(),
                compute: (0, 0),
            };
            write_frame(&mut stream, &encode_reply(&ready)).unwrap();

            let first = read_incoming(&mut reader, &mut staging).unwrap();
            write_payload_reply(&mut stream, &staged).unwrap();
            let second = read_incoming(&mut reader, &mut staging).unwrap();
            let failed = Reply::Failed { code: 700, message: "illegal address".into() };
            write_frame(&mut stream, &encode_reply(&failed)).unwrap();
            (first, second)
        });

        let mut client = super::Client::connect(&address).unwrap();
        client.send(Request::Free { handle: 9 }).unwrap();
        let mut destination = vec![0u8; payload.len()];
        let filled = client.request_copy_to_host(5, 32, &mut destination).unwrap();
        let mut small = [0u8; 8];
        let refused = client.request_copy_to_host(6, 0, &mut small).unwrap();
        let (first, second) = agent.join().unwrap();

        assert_eq!(filled, HostCopy::Filled);
        assert_eq!(destination, payload);
        assert!(matches!(refused, HostCopy::Reply(Reply::Failed { code: 700, .. })));
        // The queued Free was flushed ahead of the copy, so the agent read it first.
        assert_eq!(first, Incoming::Request(Request::Free { handle: 9 }));
        assert_eq!(
            second,
            Incoming::Request(Request::CopyToHost { handle: 5, offset: 32, bytes: 300_000 })
        );
        assert_eq!(client.sent, 4);
        assert_eq!(client.round_trips, 3);
    }

    #[test]
    fn the_address_has_a_default() {
        assert!(super::agent_address().contains(':'));
    }
}
