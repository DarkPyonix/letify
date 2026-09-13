//! The protocol between letify-core and the agent.
//!
//! Every CUDA driver call letify-core intercepts becomes one message. The whole point of
//! the design is that most of them do not wait for a reply: a launch, a copy or an
//! allocation is recorded and sent, and only a call whose result the host actually
//! reads forces a round trip.
//!
//! That is why the efficiency of forwarding is `T / (T + k * RTT)` rather than
//! `T / (T + calls * RTT)`. `k` counts the synchronizing calls, not all of them.
//!
//! Framing is an eight byte little-endian length followed by that many bytes of payload.
//! The payload is a tag byte and then fixed-width fields, hand encoded rather than
//! going through a serialization crate, because this sits on the hot path and the shapes
//! are small and fixed.

use std::io::{self, IoSlice, Read, Write};

/// Bumped when a request or reply layout changes in a breaking way.
pub const PROTOCOL_VERSION: u32 = 2;

/// What the agent is being asked to do.
///
/// The split that matters is in [`Request::needs_reply`]: everything that does not need
/// one is batched, and that is what keeps the round trip count at the number of host
/// synchronizations instead of the number of calls.
#[derive(Debug, Clone, PartialEq)]
pub enum Request {
    /// Check that the agent speaks the same protocol and has a device.
    Hello { version: u32 },
    /// Read a device attribute, such as compute capability.
    DeviceAttribute { device: i32, attribute: i32 },
    /// Total and free memory on the device.
    MemoryInfo,
    /// Allocate device memory. letify-core hands back a virtual pointer immediately and
    /// reconciles it here, so an allocation does not cost a round trip.
    Allocate { bytes: u64, handle: u64 },
    /// Release device memory.
    Free { handle: u64 },
    /// Copy host memory to the device.
    CopyToDevice { handle: u64, offset: u64, payload: Vec<u8> },
    /// Copy device memory back to the host. Always needs a reply, because the host is
    /// about to read it.
    CopyToHost { handle: u64, offset: u64, bytes: u64 },
    /// Load a compiled module. Content addressed, so the same fatbin travels once.
    LoadModule { digest: [u8; 16], payload: Vec<u8> },
    /// Look up a kernel inside a loaded module.
    GetFunction { module: u64, name: String },
    /// Launch a kernel.
    LaunchKernel {
        function: u64,
        grid: [u32; 3],
        block: [u32; 3],
        shared_bytes: u32,
        stream: u64,
        params: Vec<u8>,
    },
    /// Wait for a stream to drain. This is a synchronization, so it pays a round trip.
    SynchronizeStream { stream: u64 },
    /// Record an event on a stream.
    RecordEvent { event: u64, stream: u64 },
    /// Wait for an event. Another synchronization.
    SynchronizeEvent { event: u64 },
    /// Elapsed milliseconds between two events, which the host reads.
    ElapsedTime { start: u64, end: u64 },
    /// Tell the agent to exit.
    Shutdown,
}

impl Request {
    /// Whether the caller has to wait for the agent's answer.
    ///
    /// A call that only changes device state is fire and forget. A call whose result the
    /// host reads cannot be, and every one of those is a round trip on the critical
    /// path.
    pub fn needs_reply(&self) -> bool {
        matches!(
            self,
            Request::Hello { .. }
                | Request::DeviceAttribute { .. }
                | Request::MemoryInfo
                | Request::CopyToHost { .. }
                | Request::GetFunction { .. }
                | Request::SynchronizeStream { .. }
                | Request::SynchronizeEvent { .. }
                | Request::ElapsedTime { .. }
        )
    }

    fn tag(&self) -> u8 {
        match self {
            Request::Hello { .. } => 1,
            Request::DeviceAttribute { .. } => 2,
            Request::MemoryInfo => 3,
            Request::Allocate { .. } => 4,
            Request::Free { .. } => 5,
            Request::CopyToDevice { .. } => 6,
            Request::CopyToHost { .. } => 7,
            Request::LoadModule { .. } => 8,
            Request::GetFunction { .. } => 9,
            Request::LaunchKernel { .. } => 10,
            Request::SynchronizeStream { .. } => 11,
            Request::RecordEvent { .. } => 12,
            Request::SynchronizeEvent { .. } => 13,
            Request::ElapsedTime { .. } => 14,
            Request::Shutdown => 15,
        }
    }
}

/// What the agent answers.
#[derive(Debug, Clone, PartialEq)]
pub enum Reply {
    /// The agent is ready, and this is the device it holds.
    Ready { version: u32, device_name: String, compute: (i32, i32) },
    /// One integer value, for an attribute query.
    Value { value: i64 },
    /// Two integers, for a memory query.
    Pair { first: u64, second: u64 },
    /// A handle the agent assigned, for a module or a kernel.
    Handle { handle: u64 },
    /// Device memory copied back to the host.
    Payload { payload: Vec<u8> },
    /// A floating point result, for elapsed time.
    Elapsed { milliseconds: f32 },
    /// The request succeeded and returns nothing.
    Done,
    /// The driver reported an error, carrying its code and message.
    Failed { code: i32, message: String },
}

impl Reply {
    fn tag(&self) -> u8 {
        match self {
            Reply::Ready { .. } => 1,
            Reply::Value { .. } => 2,
            Reply::Pair { .. } => 3,
            Reply::Handle { .. } => 4,
            Reply::Payload { .. } => 5,
            Reply::Elapsed { .. } => 6,
            Reply::Done => 7,
            Reply::Failed { .. } => 8,
        }
    }
}

// -- encoding -----------------------------------------------------------------

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_i32(out: &mut Vec<u8>, value: i32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_bytes(out: &mut Vec<u8>, value: &[u8]) {
    put_u64(out, value.len() as u64);
    out.extend_from_slice(value);
}

fn put_str(out: &mut Vec<u8>, value: &str) {
    put_bytes(out, value.as_bytes());
}

/// Encode a request, without its length prefix.
pub fn encode_request(request: &Request) -> Vec<u8> {
    let mut out = vec![request.tag()];
    match request {
        Request::Hello { version } => put_u32(&mut out, *version),
        Request::DeviceAttribute { device, attribute } => {
            put_i32(&mut out, *device);
            put_i32(&mut out, *attribute);
        }
        Request::MemoryInfo | Request::Shutdown => {}
        Request::Allocate { bytes, handle } => {
            put_u64(&mut out, *bytes);
            put_u64(&mut out, *handle);
        }
        Request::Free { handle } => put_u64(&mut out, *handle),
        Request::CopyToDevice { handle, offset, payload } => {
            put_u64(&mut out, *handle);
            put_u64(&mut out, *offset);
            put_bytes(&mut out, payload);
        }
        Request::CopyToHost { handle, offset, bytes } => {
            put_u64(&mut out, *handle);
            put_u64(&mut out, *offset);
            put_u64(&mut out, *bytes);
        }
        Request::LoadModule { digest, payload } => {
            out.extend_from_slice(digest);
            put_bytes(&mut out, payload);
        }
        Request::GetFunction { module, name } => {
            put_u64(&mut out, *module);
            put_str(&mut out, name);
        }
        Request::LaunchKernel { function, grid, block, shared_bytes, stream, params } => {
            put_u64(&mut out, *function);
            for value in grid {
                put_u32(&mut out, *value);
            }
            for value in block {
                put_u32(&mut out, *value);
            }
            put_u32(&mut out, *shared_bytes);
            put_u64(&mut out, *stream);
            put_bytes(&mut out, params);
        }
        Request::SynchronizeStream { stream } => put_u64(&mut out, *stream),
        Request::RecordEvent { event, stream } => {
            put_u64(&mut out, *event);
            put_u64(&mut out, *stream);
        }
        Request::SynchronizeEvent { event } => put_u64(&mut out, *event),
        Request::ElapsedTime { start, end } => {
            put_u64(&mut out, *start);
            put_u64(&mut out, *end);
        }
    }
    out
}

/// Encode a reply, without its length prefix.
pub fn encode_reply(reply: &Reply) -> Vec<u8> {
    let mut out = vec![reply.tag()];
    match reply {
        Reply::Ready { version, device_name, compute } => {
            put_u32(&mut out, *version);
            put_str(&mut out, device_name);
            put_i32(&mut out, compute.0);
            put_i32(&mut out, compute.1);
        }
        Reply::Value { value } => out.extend_from_slice(&value.to_le_bytes()),
        Reply::Pair { first, second } => {
            put_u64(&mut out, *first);
            put_u64(&mut out, *second);
        }
        Reply::Handle { handle } => put_u64(&mut out, *handle),
        Reply::Payload { payload } => put_bytes(&mut out, payload),
        Reply::Elapsed { milliseconds } => out.extend_from_slice(&milliseconds.to_le_bytes()),
        Reply::Done => {}
        Reply::Failed { code, message } => {
            put_i32(&mut out, *code);
            put_str(&mut out, message);
        }
    }
    out
}

// -- decoding -----------------------------------------------------------------

/// Reads fixed-width fields out of a frame, reporting a short frame rather than
/// panicking, because the far side may be a different version.
struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Cursor { bytes, at: 0 }
    }

    fn take(&mut self, count: usize) -> io::Result<&'a [u8]> {
        if self.at + count > self.bytes.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "frame is short"));
        }
        let slice = &self.bytes[self.at..self.at + count];
        self.at += count;
        Ok(slice)
    }

    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn i32(&mut self) -> io::Result<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn i64(&mut self) -> io::Result<i64> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn f32(&mut self) -> io::Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn bytes(&mut self) -> io::Result<Vec<u8>> {
        let length = self.u64()? as usize;
        Ok(self.take(length)?.to_vec())
    }

    fn string(&mut self) -> io::Result<String> {
        String::from_utf8(self.bytes()?)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "field is not UTF-8"))
    }

    fn triple(&mut self) -> io::Result<[u32; 3]> {
        Ok([self.u32()?, self.u32()?, self.u32()?])
    }
}

fn unknown_tag(tag: u8) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("unknown tag {tag}"))
}

/// Decode a request from a frame body.
pub fn decode_request(frame: &[u8]) -> io::Result<Request> {
    let (tag, body) = frame
        .split_first()
        .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "empty frame"))?;
    let mut cursor = Cursor::new(body);
    Ok(match tag {
        1 => Request::Hello { version: cursor.u32()? },
        2 => Request::DeviceAttribute { device: cursor.i32()?, attribute: cursor.i32()? },
        3 => Request::MemoryInfo,
        4 => Request::Allocate { bytes: cursor.u64()?, handle: cursor.u64()? },
        5 => Request::Free { handle: cursor.u64()? },
        6 => Request::CopyToDevice {
            handle: cursor.u64()?,
            offset: cursor.u64()?,
            payload: cursor.bytes()?,
        },
        7 => Request::CopyToHost {
            handle: cursor.u64()?,
            offset: cursor.u64()?,
            bytes: cursor.u64()?,
        },
        8 => Request::LoadModule {
            digest: cursor.take(16)?.try_into().unwrap(),
            payload: cursor.bytes()?,
        },
        9 => Request::GetFunction { module: cursor.u64()?, name: cursor.string()? },
        10 => Request::LaunchKernel {
            function: cursor.u64()?,
            grid: cursor.triple()?,
            block: cursor.triple()?,
            shared_bytes: cursor.u32()?,
            stream: cursor.u64()?,
            params: cursor.bytes()?,
        },
        11 => Request::SynchronizeStream { stream: cursor.u64()? },
        12 => Request::RecordEvent { event: cursor.u64()?, stream: cursor.u64()? },
        13 => Request::SynchronizeEvent { event: cursor.u64()? },
        14 => Request::ElapsedTime { start: cursor.u64()?, end: cursor.u64()? },
        15 => Request::Shutdown,
        other => return Err(unknown_tag(*other)),
    })
}

/// Decode a reply from a frame body.
pub fn decode_reply(frame: &[u8]) -> io::Result<Reply> {
    let (tag, body) = frame
        .split_first()
        .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "empty frame"))?;
    let mut cursor = Cursor::new(body);
    Ok(match tag {
        1 => Reply::Ready {
            version: cursor.u32()?,
            device_name: cursor.string()?,
            compute: (cursor.i32()?, cursor.i32()?),
        },
        2 => Reply::Value { value: cursor.i64()? },
        3 => Reply::Pair { first: cursor.u64()?, second: cursor.u64()? },
        4 => Reply::Handle { handle: cursor.u64()? },
        5 => Reply::Payload { payload: cursor.bytes()? },
        6 => Reply::Elapsed { milliseconds: cursor.f32()? },
        7 => Reply::Done,
        8 => Reply::Failed { code: cursor.i32()?, message: cursor.string()? },
        other => return Err(unknown_tag(*other)),
    })
}

// -- framing ------------------------------------------------------------------

/// Size of the length prefix in front of every frame body.
///
/// Eight bytes, because a single copy to the device may exceed 4 GiB and a 32 bit
/// length would wrap without an error.
pub const FRAME_HEADER_BYTES: usize = 8;

/// Encode the length prefix of a frame whose body is `length` bytes.
pub fn encode_frame_header(length: u64) -> [u8; FRAME_HEADER_BYTES] {
    length.to_le_bytes()
}

/// Decode the body length a frame header carries.
pub fn decode_frame_header(header: [u8; FRAME_HEADER_BYTES]) -> u64 {
    u64::from_le_bytes(header)
}

/// Write one length-prefixed frame.
pub fn write_frame<W: Write>(writer: &mut W, body: &[u8]) -> io::Result<()> {
    writer.write_all(&encode_frame_header(body.len() as u64))?;
    writer.write_all(body)
}

/// Read one length-prefixed frame.
///
/// The body grows as bytes arrive rather than being allocated from the header, so a
/// corrupt length is a short read instead of an allocation failure.
pub fn read_frame<R: Read>(reader: &mut R) -> io::Result<Vec<u8>> {
    let mut header = [0u8; FRAME_HEADER_BYTES];
    reader.read_exact(&mut header)?;
    let mut body = Vec::new();
    read_body(reader, decode_frame_header(header), &mut body)?;
    Ok(body)
}

/// Append exactly `length` bytes from `reader` to `body`.
fn read_body<R: Read>(reader: &mut R, length: u64, body: &mut Vec<u8>) -> io::Result<()> {
    let before = body.len();
    reader.by_ref().take(length).read_to_end(body)?;
    if (body.len() - before) as u64 != length {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "frame is short"));
    }
    Ok(())
}

/// Tag byte of a `CopyToDevice` request.
const TAG_COPY_TO_DEVICE: u8 = 6;

/// Bytes of a `CopyToDevice` body before its payload: tag, handle, offset, length.
const COPY_TO_DEVICE_FIXED_BYTES: usize = 1 + 8 + 8 + 8;

/// How much the staging buffer grows by at a time, so a corrupt length cannot make the
/// agent allocate more than it has actually received plus one step.
const STAGING_STEP: usize = 64 * 1024 * 1024;

/// What [`read_incoming`] took off the wire.
#[derive(Debug, Clone, PartialEq)]
pub enum Incoming {
    /// A copy to the device whose payload is the first `bytes` bytes of the staging
    /// buffer.
    CopyToDevice { handle: u64, offset: u64, bytes: usize },
    /// Any other request, decoded whole.
    Request(Request),
}

/// Write a `CopyToDevice` frame for the caller's bytes.
///
/// Produces the same bytes as `write_frame(encode_request(CopyToDevice { .. }))`, but
/// the payload is handed to `write_vectored` as the caller's own slice. Behind a
/// `BufWriter`, a payload larger than its buffer goes to the socket without being
/// copied into it.
pub fn write_copy_to_device<W: Write>(
    writer: &mut W,
    handle: u64,
    offset: u64,
    payload: &[u8],
) -> io::Result<()> {
    let length = payload.len() as u64;
    let mut head = [0u8; FRAME_HEADER_BYTES + COPY_TO_DEVICE_FIXED_BYTES];
    head[..8].copy_from_slice(&encode_frame_header(COPY_TO_DEVICE_FIXED_BYTES as u64 + length));
    head[8] = TAG_COPY_TO_DEVICE;
    head[9..17].copy_from_slice(&handle.to_le_bytes());
    head[17..25].copy_from_slice(&offset.to_le_bytes());
    head[25..33].copy_from_slice(&length.to_le_bytes());
    write_all_vectored(writer, &mut [IoSlice::new(&head), IoSlice::new(payload)])
}

fn write_all_vectored<W: Write>(writer: &mut W, mut slices: &mut [IoSlice<'_>]) -> io::Result<()> {
    IoSlice::advance_slices(&mut slices, 0);
    while !slices.is_empty() {
        match writer.write_vectored(slices) {
            Ok(0) => {
                return Err(io::Error::new(io::ErrorKind::WriteZero, "the frame was not written"));
            }
            Ok(written) => IoSlice::advance_slices(&mut slices, written),
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

/// Read one request, putting the payload of a copy to the device into `staging`.
///
/// The tag is read before the body, so a copy's payload is read with `read_exact`
/// straight into `staging`, which the caller keeps across calls. `staging` only grows,
/// so steady state copies neither allocate nor zero. Any other request is decoded
/// whole. A body that does not decode is reported as `InvalidData` with the frame
/// fully consumed, so the caller may answer and carry on. A copy whose length fields
/// disagree is `InvalidInput`, after which the stream cannot be trusted.
pub fn read_incoming<R: Read>(reader: &mut R, staging: &mut Vec<u8>) -> io::Result<Incoming> {
    let mut header = [0u8; FRAME_HEADER_BYTES];
    reader.read_exact(&mut header)?;
    let length = decode_frame_header(header);
    if length == 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "empty frame"));
    }
    let mut tag = [0u8; 1];
    reader.read_exact(&mut tag)?;

    if tag[0] == TAG_COPY_TO_DEVICE && length >= COPY_TO_DEVICE_FIXED_BYTES as u64 {
        let mut fixed = [0u8; COPY_TO_DEVICE_FIXED_BYTES - 1];
        reader.read_exact(&mut fixed)?;
        let handle = u64::from_le_bytes(fixed[0..8].try_into().unwrap());
        let offset = u64::from_le_bytes(fixed[8..16].try_into().unwrap());
        let declared = u64::from_le_bytes(fixed[16..24].try_into().unwrap());
        if declared != length - COPY_TO_DEVICE_FIXED_BYTES as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "a copy to the device declares a payload that disagrees with its frame",
            ));
        }
        let bytes = usize::try_from(declared)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "payload is too large"))?;
        let mut filled = 0;
        while filled < bytes {
            let end = bytes.min(filled + STAGING_STEP);
            if staging.len() < end {
                staging.resize(end, 0);
            }
            reader.read_exact(&mut staging[filled..end])?;
            filled = end;
        }
        return Ok(Incoming::CopyToDevice { handle, offset, bytes });
    }

    let mut frame = vec![tag[0]];
    read_body(reader, length - 1, &mut frame)?;
    decode_request(&frame)
        .map(Incoming::Request)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))
}

/// What [`read_copy_to_host`] took off the wire.
#[derive(Debug, Clone, PartialEq)]
pub enum HostCopy {
    /// A `Payload` of exactly the requested length, now in the destination.
    Filled,
    /// Any other reply, decoded whole.
    Reply(Reply),
}

/// Write a `Payload` reply for bytes the agent already holds.
///
/// Produces the same bytes as `write_frame(encode_reply(Payload { .. }))`.
pub fn write_payload_reply<W: Write>(writer: &mut W, payload: &[u8]) -> io::Result<()> {
    write_frame(writer, &encode_reply(&Reply::Payload { payload: payload.to_vec() }))
}

/// Read the reply to a copy to the host, putting a `Payload` into `destination`.
///
/// A `Payload` whose length differs from `destination` is discarded and reported as
/// `InvalidInput`, with the stream still in step.
pub fn read_copy_to_host<R: Read>(reader: &mut R, destination: &mut [u8]) -> io::Result<HostCopy> {
    match decode_reply(&read_frame(reader)?)? {
        Reply::Payload { payload } if payload.len() == destination.len() => {
            destination.copy_from_slice(&payload);
            Ok(HostCopy::Filled)
        }
        Reply::Payload { .. } => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "a copy to the host answered with a payload of another length",
        )),
        other => Ok(HostCopy::Reply(other)),
    }
}

/// Content address of a payload, used so the same module travels once.
///
/// This is not a cryptographic decision. A module is identified by its own bytes, and a
/// 128 bit digest of them is enough to tell two fatbins apart.
pub fn digest(payload: &[u8]) -> [u8; 16] {
    // FNV-1a over two lanes, which is fast, dependency free, and plenty for a lookup
    // key. Collisions here would mean two different modules with identical digests,
    // which at 128 bits is not a practical concern.
    let mut low: u64 = 0xcbf29ce484222325;
    let mut high: u64 = 0x9e3779b97f4a7c15;
    for (index, byte) in payload.iter().enumerate() {
        if index % 2 == 0 {
            low ^= *byte as u64;
            low = low.wrapping_mul(0x100000001b3);
        } else {
            high ^= *byte as u64;
            high = high.wrapping_mul(0x100000001b3);
        }
    }
    high ^= payload.len() as u64;
    let mut out = [0u8; 16];
    out[..8].copy_from_slice(&low.to_le_bytes());
    out[8..].copy_from_slice(&high.to_le_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip_request(request: Request) {
        let encoded = encode_request(&request);
        assert_eq!(decode_request(&encoded).unwrap(), request);
    }

    #[test]
    fn requests_survive_a_round_trip() {
        round_trip_request(Request::Hello { version: PROTOCOL_VERSION });
        round_trip_request(Request::DeviceAttribute { device: 0, attribute: 75 });
        round_trip_request(Request::MemoryInfo);
        round_trip_request(Request::Allocate { bytes: 4096, handle: 7 });
        round_trip_request(Request::Free { handle: 7 });
        round_trip_request(Request::CopyToDevice {
            handle: 7,
            offset: 16,
            payload: vec![1, 2, 3, 4],
        });
        round_trip_request(Request::CopyToHost { handle: 7, offset: 0, bytes: 64 });
        round_trip_request(Request::LoadModule { digest: [9u8; 16], payload: vec![0xff; 32] });
        round_trip_request(Request::GetFunction { module: 3, name: "add_kernel".into() });
        round_trip_request(Request::LaunchKernel {
            function: 4,
            grid: [8, 1, 1],
            block: [256, 1, 1],
            shared_bytes: 0,
            stream: 0,
            params: vec![7; 24],
        });
        round_trip_request(Request::SynchronizeStream { stream: 0 });
        round_trip_request(Request::RecordEvent { event: 1, stream: 0 });
        round_trip_request(Request::SynchronizeEvent { event: 1 });
        round_trip_request(Request::ElapsedTime { start: 1, end: 2 });
        round_trip_request(Request::Shutdown);
    }

    fn round_trip_reply(reply: Reply) {
        let encoded = encode_reply(&reply);
        assert_eq!(decode_reply(&encoded).unwrap(), reply);
    }

    #[test]
    fn replies_survive_a_round_trip() {
        round_trip_reply(Reply::Ready {
            version: PROTOCOL_VERSION,
            device_name: "NVIDIA RTX PRO 6000".into(),
            compute: (12, 0),
        });
        round_trip_reply(Reply::Value { value: -3 });
        round_trip_reply(Reply::Pair { first: 1024, second: 2048 });
        round_trip_reply(Reply::Handle { handle: 12 });
        round_trip_reply(Reply::Payload { payload: vec![5; 10] });
        round_trip_reply(Reply::Elapsed { milliseconds: 1.5 });
        round_trip_reply(Reply::Done);
        round_trip_reply(Reply::Failed { code: 2, message: "out of memory".into() });
    }

    #[test]
    fn only_reads_force_a_round_trip() {
        // This split is the whole performance argument: a launch or a copy to the device
        // is batched, and a read back is not.
        assert!(!Request::LaunchKernel {
            function: 1,
            grid: [1, 1, 1],
            block: [1, 1, 1],
            shared_bytes: 0,
            stream: 0,
            params: vec![],
        }
        .needs_reply());
        assert!(!Request::CopyToDevice { handle: 1, offset: 0, payload: vec![] }.needs_reply());
        assert!(!Request::Allocate { bytes: 1, handle: 1 }.needs_reply());
        assert!(Request::CopyToHost { handle: 1, offset: 0, bytes: 1 }.needs_reply());
        assert!(Request::SynchronizeStream { stream: 0 }.needs_reply());
    }

    #[test]
    fn frames_carry_their_length() {
        let mut buffer = Vec::new();
        write_frame(&mut buffer, &[1, 2, 3]).unwrap();
        let mut reader = buffer.as_slice();
        assert_eq!(read_frame(&mut reader).unwrap(), vec![1, 2, 3]);
    }

    /// Records every buffer a writer is handed, by address and length.
    #[derive(Default)]
    struct RecordingWriter {
        bytes: Vec<u8>,
        slices: Vec<(usize, usize)>,
    }

    impl Write for RecordingWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.slices.push((buf.as_ptr() as usize, buf.len()));
            self.bytes.extend_from_slice(buf);
            Ok(buf.len())
        }

        fn write_vectored(&mut self, bufs: &[io::IoSlice<'_>]) -> io::Result<usize> {
            let mut written = 0;
            for buf in bufs {
                written += self.write(buf)?;
            }
            Ok(written)
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    /// Records the address of every buffer a reader fills and how many bytes went in.
    struct RecordingReader<'a> {
        bytes: &'a [u8],
        destinations: Vec<(usize, usize)>,
    }

    impl Read for RecordingReader<'_> {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let count = buf.len().min(self.bytes.len());
            buf[..count].copy_from_slice(&self.bytes[..count]);
            self.bytes = &self.bytes[count..];
            self.destinations.push((buf.as_ptr() as usize, count));
            Ok(count)
        }
    }

    fn patterned(bytes: usize) -> Vec<u8> {
        (0..bytes).map(|index| (index % 251) as u8).collect()
    }

    fn encoded_frame(request: &Request) -> Vec<u8> {
        let mut frame = Vec::new();
        write_frame(&mut frame, &encode_request(request)).unwrap();
        frame
    }

    #[test]
    fn a_streamed_copy_writes_the_same_bytes_as_an_encoded_frame() {
        let payload = patterned(100_000);
        let mut writer = RecordingWriter::default();
        write_copy_to_device(&mut writer, 3, 16, &payload).unwrap();
        let expected =
            encoded_frame(&Request::CopyToDevice { handle: 3, offset: 16, payload: payload.clone() });
        assert_eq!(writer.bytes, expected);
    }

    #[test]
    fn a_streamed_copy_hands_the_callers_bytes_to_the_writer_uncopied() {
        let payload = patterned(100_000);
        let mut writer = RecordingWriter::default();
        write_copy_to_device(&mut writer, 3, 16, &payload).unwrap();
        assert!(
            writer.slices.contains(&(payload.as_ptr() as usize, payload.len())),
            "the payload reached the writer through another buffer: {:?}",
            writer.slices
        );
    }

    #[test]
    fn a_copy_payload_is_read_straight_into_the_staging_buffer() {
        let payload = patterned(100_000);
        let frame =
            encoded_frame(&Request::CopyToDevice { handle: 3, offset: 16, payload: payload.clone() });
        let mut reader = RecordingReader { bytes: &frame, destinations: Vec::new() };
        // Sized ahead so the staging buffer does not move while it is filled.
        let mut staging = vec![0u8; 2 * payload.len()];
        let incoming = read_incoming(&mut reader, &mut staging).unwrap();

        assert_eq!(incoming, Incoming::CopyToDevice { handle: 3, offset: 16, bytes: payload.len() });
        assert_eq!(&staging[..payload.len()], &payload[..]);
        let start = staging.as_ptr() as usize;
        let end = start + staging.len();
        let elsewhere: usize = reader
            .destinations
            .iter()
            .filter(|(address, _)| *address < start || *address >= end)
            .map(|(_, count)| count)
            .sum();
        // Only the frame header and the fixed fields may land outside the staging buffer.
        assert!(elsewhere <= 64, "{elsewhere} bytes were read into an intermediate buffer");
    }

    #[test]
    fn other_requests_arrive_decoded_through_read_incoming() {
        let frame = encoded_frame(&Request::GetFunction { module: 3, name: "add_kernel".into() });
        let mut staging = Vec::new();
        assert_eq!(
            read_incoming(&mut frame.as_slice(), &mut staging).unwrap(),
            Incoming::Request(Request::GetFunction { module: 3, name: "add_kernel".into() })
        );
    }

    #[test]
    fn a_corrupt_frame_length_is_an_error_not_an_allocation() {
        let mut frame = encode_frame_header(1 << 62).to_vec();
        frame.extend_from_slice(&[2, 0, 0]);
        let mut staging = Vec::new();
        assert!(read_incoming(&mut frame.as_slice(), &mut staging).is_err());
        assert!(read_frame(&mut frame.as_slice()).is_err());
    }

    fn encoded_reply_frame(reply: &Reply) -> Vec<u8> {
        let mut frame = Vec::new();
        write_frame(&mut frame, &encode_reply(reply)).unwrap();
        frame
    }

    #[test]
    fn a_streamed_payload_reply_writes_the_same_bytes_as_an_encoded_frame() {
        let payload = patterned(100_000);
        let mut writer = RecordingWriter::default();
        write_payload_reply(&mut writer, &payload).unwrap();
        assert_eq!(writer.bytes, encoded_reply_frame(&Reply::Payload { payload: payload.clone() }));
    }

    #[test]
    fn a_streamed_payload_reply_hands_the_staged_bytes_to_the_writer_uncopied() {
        let payload = patterned(100_000);
        let mut writer = RecordingWriter::default();
        write_payload_reply(&mut writer, &payload).unwrap();
        assert!(
            writer.slices.contains(&(payload.as_ptr() as usize, payload.len())),
            "the payload reached the writer through another buffer: {:?}",
            writer.slices
        );
    }

    #[test]
    fn a_payload_reply_is_read_straight_into_the_destination() {
        let payload = patterned(100_000);
        let frame = encoded_reply_frame(&Reply::Payload { payload: payload.clone() });
        let mut reader = RecordingReader { bytes: &frame, destinations: Vec::new() };
        let mut destination = vec![0u8; payload.len()];
        let copied = read_copy_to_host(&mut reader, &mut destination).unwrap();

        assert_eq!(copied, HostCopy::Filled);
        assert_eq!(destination, payload);
        let start = destination.as_ptr() as usize;
        let end = start + destination.len();
        let elsewhere: usize = reader
            .destinations
            .iter()
            .filter(|(address, _)| *address < start || *address >= end)
            .map(|(_, count)| count)
            .sum();
        // Only the frame header and the fixed fields may land outside the destination.
        assert!(elsewhere <= 64, "{elsewhere} bytes were read into an intermediate buffer");
    }

    #[test]
    fn a_payload_of_another_length_is_refused_and_the_stream_stays_in_step() {
        let mut frames = encoded_reply_frame(&Reply::Payload { payload: patterned(1000) });
        frames.extend(encoded_reply_frame(&Reply::Payload { payload: patterned(10) }));
        let mut reader = frames.as_slice();
        let mut destination = vec![0u8; 10];

        let refused = read_copy_to_host(&mut reader, &mut destination).unwrap_err();
        assert_eq!(refused.kind(), io::ErrorKind::InvalidInput);
        assert_eq!(read_copy_to_host(&mut reader, &mut destination).unwrap(), HostCopy::Filled);
        assert_eq!(destination, patterned(10));
    }

    #[test]
    fn a_failed_copy_to_the_host_arrives_decoded() {
        let failed = Reply::Failed { code: 700, message: "illegal address".into() };
        let frame = encoded_reply_frame(&failed);
        let mut destination = vec![0u8; 16];
        assert_eq!(
            read_copy_to_host(&mut frame.as_slice(), &mut destination).unwrap(),
            HostCopy::Reply(failed)
        );
    }

    #[test]
    fn a_corrupt_payload_length_is_an_error_not_an_allocation() {
        let mut frame = encode_frame_header(1 << 62).to_vec();
        frame.push(5);
        frame.extend_from_slice(&((1u64 << 62) - 9).to_le_bytes());
        let mut destination = vec![0u8; 16];
        assert!(read_copy_to_host(&mut frame.as_slice(), &mut destination).is_err());
    }

    #[test]
    fn a_frame_length_above_four_gib_survives_the_header() {
        // A batch copied to the device can exceed 4 GiB. The header codec is tested
        // directly so the test does not allocate the body.
        for length in [u32::MAX as u64 + 1, 5 * (1u64 << 30), u64::MAX] {
            assert_eq!(decode_frame_header(encode_frame_header(length)), length);
        }
    }

    #[test]
    fn the_protocol_version_names_the_64_bit_frame_layout() {
        assert_eq!(PROTOCOL_VERSION, 2);
        assert_eq!(FRAME_HEADER_BYTES, 8);
    }

    #[test]
    fn a_short_frame_is_an_error_not_a_panic() {
        // The far side may be a different version, so a malformed frame has to be
        // reported rather than crash the process it was injected into.
        assert!(decode_request(&[2, 0, 0]).is_err());
        assert!(decode_request(&[]).is_err());
        assert!(decode_request(&[99]).is_err());
    }

    #[test]
    fn the_digest_separates_different_payloads() {
        assert_eq!(digest(b"same"), digest(b"same"));
        assert_ne!(digest(b"same"), digest(b"other"));
        assert_ne!(digest(b"ab"), digest(b"ba"));
    }
}
