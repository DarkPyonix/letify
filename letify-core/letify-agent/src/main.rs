//! The agent that holds the real device.
//!
//! It listens on a TCP port, accepts one core at a time, and executes the driver calls
//! that arrive. letify starts it on the machine with the GPU and tells the shim where to
//! find it.
//!
//! Two tables live here. Allocations map the shim's handles onto real device pointers, so
//! the shim can hand out a pointer before this side has allocated anything. Modules are
//! keyed by content, so a fatbin that arrived once is not sent again.
//!
//! Requests arrive in order and are executed in order. A batch of queued launches is
//! simply a run of requests with no reply between them, which is why the batching on the
//! the local side needs nothing special here.

mod driver;

use std::collections::HashMap;
use std::ffi::c_void;
use std::io::{BufReader, BufWriter, Write};
use std::net::{TcpListener, TcpStream};

use letify_wire::{
    PROTOCOL_VERSION, Reply, Request, decode_request, encode_reply, read_frame, write_frame,
};

use crate::driver::Driver;

fn main() {
    let address = std::env::args()
        .nth(1)
        .or_else(|| std::env::var("LETIFY_AGENT_BIND").ok())
        .unwrap_or_else(|| "0.0.0.0:7654".to_string());

    let driver = match Driver::open(0) {
        Ok(driver) => {
            eprintln!(
                "letify-agent: holding {} with compute capability {}.{}",
                driver.name, driver.compute.0, driver.compute.1
            );
            driver
        }
        Err(error) => {
            eprintln!("letify-agent: {error}");
            std::process::exit(1);
        }
    };

    let listener = match TcpListener::bind(&address) {
        Ok(listener) => listener,
        Err(error) => {
            eprintln!("letify-agent: could not bind {address}: {error}");
            std::process::exit(1);
        }
    };
    eprintln!("letify-agent: listening on {address}");

    for incoming in listener.incoming() {
        match incoming {
            Ok(stream) => {
                if let Err(error) = serve(stream, &driver) {
                    eprintln!("letify-agent: the connection ended: {error}");
                }
            }
            Err(error) => eprintln!("letify-agent: could not accept: {error}"),
        }
    }
}

/// State that belongs to one shim connection.
struct Session {
    /// the shim's handle to the real device pointer behind it.
    allocations: HashMap<u64, u64>,
    /// Content address to the module handle, so a fatbin travels once.
    modules: HashMap<[u8; 16], u64>,
    /// the shim's event handle to the real one.
    events: HashMap<u64, u64>,
}

impl Session {
    fn new() -> Self {
        Session {
            allocations: HashMap::new(),
            modules: HashMap::new(),
            events: HashMap::new(),
        }
    }

    /// The real pointer a handle and offset name.
    fn address(&self, handle: u64, offset: u64) -> Option<u64> {
        self.allocations.get(&handle).map(|base| base + offset)
    }

    /// The real event behind a stand-in library handle, creating it on first use.
    fn event(&mut self, driver: &Driver, handle: u64) -> Result<u64, String> {
        if let Some(found) = self.events.get(&handle) {
            return Ok(*found);
        }
        let created = driver.create_event()?;
        self.events.insert(handle, created);
        Ok(created)
    }
}

fn serve(stream: TcpStream, driver: &Driver) -> std::io::Result<()> {
    stream.set_nodelay(true)?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut writer = BufWriter::new(stream);
    let mut session = Session::new();

    loop {
        let frame = match read_frame(&mut reader) {
            Ok(frame) => frame,
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(error) => return Err(error),
        };
        let request = match decode_request(&frame) {
            Ok(request) => request,
            Err(error) => {
                reply(&mut writer, &Reply::Failed { code: 1, message: error.to_string() })?;
                continue;
            }
        };
        if matches!(request, Request::Shutdown) {
            return Ok(());
        }
        let needs_reply = request.needs_reply();
        let outcome = handle(&request, driver, &mut session);
        // Only answer what asked for an answer. A queued launch that failed is reported
        // at the next synchronization, the same way a real driver reports it.
        match outcome {
            Ok(answer) if needs_reply => reply(&mut writer, &answer)?,
            Ok(_) => {}
            Err(message) => {
                eprintln!("letify-agent: {message}");
                if needs_reply {
                    reply(&mut writer, &Reply::Failed { code: 999, message })?;
                }
            }
        }
    }
}

fn reply<W: Write>(writer: &mut W, answer: &Reply) -> std::io::Result<()> {
    write_frame(writer, &encode_reply(answer))?;
    writer.flush()
}

fn handle(request: &Request, driver: &Driver, session: &mut Session) -> Result<Reply, String> {
    match request {
        Request::Hello { version } => {
            if *version != PROTOCOL_VERSION {
                eprintln!(
                    "letify-agent: the shim speaks protocol {version} and this agent speaks \
                     {PROTOCOL_VERSION}"
                );
            }
            Ok(Reply::Ready {
                version: PROTOCOL_VERSION,
                device_name: driver.name.clone(),
                compute: driver.compute,
            })
        }
        Request::DeviceAttribute { attribute, .. } => {
            Ok(Reply::Value { value: driver.attribute(*attribute)? as i64 })
        }
        Request::MemoryInfo => {
            let (free, total) = driver.memory_info()?;
            Ok(Reply::Pair { first: free, second: total })
        }
        Request::Allocate { bytes, handle } => {
            let pointer = driver.allocate(*bytes)?;
            session.allocations.insert(*handle, pointer);
            Ok(Reply::Done)
        }
        Request::Free { handle } => {
            if let Some(pointer) = session.allocations.remove(handle) {
                driver.free(pointer)?;
            }
            Ok(Reply::Done)
        }
        Request::CopyToDevice { handle, offset, payload } => {
            let pointer = session
                .address(*handle, *offset)
                .ok_or_else(|| format!("handle {handle} was never allocated"))?;
            driver.copy_to_device(pointer, payload)?;
            Ok(Reply::Done)
        }
        Request::CopyToHost { handle, offset, bytes } => {
            let pointer = session
                .address(*handle, *offset)
                .ok_or_else(|| format!("handle {handle} was never allocated"))?;
            Ok(Reply::Payload { payload: driver.copy_to_host(pointer, *bytes)? })
        }
        Request::LoadModule { digest, payload } => {
            if let Some(found) = session.modules.get(digest) {
                return Ok(Reply::Handle { handle: *found });
            }
            let module = driver.load_module(payload)?;
            session.modules.insert(*digest, module);
            Ok(Reply::Handle { handle: module })
        }
        Request::GetFunction { module, name } => {
            Ok(Reply::Handle { handle: driver.function(*module, name)? })
        }
        Request::LaunchKernel { function, grid, block, shared_bytes, stream, params } => {
            // the shim sends the pointer list it was given. Each entry is an address in
            // the caller's own space, so it is translated here through the allocation
            // table before the launch.
            let mut translated: Vec<*mut c_void> = params
                .chunks_exact(8)
                .map(|chunk| {
                    let raw = u64::from_le_bytes(chunk.try_into().unwrap());
                    let handle = raw / 0x4000_0000;
                    let offset = raw % 0x4000_0000;
                    session.address(handle, offset).unwrap_or(raw) as *mut c_void
                })
                .collect();
            driver.launch(*function, *grid, *block, *shared_bytes, *stream, &mut translated)?;
            Ok(Reply::Done)
        }
        Request::SynchronizeStream { stream } => {
            driver.synchronize_stream(*stream)?;
            Ok(Reply::Done)
        }
        Request::RecordEvent { event, stream } => {
            let real = session.event(driver, *event)?;
            driver.record_event(real, *stream)?;
            Ok(Reply::Done)
        }
        Request::SynchronizeEvent { event } => {
            let real = session.event(driver, *event)?;
            driver.synchronize_event(real)?;
            Ok(Reply::Done)
        }
        Request::ElapsedTime { start, end } => {
            let first = session.event(driver, *start)?;
            let second = session.event(driver, *end)?;
            Ok(Reply::Elapsed { milliseconds: driver.elapsed(first, second)? })
        }
        Request::Shutdown => Ok(Reply::Done),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_allocation_maps_a_handle_to_a_real_pointer() {
        let mut session = Session::new();
        session.allocations.insert(7, 0xdead_0000);
        assert_eq!(session.address(7, 0), Some(0xdead_0000));
        assert_eq!(session.address(7, 256), Some(0xdead_0100));
        assert_eq!(session.address(8, 0), None);
    }

    #[test]
    fn a_module_is_remembered_by_its_contents() {
        // This is what keeps a fatbin from being sent on every process start.
        let mut session = Session::new();
        let digest = letify_wire::digest(b"a fatbin");
        session.modules.insert(digest, 42);
        assert_eq!(session.modules.get(&digest), Some(&42));
        assert_eq!(session.modules.get(&letify_wire::digest(b"another")), None);
    }
}
