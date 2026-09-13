//! letify-shim, the stand-in for the CUDA driver.
//!
//! This library stands in for the CUDA driver on the machine that runs the Python code.
//! Every call an application makes is forwarded to an agent on the machine that has the
//! device, so the code and the data stay where they already are and only driver traffic
//! crosses the network.
//!
//! How it gets loaded differs by platform, and both paths come down to the same trick of
//! being found before the real driver.
//!
//! On Linux and WSL2 it is built as `libcuda.so.1` and placed ahead of the real one with
//! `LD_PRELOAD`.
//!
//! On Windows it is built as `nvcuda.dll`. There is no `LD_PRELOAD`, so letify calls
//! `os.add_dll_directory` on the shim's directory before importing torch, which puts it
//! first in the loader's search order. That is why the Python side owns the injection:
//! it has to happen before the first CUDA library is loaded.
//!
//! What is implemented here is the first milestone, which is the set of entry points a
//! PyTorch process touches to start up and run one kernel: initialization, device
//! queries, context handling, allocation and copies, module loading, launches, streams
//! and events. Everything else returns `CUDA_ERROR_NOT_SUPPORTED` and names itself in
//! the log, so the next thing to implement is whatever a real run prints.

mod client;
mod exports;
mod status;
mod table;

pub use client::{Client, agent_address, client};
pub use status::{CUresult, log};
pub use table::{Table, handle_of, offset_of, pointer_of, table};

/// Connect to the agent and learn the device size.
///
/// Called from `cuInit`, which every CUDA application calls first, so by the time
/// anything else arrives the connection exists and the memory budget is known.
pub fn attach() -> CUresult {
    let address = agent_address();
    let mut guard = match client().lock() {
        Ok(guard) => guard,
        Err(_) => return status::CUDA_ERROR_UNKNOWN,
    };
    if guard.is_some() {
        return status::CUDA_SUCCESS;
    }
    match Client::connect(&address) {
        Ok(mut connected) => {
            // Ask once for the device size, so the local memory accounting can refuse an
            // allocation the way a real driver would.
            if let Ok(letify_wire::Reply::Pair { first: _free, second: total }) =
                connected.request(letify_wire::Request::MemoryInfo)
                && let Ok(mut allocations) = table().lock()
            {
                allocations.set_total(total);
            }
            *guard = Some(connected);
            log(&format!("attached to the agent at {address}"));
            status::CUDA_SUCCESS
        }
        Err(error) => {
            log(&format!(
                "could not reach the agent at {address}: {error}. Set LETIFY_AGENT to \
                 where it is listening, or declare host='remote' to ship the function \
                 instead."
            ));
            status::CUDA_ERROR_NOT_INITIALIZED
        }
    }
}

/// Report what the shim has done, for the Python side to read back.
///
/// The ratio of calls to round trips is the number that says whether batching is working:
/// a training step issues thousands of calls and should only pay a handful of round trips.
pub fn report() -> (u64, u64, usize, u64) {
    let (sent, round_trips) = match client().lock() {
        Ok(guard) => guard
            .as_ref()
            .map(|found| (found.sent, found.round_trips))
            .unwrap_or((0, 0)),
        Err(_) => (0, 0),
    };
    let (live, used) = match table().lock() {
        Ok(guard) => (guard.live(), guard.used()),
        Err(_) => (0, 0),
    };
    (sent, round_trips, live, used)
}
