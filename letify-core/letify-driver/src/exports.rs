//! The CUDA driver entry points this library stands in for.
//!
//! Each one has the signature the real driver has, because the caller was compiled
//! against that header and does not know it is talking to us. Everything below converts
//! its arguments into a request and lets [`crate::client`] decide whether it waits.
//!
//! The milestone here is what a PyTorch process touches to start up and run one kernel.
//! Anything outside it reports its own name and returns `CUDA_ERROR_NOT_SUPPORTED`,
//! which is how the next thing to implement gets identified.

use std::ffi::{CStr, c_char, c_void};

use letify_wire::{HostCopy, Reply, Request};

use crate::client::client;
use crate::status::{
    CUDA_ERROR_INVALID_DEVICE, CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_INVALID_VALUE,
    CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_UNKNOWN, CUDA_SUCCESS,
    CUresult, log, unimplemented,
};
use crate::table::{handle_of, offset_of, table};

/// Run a body with the connected client, or fail the way an uninitialized driver does.
macro_rules! with_client {
    (|$name:ident| $body:block) => {{
        let mut guard = match client().lock() {
            Ok(guard) => guard,
            Err(_) => return CUDA_ERROR_UNKNOWN,
        };
        match guard.as_mut() {
            Some($name) => $body,
            None => CUDA_ERROR_NOT_INITIALIZED,
        }
    }};
}

// -- initialization and devices -----------------------------------------------

/// Every CUDA application calls this first, so this is where the connection is made.
#[unsafe(no_mangle)]
pub extern "C" fn cuInit(_flags: u32) -> CUresult {
    crate::attach()
}

#[unsafe(no_mangle)]
pub extern "C" fn cuDriverGetVersion(version: *mut i32) -> CUresult {
    if version.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Report a version new enough that a Blackwell capable stack does not refuse to
    // start. The agent's real driver is what actually runs the work.
    unsafe { *version = 12080 };
    CUDA_SUCCESS
}

#[unsafe(no_mangle)]
pub extern "C" fn cuDeviceGetCount(count: *mut i32) -> CUresult {
    if count.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // One agent holds one device. Presenting more would mean deciding how to place work
    // across them, which belongs in letify rather than here.
    unsafe { *count = 1 };
    CUDA_SUCCESS
}

#[unsafe(no_mangle)]
pub extern "C" fn cuDeviceGet(device: *mut i32, ordinal: i32) -> CUresult {
    if device.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if ordinal != 0 {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    unsafe { *device = 0 };
    CUDA_SUCCESS
}

#[unsafe(no_mangle)]
pub extern "C" fn cuDeviceGetAttribute(value: *mut i32, attribute: i32, device: i32) -> CUresult {
    if value.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    with_client!(|connection| {
        match connection.request(Request::DeviceAttribute { device, attribute }) {
            Ok(Reply::Value { value: answer }) => {
                unsafe { *value = answer as i32 };
                CUDA_SUCCESS
            }
            Ok(Reply::Failed { code, message }) => {
                log(&format!("cuDeviceGetAttribute({attribute}) failed: {message}"));
                code
            }
            _ => CUDA_ERROR_UNKNOWN,
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CUresult {
    if free.is_null() || total.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    with_client!(|connection| {
        match connection.request(Request::MemoryInfo) {
            Ok(Reply::Pair { first, second }) => {
                unsafe {
                    *free = first as usize;
                    *total = second as usize;
                }
                CUDA_SUCCESS
            }
            _ => CUDA_ERROR_UNKNOWN,
        }
    })
}

// -- memory -------------------------------------------------------------------

/// Allocate device memory.
///
/// Returns immediately with a virtual pointer. Waiting for the agent would put a round
/// trip in front of every allocation, and the caching allocator makes many. The budget
/// check is local so that an out of memory condition still surfaces here, where the
/// caller can free its cache and retry.
#[unsafe(no_mangle)]
pub extern "C" fn cuMemAlloc_v2(pointer: *mut u64, bytes: usize) -> CUresult {
    if pointer.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let (address, handle) = {
        let mut allocations = match table().lock() {
            Ok(guard) => guard,
            Err(_) => return CUDA_ERROR_UNKNOWN,
        };
        match allocations.allocate(bytes as u64) {
            Some(pair) => pair,
            None => return CUDA_ERROR_OUT_OF_MEMORY,
        }
    };
    let result = with_client!(|connection| {
        match connection.send(Request::Allocate { bytes: bytes as u64, handle }) {
            Ok(()) => CUDA_SUCCESS,
            Err(_) => CUDA_ERROR_UNKNOWN,
        }
    });
    if result == CUDA_SUCCESS {
        unsafe { *pointer = address };
    }
    result
}

#[unsafe(no_mangle)]
pub extern "C" fn cuMemFree_v2(pointer: u64) -> CUresult {
    let handle = {
        let mut allocations = match table().lock() {
            Ok(guard) => guard,
            Err(_) => return CUDA_ERROR_UNKNOWN,
        };
        match allocations.free(pointer) {
            Some(handle) => handle,
            None => return CUDA_ERROR_INVALID_HANDLE,
        }
    };
    with_client!(|connection| {
        match connection.send(Request::Free { handle }) {
            Ok(()) => CUDA_SUCCESS,
            Err(_) => CUDA_ERROR_UNKNOWN,
        }
    })
}

/// Copy host memory to the device. Queued, because nothing reads the result.
#[unsafe(no_mangle)]
pub extern "C" fn cuMemcpyHtoD_v2(
    destination: u64,
    source: *const c_void,
    bytes: usize,
) -> CUresult {
    let Some(handle) = handle_of(destination) else {
        return CUDA_ERROR_INVALID_HANDLE;
    };
    if source.is_null() && bytes > 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Borrowed, not copied: the bytes go from here to the socket.
    let payload: &[u8] = if bytes == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(source as *const u8, bytes) }
    };
    with_client!(|connection| {
        match connection.send_copy_to_device(handle, offset_of(destination), payload) {
            Ok(()) => CUDA_SUCCESS,
            Err(_) => CUDA_ERROR_UNKNOWN,
        }
    })
}

/// Copy device memory back to the host.
///
/// This one always waits, because the caller is about to read the bytes. Every call
/// here is one of the round trips in the efficiency formula.
#[unsafe(no_mangle)]
pub extern "C" fn cuMemcpyDtoH_v2(
    destination: *mut c_void,
    source: u64,
    bytes: usize,
) -> CUresult {
    let Some(handle) = handle_of(source) else {
        return CUDA_ERROR_INVALID_HANDLE;
    };
    if destination.is_null() && bytes > 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // The reply is read straight into the caller's buffer, with nothing in between.
    let destination: &mut [u8] = if bytes == 0 {
        &mut []
    } else {
        unsafe { std::slice::from_raw_parts_mut(destination as *mut u8, bytes) }
    };
    with_client!(|connection| {
        match connection.request_copy_to_host(handle, offset_of(source), destination) {
            Ok(HostCopy::Filled) => CUDA_SUCCESS,
            Ok(HostCopy::Reply(Reply::Failed { code, .. })) => code,
            Err(error) if error.kind() == std::io::ErrorKind::InvalidInput => {
                CUDA_ERROR_INVALID_VALUE
            }
            _ => CUDA_ERROR_UNKNOWN,
        }
    })
}

// -- modules and kernels --------------------------------------------------------

/// Load a compiled module.
///
/// Content addressed, so a fatbin that the agent already holds is named rather than
/// sent. PyTorch loads the same modules on every process start, and they are large.
#[unsafe(no_mangle)]
pub extern "C" fn cuModuleLoadData(module: *mut u64, image: *const c_void) -> CUresult {
    if module.is_null() || image.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // There is no length argument here, so the size comes from the image itself.
    let payload = unsafe { crate::image::module_image(image as *const u8) }.to_vec();
    let digest = letify_wire::digest(&payload);
    with_client!(|connection| {
        match connection.request(Request::LoadModule { digest, payload }) {
            Ok(Reply::Handle { handle }) => {
                unsafe { *module = handle };
                CUDA_SUCCESS
            }
            Ok(Reply::Failed { code, message }) => {
                log(&format!("cuModuleLoadData failed: {message}"));
                code
            }
            _ => CUDA_ERROR_UNKNOWN,
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn cuModuleGetFunction(
    function: *mut u64,
    module: u64,
    name: *const c_char,
) -> CUresult {
    if function.is_null() || name.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let Ok(symbol) = unsafe { CStr::from_ptr(name) }.to_str() else {
        return CUDA_ERROR_INVALID_VALUE;
    };
    with_client!(|connection| {
        match connection.request(Request::GetFunction { module, name: symbol.to_string() }) {
            Ok(Reply::Handle { handle }) => {
                unsafe { *function = handle };
                CUDA_SUCCESS
            }
            Ok(Reply::Failed { code, .. }) => code,
            _ => CUDA_ERROR_UNKNOWN,
        }
    })
}

/// Launch a kernel. Queued, which is what makes a training step affordable.
///
/// A step issues thousands of these. If each one waited, the round trip count would be
/// the call count and no amount of tuning would help.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn cuLaunchKernel(
    function: u64,
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    block_x: u32,
    block_y: u32,
    block_z: u32,
    shared_bytes: u32,
    stream: u64,
    parameters: *mut *mut c_void,
    _extra: *mut *mut c_void,
) -> CUresult {
    // Kernel parameters arrive as an array of pointers with no count and no sizes, so
    // the agent is told the pointer list and reconstructs the arguments from the
    // function's own signature.
    let params = if parameters.is_null() {
        Vec::new()
    } else {
        let mut encoded = Vec::new();
        let mut index = 0usize;
        // The array is null terminated in practice for the launches PyTorch makes.
        while index < 64 {
            let slot = unsafe { *parameters.add(index) };
            if slot.is_null() {
                break;
            }
            encoded.extend_from_slice(&(slot as u64).to_le_bytes());
            index += 1;
        }
        encoded
    };
    with_client!(|connection| {
        match connection.send(Request::LaunchKernel {
            function,
            grid: [grid_x, grid_y, grid_z],
            block: [block_x, block_y, block_z],
            shared_bytes,
            stream,
            params,
        }) {
            Ok(()) => CUDA_SUCCESS,
            Err(_) => CUDA_ERROR_UNKNOWN,
        }
    })
}

// -- streams and events ---------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn cuStreamSynchronize(stream: u64) -> CUresult {
    with_client!(|connection| {
        match connection.request(Request::SynchronizeStream { stream }) {
            Ok(Reply::Done) => CUDA_SUCCESS,
            Ok(Reply::Failed { code, .. }) => code,
            _ => CUDA_ERROR_UNKNOWN,
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn cuEventRecord(event: u64, stream: u64) -> CUresult {
    with_client!(|connection| {
        match connection.send(Request::RecordEvent { event, stream }) {
            Ok(()) => CUDA_SUCCESS,
            Err(_) => CUDA_ERROR_UNKNOWN,
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn cuEventSynchronize(event: u64) -> CUresult {
    with_client!(|connection| {
        match connection.request(Request::SynchronizeEvent { event }) {
            Ok(Reply::Done) => CUDA_SUCCESS,
            Ok(Reply::Failed { code, .. }) => code,
            _ => CUDA_ERROR_UNKNOWN,
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn cuEventElapsedTime(
    milliseconds: *mut f32,
    start: u64,
    end: u64,
) -> CUresult {
    if milliseconds.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    with_client!(|connection| {
        match connection.request(Request::ElapsedTime { start, end }) {
            Ok(Reply::Elapsed { milliseconds: value }) => {
                unsafe { *milliseconds = value };
                CUDA_SUCCESS
            }
            Ok(Reply::Failed { code, .. }) => code,
            _ => CUDA_ERROR_UNKNOWN,
        }
    })
}

// -- not implemented yet ---------------------------------------------------------

/// Unified memory cannot be forwarded at all.
///
/// Managed memory works by letting the device fault on host pages, which needs the two
/// to share an address space. Across a network there is nothing to fault into. This is
/// the one limit that no amount of implementation removes, and it is why a paged
/// optimizer cannot run under forwarding.
#[unsafe(no_mangle)]
pub extern "C" fn cuMemAllocManaged(_pointer: *mut u64, _bytes: usize, _flags: u32) -> CUresult {
    eprintln!(
        "letify-driver: cuMemAllocManaged cannot be forwarded. Unified memory relies on the \
         device faulting into host pages, which needs one address space, and there is no \
         such thing across a network. A paged optimizer such as bitsandbytes PagedAdamW \
         uses this. Use host='remote' to ship the function instead."
    );
    crate::status::CUDA_ERROR_NOT_SUPPORTED
}

#[unsafe(no_mangle)]
pub extern "C" fn cuMemHostAlloc(
    _pointer: *mut *mut c_void,
    _bytes: usize,
    _flags: u32,
) -> CUresult {
    unimplemented("cuMemHostAlloc")
}

#[unsafe(no_mangle)]
pub extern "C" fn cuGraphLaunch(_graph: u64, _stream: u64) -> CUresult {
    unimplemented("cuGraphLaunch")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_null_out_pointer_is_rejected_rather_than_dereferenced() {
        assert_eq!(cuDeviceGetCount(std::ptr::null_mut()), CUDA_ERROR_INVALID_VALUE);
        assert_eq!(cuDriverGetVersion(std::ptr::null_mut()), CUDA_ERROR_INVALID_VALUE);
    }

    #[test]
    fn calls_before_init_say_so() {
        // Without a connection there is nothing to forward to, and the driver's own code
        // for that is what callers already handle.
        assert_eq!(cuStreamSynchronize(0), CUDA_ERROR_NOT_INITIALIZED);
    }

    #[test]
    fn only_one_device_is_presented() {
        let mut count = 0;
        assert_eq!(cuDeviceGetCount(&mut count), CUDA_SUCCESS);
        assert_eq!(count, 1);

        let mut device = -1;
        assert_eq!(cuDeviceGet(&mut device, 0), CUDA_SUCCESS);
        assert_eq!(device, 0);
        assert_eq!(cuDeviceGet(&mut device, 1), CUDA_ERROR_INVALID_DEVICE);
    }

    #[test]
    fn managed_memory_is_refused_with_the_reason() {
        assert_eq!(
            cuMemAllocManaged(std::ptr::null_mut(), 1024, 0),
            crate::status::CUDA_ERROR_NOT_SUPPORTED
        );
    }
}
