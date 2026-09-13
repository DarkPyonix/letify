//! The real CUDA driver, loaded at runtime.
//!
//! The driver is opened with `libloading` rather than linked, for two reasons. The agent
//! then builds on a machine with no CUDA toolkit installed, which is what lets it be
//! tested and packaged anywhere. And the same binary runs against whatever driver
//! version the machine happens to have, instead of being pinned at build time.
//!
//! Only the entry points the shim actually forwards are looked up. A missing symbol is
//! reported by name, which is the same discipline the shim follows.

use std::ffi::{CString, c_char, c_void};

use libloading::{Library, Symbol};

/// Names the driver library goes by, in the order they are tried.
const CANDIDATES: &[&str] = &[
    // Windows.
    "nvcuda.dll",
    // Linux, including WSL2, where the driver is provided by the Windows host.
    "libcuda.so.1",
    "libcuda.so",
    "/usr/lib/wsl/lib/libcuda.so.1",
];

type CuInit = unsafe extern "C" fn(u32) -> i32;
type CuDeviceGet = unsafe extern "C" fn(*mut i32, i32) -> i32;
type CuDeviceGetName = unsafe extern "C" fn(*mut c_char, i32, i32) -> i32;
type CuDeviceGetAttribute = unsafe extern "C" fn(*mut i32, i32, i32) -> i32;
type CuCtxCreate = unsafe extern "C" fn(*mut u64, u32, i32) -> i32;
type CuMemGetInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> i32;
type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
type CuMemFree = unsafe extern "C" fn(u64) -> i32;
type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
type CuModuleLoadData = unsafe extern "C" fn(*mut u64, *const c_void) -> i32;
type CuModuleGetFunction = unsafe extern "C" fn(*mut u64, u64, *const c_char) -> i32;
type CuLaunchKernel = unsafe extern "C" fn(
    u64,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u64,
    *mut *mut c_void,
    *mut *mut c_void,
) -> i32;
type CuStreamSynchronize = unsafe extern "C" fn(u64) -> i32;
type CuEventCreate = unsafe extern "C" fn(*mut u64, u32) -> i32;
type CuEventRecord = unsafe extern "C" fn(u64, u64) -> i32;
type CuEventSynchronize = unsafe extern "C" fn(u64) -> i32;
type CuEventElapsedTime = unsafe extern "C" fn(*mut f32, u64, u64) -> i32;

/// Compute capability attribute numbers, from the driver's own header.
pub const ATTRIBUTE_COMPUTE_MAJOR: i32 = 75;
pub const ATTRIBUTE_COMPUTE_MINOR: i32 = 76;

/// The loaded driver, with a context on device zero.
pub struct Driver {
    library: Library,
    #[allow(dead_code)]
    context: u64,
    pub device: i32,
    pub name: String,
    pub compute: (i32, i32),
}

impl Driver {
    /// Open the driver, create a context and read what the device is.
    pub fn open(ordinal: i32) -> Result<Self, String> {
        let library = load_library()?;
        let mut driver = Driver {
            library,
            context: 0,
            device: 0,
            name: String::new(),
            compute: (0, 0),
        };

        let status = unsafe { driver.symbol::<CuInit>("cuInit")?(0) };
        if status != 0 {
            return Err(format!("cuInit returned {status}"));
        }

        let mut device = 0;
        let status = unsafe { driver.symbol::<CuDeviceGet>("cuDeviceGet")?(&mut device, ordinal) };
        if status != 0 {
            return Err(format!("cuDeviceGet({ordinal}) returned {status}"));
        }
        driver.device = device;

        let mut context = 0u64;
        let status =
            unsafe { driver.symbol::<CuCtxCreate>("cuCtxCreate_v2")?(&mut context, 0, device) };
        if status != 0 {
            return Err(format!("cuCtxCreate returned {status}"));
        }
        driver.context = context;

        driver.name = driver.device_name()?;
        driver.compute = (
            driver.attribute(ATTRIBUTE_COMPUTE_MAJOR)?,
            driver.attribute(ATTRIBUTE_COMPUTE_MINOR)?,
        );
        Ok(driver)
    }

    fn symbol<T>(&self, name: &str) -> Result<Symbol<'_, T>, String> {
        unsafe { self.library.get(format!("{name}\0").as_bytes()) }
            .map_err(|error| format!("the driver has no {name}: {error}"))
    }

    fn device_name(&self) -> Result<String, String> {
        let mut buffer = vec![0i8; 256];
        let status = unsafe {
            self.symbol::<CuDeviceGetName>("cuDeviceGetName")?(
                buffer.as_mut_ptr() as *mut c_char,
                buffer.len() as i32,
                self.device,
            )
        };
        if status != 0 {
            return Err(format!("cuDeviceGetName returned {status}"));
        }
        let bytes: Vec<u8> = buffer.iter().take_while(|b| **b != 0).map(|b| *b as u8).collect();
        Ok(String::from_utf8_lossy(&bytes).to_string())
    }

    pub fn attribute(&self, attribute: i32) -> Result<i32, String> {
        let mut value = 0;
        let status = unsafe {
            self.symbol::<CuDeviceGetAttribute>("cuDeviceGetAttribute")?(
                &mut value,
                attribute,
                self.device,
            )
        };
        if status != 0 {
            return Err(format!("cuDeviceGetAttribute({attribute}) returned {status}"));
        }
        Ok(value)
    }

    pub fn memory_info(&self) -> Result<(u64, u64), String> {
        let mut free = 0usize;
        let mut total = 0usize;
        let status =
            unsafe { self.symbol::<CuMemGetInfo>("cuMemGetInfo_v2")?(&mut free, &mut total) };
        if status != 0 {
            return Err(format!("cuMemGetInfo returned {status}"));
        }
        Ok((free as u64, total as u64))
    }

    pub fn allocate(&self, bytes: u64) -> Result<u64, String> {
        let mut pointer = 0u64;
        let status =
            unsafe { self.symbol::<CuMemAlloc>("cuMemAlloc_v2")?(&mut pointer, bytes as usize) };
        if status != 0 {
            return Err(format!("cuMemAlloc returned {status}"));
        }
        Ok(pointer)
    }

    pub fn free(&self, pointer: u64) -> Result<(), String> {
        let status = unsafe { self.symbol::<CuMemFree>("cuMemFree_v2")?(pointer) };
        if status != 0 {
            return Err(format!("cuMemFree returned {status}"));
        }
        Ok(())
    }

    pub fn copy_to_device(&self, pointer: u64, payload: &[u8]) -> Result<(), String> {
        let status = unsafe {
            self.symbol::<CuMemcpyHtoD>("cuMemcpyHtoD_v2")?(
                pointer,
                payload.as_ptr() as *const c_void,
                payload.len(),
            )
        };
        if status != 0 {
            return Err(format!("cuMemcpyHtoD returned {status}"));
        }
        Ok(())
    }

    pub fn copy_to_host(&self, pointer: u64, bytes: u64) -> Result<Vec<u8>, String> {
        let mut buffer = vec![0u8; bytes as usize];
        let status = unsafe {
            self.symbol::<CuMemcpyDtoH>("cuMemcpyDtoH_v2")?(
                buffer.as_mut_ptr() as *mut c_void,
                pointer,
                buffer.len(),
            )
        };
        if status != 0 {
            return Err(format!("cuMemcpyDtoH returned {status}"));
        }
        Ok(buffer)
    }

    pub fn load_module(&self, image: &[u8]) -> Result<u64, String> {
        let mut module = 0u64;
        let status = unsafe {
            self.symbol::<CuModuleLoadData>("cuModuleLoadData")?(
                &mut module,
                image.as_ptr() as *const c_void,
            )
        };
        if status != 0 {
            return Err(format!("cuModuleLoadData returned {status}"));
        }
        Ok(module)
    }

    pub fn function(&self, module: u64, name: &str) -> Result<u64, String> {
        let symbol = CString::new(name).map_err(|_| "the kernel name has a null byte".to_string())?;
        let mut function = 0u64;
        let status = unsafe {
            self.symbol::<CuModuleGetFunction>("cuModuleGetFunction")?(
                &mut function,
                module,
                symbol.as_ptr(),
            )
        };
        if status != 0 {
            return Err(format!("cuModuleGetFunction({name}) returned {status}"));
        }
        Ok(function)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch(
        &self,
        function: u64,
        grid: [u32; 3],
        block: [u32; 3],
        shared_bytes: u32,
        stream: u64,
        parameters: &mut [*mut c_void],
    ) -> Result<(), String> {
        let status = unsafe {
            self.symbol::<CuLaunchKernel>("cuLaunchKernel")?(
                function,
                grid[0],
                grid[1],
                grid[2],
                block[0],
                block[1],
                block[2],
                shared_bytes,
                stream,
                parameters.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if status != 0 {
            return Err(format!("cuLaunchKernel returned {status}"));
        }
        Ok(())
    }

    pub fn synchronize_stream(&self, stream: u64) -> Result<(), String> {
        let status =
            unsafe { self.symbol::<CuStreamSynchronize>("cuStreamSynchronize")?(stream) };
        if status != 0 {
            return Err(format!("cuStreamSynchronize returned {status}"));
        }
        Ok(())
    }

    pub fn create_event(&self) -> Result<u64, String> {
        let mut event = 0u64;
        let status = unsafe { self.symbol::<CuEventCreate>("cuEventCreate")?(&mut event, 0) };
        if status != 0 {
            return Err(format!("cuEventCreate returned {status}"));
        }
        Ok(event)
    }

    pub fn record_event(&self, event: u64, stream: u64) -> Result<(), String> {
        let status = unsafe { self.symbol::<CuEventRecord>("cuEventRecord")?(event, stream) };
        if status != 0 {
            return Err(format!("cuEventRecord returned {status}"));
        }
        Ok(())
    }

    pub fn synchronize_event(&self, event: u64) -> Result<(), String> {
        let status = unsafe { self.symbol::<CuEventSynchronize>("cuEventSynchronize")?(event) };
        if status != 0 {
            return Err(format!("cuEventSynchronize returned {status}"));
        }
        Ok(())
    }

    pub fn elapsed(&self, start: u64, end: u64) -> Result<f32, String> {
        let mut milliseconds = 0f32;
        let status = unsafe {
            self.symbol::<CuEventElapsedTime>("cuEventElapsedTime")?(&mut milliseconds, start, end)
        };
        if status != 0 {
            return Err(format!("cuEventElapsedTime returned {status}"));
        }
        Ok(milliseconds)
    }
}

fn load_library() -> Result<Library, String> {
    let mut tried = Vec::new();
    if let Ok(override_path) = std::env::var("LETIFY_CUDA_LIBRARY") {
        tried.push(override_path.clone());
        if let Ok(library) = unsafe { Library::new(&override_path) } {
            return Ok(library);
        }
    }
    for candidate in CANDIDATES {
        tried.push((*candidate).to_string());
        if let Ok(library) = unsafe { Library::new(candidate) } {
            return Ok(library);
        }
    }
    Err(format!(
        "no CUDA driver library could be loaded. Tried: {}. Set LETIFY_CUDA_LIBRARY to its \
         path if it lives somewhere else.",
        tried.join(", ")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_driver_is_reported_with_what_was_tried() {
        // The agent has to build and run on a machine with no CUDA at all, so this path
        // is an error message rather than a link failure.
        if load_library().is_err() {
            let message = load_library().unwrap_err();
            assert!(message.contains("nvcuda.dll"));
            assert!(message.contains("LETIFY_CUDA_LIBRARY"));
        }
    }

    #[test]
    fn the_capability_attributes_match_the_header() {
        assert_eq!(ATTRIBUTE_COMPUTE_MAJOR, 75);
        assert_eq!(ATTRIBUTE_COMPUTE_MINOR, 76);
    }
}
