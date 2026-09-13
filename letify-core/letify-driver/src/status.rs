//! Driver status codes, and saying plainly what is not implemented yet.
//!
//! The values match the CUDA driver's own, because callers compare against them and
//! PyTorch's error handling reads them directly.
//!
//! An unimplemented entry point returns `CUDA_ERROR_NOT_SUPPORTED` and logs its own name.
//! That is deliberate: the honest way to find out what a real workload needs is to run
//! one and read the list, rather than guessing at several hundred symbols up front.

/// The driver's own result type.
pub type CUresult = i32;

// The full set is mirrored even where the local driver does not return one yet, because these
// are the driver's numbers and a caller may compare against any of them.
pub const CUDA_SUCCESS: CUresult = 0;
pub const CUDA_ERROR_INVALID_VALUE: CUresult = 1;
pub const CUDA_ERROR_OUT_OF_MEMORY: CUresult = 2;
pub const CUDA_ERROR_NOT_INITIALIZED: CUresult = 3;
pub const CUDA_ERROR_INVALID_DEVICE: CUresult = 101;
pub const CUDA_ERROR_INVALID_HANDLE: CUresult = 400;
#[allow(dead_code)]
pub const CUDA_ERROR_NOT_FOUND: CUresult = 500;
pub const CUDA_ERROR_NOT_SUPPORTED: CUresult = 801;
pub const CUDA_ERROR_UNKNOWN: CUresult = 999;

/// Write a line to the driver log.
///
/// Off unless `LETIFY_DRIVER_LOG` is set, because this sits on the path of every driver
/// call and a process can make millions of them.
pub fn log(message: &str) {
    if std::env::var_os("LETIFY_DRIVER_LOG").is_none() {
        return;
    }
    eprintln!("letify-driver: {message}");
}

/// Record that an entry point was reached before it was implemented.
///
/// Always logged, even without `LETIFY_DRIVER_LOG`, because a missing symbol is the reason
/// a run failed and the name of it is the next thing to build.
pub fn unimplemented(symbol: &str) -> CUresult {
    eprintln!(
        "letify-driver: {symbol} is not implemented yet, so this call returns \
         CUDA_ERROR_NOT_SUPPORTED. Report the symbol name; the driver surface is being \
         filled in from what real workloads actually reach for."
    );
    CUDA_ERROR_NOT_SUPPORTED
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_codes_match_the_driver() {
        // Callers compare against these numbers, so they are not ours to choose.
        assert_eq!(CUDA_SUCCESS, 0);
        assert_eq!(CUDA_ERROR_OUT_OF_MEMORY, 2);
        assert_eq!(CUDA_ERROR_NOT_SUPPORTED, 801);
    }
}
