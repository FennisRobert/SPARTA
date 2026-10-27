//! C and C++ binding for SPARTA. The header this implements is
//! `include/sparta.h`, which is also the documentation a caller reads.
//!
//! Every exported function follows the same three rules:
//!
//! * it returns a `sparta_status`, or a value with a harmless fallback;
//! * it checks its pointers before touching them, and treats null as a
//!   reportable mistake rather than a crash;
//! * it runs inside [`status::guard`], so a panic becomes an error code
//!   instead of unwinding into C, where it would be undefined behaviour.

mod analysis;
mod config;
mod solver;
mod status;

use std::ffi::{c_char, CStr};

/// Returns the library version as text, for example "0.3.1".
#[no_mangle]
pub extern "C" fn sparta_version() -> *const c_char {
    const VERSION: &CStr = match CStr::from_bytes_with_nul(concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes()) {
        Ok(text) => text,
        Err(_) => unreachable!(),
    };
    VERSION.as_ptr()
}

/// Returns the GPU backend this library was built for, as a static string.
///
/// One build carries one backend's kernels, so this is a property of the
/// library rather than of the machine. Pair it with `sparta_gpu_available`,
/// which reports whether this machine can actually run them.
#[no_mangle]
pub extern "C" fn sparta_device() -> *const std::ffi::c_char {
    // Static and NUL-terminated, so the pointer stays valid for the process.
    match sparta::Device::COMPILED {
        sparta::Device::Metal => c"metal".as_ptr(),
        sparta::Device::Vulkan => c"vulkan".as_ptr(),
    }
}

/// Returns 1 when a usable GPU is visible to this process, 0 otherwise.
#[no_mangle]
pub extern "C" fn sparta_gpu_available() -> i32 {
    status::guard_value(0, || i32::from(sparta::gpu_available()))
}
