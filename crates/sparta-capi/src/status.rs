//! Result codes and the per-thread message that explains the last failure.

use std::cell::RefCell;
use std::ffi::{c_char, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

/// Mirrors `sparta_status` in `sparta.h`.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Status {
    Ok = 0,
    Invalid = 1,
    Unsupported = 2,
    /// `SPARTA_ERROR_DEVICE`, still spelled `SPARTA_ERROR_METAL` in C for
    /// compatibility. The value is the ABI; the name is not.
    Device = 3,
    Pivot = 4,
    Internal = 5,
}

thread_local! {
    /// Held until the next failure on this thread, which is exactly how long
    /// `sparta_last_error` promises the text stays readable.
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

/// Records `message` as this thread's last failure and returns `status`, so a
/// failing branch can read `return fail(Status::Invalid, "...")`.
pub fn fail(status: Status, message: impl AsRef<str>) -> Status {
    let text = CString::new(message.as_ref()).unwrap_or_default();
    LAST_ERROR.with(|slot| *slot.borrow_mut() = text);
    status
}

/// Translates a solver error into its result code and message.
pub fn from_error(error: sparta::Error) -> Status {
    let status = match error {
        sparta::Error::Invalid(_) => Status::Invalid,
        sparta::Error::Unsupported(_) => Status::Unsupported,
        sparta::Error::Device(_) => Status::Device,
        sparta::Error::Pivot { .. } => Status::Pivot,
    };
    fail(status, error.to_string())
}

/// Runs `body` as the whole of an exported function.
///
/// A Rust panic must never cross back into C, so one is caught here and
/// reported as [`Status::Internal`] instead.
pub fn guard(body: impl FnOnce() -> Status) -> Status {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(status) => status,
        Err(panic) => {
            let detail = panic
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown cause".into());
            fail(Status::Internal, format!("internal error: {detail}"))
        }
    }
}

/// Same as [`guard`], for a function that returns a value rather than a status.
/// A panic yields `fallback`.
pub fn guard_value<T>(fallback: T, body: impl FnOnce() -> T) -> T {
    catch_unwind(AssertUnwindSafe(body)).unwrap_or(fallback)
}

/// Returns the message describing this thread's last failure.
#[no_mangle]
pub extern "C" fn sparta_last_error() -> *const c_char {
    guard_value(ptr::null(), || {
        LAST_ERROR.with(|slot| slot.borrow().as_ptr())
    })
}
