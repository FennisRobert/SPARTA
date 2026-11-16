//! Whether a call may spread over the Rayon pool, set per calling thread.
//!
//! The kernels never start threads of their own. A call made with
//! [`Threading::Machine`] splits its work over the Rayon pool it runs in; one
//! made with [`Threading::Caller`] stays on the calling thread. A solver that
//! already spreads independent work over the pool sets `Caller` on its workers
//! and `Machine` around a lone large call, and the two never compete.
use std::cell::Cell;

/// The threading of calls made from this thread.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Threading {
    /// Spread large calls over the Rayon pool. The default.
    Machine,
    /// Run every call on the calling thread.
    Caller,
}

thread_local! {
    static CALLER: Cell<bool> = const { Cell::new(false) };
}

/// Set the threading of later calls made from this thread.
pub fn set(threading: Threading) {
    CALLER.with(|c| c.set(threading == Threading::Caller));
}

/// This thread's setting.
pub fn get() -> Threading {
    if CALLER.with(|c| c.get()) {
        Threading::Caller
    } else {
        Threading::Machine
    }
}

/// Whether calls from this thread may spread.
pub(crate) fn machine() -> bool {
    get() == Threading::Machine
}
