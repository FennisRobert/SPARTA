//! Opt-in wall-clock timing for the symbolic analysis phases.
//!
//! Enabled by a nonzero `SPARTA_PROFILE`, the same switch `src/metal.m` reads
//! for per-kernel GPU timing. When it is unset every function here is a branch
//! on a cached bool, so instrumented call sites cost nothing in normal runs.
//!
//! Phases accumulate in a thread-local, so the timed call sites do not appear
//! in any signature and the public symbolic API is unchanged. One analysis runs
//! start to report on a single thread; SCALPEL's Rayon workers are nested
//! inside the `order` phase and record nothing of their own.

use std::cell::RefCell;
use std::sync::OnceLock;
use std::time::Instant;

/// Reports whether `SPARTA_PROFILE` is set to something other than `0`.
/// The environment is read once and cached for the life of the process.
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("SPARTA_PROFILE").is_ok_and(|v| !v.is_empty() && v != "0"))
}

/// A timing run: when the previous mark landed, and the phases closed so far.
type Run = (Instant, Vec<(&'static str, f64)>);

thread_local! {
    static RUN: RefCell<Option<Run>> = const { RefCell::new(None) };
}

/// Begins a timing run on this thread, discarding any unreported phases.
pub(crate) fn start() {
    if !enabled() {
        return;
    }
    RUN.with(|r| *r.borrow_mut() = Some((Instant::now(), Vec::new())));
}

/// Records the time since the previous mark, or since `start`, under `name`.
pub(crate) fn mark(name: &'static str) {
    if !enabled() {
        return;
    }
    RUN.with(|r| {
        if let Some((last, rows)) = r.borrow_mut().as_mut() {
            let now = Instant::now();
            rows.push((name, now.duration_since(*last).as_secs_f64()));
            *last = now;
        }
    });
}

/// Prints one table of phases, their milliseconds and their share of the
/// total, then ends the run. `label` names the analysis, such as the ordering
/// backend in use.
pub(crate) fn report(label: &str) {
    if !enabled() {
        return;
    }
    let Some((_, rows)) = RUN.with(|r| r.borrow_mut().take()) else {
        return;
    };
    if rows.is_empty() {
        return;
    }
    let total: f64 = rows.iter().map(|r| r.1).sum();
    eprintln!("\nSPARTA_PROFILE  symbolic analysis: {label}");
    eprintln!("  {:<28} {:>10} {:>8}", "phase", "ms", "share");
    for (name, seconds) in &rows {
        eprintln!(
            "  {:<28} {:>10.3} {:>7.1}%",
            name,
            seconds * 1e3,
            100.0 * seconds / total.max(f64::MIN_POSITIVE)
        );
    }
    eprintln!("  {:<28} {:>10.3} {:>7.1}%", "total", total * 1e3, 100.0);
}
