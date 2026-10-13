//! Opt-in accounting for the CPU factorization, printed when `SPARTA_PROFILE`
//! is set -- the same switch `src/metal.m` reads for per-kernel GPU timing.
//!
//! A [`Report`] is threaded through the factorization by value, so fronts
//! running on different threads never share one, and the totals are summed at
//! the end. When the switch is off every call site is a branch on a cached
//! bool and the clock is never read.
use crate::profile::enabled;
use std::time::Instant;

/// Phases of one front, in the order they run.
#[derive(Clone, Copy)]
pub(super) enum Phase {
    /// Zero the reused front slot.
    Blank,
    /// Scatter input entries and add every child's Schur complement.
    Assemble,
    /// Factor the fully summed block: `zsytrf` or `zgetrf` and its bookkeeping.
    Leading,
    /// Form the off-diagonal factor: gather, triangular solve, pivot scaling.
    Offdiag,
    /// Subtract the Schur complement from the trailing block.
    Schur,
    /// The `panel.rs` fallback, whole.
    Panels,
    /// The serial narrow-front kernel, whole.
    Serial,
    /// Copy the finished columns into permanent factor storage.
    Store,
}
const NAMES: [&str; 8] = [
    "blank",
    "assemble",
    "leading",
    "offdiag",
    "schur",
    "panels",
    "serial",
    "store",
];

/// Wall time and complex multiply-adds per phase, and the front counts.
#[derive(Clone, Default)]
pub(super) struct Report {
    seconds: [f64; NAMES.len()],
    work: [f64; NAMES.len()],
    /// Fronts taking each route.
    blocked: u64,
    fallback: u64,
    serial: u64,
    /// Wall seconds in levels spread across the pool, and in the rest.
    spread: f64,
    narrow: f64,
    /// Rows and columns of the widest front.
    widest: (usize, usize),
}

impl Report {
    /// Start a phase clock, or `None` when profiling is off.
    pub(super) fn clock() -> Option<Instant> {
        enabled().then(Instant::now)
    }

    /// Close a clock opened by [`Report::clock`] into `phase`.
    pub(super) fn stop(&mut self, phase: Phase, started: Option<Instant>) {
        if let Some(started) = started {
            self.seconds[phase as usize] += started.elapsed().as_secs_f64();
        }
    }

    /// Record one front of `rows` by `columns` taking `route`, and the complex
    /// multiply-adds each phase of it performs.
    ///
    /// The three terms are the fully summed block's own factorization, the
    /// triangular solve that forms the off-diagonal factor, and the rank-`nc`
    /// Schur update. A fallback or serial front does all three in one phase.
    pub(super) fn front(&mut self, route: Phase, rows: usize, columns: usize) {
        if !enabled() {
            return;
        }
        let (tail, n) = ((rows - columns) as f64, columns as f64);
        let (leading, offdiag, schur) = (n * n * n / 3.0, tail * n * n / 2.0, tail * tail * n);
        match route {
            Phase::Panels => {
                self.fallback += 1;
                self.work[Phase::Panels as usize] += leading + offdiag + schur;
            }
            Phase::Serial => {
                self.serial += 1;
                self.work[Phase::Serial as usize] += leading + offdiag + schur;
            }
            _ => {
                self.blocked += 1;
                self.work[Phase::Leading as usize] += leading;
                self.work[Phase::Offdiag as usize] += offdiag;
                self.work[Phase::Schur as usize] += schur;
            }
        }
        self.widest = self.widest.max((rows, columns));
    }

    /// Add `seconds` of wall time spent in a level, by how it ran.
    pub(super) fn level(&mut self, spread: bool, started: Option<Instant>) {
        if let Some(started) = started {
            let seconds = started.elapsed().as_secs_f64();
            if spread {
                self.spread += seconds;
            } else {
                self.narrow += seconds;
            }
        }
    }

    /// Fold a worker's totals into this one.
    pub(super) fn merge(&mut self, other: &Report) {
        for (row, add) in self.seconds.iter_mut().zip(other.seconds) {
            *row += add;
        }
        for (row, add) in self.work.iter_mut().zip(other.work) {
            *row += add;
        }
        self.widest = self.widest.max(other.widest);
        self.spread += other.spread;
        self.narrow += other.narrow;
        self.blocked += other.blocked;
        self.fallback += other.fallback;
        self.serial += other.serial;
    }

    /// Print one table against the `wall` seconds the factorization took.
    pub(super) fn print(&self, wall: f64) {
        if !enabled() {
            return;
        }
        let total: f64 = self.seconds.iter().sum();
        // Eight real flops per complex multiply-add.
        let rate = |work: f64, seconds: f64| 8.0 * work / seconds.max(f64::MIN_POSITIVE) / 1e9;
        eprintln!("\nSPARTA_PROFILE  CPU factorization");
        eprintln!(
            "  {:<12} {:>10} {:>8} {:>10} {:>10}",
            "phase", "cpu ms", "share", "GFLOP", "GFLOP/s"
        );
        for (i, name) in NAMES.iter().enumerate() {
            let (seconds, work) = (self.seconds[i], self.work[i]);
            eprintln!(
                "  {:<12} {:>10.1} {:>7.1}% {:>10.1} {:>10.0}",
                name,
                seconds * 1e3,
                100.0 * seconds / total.max(f64::MIN_POSITIVE),
                8.0 * work / 1e9,
                rate(work, seconds)
            );
        }
        let work: f64 = self.work.iter().sum();
        eprintln!(
            "  {:<12} {:>10.1} {:>7.1}% {:>10.1} {:>10.0}",
            "total",
            total * 1e3,
            100.0,
            8.0 * work / 1e9,
            rate(work, total)
        );
        eprintln!(
            "  {:<12} {:>10.1} {:>8} {:>10} {:>10.0}",
            "wall",
            wall * 1e3,
            format!("x{:.2}", total / wall.max(f64::MIN_POSITIVE)),
            "",
            rate(work, wall)
        );
        eprintln!(
            "  levels       {:.1} ms spread across the pool, {:.1} ms one front at a time",
            self.spread * 1e3,
            self.narrow * 1e3
        );
        eprintln!(
            "  fronts       {} blocked, {} panel fallback, {} serial; widest {}x{}",
            self.blocked, self.fallback, self.serial, self.widest.0, self.widest.1
        );
    }
}
