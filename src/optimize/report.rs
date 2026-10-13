//! Progress reports during a search, and the outcome at its end.

use super::{Backend, Trial};
use crate::config::Config;
use crate::profiles::{Profile, Tuning};
use crate::settings::Value;
use std::fmt;
use std::time::Duration;

/// What happened, for a progress report.
#[derive(Clone, Debug)]
pub enum Event {
    /// The search is about to start.
    Begin { problems: usize, backends: Vec<Backend> },
    /// The starting settings, measured on one backend.
    Baseline { backend: Backend, seconds: f64 },
    /// A backend could not run the starting settings and is left out.
    Skipped { backend: Backend, reason: String },
    /// A new phase of the search.
    Phase { backend: Backend, what: &'static str },
    /// One candidate measured. `current` is the time of the setting's
    /// current value in this pass, None while that is what is being measured.
    Measured { trial: Trial, current: Option<f64> },
    /// A setting settled: moved to `value` when `kept`, else unchanged.
    Settled { backend: Backend, setting: &'static str, value: Value, kept: bool, gain: f64 },
    /// `value` looked faster than `current` but lost the recheck, so
    /// `current` stays.
    Unconfirmed { backend: Backend, setting: &'static str, value: Value, current: Value },
    /// The time limit ran out; the rest of the search is skipped.
    OutOfTime,
    /// Iteration `number` of `of` begins, from the settings found so far.
    Iteration { number: usize, of: usize },
    /// Iteration `number` moved no setting, so later ones would repeat it.
    Converged { number: usize },
    /// The closing like-for-like measurement of start and result.
    Compared { backend: Backend, start: f64, tuned: f64 },
}

/// One progress report: an event, and how far along the search is.
#[derive(Clone, Debug)]
pub struct Progress {
    pub event: Event,
    /// Measurements done, and the total now expected.
    pub done: usize,
    pub total: usize,
    pub elapsed: Duration,
    /// Estimated time left, once there is anything to estimate from.
    pub remaining: Option<Duration>,
}

impl fmt::Display for Progress {
    /// One line for a terminal.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let width = self.total.to_string().len();
        let left = self.remaining.map_or("estimating".into(), |r| format!("~{} left", clock(r)));
        write!(f, "[{:>width$}/{} {} | {}] ", self.done, self.total, clock(self.elapsed), left)?;
        match &self.event {
            Event::Begin { problems, backends } => {
                let names: Vec<_> = backends.iter().map(|b| b.to_string()).collect();
                write!(f, "tuning {} on {problems} problem(s)", names.join(" and "))
            }
            Event::Baseline { backend, seconds } => {
                write!(f, "{backend} starting settings: {}", secs(*seconds))
            }
            Event::Skipped { backend, reason } => write!(f, "{backend} left out: {reason}"),
            Event::Phase { backend, what } => write!(f, "{backend}: tuning {what}"),
            Event::Measured { trial, current } => {
                let what = format!("{} {}={}", trial.backend, trial.setting, trial.value);
                match (trial.seconds, current) {
                    (None, _) => write!(f, "{what:<32} rejected: {}", trial.note),
                    (Some(s), None) => write!(f, "{what:<32} {:>9}  (current)", secs(s)),
                    (Some(s), Some(c)) => write!(f, "{what:<32} {:>9}  ({:+.1}% vs current)", secs(s), (s / c - 1.0) * 100.0),
                }
            }
            Event::Settled { backend, setting, value, kept, gain } => {
                if *kept {
                    write!(f, "{backend} {setting} -> {value}: {:.1}% faster on a recheck, kept", gain * 100.0)
                } else {
                    write!(f, "{backend} {setting} stays {value}")
                }
            }
            Event::Unconfirmed { backend, setting, value, current } => write!(
                f,
                "{backend} {setting}={value} was not faster on a recheck; stays {current}"
            ),
            Event::OutOfTime => write!(f, "time limit reached; keeping the best so far"),
            Event::Iteration { number, of } => {
                write!(f, "iteration {number} of {of}: starting again from the settings found so far")
            }
            Event::Converged { number } => write!(
                f,
                "iteration {number} changed nothing: the settings have settled, skipping the rest"
            ),
            Event::Compared { backend, start, tuned } => write!(
                f,
                "{backend} final: {} -> {}  ({:.2}x)",
                secs(*start),
                secs(*tuned),
                start / tuned
            ),
        }
    }
}

/// What [`optimize`] found.
#[derive(Clone, Debug)]
pub struct Outcome {
    /// The tuned settings, selecting the faster backend with `force_cpu`.
    /// Both backends keep their own tuned settings in it.
    pub config: Config,
    /// Iterations run, and asked for. Fewer ran when the settings settled
    /// or the time limit ran out.
    pub iterations: (usize, usize),
    /// The settings the search started from.
    pub start: Config,
    /// The backend `config` selects.
    pub backend: Backend,
    /// The backend `start` selects, which the headline gain compares against.
    pub start_backend: Backend,
    /// Per backend: summed seconds under the start and under the result,
    /// measured back to back at the end.
    pub seconds: Vec<(Backend, f64, f64)>,
    pub trials: Vec<Trial>,
    pub elapsed: Duration,
    /// Order and stored entries of each problem.
    pub problems: Vec<(usize, usize)>,
    pub analysis: bool,
}

impl Outcome {
    fn time(&self, backend: Backend, tuned: bool) -> f64 {
        let row = self.seconds.iter().find(|r| r.0 == backend).expect("measured backend");
        if tuned { row.2 } else { row.1 }
    }

    /// How many times faster the result is than the start, each on the
    /// backend it selects.
    pub fn speedup(&self) -> f64 {
        self.time(self.start_backend, false) / self.time(self.backend, true)
    }

    /// Every setting that moved, as `(name, start, result)`.
    pub fn changes(&self) -> Vec<(&'static str, Value, Value)> {
        self.start
            .settings()
            .into_iter()
            .zip(self.config.settings())
            .filter(|((name, a), (_, b))| a != b && *name != "force_cpu")
            .map(|((name, a), (_, b))| (name, a, b))
            .collect()
    }

    /// The result as a profile, ready for [`profiles::save`](crate::profiles::save).
    pub fn profile(&self) -> Profile {
        let mut profile = Profile::new(self.config.clone());
        profile.tuning = Some(Tuning {
            backend: self.backend.name().into(),
            speedup: self.speedup(),
            default_seconds: self.time(self.start_backend, false),
            tuned_seconds: self.time(self.backend, true),
            analysis: self.analysis,
            problems: self.problems.clone(),
        });
        profile
    }
}

impl fmt::Display for Outcome {
    /// A summary: per-backend timings, the headline gain and what changed.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let chip = crate::profiles::machine().chip;
        writeln!(f, "SPARTA optimize: {} problem(s) on {chip}, {}", self.problems.len(), clock(self.elapsed))?;
        writeln!(f, "  {:<5}{:>12}{:>12}{:>9}", "", "default", "tuned", "gain")?;
        for &(backend, start, tuned) in &self.seconds {
            writeln!(f, "  {:<5}{:>12}{:>12}{:>8.2}x", backend.to_string(), secs(start), secs(tuned), start / tuned)?;
        }
        let speedup = self.speedup();
        writeln!(
            f,
            "  Result: uses the {}, {speedup:.2}x the speed of the default configuration ({}).",
            self.backend,
            if speedup >= 1.0 {
                format!("{:.0}% less time", (1.0 - 1.0 / speedup) * 100.0)
            } else {
                format!("{:.0}% more time", (1.0 / speedup - 1.0) * 100.0)
            }
        )?;
        let (ran, asked) = self.iterations;
        if asked > 1 {
            writeln!(f, "  Iterations: {ran} of {asked}{}.", if ran < asked { ", stopped early" } else { "" })?;
        }
        let changes = self.changes();
        if changes.is_empty() {
            write!(f, "  No setting beat the defaults by more than the noise; they stand.")
        } else {
            let moved: Vec<_> = changes.iter().map(|(n, a, b)| format!("{n} {a} -> {b}")).collect();
            write!(f, "  Changed: {}", moved.join(", "))
        }
    }
}

pub(super) fn first_line(text: &str) -> String {
    text.lines().next().unwrap_or("").chars().take(120).collect()
}

/// Seconds with a unit that suits their size.
fn secs(s: f64) -> String {
    if !s.is_finite() {
        "failed".into()
    } else if s < 1.0 {
        format!("{:.1} ms", s * 1e3)
    } else {
        format!("{s:.3} s")
    }
}

/// A duration as `m:ss`, or `h:mm:ss` past an hour.
fn clock(d: Duration) -> String {
    let s = d.as_secs();
    if s >= 3600 {
        format!("{}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60)
    } else {
        format!("{}:{:02}", s / 60, s % 60)
    }
}

