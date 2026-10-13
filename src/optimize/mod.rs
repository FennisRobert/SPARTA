//! Tunes a [`Config`] to this computer and to the matrices it will factor.
//!
//! The best settings depend on the hardware and on the matrices, and the only
//! way to find them is to measure. [`optimize`] factors the given problems
//! under candidate settings, keeps what is faster, and returns an ordinary
//! `Config`, which [`profiles::save`](crate::profiles::save) can keep under a
//! name.
//!
//! The search is coordinate descent, not a grid: the grid is far too large
//! and the settings are only loosely coupled. Settings that shape the plan are
//! tuned first, on whichever backend is faster at the start, since both
//! backends share one plan; each backend's own settings follow. Every
//! candidate is measured against the current value in the same pass, so slow
//! thermal drift does not pass for a win, and a change is kept only when it
//! beats the current value by more than [`Options::min_gain`], and by more than the
//! timing noise of either, twice: once in the pass
//! and again in a head-to-head recheck, so one lucky measurement is not
//! enough. [`Options::iterations`] repeats the whole pass from the settings
//! found, until one changes nothing.
//!
//! A measurement is the median wall time of factor plus solve on an analyzed
//! plan, the cost paid for every new set of values, summed over the problems.
//! [`Options::analysis`] adds the analysis for patterns factored once. The
//! first factorization of each plan is not timed: it allocates and, on the
//! GPU, may compile kernels for new geometry, and neither recurs. A candidate
//! whose residual is more than [`ACCURACY`] times the starting settings' is
//! rejected, so no setting is bought with accuracy, and `lu_diag_threshold`,
//! which trades stability for speed by design, is never searched.
//!
//! Progress goes to a callback after every measurement, with the count done,
//! the count left and an estimate of the time left; returning `false` from it
//! stops the search.

mod problem;
mod report;
mod search;
mod space;

pub use problem::Problem;
pub use report::{Event, Outcome, Progress};

use crate::config::Config;
use crate::settings::Value;
use crate::{gpu_available, Error, Result};
use search::Search;
use space::PLAN;
use std::fmt;
use std::time::{Duration, Instant};

/// The default [`Options::min_gain`].
pub const MIN_GAIN: f64 = 0.03;
/// A candidate's residual may be at most this multiple of the start's.
pub const ACCURACY: f64 = 10.0;

/// Where factorization runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Backend {
    Gpu,
    Cpu,
}

impl Backend {
    fn own(self) -> &'static [(&'static str, space::Choices)] {
        match self {
            Backend::Gpu => space::GPU,
            Backend::Cpu => space::CPU,
        }
    }
    fn select(self, config: &Config) -> Config {
        Config {
            force_cpu: self == Backend::Cpu,
            ..config.clone()
        }
    }
    /// `"gpu"` or `"cpu"`.
    pub fn name(self) -> &'static str {
        match self {
            Backend::Gpu => "gpu",
            Backend::Cpu => "cpu",
        }
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Backend::Gpu => "GPU",
            Backend::Cpu => "CPU",
        })
    }
}

/// What to tune, and within what limits.
#[derive(Clone, Debug)]
pub struct Options {
    /// Where the search starts, and what the report compares against.
    pub start: Config,
    /// Tune the GPU backend; skipped when no GPU is available.
    pub gpu: bool,
    /// Tune the CPU backend.
    pub cpu: bool,
    /// Count analysis time too. Leave off when a pattern is factored many
    /// times; turn on when each pattern is factored once.
    pub analysis: bool,
    /// Stop searching after this long and keep the best found so far. The
    /// closing comparison against the start runs regardless.
    pub time_limit: Option<Duration>,
    /// Full passes over every setting. Each pass after the first starts from
    /// the settings the previous one found, so a setting whose best value
    /// depends on another is revisited once that other has moved. The search
    /// stops early after a pass that changes nothing. At least 1.
    pub iterations: usize,
    /// The smallest improvement worth keeping, as a fraction: 0.03 keeps a
    /// change only when it saves more than 3%. Noisier timings raise the bar
    /// to their noise, so small wins that may be chance are left alone.
    pub min_gain: f64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            start: Config::default(),
            gpu: true,
            cpu: true,
            analysis: false,
            time_limit: None,
            iterations: 1,
            min_gain: MIN_GAIN,
        }
    }
}

/// One measured candidate.
#[derive(Clone, Debug, PartialEq)]
pub struct Trial {
    pub backend: Backend,
    pub setting: &'static str,
    pub value: Value,
    /// Summed seconds over the problems, or None when it failed or lost accuracy.
    pub seconds: Option<f64>,
    /// Why it has no time.
    pub note: String,
}

/// Finds the fastest settings for `problems` on this computer.
///
/// `report` receives a [`Progress`] after every step; returning `false`
/// stops the search with an error. See the module documentation for how the
/// search runs.
pub fn optimize(
    problems: &[Problem],
    options: &Options,
    report: &mut dyn FnMut(&Progress) -> bool,
) -> Result<Outcome> {
    if problems.is_empty() {
        return Err(Error::Invalid("give at least one problem to tune on".into()));
    }
    options.start.check().map_err(Error::Invalid)?;
    if !(0.0..1.0).contains(&options.min_gain) {
        return Err(Error::Invalid("min_gain must be at least 0 and below 1".into()));
    }
    if options.iterations == 0 {
        return Err(Error::Invalid("iterations must be at least 1".into()));
    }
    let iterations = options.iterations;
    let mut search = Search {
        problems,
        analysis: options.analysis,
        began: Instant::now(),
        deadline: options.time_limit.map(|t| Instant::now() + t),
        min_gain: options.min_gain,
        stopped: false,
        done: 0,
        total: 0,
        cost: Vec::new(),
        planned: Vec::new(),
        trials: Vec::new(),
        reference: Vec::new(),
        report,
    };
    let mut backends = Vec::new();
    if options.gpu && gpu_available() {
        backends.push(Backend::Gpu);
    }
    if options.cpu {
        backends.push(Backend::Cpu);
    }
    if backends.is_empty() {
        return Err(Error::Invalid(if options.gpu {
            "no GPU is available and the CPU is not selected".into()
        } else {
            "select the GPU, the CPU or both".into()
        }));
    }
    // Everything the search will measure, so progress has a total from the
    // start: a baseline and a closing pair per backend and, per iteration,
    // the plan settings on one backend and each backend's own settings on it.
    for &b in &backends {
        let own: usize = b.own().iter().map(|(_, c)| c.values().len()).sum();
        search.planned.push((b, 1 + own * iterations + 2));
    }
    let plan: usize = PLAN.iter().map(|(_, c)| c.values().len()).sum::<usize>() * iterations;
    search.planned[0].1 += plan;
    search.total = search.planned.iter().map(|p| p.1).sum();
    search.emit(Event::Begin { problems: problems.len(), backends: backends.clone() })?;

    let start = &options.start;
    let mut baseline = Vec::new();
    for &b in &backends.clone() {
        match search.score(start, b) {
            Ok(score) => {
                let seconds = score.seconds;
                search.reference.push((b, score.residuals));
                baseline.push((b, seconds));
                search.emit(Event::Baseline { backend: b, seconds })?;
            }
            Err(reason) => {
                backends.retain(|&x| x != b);
                search.forget(b);
                search.emit(Event::Skipped { backend: b, reason })?;
            }
        }
    }
    let Some(&(primary, _)) = baseline.iter().min_by(|a, b| a.1.total_cmp(&b.1)) else {
        return Err(Error::Invalid("no backend can factor these problems with the starting settings".into()));
    };
    if primary != backends[0] {
        // The plan is tuned on the faster backend; move its share of the total.
        let share = plan.min(search.remaining_for(backends[0]));
        search.shift(backends[0], primary, share);
    }
    let start_backend = if start.force_cpu || !backends.contains(&Backend::Gpu) {
        Backend::Cpu
    } else {
        Backend::Gpu
    };
    if !backends.contains(&start_backend) {
        return Err(Error::Invalid(format!("the starting settings select the {start_backend}, which is not being tuned")));
    }

    let mut best = start.clone();
    let mut ran = 0;
    for number in 1..=iterations {
        if search.stopped {
            break;
        }
        if iterations > 1 {
            search.emit(Event::Iteration { number, of: iterations })?;
        }
        let before = best.clone();
        search.emit(Event::Phase { backend: primary, what: "the plan (ordering and supernodes)" })?;
        search.sweep(&mut best, primary, PLAN)?;
        for &b in &backends {
            if search.stopped {
                break;
            }
            let what = match b {
                Backend::Gpu => "the GPU kernels and their geometry",
                Backend::Cpu => "the CPU dense updates",
            };
            search.emit(Event::Phase { backend: b, what })?;
            search.sweep(&mut best, b, b.own())?;
        }
        ran = number;
        if best == before && number < iterations && !search.stopped {
            search.only_closing_left();
            search.emit(Event::Converged { number })?;
            break;
        }
    }

    // Start and result back to back, so the reported gain is like for like.
    let mut seconds = Vec::new();
    for &b in &backends {
        let before = search.score(start, b).map(|s| s.seconds).unwrap_or_else(|_| {
            baseline.iter().find(|x| x.0 == b).map_or(f64::INFINITY, |x| x.1)
        });
        let after = search.score(&best, b).map_or(f64::INFINITY, |s| s.seconds);
        search.emit(Event::Compared { backend: b, start: before, tuned: after })?;
        seconds.push((b, before, after));
    }
    let (winner, _, tuned) = *seconds
        .iter()
        .min_by(|a, b| a.2.total_cmp(&b.2))
        .expect("at least one backend");
    let reference = seconds.iter().find(|s| s.0 == start_backend).expect("start backend").1;
    let (config, backend) = if tuned < reference {
        (winner.select(&best), winner)
    } else {
        // Nothing survived the closing comparison: the start stands.
        for row in &mut seconds {
            row.2 = row.1;
        }
        (start_backend.select(start), start_backend)
    };
    Ok(Outcome {
        config,
        iterations: (ran, iterations),
        start: start.clone(),
        backend,
        start_backend,
        seconds,
        trials: search.trials,
        elapsed: search.began.elapsed(),
        problems: problems.iter().map(|p| (p.order(), p.stored())).collect(),
        analysis: options.analysis,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// A CPU-only optimization of a small problem finishes, reports progress
    /// that ends where it said it would, and never loses accuracy.
    fn optimizes_a_small_problem_on_the_cpu() {
        let problems = vec![Problem::helmholtz(8)];
        let options = Options {
            gpu: false,
            time_limit: Some(Duration::from_secs(20)),
            ..Options::default()
        };
        let mut last = None;
        let outcome = optimize(&problems, &options, &mut |p| {
            assert!(p.done <= p.total);
            last = Some((p.done, p.total));
            true
        })
        .unwrap();
        let (done, total) = last.unwrap();
        assert_eq!(done, total, "the estimate of the total must come true");
        assert_eq!(outcome.backend, Backend::Cpu);
        assert!(outcome.config.force_cpu);
        assert!(outcome.speedup() > 0.0);
        assert!(outcome.config.check().is_ok());
    }

    #[test]
    /// A second iteration runs from the first one's result and, when the
    /// first changed nothing, is skipped as settled; either way the progress
    /// total comes true.
    fn iterations_repeat_until_settled() {
        let problems = vec![Problem::helmholtz(6)];
        let options = Options {
            gpu: false,
            iterations: 3,
            ..Options::default()
        };
        let (mut last, mut iterations, mut settled) = (None, 0, false);
        let outcome = optimize(&problems, &options, &mut |p| {
            match p.event {
                Event::Iteration { .. } => iterations += 1,
                Event::Converged { .. } => settled = true,
                _ => {}
            }
            last = Some((p.done, p.total));
            true
        })
        .unwrap();
        let (done, total) = last.unwrap();
        assert_eq!(done, total);
        assert_eq!(outcome.iterations, (iterations, 3));
        assert_eq!(settled, iterations < 3);
        assert!(optimize(&problems, &Options { iterations: 0, ..options }, &mut |_| true).is_err());
    }

    #[test]
    /// Returning false from the callback stops the search.
    fn the_callback_can_stop_it() {
        let problems = vec![Problem::helmholtz(6)];
        let options = Options { gpu: false, ..Options::default() };
        let mut calls = 0;
        let result = optimize(&problems, &options, &mut |_| {
            calls += 1;
            calls < 3
        });
        assert!(result.is_err());
        assert_eq!(calls, 3);
    }
}
