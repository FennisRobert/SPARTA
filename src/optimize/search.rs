//! The coordinate-descent search and its progress accounting.

use super::report::first_line;
use super::space::Choices;
use super::{Backend, Event, Problem, Progress, Trial, ACCURACY};
use crate::config::Config;
use crate::settings::Value;
use crate::{Error, Result};
use std::time::{Duration, Instant};

/// The state of one optimization.
pub(super) struct Search<'a> {
    pub(super) problems: &'a [Problem],
    pub(super) analysis: bool,
    pub(super) began: Instant,
    pub(super) deadline: Option<Instant>,
    /// The smallest improvement worth keeping, as a fraction.
    pub(super) min_gain: f64,
    /// Set once the time limit has run out.
    pub(super) stopped: bool,
    pub(super) done: usize,
    pub(super) total: usize,
    /// Per backend: measurements made and their summed wall time, for the estimate.
    pub(super) cost: Vec<(Backend, usize, f64)>,
    /// Per backend: measurements still expected.
    pub(super) planned: Vec<(Backend, usize)>,
    pub(super) trials: Vec<Trial>,
    /// Per backend: residuals under the starting settings.
    pub(super) reference: Vec<(Backend, Vec<f64>)>,
    pub(super) report: &'a mut dyn FnMut(&Progress) -> bool,
}

/// One measurement of a configuration over every problem.
pub(super) struct Score {
    /// Summed median seconds.
    pub(super) seconds: f64,
    /// The noisiest problem's interquartile range over its median.
    pub(super) spread: f64,
    /// Per problem, the worst relative residual.
    pub(super) residuals: Vec<f64>,
}


impl Search<'_> {
    /// Measures `config` on `backend`, or says why it could not run. Counts
    /// as one measurement for progress.
    pub(super) fn score(&mut self, config: &Config, backend: Backend) -> std::result::Result<Score, String> {
        let config = backend.select(config);
        let begin = Instant::now();
        let result = config.check().and_then(|()| {
            let mut score = Score { seconds: 0.0, spread: 0.0, residuals: Vec::new() };
            for problem in self.problems {
                let (seconds, residual, spread) =
                    problem.measure(&config, self.analysis).map_err(|e| e.to_string())?;
                score.seconds += seconds;
                score.spread = score.spread.max(spread);
                score.residuals.push(residual);
            }
            Ok(score)
        });
        let spent = begin.elapsed().as_secs_f64();
        match self.cost.iter_mut().find(|c| c.0 == backend) {
            Some(c) => {
                c.1 += 1;
                c.2 += spent;
            }
            None => self.cost.push((backend, 1, spent)),
        }
        self.done += 1;
        if let Some(p) = self.planned.iter_mut().find(|p| p.0 == backend) {
            p.1 = p.1.saturating_sub(1);
        }
        result
    }

    /// Tunes each setting of `space` in turn on `backend`, moving `best`.
    pub(super) fn sweep(&mut self, best: &mut Config, backend: Backend, space: &[(&'static str, Choices)]) -> Result<()> {
        for (setting, choices) in space {
            let values = choices.values();
            if self.out_of_time() {
                return self.stop();
            }
            let current = best.get(setting).expect("known setting");
            let mut candidates = vec![current.clone()];
            candidates.extend(values.into_iter().filter(|v| *v != current));
            // The list holds the current value once more than planned when it
            // is not one of the choices.
            let planned = choices.values().len();
            if candidates.len() > planned {
                self.add_planned(backend, candidates.len() - planned);
            }
            let mut timed: Vec<(Value, f64, f64)> = Vec::new();
            let mut current_seconds = None;
            for (index, value) in candidates.into_iter().enumerate() {
                if self.out_of_time() {
                    return self.stop();
                }
                let candidate = best.with(setting, value.clone()).map_err(Error::Invalid)?;
                let trial = match self.score(&candidate, backend) {
                    Err(note) => Trial { backend, setting, value, seconds: None, note: first_line(&note) },
                    Ok(score) if !self.accurate(backend, &score.residuals) => Trial {
                        backend,
                        setting,
                        value,
                        seconds: None,
                        note: format!("residual {:.1e} is over {ACCURACY}x the start's", score.residuals.iter().fold(0.0, |a: f64, &b| a.max(b))),
                    },
                    Ok(score) => {
                        timed.push((value.clone(), score.seconds, score.spread));
                        Trial { backend, setting, value, seconds: Some(score.seconds), note: String::new() }
                    }
                };
                self.trials.push(trial.clone());
                let seconds = trial.seconds;
                self.emit(Event::Measured { trial, current: current_seconds })?;
                if index == 0 {
                    current_seconds = seconds;
                }
            }
            let (current_seconds, current_spread) =
                timed.iter().find(|t| t.0 == current).map_or((f64::INFINITY, 0.0), |t| (t.1, t.2));
            let winner = timed.iter().min_by(|a, b| a.1.total_cmp(&b.1)).cloned();
            match winner {
                Some((value, seconds, spread))
                    if value != current
                        && seconds < current_seconds * (1.0 - self.needed(spread, current_spread)) =>
                {
                    // One lucky measurement is not enough: time both again,
                    // head to head, and keep the change only if it holds.
                    let gain = if current_seconds.is_finite() {
                        match self.recheck(best, backend, setting, &value, &current)? {
                            Some(gain) => gain,
                            None => {
                                self.emit(Event::Unconfirmed { backend, setting, value, current: current.clone() })?;
                                continue;
                            }
                        }
                    } else {
                        0.0
                    };
                    best.set(setting, value.clone()).map_err(Error::Invalid)?;
                    self.emit(Event::Settled { backend, setting, value, kept: true, gain })?;
                }
                _ => self.emit(Event::Settled { backend, setting, value: current, kept: false, gain: 0.0 })?,
            }
        }
        Ok(())
    }

    /// Times `value` and `current` of `setting` once more each. Returns the
    /// gain when `value` is still faster by more than the minimum gain and the
    /// timing noise of both, else None.
    fn recheck(
        &mut self,
        best: &Config,
        backend: Backend,
        setting: &str,
        value: &Value,
        current: &Value,
    ) -> Result<Option<f64>> {
        self.add_planned(backend, 2);
        let time = |search: &mut Self, v: &Value| -> Result<(f64, f64)> {
            let config = best.with(setting, v.clone()).map_err(Error::Invalid)?;
            Ok(search.score(&config, backend).map_or((f64::INFINITY, 0.0), |s| (s.seconds, s.spread)))
        };
        let (proposed, a) = time(self, value)?;
        let (kept, b) = time(self, current)?;
        Ok((proposed < kept * (1.0 - self.needed(a, b))).then(|| 1.0 - proposed / kept))
    }

    /// The fraction by which one time must beat another to count as faster:
    /// the minimum gain, or the timing noise of either when that is larger.
    fn needed(&self, a: f64, b: f64) -> f64 {
        self.min_gain.max(a).max(b)
    }

    pub(super) fn accurate(&self, backend: Backend, residuals: &[f64]) -> bool {
        let reference = &self.reference.iter().find(|r| r.0 == backend).expect("baseline").1;
        residuals
            .iter()
            .zip(reference)
            .all(|(&r, &start)| r.is_finite() && r <= ACCURACY * start + 1e-14)
    }

    pub(super) fn out_of_time(&self) -> bool {
        self.stopped || self.deadline.is_some_and(|d| Instant::now() > d)
    }

    /// Ends the search early, once: only the closing comparison remains.
    pub(super) fn stop(&mut self) -> Result<()> {
        if self.stopped {
            return Ok(());
        }
        self.stopped = true;
        self.only_closing_left();
        self.emit(Event::OutOfTime)
    }

    /// Drops every planned measurement but each backend's closing pair.
    pub(super) fn only_closing_left(&mut self) {
        for p in &mut self.planned {
            p.1 = p.1.min(2);
        }
        self.total = self.done + self.planned.iter().map(|p| p.1).sum::<usize>();
    }

    pub(super) fn remaining_for(&self, backend: Backend) -> usize {
        self.planned.iter().find(|p| p.0 == backend).map_or(0, |p| p.1)
    }
    pub(super) fn drop_planned(&mut self, backend: Backend, count: usize) {
        if let Some(p) = self.planned.iter_mut().find(|p| p.0 == backend) {
            let taken = count.min(p.1);
            p.1 -= taken;
            self.total -= taken;
        }
    }
    pub(super) fn add_planned(&mut self, backend: Backend, count: usize) {
        if let Some(p) = self.planned.iter_mut().find(|p| p.0 == backend) {
            p.1 += count;
            self.total += count;
        }
    }
    pub(super) fn forget(&mut self, backend: Backend) {
        let left = self.remaining_for(backend);
        self.drop_planned(backend, left);
    }
    pub(super) fn shift(&mut self, from: Backend, to: Backend, count: usize) {
        self.drop_planned(from, count);
        self.add_planned(to, count);
    }

    /// The time left: each backend's planned measurements at its average cost
    /// so far, or the overall average for a backend not yet measured.
    pub(super) fn remaining(&self) -> Option<Duration> {
        let (count, spent) = self.cost.iter().fold((0, 0.0), |a, c| (a.0 + c.1, a.1 + c.2));
        if count == 0 {
            return None;
        }
        let overall = spent / count as f64;
        let left: f64 = self
            .planned
            .iter()
            .map(|&(b, n)| {
                let each = self.cost.iter().find(|c| c.0 == b).map_or(overall, |c| c.2 / c.1 as f64);
                n as f64 * each
            })
            .sum();
        Some(Duration::from_secs_f64(left))
    }

    pub(super) fn emit(&mut self, event: Event) -> Result<()> {
        let progress = Progress {
            event,
            done: self.done,
            total: self.total.max(self.done),
            elapsed: self.began.elapsed(),
            remaining: self.remaining(),
        };
        if (self.report)(&progress) {
            Ok(())
        } else {
            Err(Error::Invalid("optimization stopped".into()))
        }
    }
}
