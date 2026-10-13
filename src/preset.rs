//! Named starting points for [`Config`], and the text format that ships them.
//!
//! A preset is a [`Config`] with a few fields moved off their defaults for a
//! recognisable kind of workload. It is a starting point, not a
//! recommendation: the right settings depend on the matrix, and the only way
//! to know is to measure. Start from the closest preset, change one thing at a
//! time, and keep what helps.

use crate::config::Config;
use crate::symbolic::OrderingMethod;
use std::fmt;

/// A named set of starting settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preset {
    /// The measured defaults. Balanced for a single factorization of an
    /// unstructured mesh.
    Balanced,
    /// One pattern factored at many right-hand sides or frequencies. Analysis
    /// is paid once and every factorization benefits, so ordering quality is
    /// bought freely.
    Sweep,
    /// One factorization of one pattern, then discard. Ordering quality is
    /// only worth buying while it saves more factorization time than it costs,
    /// so the expensive refinement is off.
    OneShot,
    /// Structured grids, where wider supernodes keep paying off.
    StructuredGrid,
    /// Ill-conditioned or nearly singular systems. Keeps strict pivoting and
    /// favours a balanced elimination tree over the smallest separator.
    Careful,
}

impl Preset {
    /// Every preset, in a stable order for listing them.
    pub const ALL: [Preset; 5] = [
        Preset::Balanced,
        Preset::Sweep,
        Preset::OneShot,
        Preset::StructuredGrid,
        Preset::Careful,
    ];

    /// The name used by [`Preset::parse`] and the Python bindings.
    pub fn name(self) -> &'static str {
        match self {
            Preset::Balanced => "balanced",
            Preset::Sweep => "sweep",
            Preset::OneShot => "one-shot",
            Preset::StructuredGrid => "structured-grid",
            Preset::Careful => "careful",
        }
    }

    /// One line on when to choose this preset.
    pub fn summary(self) -> &'static str {
        match self {
            Preset::Balanced => "measured defaults; a single factorization of an unstructured mesh",
            Preset::Sweep => "one pattern factored many times; spend on ordering quality",
            Preset::OneShot => "factor once and discard; order fast, accept more fill",
            Preset::StructuredGrid => "regular grids; wider supernodes keep paying",
            Preset::Careful => "ill-conditioned systems; strict pivoting, balanced tree",
        }
    }

    /// Parses a preset name, accepting either `-` or `_` as the separator.
    pub fn parse(name: &str) -> Option<Preset> {
        let wanted = name.trim().to_ascii_lowercase().replace('_', "-");
        Preset::ALL.into_iter().find(|p| p.name() == wanted)
    }

    /// Returns the configuration this preset stands for.
    pub fn config(self) -> Config {
        let base = Config::default();
        match self {
            Preset::Balanced => base,
            Preset::Sweep => Config {
                scalpel: scalpel::Options {
                    wide_cycles: 6,
                    separator_trials: 8,
                    ..base.scalpel
                },
                ordering: OrderingMethod::Scalpel,
                ..base
            },
            Preset::OneShot => Config {
                scalpel: scalpel::Options {
                    wide_cycles: 0,
                    separator_trials: 1,
                    spectral_candidates: 0,
                    refine_all_candidates: false,
                    ..base.scalpel
                },
                ordering: OrderingMethod::Scalpel,
                ..base
            },
            Preset::StructuredGrid => Config {
                nemin: 32,
                ordering: OrderingMethod::Scalpel,
                ..base
            },
            Preset::Careful => Config {
                lu_diag_threshold: 0.0,
                scalpel: scalpel::Options {
                    imbalance: 0.6,
                    ..base.scalpel
                },
                ..base
            },
        }
    }
}

impl fmt::Display for Preset {
    /// Writes the preset's file name.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// Every preset is valid, names round-trip, and each differs from the
    /// balanced one so that choosing it actually changes something.
    fn presets_are_valid_and_distinct() {
        for preset in Preset::ALL {
            assert_eq!(preset.config().check(), Ok(()), "{preset} is invalid");
            assert_eq!(Preset::parse(preset.name()), Some(preset));
            assert_eq!(
                Preset::parse(&preset.name().replace('-', "_")),
                Some(preset)
            );
            if preset != Preset::Balanced {
                assert_ne!(
                    preset.config(),
                    Preset::Balanced.config(),
                    "{preset} matches the balanced preset, so it has no effect"
                );
            }
        }
        assert_eq!(Preset::parse("nope"), None);
        assert_eq!(Preset::Balanced.config(), Config::default());
    }
}
