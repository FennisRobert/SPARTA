//! [`Config`] fields by name, for code that handles settings as data.
//!
//! The names are the Python keyword names, which are also the Rust field
//! names apart from the SCALPEL options, flattened out of `scalpel`. A saved
//! profile, the optimizer's search space and the Python bindings all speak
//! this one vocabulary, so a setting has the same name everywhere.
//!
//! `device` is absent on purpose: a profile or a search belongs to the build
//! running it.

use crate::config::Config;
use crate::symbolic::OrderingMethod;
use std::fmt;

/// One setting's value.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Bool(bool),
    Int(u64),
    Float(f64),
    Text(String),
    /// Only `max_front`, where it means unlimited.
    None,
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Bool(v) => write!(f, "{v}"),
            Value::Int(v) => write!(f, "{v}"),
            Value::Float(v) => write!(f, "{v}"),
            Value::Text(v) => f.write_str(v),
            Value::None => f.write_str("none"),
        }
    }
}

impl From<u64> for Value {
    fn from(v: u64) -> Self {
        Value::Int(v)
    }
}
impl From<f64> for Value {
    fn from(v: f64) -> Self {
        Value::Float(v)
    }
}
impl From<&str> for Value {
    fn from(v: &str) -> Self {
        Value::Text(v.to_string())
    }
}
impl From<bool> for Value {
    fn from(v: bool) -> Self {
        Value::Bool(v)
    }
}

/// Every setting name, in the order the Python `Config` lists them.
pub const NAMES: [&str; 23] = [
    "ordering",
    "max_front",
    "nemin",
    "small_front_rows",
    "imbalance",
    "wide_cycles",
    "fm_passes",
    "separator_trials",
    "schur_batch",
    "panel_threads",
    "solve_threads",
    "lu_diag_threshold",
    "panel",
    "schur_block",
    "strip_block",
    "narrow_cols",
    "pivot_threads",
    "schur_chunk",
    "force_cpu",
    "cpu_panel",
    "cpu_block_rows",
    "cpu_blas_min_rows",
    "hybrid_cpu",
];

impl Config {
    /// Returns the setting called `name`, or None for an unknown name.
    pub fn get(&self, name: &str) -> Option<Value> {
        let int = |v: u64| Some(Value::Int(v));
        match name {
            "ordering" => Some(Value::Text(ordering_name(self.ordering).into())),
            "max_front" => Some(self.max_front.map_or(Value::None, |f| Value::Int(f as u64))),
            "nemin" => int(self.nemin as u64),
            "small_front_rows" => int(self.small_front_rows.into()),
            "imbalance" => Some(Value::Float(self.scalpel.imbalance)),
            "wide_cycles" => int(self.scalpel.wide_cycles as u64),
            "fm_passes" => int(self.scalpel.fm_passes as u64),
            "separator_trials" => int(self.scalpel.separator_trials as u64),
            "schur_batch" => int(self.schur_batch.into()),
            "panel_threads" => int(self.panel_threads.into()),
            "solve_threads" => int(self.solve_threads.into()),
            "lu_diag_threshold" => Some(Value::Float(self.lu_diag_threshold)),
            "panel" => int(self.panel.into()),
            "schur_block" => int(self.schur_block.into()),
            "strip_block" => int(self.strip_block.into()),
            "narrow_cols" => int(self.narrow_cols.into()),
            "pivot_threads" => int(self.pivot_threads.into()),
            "schur_chunk" => int(self.schur_chunk.into()),
            "force_cpu" => Some(Value::Bool(self.force_cpu)),
            "cpu_panel" => int(self.cpu_panel as u64),
            "cpu_block_rows" => int(self.cpu_block_rows as u64),
            "cpu_blas_min_rows" => int(self.cpu_blas_min_rows as u64),
            "hybrid_cpu" => Some(Value::Float(self.hybrid_cpu)),
            _ => None,
        }
    }

    /// Sets the setting called `name`. The value must have the setting's
    /// type; ranges are left to [`Config::check`], which sees every setting
    /// at once.
    pub fn set(&mut self, name: &str, value: Value) -> Result<(), String> {
        let wrong = || format!("{name} cannot be {value}");
        let int = || match value {
            Value::Int(v) => Ok(v),
            _ => Err(wrong()),
        };
        let small = || int().and_then(|v| u32::try_from(v).map_err(|_| wrong()));
        let size = || int().and_then(|v| usize::try_from(v).map_err(|_| wrong()));
        let float = || match value {
            Value::Float(v) => Ok(v),
            Value::Int(v) => Ok(v as f64),
            _ => Err(wrong()),
        };
        match name {
            "ordering" => {
                self.ordering = match &value {
                    Value::Text(t) if t == "scalpel" => OrderingMethod::Scalpel,
                    _ => return Err(format!("ordering must be scalpel, not {value}")),
                }
            }
            "max_front" => {
                self.max_front = match value {
                    Value::None => None,
                    _ => Some(size()?),
                }
            }
            "nemin" => self.nemin = size()?,
            "small_front_rows" => self.small_front_rows = small()?,
            "imbalance" => self.scalpel.imbalance = float()?,
            "wide_cycles" => self.scalpel.wide_cycles = size()?,
            "fm_passes" => self.scalpel.fm_passes = size()?,
            "separator_trials" => self.scalpel.separator_trials = size()?,
            "schur_batch" => self.schur_batch = small()?,
            "panel_threads" => self.panel_threads = small()?,
            "solve_threads" => self.solve_threads = small()?,
            "lu_diag_threshold" => self.lu_diag_threshold = float()?,
            "panel" => self.panel = small()?,
            "schur_block" => self.schur_block = small()?,
            "strip_block" => self.strip_block = small()?,
            "narrow_cols" => self.narrow_cols = small()?,
            "pivot_threads" => self.pivot_threads = small()?,
            "schur_chunk" => self.schur_chunk = small()?,
            "force_cpu" => {
                self.force_cpu = match value {
                    Value::Bool(v) => v,
                    _ => return Err(wrong()),
                }
            }
            "cpu_panel" => self.cpu_panel = size()?,
            "cpu_block_rows" => self.cpu_block_rows = size()?,
            "cpu_blas_min_rows" => self.cpu_blas_min_rows = size()?,
            "hybrid_cpu" => self.hybrid_cpu = float()?,
            _ => return Err(format!("unknown setting {name}")),
        }
        Ok(())
    }

    /// Returns a copy with `name` set to `value`.
    pub fn with(&self, name: &str, value: impl Into<Value>) -> Result<Config, String> {
        let mut config = self.clone();
        config.set(name, value.into())?;
        Ok(config)
    }

    /// Every setting as `(name, value)`, in [`NAMES`] order.
    pub fn settings(&self) -> Vec<(&'static str, Value)> {
        NAMES.iter().map(|&n| (n, self.get(n).expect("every name is known"))).collect()
    }
}

/// The settings name of an ordering backend.
pub fn ordering_name(ordering: OrderingMethod) -> &'static str {
    match ordering {
        OrderingMethod::Scalpel => "scalpel",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// Every name reads back what it writes, and names cover distinct fields.
    fn settings_round_trip() {
        let base = Config::default();
        let mut copy = Config {
            nemin: 1,
            ..base.clone()
        };
        for (name, value) in base.settings() {
            copy.set(name, value).unwrap();
        }
        assert_eq!(copy, base);
        let changed = base.with("schur_batch", 7u64).unwrap();
        assert_eq!(changed.schur_batch, 7);
        assert_eq!(base.with("ordering", "scalpel").unwrap().ordering, OrderingMethod::Scalpel);
        assert!(base.with("nemin", "wide").is_err());
        assert!(base.with("nope", 1u64).is_err());
        assert_eq!(base.with("max_front", Value::None).unwrap().max_front, None);
    }
}
