//! What the optimizer searches: candidate values per setting, by group.

use crate::settings::Value;

/// Candidate values of one setting, tried in this order.
pub(super) enum Choices {
    Ints(&'static [u64]),
    Floats(&'static [f64]),
}
use Choices::{Floats, Ints};

impl Choices {
    pub(super) fn values(&self) -> Vec<Value> {
        match self {
            Ints(v) => v.iter().map(|&x| Value::Int(x)).collect(),
            Floats(v) => v.iter().map(|&x| Value::Float(x)).collect(),
        }
    }
}

/// Settings that shape the plan, shared by both backends.
pub(super) const PLAN: &[(&str, Choices)] = &[
    ("nemin", Ints(&[8, 16, 24, 32, 48, 64])),
    ("imbalance", Floats(&[0.6, 0.7, 0.8, 0.9])),
    ("separator_trials", Ints(&[1, 2, 4, 8])),
    ("wide_cycles", Ints(&[0, 2, 4, 6])),
];
/// How the GPU is driven, and the frontal geometry compiled into its kernels.
pub(super) const GPU: &[(&str, Choices)] = &[
    ("small_front_rows", Ints(&[16, 32, 64])),
    ("schur_batch", Ints(&[0, 1, 2, 4, 8])),
    ("panel_threads", Ints(&[256, 384, 512, 768, 1024])),
    ("solve_threads", Ints(&[0, 128, 256, 512])),
    ("schur_block", Ints(&[2, 3, 4, 5, 6, 8])),
    ("strip_block", Ints(&[1, 2, 4, 8])),
    ("schur_chunk", Ints(&[8, 16, 32])),
    ("pivot_threads", Ints(&[64, 128, 256])),
    ("narrow_cols", Ints(&[16, 32, 64, 128])),
];
/// The CPU backend's dense updates.
pub(super) const CPU: &[(&str, Choices)] = &[
    ("cpu_panel", Ints(&[16, 24, 32, 48, 64])),
    ("cpu_block_rows", Ints(&[48, 64, 96, 128, 192])),
    ("cpu_blas_min_rows", Ints(&[16, 32, 64, 128, 256])),
];
