//! The `sparta_config` struct and its translation to and from [`sparta::Config`].

use crate::status::{fail, guard, guard_value, Status};
use sparta::{Config, OrderingMethod, Preset};
use std::mem::size_of;

/// Mirrors `sparta_ordering` in `sparta.h`.
const ORDER_SCALPEL: i32 = 0;

/// Mirrors `sparta_config` in `sparta.h`, field for field.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CConfig {
    pub struct_size: usize,
    pub ordering: i32,
    pub nemin: usize,
    pub max_front: usize,
    pub imbalance: f64,
    pub wide_cycles: u32,
    pub fm_passes: u32,
    pub separator_trials: u32,
    pub lu_diag_threshold: f64,
    pub small_front_rows: u32,
    pub schur_batch: u32,
    pub panel_threads: u32,
    pub solve_threads: u32,
    pub panel: u32,
    pub schur_block: u32,
    pub strip_block: u32,
    pub narrow_cols: u32,
    pub pivot_threads: u32,
    pub schur_chunk: u32,
    pub hybrid_cpu: f64,
    pub threads: u32,
}

impl From<Config> for CConfig {
    fn from(config: Config) -> Self {
        Self {
            struct_size: size_of::<Self>(),
            ordering: match config.ordering {
                OrderingMethod::Scalpel => ORDER_SCALPEL,
            },
            nemin: config.nemin,
            max_front: config.max_front.unwrap_or(0),
            imbalance: config.scalpel.imbalance,
            wide_cycles: config.scalpel.wide_cycles as u32,
            fm_passes: config.scalpel.fm_passes as u32,
            separator_trials: config.scalpel.separator_trials as u32,
            lu_diag_threshold: config.lu_diag_threshold,
            small_front_rows: config.small_front_rows,
            schur_batch: config.schur_batch,
            panel_threads: config.panel_threads,
            solve_threads: config.solve_threads,
            panel: config.panel,
            schur_block: config.schur_block,
            strip_block: config.strip_block,
            narrow_cols: config.narrow_cols,
            pivot_threads: config.pivot_threads,
            schur_chunk: config.schur_chunk,
            hybrid_cpu: config.hybrid_cpu,
            threads: config.threads.min(u32::MAX as usize) as u32,
        }
    }
}

impl CConfig {
    /// Builds the solver's own settings from this struct.
    /// Returns the offending field and rule when a value is out of range.
    fn to_config(self) -> Result<Config, String> {
        let ordering = match self.ordering {
            ORDER_SCALPEL => OrderingMethod::Scalpel,
            other => return Err(format!("ordering must be SPARTA_ORDER_SCALPEL (0), not {other}")),
        };
        let base = Config::default();
        let config = Config {
            ordering,
            nemin: self.nemin,
            max_front: (self.max_front > 0).then_some(self.max_front),
            scalpel: sparta::scalpel::Options {
                imbalance: self.imbalance,
                wide_cycles: self.wide_cycles as usize,
                fm_passes: self.fm_passes as usize,
                separator_trials: self.separator_trials as usize,
                ..base.scalpel
            },
            lu_diag_threshold: self.lu_diag_threshold,
            small_front_rows: self.small_front_rows,
            schur_batch: self.schur_batch,
            panel_threads: self.panel_threads,
            solve_threads: self.solve_threads,
            panel: self.panel,
            schur_block: self.schur_block,
            strip_block: self.strip_block,
            narrow_cols: self.narrow_cols,
            pivot_threads: self.pivot_threads,
            schur_chunk: self.schur_chunk,
            // One shipped library is built for one backend, so there is
            // nothing here for a caller to choose; `sparta_device` reports it.
            device: base.device,
            force_cpu: false,
            cpu_panel: base.cpu_panel,
            cpu_block_rows: base.cpu_block_rows,
            cpu_blas_min_rows: base.cpu_blas_min_rows,
            hybrid_cpu: self.hybrid_cpu,
            threads: self.threads as usize,
        };
        config.check().map(|()| config)
    }
}

/// Reads a caller's `sparta_config`, or the defaults when it passed NULL.
///
/// A struct shorter than the current one comes from a program built against an
/// older header: the fields it does carry are taken and the rest keep their
/// defaults. A longer one comes from a newer header than this library knows,
/// which it cannot honour and so refuses.
///
/// # Safety
/// `pointer` must be null or point to at least `struct_size` readable bytes.
pub unsafe fn read(pointer: *const CConfig) -> Result<Config, Status> {
    if pointer.is_null() {
        return Ok(Config::default());
    }
    let declared = unsafe { std::ptr::read_unaligned(std::ptr::addr_of!((*pointer).struct_size)) };
    let current = size_of::<CConfig>();
    if declared < size_of::<usize>() || declared > current {
        return Err(fail(
            Status::Invalid,
            format!("config.struct_size is {declared}; expected at most {current}. \
                     Set it to sizeof(sparta_config)."),
        ));
    }
    let mut config = CConfig::from(Config::default());
    unsafe {
        std::ptr::copy_nonoverlapping(
            pointer.cast::<u8>(),
            std::ptr::addr_of_mut!(config).cast::<u8>(),
            declared,
        );
    }
    config.struct_size = current;
    config.to_config().map_err(|why| fail(Status::Invalid, why))
}

/// Fills `config` with the default settings.
#[no_mangle]
pub unsafe extern "C" fn sparta_config_default(config: *mut CConfig) {
    guard_value((), || {
        if let Some(slot) = unsafe { config.as_mut() } {
            *slot = CConfig::from(Config::default());
        }
    });
}

/// Fills `config` with the settings a named preset stands for.
#[no_mangle]
pub unsafe extern "C" fn sparta_config_preset(config: *mut CConfig, preset: i32) -> Status {
    guard(|| {
        let Some(slot) = (unsafe { config.as_mut() }) else {
            return fail(Status::Invalid, "sparta_config_preset needs a config to fill");
        };
        let Some(&preset) = Preset::ALL.get(preset as usize) else {
            return fail(
                Status::Invalid,
                format!("no preset numbered {preset}; see sparta_preset"),
            );
        };
        *slot = CConfig::from(preset.config());
        Status::Ok
    })
}
