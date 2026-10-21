//! Settings a caller may choose when analyzing a matrix.
//!
//! Every field is free to change per call. Most only shape the plan, which is
//! data the kernels read. The frontal geometry in [`Config::panel`] and its
//! neighbours additionally decides how the Metal kernels are compiled: the
//! shader is built from source when a `Solver` is created, and these arrive as
//! preprocessor macros. Because one value feeds both the plan and the shader,
//! the two cannot disagree.
//!
//! Changing geometry away from the default costs a shader compile the first
//! time a process sees that combination, roughly 300 ms; macOS caches it per
//! source and macro set, so repeats are a few milliseconds.
//!
//! CPU execution is selected explicitly by `force_cpu`; `cpu_*` fields tune
//! its dense updates independently of GPU geometry.

use crate::device::Device;
use crate::symbolic::OrderingMethod;

/// How a matrix is ordered, grouped into fronts, and executed on Metal or CPU.
///
/// Start from [`Config::default`], which carries the compiled-in build
/// profile, and override only what a workload needs:
///
/// ```no_run
/// # use sparta::Config;
/// let config = Config {
///     nemin: 32,
///     ..Config::default()
/// };
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    /// Which GPU backend's kernels and geometry this plan is built for.
    ///
    /// Defaults to the backend this build compiled, which is what almost every
    /// caller wants. Setting it to another device builds a plan in that
    /// device's geometry, which is how one backend's shapes and memory
    /// estimates are compared without that hardware, but
    /// [`Solver::new`](crate::Solver::new) refuses to execute one: the kernels
    /// are not in this binary. Ignored when `force_cpu` is set.
    pub device: Device,
    /// Run factorization and solves in Rust complex128 instead of on the GPU.
    pub force_cpu: bool,
    /// CPU pivots accumulated before updating a trailing Schur block, on the
    /// panel path only. May be exceeded by one column to keep a 2x2 pivot
    /// together. The LDL^T path normally eliminates the whole fully summed
    /// block through LAPACK and does not use this; it applies to `LU` fronts
    /// and to any front LAPACK reports as exactly singular.
    pub cpu_panel: usize,
    /// Rows per block in the CPU trailing update, and the unit of work when one
    /// front is split across threads. The blocked update forms only the lower
    /// triangle, one block of rows at a time, so a block wastes
    /// `cpu_block_rows / (2 * trailing rows)` of the product and small values
    /// waste less. Measured on an M4 Pro `curl_curl_30^3`, 96 rows: 64 and 128
    /// both cost about 5%, and 256 about 20%.
    pub cpu_block_rows: usize,
    /// Minimum trailing rows before updates use Apple Accelerate. Raise to avoid BLAS overhead on small blocks.
    pub cpu_blas_min_rows: usize,
    /// Minimum eliminated columns before two supernodes may merge. Larger
    /// values give fewer, wider fronts for more factor storage.
    pub nemin: usize,
    /// Front height at which execution switches from the serial complete-pivot
    /// route to Bunch–Kaufman panels (cooperative kernels on GPU), in rows.
    pub small_front_rows: u32,
    /// Largest front the plan may build, in rows. `None` is unlimited.
    /// A front bigger than this is a structural failure rather than a slow
    /// path, so the limit exists to fail early on a pathological ordering
    /// instead of allocating for it.
    pub max_front: Option<usize>,
    /// Which nested-dissection backend orders the matrix.
    pub ordering: OrderingMethod,
    /// Ordering quality and cost.
    pub scalpel: scalpel::Options,
    /// Pivot panels completed between two Schur drains; 0 drains once per
    /// level.
    pub schur_batch: u32,
    /// Threads per threadgroup in the panel factorization.
    pub panel_threads: u32,
    /// Threads per front in the triangular solves; 0 picks per front width.
    pub solve_threads: u32,
    /// LU diagonal preference between 0 and 1. 0 keeps strict partial
    /// pivoting. Above 0 the factorization accepts a diagonal pivot within
    /// this fraction of the column's best candidate.
    pub lu_diag_threshold: f64,
    /// Share of the factorization work, between 0 and 1, that the CPU takes
    /// while the GPU factors the rest at the same time (the hybrid
    /// factorization). The CPU factors whole subtrees of the elimination tree
    /// in the GPU's precision and hands their results to the GPU, which then
    /// factors the fronts above them. The share is rebalanced after every
    /// factorization from the measured CPU and GPU times, so this is the
    /// starting point of a sweep. 0, the default, leaves the whole
    /// factorization on the GPU. LDL^T and LU alike. Only the Metal backend
    /// has a hybrid path; everywhere else, including the CPU backend, the
    /// setting is ignored.
    pub hybrid_cpu: f64,

    // ---- Frontal geometry. Compiled into the Metal kernels. ----
    /// Pivots accumulated per panel before the trailing update runs.
    pub panel: u32,
    /// Per-thread register block edge in the Schur update. The tile each
    /// threadgroup owns is `16 * schur_block` square.
    pub schur_block: u32,
    /// Per-thread register block height in the strip update. One threadgroup
    /// owns `16 * strip_block` rows by `panel` columns.
    pub strip_block: u32,
    /// Fully summed width above which a front takes the wide kernel path.
    pub narrow_cols: u32,
    /// Threads cooperating on one pivot search.
    pub pivot_threads: u32,
    /// Panel columns staged per Schur pass.
    pub schur_chunk: u32,
}

impl Config {
    /// Edge of one Schur update tile, in rows and columns.
    #[inline]
    pub fn schur_tile(&self) -> u32 {
        16 * self.schur_block
    }
    /// Rows one strip update threadgroup owns.
    #[inline]
    pub fn strip_rows(&self) -> u32 {
        16 * self.strip_block
    }
    /// Column tiles one strip row block spans.
    #[inline]
    pub fn strip_span(&self) -> u32 {
        self.strip_rows() / self.panel
    }
    /// Panel workspace columns per wide front: `panel` pivots plus the extra
    /// column a 2x2 pivot straddling the boundary needs.
    #[inline]
    pub fn panel_cols(&self) -> u32 {
        self.panel + 1
    }
}

/// Reads `name` from the environment as a `T`, or returns `fallback`.
/// A value that will not parse is ignored rather than failing the run, since
/// these variables exist for ad-hoc sweeps.
fn from_env<T: std::str::FromStr>(name: &str, fallback: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(fallback)
}

impl Default for Config {
    /// Returns the measured defaults, with the dispatch environment variables
    /// applied on top when they are set.
    ///
    /// The environment is read here and nowhere else. `metal.m` takes these
    /// values as arguments from the plan, so the schedule `dispatch_audit`
    /// counts is always the schedule that runs.
    fn default() -> Self {
        Config::for_device(Device::COMPILED)
    }
}

impl Config {
    /// Returns the measured defaults for `device`, with the dispatch
    /// environment variables applied on top when they are set.
    ///
    /// [`Config::default`] is this for the compiled-in backend. Use it
    /// directly to build a plan in another device's geometry.
    pub fn for_device(device: Device) -> Self {
        let profile = device.profile();
        Self {
            device,
            force_cpu: false,
            // Conservative starting geometry, independent of Metal threadgroup limits.
            // These defaults are not a hardware performance claim.
            cpu_panel: 32,
            cpu_block_rows: 96,
            cpu_blas_min_rows: 64,
            nemin: 16,
            small_front_rows: profile.small_front_rows,
            max_front: None,
            // Not a build-profile setting. Which backend to order with is a
            // first-class API choice with its own default in every language
            // binding; a third source of truth here would silently lose to
            // them. SCALPEL is the only ordering; see `OrderingMethod`.
            ordering: OrderingMethod::Scalpel,
            scalpel: scalpel::Options {
                imbalance: 0.8,
                wide_cycles: 4,
                fm_passes: 4,
                separator_trials: 4,
                // Chosen for GPU factorization and solve time, not fill. The
                // level-set candidates alone cut lopsided, and the slab-shaped
                // halves that leaves get larger separators further down: a tree
                // three levels deeper than METIS's, with a longer chain of
                // panel steps. Spectral cuts, compared after full refinement,
                // and two-ended level sets give a shallower tree. Device and
                // solve time against the level-set candidates alone, M4 Pro:
                // em_m2_sym1 -8.8% / -12.9%, em_m1_sym1 -3.8% / -11.2%,
                // em_m3_asym1 -0.8% / -0.2%, em_bandcond_sym1 -5.2% / -17.3%,
                // for 1.3-1.5x the ordering time.
                spectral_candidates: 4,
                refine_all_candidates: true,
                half_levels: true,
                ..scalpel::Options::default()
            },
            schur_batch: from_env("SPARTA_SCHUR_BATCH", profile.schur_batch),
            panel_threads: from_env("SPARTA_PANEL_THREADS", profile.panel_threads),
            solve_threads: from_env("SPARTA_SOLVE_THREADS", profile.solve_threads),
            // Not a build-profile setting either: it changes results rather
            // than speed, so it should be visible at the call site that asked
            // for it, never compiled into a binary as a silent default.
            lu_diag_threshold: from_env("SPARTA_LU_DIAG_THRESHOLD", 0.0),
            hybrid_cpu: from_env("SPARTA_HYBRID_CPU", 0.0),
            panel: profile.panel,
            schur_block: profile.schur_block,
            strip_block: profile.strip_block,
            narrow_cols: profile.narrow_cols,
            pivot_threads: profile.pivot_threads,
            schur_chunk: profile.schur_chunk,
        }
    }

    /// Rejects settings that would produce an unusable plan.
    /// Returns the offending field's name and the rule it broke.
    pub fn check(&self) -> Result<(), String> {
        let rule = |ok: bool, what: &str| if ok { None } else { Some(what.to_string()) };
        // Every geometry limit below belongs to a device, not to the crate.
        let (shared, lane, group) = (
            self.device.shared_bytes(),
            self.device.lane_width(),
            self.device.max_group(),
        );
        let device = self.device;
        let broken = rule(self.cpu_panel >= 1 && self.cpu_panel <= u32::MAX as usize, "cpu_panel must be between 1 and u32::MAX")
            .or(rule(self.cpu_block_rows >= 1 && self.cpu_block_rows <= u32::MAX as usize, "cpu_block_rows must be between 1 and u32::MAX"))
            .or(rule(self.cpu_blas_min_rows >= 1, "cpu_blas_min_rows must be at least 1"))
            .or(rule(self.nemin >= 1, "nemin must be at least 1"))
            .or(rule(
                (0.0..1.0).contains(&self.hybrid_cpu),
                "hybrid_cpu must be at least 0 and below 1",
            ))
            .or(rule(
                self.max_front.is_none_or(|f| f >= 1),
                "max_front must be None or at least 1",
            ))
            .or(rule(
                self.small_front_rows >= 1,
                "small_front_rows must be at least 1",
            ))

            .or(rule(
                self.panel_threads == 0 || self.panel_threads >= 32,
                "panel_threads must be 0 or at least 32",
            ))
            .or(rule(
                self.panel_threads <= group,
                &format!("panel_threads must be at most {group} on {device}"),
            ))
            .or(rule(
                self.solve_threads == 0 || self.solve_threads >= 32,
                "solve_threads must be 0 or at least 32",
            ))
            .or(rule(
                self.solve_threads <= group,
                &format!("solve_threads must be at most {group} on {device}"),
            ))
            .or(rule(
                (0.0..=1.0).contains(&self.lu_diag_threshold),
                "lu_diag_threshold must be between 0 and 1",
            ))
            // The strip tile is `16 * strip_block` rows by `panel` columns and
            // is indexed in whole column tiles, so the rows must divide evenly.
            .or(rule(
                self.panel >= 1 && self.panel % 16 == 0 || self.panel == 16,
                "panel must be 16 (other widths need matching kernel changes)",
            ))
            .or(rule(
                (1..=8).contains(&self.schur_block),
                "schur_block must be between 1 and 8",
            ))
            .or(rule(
                (1..=8).contains(&self.strip_block),
                "strip_block must be between 1 and 8",
            ))
            .or(rule(
                16 * self.strip_block % self.panel == 0,
                "16 * strip_block must be a multiple of panel",
            ))
            .or(rule(self.narrow_cols >= 1, "narrow_cols must be at least 1"))
            // The Vulkan narrow-front tail kernels stage the fully summed block
            // in shared memory: `factor_narrow_tail` its packed coefficients
            // plus 32 rows, `factor_narrow_tail_lu` the whole block plus 32
            // rows. Metal's kernels stage less and are not bounded by this.
            .or(rule(
                self.device != Device::Vulkan || {
                    let n = self.narrow_cols as usize;
                    let ldlt = (n * n.saturating_sub(1) / 2 + 1) * 8 + n * 4 + 32 * n * 8;
                    let lu = n * n * 8 + 32 * n * 8;
                    ldlt.max(lu) <= shared as usize
                },
                &format!(
                    "narrow_cols complex columns exceed {} KiB of shared memory in \
                     factor_narrow_tail on {device}: lower narrow_cols",
                    shared / 1024
                ),
            ))
            // A width that is not a whole number of lanes leaves part of the
            // last one idle for the whole search.
            .or(rule(
                self.pivot_threads >= lane && self.pivot_threads % lane == 0,
                &format!("pivot_threads must be a positive multiple of {lane} on {device}"),
            ))
            .or(rule(
                self.pivot_threads <= group,
                &format!("pivot_threads must be at most {group} on {device}"),
            ))
            .or(rule(self.schur_chunk >= 1, "schur_chunk must be at least 1"))
            // Both update kernels stage their operands in shared memory, which
            // every device caps. `update_schur` declares
            // [SCHUR_TILE][SCHUR_CHUNK] twice; `update_strip` declares
            // [STRIP_ROWS][SCHUR_CHUNK] plus [16][SCHUR_CHUNK]. Exceeding
            // either only shows up as a pipeline that will not build, so both
            // are checked here where the message can name the setting.
            .or(rule(
                2 * self.schur_tile() * self.schur_chunk * 8 <= shared,
                &format!(
                    "2 * schur_tile * schur_chunk complex floats exceed {} KiB of \
                     shared memory in update_schur on {device}: lower schur_block or schur_chunk",
                    shared / 1024
                ),
            ))
            .or(rule(
                (self.strip_rows() + 16) * self.schur_chunk * 8 <= shared,
                &format!(
                    "(strip_rows + 16) * schur_chunk complex floats exceed {} KiB of \
                     shared memory in update_strip on {device}: lower strip_block or schur_chunk",
                    shared / 1024
                ),
            ));
        match broken {
            Some(what) => Err(what),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// Dispatch variables override the profile; others are unaffected.
    fn environment_overrides_dispatch_settings() {
        // SAFETY-adjacent: this test owns these variables for its duration.
        // Run with --test-threads=1 if more tests start reading them.
        std::env::set_var("SPARTA_SCHUR_BATCH", "7");
        assert_eq!(Config::default().schur_batch, 7);
        std::env::remove_var("SPARTA_SCHUR_BATCH");
    }

    #[test]
    /// The defaults are usable as delivered.
    fn defaults_are_valid() {
        assert_eq!(Config::default().check(), Ok(()));
    }

    #[test]
    /// A narrow front wider than the Vulkan tail kernels can stage is refused
    /// at analysis rather than as a pipeline that will not build; Metal's
    /// kernels take the same width.
    fn vulkan_narrow_cols_is_bounded_by_shared_memory() {
        let vulkan = Config::for_device(crate::Device::Vulkan);
        assert_eq!(vulkan.check(), Ok(()));
        assert!(Config { narrow_cols: 64, ..vulkan }.check().is_err());
        let metal = Config::for_device(crate::Device::Metal);
        assert_eq!(Config { narrow_cols: 64, ..metal }.check(), Ok(()));
    }

    #[test]
    /// Each rule in `check` rejects the value it names.
    fn invalid_settings_are_rejected() {
        // The limits below are Metal's (32 KiB of threadgroup memory), whichever
        // backend this build compiled.
        let metal = || Config::for_device(crate::Device::Metal);
        let bad = |c: Config| assert!(c.check().is_err());
        bad(Config {
            nemin: 0,
            ..metal()
        });
        bad(Config {
            small_front_rows: 0,
            ..metal()
        });
        bad(Config {
            panel_threads: 8,
            ..metal()
        });
        bad(Config {
            solve_threads: 8,
            ..metal()
        });
        bad(Config {
            lu_diag_threshold: 1.5,
            ..metal()
        });
        bad(Config {
            lu_diag_threshold: -0.1,
            ..metal()
        });
        bad(Config {
            schur_block: 0,
            ..metal()
        });
        bad(Config {
            schur_block: 9,
            ..metal()
        });
        bad(Config {
            strip_block: 0,
            ..metal()
        });
        bad(Config {
            narrow_cols: 0,
            ..metal()
        });
        bad(Config {
            pivot_threads: 24,
            ..metal()
        });
        bad(Config {
            schur_chunk: 0,
            ..metal()
        });
        // Both update kernels have their own threadgroup-memory ceiling.
        bad(Config {
            schur_block: 8,
            schur_chunk: 64,
            ..metal()
        });
        bad(Config {
            strip_block: 8,
            schur_chunk: 32,
            ..metal()
        });
        // Each is fine on its own; only the combination overflows.
        assert_eq!(
            Config {
                strip_block: 8,
                ..metal()
            }
            .check(),
            Ok(())
        );
        assert_eq!(
            Config {
                schur_chunk: 32,
                ..metal()
            }
            .check(),
            Ok(())
        );
    }

    #[test]
    /// The derived geometry matches the constants the kernels ship with.
    fn derived_geometry_matches_kernel_defaults() {
        let c = Config::default();
        assert_eq!(c.schur_tile(), 64);
        assert_eq!(c.strip_rows(), 64);
        assert_eq!(c.strip_span(), 4);
        assert_eq!(c.panel_cols(), 17);
    }
}
