//! `C = alpha·A·B + beta·C` for strided views: blocking, packing, threading,
//! and the per-instruction-set entry points the microkernels are compiled into.
//!
//! The loop nest is the standard one (Goto; BLIS): `NC` columns of C at a
//! time, `KC` deep, B's `KC × NC` block packed once and kept in L3; then `MC`
//! rows at a time, A's `MC × KC` block packed and kept in L2; then the
//! microkernel walks `MR × NR` tiles, its B sliver in L1.
use crate::element::{Element, Real};
use crate::isa::{self, Isa};
use crate::kernel::{self, Target};
use crate::pack::{pack_a, pack_b, packed_len, View};
use crate::simd::{Scalar, Simd};
use crate::threading;
use rayon::prelude::*;
use std::cell::Cell;

/// One macro-kernel call: an `mc × nc` block of C from packed A and B.
#[derive(Clone, Copy)]
pub(crate) struct Block<R> {
    pub mc: usize,
    pub nc: usize,
    pub kc: usize,
    pub ap: *const R,
    pub bp: *const R,
    pub c: *mut R,
    pub rs: isize,
    pub cs: isize,
    /// (re, im); real kernels read only `.0`.
    pub alpha: (R, R),
    pub beta: (R, R),
}

/// A macro-kernel compiled for one instruction set.
pub(crate) type MacroKernel<R> = unsafe fn(&Block<R>);

/// A packing routine for one layout: `(lines, kc, p, line stride, depth
/// stride, sliver width, dst)`, strides in elements. Returns false, having
/// done nothing, when it does not handle the layout given.
pub(crate) type Pack<R> = unsafe fn(usize, usize, *const R, isize, isize, usize, *mut R) -> bool;

/// Tile and block sizes, the macro-kernel and any specialised packing, of one
/// type on one family.
#[derive(Clone, Copy)]
#[doc(hidden)]
pub struct Config<R> {
    pub(crate) mr: usize,
    pub(crate) nr: usize,
    pub(crate) mc: usize,
    pub(crate) kc: usize,
    pub(crate) nc: usize,
    pub(crate) kernel: MacroKernel<R>,
    /// Tried before the portable packing, for both operands.
    pub(crate) pack: Option<Pack<R>>,
}

/// The real macro-kernel: every `MR × NR` tile of the block, column slivers
/// outside so one B sliver serves a whole column of A slivers from L1.
#[inline(always)]
unsafe fn macro_real<S: Simd, const MV: usize, const NR: usize>(b: &Block<S::R>) {
    let mr = MV * S::L;
    for j0 in (0..b.nc).step_by(NR) {
        for i0 in (0..b.mc).step_by(mr) {
            let t = Target {
                c: b.c.offset(i0 as isize * b.rs + j0 as isize * b.cs),
                rs: b.rs,
                cs: b.cs,
                m: mr.min(b.mc - i0),
                n: NR.min(b.nc - j0),
            };
            kernel::real::<S, MV, NR>(b.kc, b.ap.add(i0 * b.kc), b.bp.add(j0 * b.kc), t, b.alpha.0, b.beta.0);
        }
    }
}

/// The complex macro-kernel, as [`macro_real`] on split-packed A.
#[inline(always)]
unsafe fn macro_complex<S: Simd, const MV: usize, const NR: usize>(b: &Block<S::R>) {
    let mr = MV * S::L;
    for j0 in (0..b.nc).step_by(NR) {
        for i0 in (0..b.mc).step_by(mr) {
            let t = Target {
                c: b.c.offset(2 * (i0 as isize * b.rs + j0 as isize * b.cs)),
                rs: b.rs,
                cs: b.cs,
                m: mr.min(b.mc - i0),
                n: NR.min(b.nc - j0),
            };
            kernel::complex::<S, MV, NR>(b.kc, b.ap.add(2 * i0 * b.kc), b.bp.add(2 * j0 * b.kc), t, b.alpha, b.beta);
        }
    }
}

/// Entry points: one function per (family, type), carrying the family's
/// `target_feature` so the generic kernel and its intrinsics inline into it.
macro_rules! entry {
    ($name:ident, $feature:literal, $simd:ty, $mv:literal, $nr:literal, $kind:ident) => {
        #[target_feature(enable = $feature)]
        pub(super) unsafe fn $name(b: &Block<<$simd as Simd>::R>) {
            $kind::<$simd, $mv, $nr>(b)
        }
    };
    ($name:ident, $simd:ty, $mv:literal, $nr:literal, $kind:ident) => {
        unsafe fn $name(b: &Block<<$simd as Simd>::R>) {
            $kind::<$simd, $mv, $nr>(b)
        }
    };
}

entry!(scalar_d, Scalar<f64>, 4, 4, macro_real);
entry!(scalar_s, Scalar<f32>, 4, 4, macro_real);
entry!(scalar_z, Scalar<f64>, 2, 4, macro_complex);
entry!(scalar_c, Scalar<f32>, 2, 4, macro_complex);

#[cfg(target_arch = "aarch64")]
mod neon_entries {
    use super::*;
    use crate::simd::neon::{NeonF32, NeonF64};
    // 32 registers: 24 accumulators, 4 (real) or 4 (split complex) A vectors.
    entry!(d, "neon", NeonF64, 4, 6, macro_real);
    entry!(s, "neon", NeonF32, 4, 6, macro_real);
    entry!(z, "neon", NeonF64, 2, 6, macro_complex);
    entry!(c, "neon", NeonF32, 2, 6, macro_complex);
}

#[cfg(target_arch = "x86_64")]
mod x86_entries {
    use super::*;
    use crate::simd::x86::{Avx2F32, Avx2F64, Avx512F32, Avx512F64};
    // AVX2, 16 registers. Real: 12 accumulators, 2 A vectors, a broadcast.
    // Complex: 10 accumulators, 2 A vectors, 2 broadcasts; six columns would
    // need all 16 and the compiler bounces two accumulators through the stack.
    entry!(avx2_d, "avx2,fma", Avx2F64, 2, 6, macro_real);
    entry!(avx2_s, "avx2,fma", Avx2F32, 2, 6, macro_real);
    entry!(avx2_z, "avx2,fma", Avx2F64, 1, 5, macro_complex);
    entry!(avx2_c, "avx2,fma", Avx2F32, 1, 5, macro_complex);
    // AVX-512, 32 registers: 24 accumulators, 2 A vectors, the broadcasts.
    entry!(avx512_d, "avx512f", Avx512F64, 2, 12, macro_real);
    entry!(avx512_s, "avx512f", Avx512F32, 2, 12, macro_real);
    entry!(avx512_z, "avx512f", Avx512F64, 1, 12, macro_complex);
    entry!(avx512_c, "avx512f", Avx512F32, 1, 12, macro_complex);
}

/// Configuration for `(mv, nr, lanes)` with block sizes for that tile.
fn config<R>(mr: usize, nr: usize, kernel: MacroKernel<R>, kc: usize) -> Config<R> {
    // MC: a multiple of MR near 192 rows; NC: a multiple of NR near 4096.
    let mc = (192 / mr).max(1) * mr;
    let nc = (4096 / nr).max(1) * nr;
    Config { mr, nr, mc, kc, nc, kernel, pack: None }
}

/// The configuration of type `T` on family `isa`, which this CPU runs.
pub(crate) fn config_for<T: Element>(isa: Isa) -> Config<T::Real> {
    T::config(isa)
}

/// Per-type configurations, reached through `Element::config`.
#[doc(hidden)]
pub trait Configs: Real {
    fn real_config(isa: Isa) -> Config<Self>;
    fn complex_config(isa: Isa) -> Config<Self>;
}

impl Configs for f64 {
    fn real_config(isa: Isa) -> Config<f64> {
        match isa {
            #[cfg(target_arch = "aarch64")]
            Isa::Sme512 => Config {
                pack: Some(crate::sme512::pack::transpose_d),
                ..config(crate::sme512::d::MR, crate::sme512::d::NR, crate::sme512::macro_d, 512)
            },
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => config(8, 6, neon_entries::d, 256),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => config(8, 6, x86_entries::avx2_d, 256),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => config(16, 12, x86_entries::avx512_d, 256),
            _ => config(4, 4, scalar_d, 256),
        }
    }
    fn complex_config(isa: Isa) -> Config<f64> {
        match isa {
            #[cfg(target_arch = "aarch64")]
            Isa::Sme512 => Config {
                pack: Some(crate::sme512::pack::split_z),
                ..config(crate::sme512::z::MR, crate::sme512::z::NR, crate::sme512::macro_z, 512)
            },
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => config(4, 6, neon_entries::z, 256),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => config(4, 5, x86_entries::avx2_z, 192),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => config(8, 12, x86_entries::avx512_z, 192),
            _ => config(2, 4, scalar_z, 256),
        }
    }
}

impl Configs for f32 {
    fn real_config(isa: Isa) -> Config<f32> {
        match isa {
            #[cfg(target_arch = "aarch64")]
            Isa::Sme512 => Config {
                pack: Some(crate::sme512::pack::transpose_s),
                ..config(crate::sme512::s::MR, crate::sme512::s::NR, crate::sme512::macro_s, 512)
            },
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => config(16, 6, neon_entries::s, 384),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => config(16, 6, x86_entries::avx2_s, 384),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => config(32, 12, x86_entries::avx512_s, 384),
            _ => config(4, 4, scalar_s, 256),
        }
    }
    fn complex_config(isa: Isa) -> Config<f32> {
        match isa {
            #[cfg(target_arch = "aarch64")]
            Isa::Sme512 => Config {
                pack: Some(crate::sme512::pack::split_c),
                ..config(crate::sme512::c::MR, crate::sme512::c::NR, crate::sme512::macro_c, 512)
            },
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => config(8, 6, neon_entries::c, 256),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => config(8, 5, x86_entries::avx2_c, 256),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => config(16, 12, x86_entries::avx512_c, 256),
            _ => config(2, 4, scalar_c, 256),
        }
    }
}

thread_local! {
    /// Packing buffers, reused across calls. Taken out for the duration of a
    /// call and put back after, so a nested call on the same thread (Rayon
    /// may run another task while this one waits) finds it empty and
    /// allocates its own rather than aliasing.
    static PACK_A: Cell<Vec<f64>> = const { Cell::new(Vec::new()) };
    static PACK_B: Cell<Vec<f64>> = const { Cell::new(Vec::new()) };
}

/// Run `f` with this thread's buffer `key` holding at least `reals` values of
/// `R`.
fn with_buffer<R, O>(key: &'static std::thread::LocalKey<Cell<Vec<f64>>>, reals: usize, f: impl FnOnce(*mut R) -> O) -> O {
    let mut buf = key.with(|b| b.take());
    let words = (reals * std::mem::size_of::<R>()).div_ceil(8);
    if buf.len() < words {
        buf.resize(words, 0.0);
    }
    let out = f(buf.as_mut_ptr().cast());
    key.with(|b| b.set(buf));
    out
}

/// Below this many multiply-adds a product runs on one thread whatever the
/// threading setting: splitting it costs more than it saves.
const PARALLEL_MIN: usize = 1 << 20;


/// `C = alpha·A·B + beta·C`, `C` is `m × n` and the product `k` deep.
///
/// # Safety
/// Every view must be valid for its shape and strides, and C must overlap
/// neither A nor B.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn gemm<T: Element>(
    threads: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    a: View<T>,
    b: View<T>,
    beta: T,
    c: *mut T,
    rsc: isize,
    csc: isize,
) {
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 || alpha == T::zero() {
        scale(m, n, beta, c, rsc, csc);
        return;
    }
    // Make C contiguous down its columns where it can be: a row-major C is
    // the transpose of a column-major one, and C^T = B^T A^T.
    let (m, n, a, b, rsc, csc) = if rsc != 1 && csc == 1 {
        let at = View { p: b.p, rs: b.cs, cs: b.rs };
        let bt = View { p: a.p, rs: a.cs, cs: a.rs };
        (n, m, at, bt, csc, rsc)
    } else {
        (m, n, a, b, rsc, csc)
    };
    let isa = isa::current();
    // On the matrix units too: raising the threshold 16- or 64-fold, or
    // capping a call at 4 or 8 tasks, changed nothing beyond noise
    // (M4 Pro, every shape in examples/vs_accelerate.rs).
    let parallel = threads && rayon::current_num_threads() > 1 && m.saturating_mul(n).saturating_mul(k) >= PARALLEL_MIN;
    // Every call uses the best family, the matrix units included, even
    // when every core calls at once: shared, they still give each of 14
    // threads more than its own NEON units would. Measured on em_m4 (M4 Pro,
    // complex128): letting only 2, 3, 4 or 6 calls at a time use SME and the
    // rest NEON factored in 2.6-3.1, 2.4-2.9, 2.6-2.9 and 2.2-2.4 s, against
    // 2.08-2.13 s for SME everywhere.
    let cfg = config_for::<T>(isa);
    let (alpha, beta) = ((alpha.re(), alpha.im()), (beta.re(), beta.im()));
    let one = (T::Real::ONE, T::Real::ZERO);
    let cr = c as *mut T::Real;

    for jc in (0..n).step_by(cfg.nc) {
        let nc = cfg.nc.min(n - jc);
        for pc in (0..k).step_by(cfg.kc) {
            let kc = cfg.kc.min(k - pc);
            let beta = if pc == 0 { beta } else { one };
            let bview = View { p: b.at(pc, jc), rs: b.rs, cs: b.cs };
            with_buffer(&PACK_B, packed_len::<T>(nc, kc, cfg.nr), |bp: *mut T::Real| {
                let special = cfg.pack.is_some_and(|f| f(nc, kc, bview.p.cast(), bview.cs, bview.rs, cfg.nr, bp));
                if !special {
                    pack_b(kc, nc, bview, cfg.nr, bp);
                }
                let bp = bp as *const T::Real;
                let w = if T::COMPLEX { 2 } else { 1 };
                // One task: rows [i0, i0 + mc) against B columns [j0, j1).
                let task = |i0: usize, j0: usize, j1: usize| {
                    let mc = cfg.mc.min(m - i0);
                    let aview = View { p: a.at(i0, pc), rs: a.rs, cs: a.cs };
                    with_buffer(&PACK_A, packed_len::<T>(mc, kc, cfg.mr), |ap: *mut T::Real| {
                        let special = cfg.pack.is_some_and(|f| f(mc, kc, aview.p.cast(), aview.rs, aview.cs, cfg.mr, ap));
                        if !special {
                            pack_a(mc, kc, aview, cfg.mr, ap);
                        }
                        let block = Block {
                            mc,
                            nc: j1 - j0,
                            kc,
                            ap,
                            bp: bp.add(j0 * kc * w),
                            c: cr.offset(w as isize * (i0 as isize * rsc + (jc + j0) as isize * csc)),
                            rs: rsc,
                            cs: csc,
                            alpha,
                            beta,
                        };
                        (cfg.kernel)(&block);
                    });
                };
                if parallel {
                    // Split rows into MC blocks and, when that gives too few
                    // tasks, columns into whole NR slivers as well.
                    let rows = m.div_ceil(cfg.mc);
                    let slivers = nc.div_ceil(cfg.nr);
                    let want = 2 * rayon::current_num_threads();
                    let cols = want.div_ceil(rows).clamp(1, slivers);
                    let per = slivers.div_ceil(cols) * cfg.nr;
                    let task = Shared(task);
                    (0..rows * cols).into_par_iter().for_each(|t| {
                        let (r, s) = (t / cols, t % cols);
                        let j0 = s * per;
                        if j0 < nc {
                            task.get()(r * cfg.mc, j0, (j0 + per).min(nc));
                        }
                    });
                } else {
                    for i0 in (0..m).step_by(cfg.mc) {
                        task(i0, 0, nc);
                    }
                }
            });
        }
    }
}

/// A task the pool's threads may share. Each call writes a disjoint part of
/// C and only reads the packed B, through raw pointers the compiler cannot
/// see are used that way.
pub(crate) struct Shared<F>(pub F);
unsafe impl<F> Send for Shared<F> {}
unsafe impl<F> Sync for Shared<F> {}
impl<F> Shared<F> {
    /// The task. A method, so closures capture the whole wrapper.
    pub fn get(&self) -> &F {
        &self.0
    }
}

/// `C = beta·C`; with `beta == 0`, `C = 0` whatever it held.
unsafe fn scale<T: Element>(m: usize, n: usize, beta: T, c: *mut T, rsc: isize, csc: isize) {
    for j in 0..n {
        for i in 0..m {
            let p = c.offset(i as isize * rsc + j as isize * csc);
            *p = if beta == T::zero() { T::zero() } else { beta * *p };
        }
    }
}

/// The public product, threaded per [`threading::set`].
///
/// `C = alpha·A·B + beta·C` with A `m × k`, B `k × n` and C `m × n`, each a
/// strided view: element `(i, j)` of A is at `a + i·rsa + j·csa`, and so on.
/// Strides may be negative. With `beta == 0` C is only written.
///
/// # Safety
/// Every view must be valid for its shape and strides, and C must overlap
/// neither A nor B.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_strided<T: Element>(
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    a: *const T,
    rsa: isize,
    csa: isize,
    b: *const T,
    rsb: isize,
    csb: isize,
    beta: T,
    c: *mut T,
    rsc: isize,
    csc: isize,
) {
    gemm(
        threading::machine(),
        m,
        n,
        k,
        alpha,
        View { p: a, rs: rsa, cs: csa },
        View { p: b, rs: rsb, cs: csb },
        beta,
        c,
        rsc,
        csc,
    )
}
