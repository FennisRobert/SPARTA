//! One vector register's worth of reals, per instruction set.
//!
//! The microkernels in `kernel.rs` are written once against [`Simd`] and
//! compiled per instruction set. Every method is `#[inline(always)]` and has no
//! `target_feature` of its own: it is inlined into a kernel entry point that
//! carries the feature (see `gemm.rs`), and only there do the intrinsics it
//! calls become inlinable. Called anywhere else the methods would be slow, so
//! nothing outside a feature-gated entry point calls them.
use crate::element::Real;

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;
#[cfg(target_arch = "x86_64")]
pub(crate) mod x86;

/// A vector of `L` reals of type `R`.
pub(crate) trait Simd: Copy {
    type R: Real;
    type V: Copy;
    /// Lanes per vector.
    const L: usize;
    unsafe fn zero() -> Self::V;
    unsafe fn splat(x: Self::R) -> Self::V;
    /// Load `L` reals from `p`, which need not be aligned.
    unsafe fn load(p: *const Self::R) -> Self::V;
    unsafe fn store(p: *mut Self::R, v: Self::V);
    /// `c + a * b`, fused.
    unsafe fn fma(a: Self::V, b: Self::V, c: Self::V) -> Self::V;
    /// `c - a * b`, fused.
    unsafe fn fnma(a: Self::V, b: Self::V, c: Self::V) -> Self::V;
    unsafe fn mul(a: Self::V, b: Self::V) -> Self::V;
    unsafe fn add(a: Self::V, b: Self::V) -> Self::V;

    /// Load `L` interleaved complex values from `p` as (real parts,
    /// imaginary parts).
    #[inline(always)]
    unsafe fn load2(p: *const Self::R) -> (Self::V, Self::V) {
        let mut re = [Self::R::ZERO; 16];
        let mut im = [Self::R::ZERO; 16];
        for i in 0..Self::L {
            re[i] = *p.add(2 * i);
            im[i] = *p.add(2 * i + 1);
        }
        (Self::load(re.as_ptr()), Self::load(im.as_ptr()))
    }

    /// Store `L` complex values to `p`, interleaved; the inverse of `load2`.
    #[inline(always)]
    unsafe fn store2(p: *mut Self::R, re: Self::V, im: Self::V) {
        let mut r = [Self::R::ZERO; 16];
        let mut i_ = [Self::R::ZERO; 16];
        Self::store(r.as_mut_ptr(), re);
        Self::store(i_.as_mut_ptr(), im);
        for i in 0..Self::L {
            *p.add(2 * i) = r[i];
            *p.add(2 * i + 1) = i_[i];
        }
    }
}

/// One real per "vector": the portable fallback, and the reference the vector
/// versions are tested against.
#[derive(Clone, Copy)]
pub(crate) struct Scalar<R>(std::marker::PhantomData<R>);

impl<R: Real> Simd for Scalar<R> {
    type R = R;
    type V = R;
    const L: usize = 1;
    #[inline(always)]
    unsafe fn zero() -> R {
        R::ZERO
    }
    #[inline(always)]
    unsafe fn splat(x: R) -> R {
        x
    }
    #[inline(always)]
    unsafe fn load(p: *const R) -> R {
        *p
    }
    #[inline(always)]
    unsafe fn store(p: *mut R, v: R) {
        *p = v
    }
    #[inline(always)]
    unsafe fn fma(a: R, b: R, c: R) -> R {
        c + a * b
    }
    #[inline(always)]
    unsafe fn fnma(a: R, b: R, c: R) -> R {
        c - a * b
    }
    #[inline(always)]
    unsafe fn mul(a: R, b: R) -> R {
        a * b
    }
    #[inline(always)]
    unsafe fn add(a: R, b: R) -> R {
        a + b
    }
}
