//! aarch64 Advanced SIMD (NEON): two `f64` or four `f32` lanes, 32 registers.
use super::Simd;
use std::arch::aarch64::*;

#[derive(Clone, Copy)]
pub(crate) struct NeonF64;
#[derive(Clone, Copy)]
pub(crate) struct NeonF32;

impl Simd for NeonF64 {
    type R = f64;
    type V = float64x2_t;
    const L: usize = 2;
    #[inline(always)]
    unsafe fn zero() -> Self::V {
        vdupq_n_f64(0.0)
    }
    #[inline(always)]
    unsafe fn splat(x: f64) -> Self::V {
        vdupq_n_f64(x)
    }
    #[inline(always)]
    unsafe fn load(p: *const f64) -> Self::V {
        vld1q_f64(p)
    }
    #[inline(always)]
    unsafe fn store(p: *mut f64, v: Self::V) {
        vst1q_f64(p, v)
    }
    #[inline(always)]
    unsafe fn fma(a: Self::V, b: Self::V, c: Self::V) -> Self::V {
        vfmaq_f64(c, a, b)
    }
    #[inline(always)]
    unsafe fn fnma(a: Self::V, b: Self::V, c: Self::V) -> Self::V {
        vfmsq_f64(c, a, b)
    }
    #[inline(always)]
    unsafe fn mul(a: Self::V, b: Self::V) -> Self::V {
        vmulq_f64(a, b)
    }
    #[inline(always)]
    unsafe fn add(a: Self::V, b: Self::V) -> Self::V {
        vaddq_f64(a, b)
    }
    #[inline(always)]
    unsafe fn load2(p: *const f64) -> (Self::V, Self::V) {
        let x = vld2q_f64(p);
        (x.0, x.1)
    }
    #[inline(always)]
    unsafe fn store2(p: *mut f64, re: Self::V, im: Self::V) {
        vst2q_f64(p, float64x2x2_t(re, im))
    }
}

impl Simd for NeonF32 {
    type R = f32;
    type V = float32x4_t;
    const L: usize = 4;
    #[inline(always)]
    unsafe fn zero() -> Self::V {
        vdupq_n_f32(0.0)
    }
    #[inline(always)]
    unsafe fn splat(x: f32) -> Self::V {
        vdupq_n_f32(x)
    }
    #[inline(always)]
    unsafe fn load(p: *const f32) -> Self::V {
        vld1q_f32(p)
    }
    #[inline(always)]
    unsafe fn store(p: *mut f32, v: Self::V) {
        vst1q_f32(p, v)
    }
    #[inline(always)]
    unsafe fn fma(a: Self::V, b: Self::V, c: Self::V) -> Self::V {
        vfmaq_f32(c, a, b)
    }
    #[inline(always)]
    unsafe fn fnma(a: Self::V, b: Self::V, c: Self::V) -> Self::V {
        vfmsq_f32(c, a, b)
    }
    #[inline(always)]
    unsafe fn mul(a: Self::V, b: Self::V) -> Self::V {
        vmulq_f32(a, b)
    }
    #[inline(always)]
    unsafe fn add(a: Self::V, b: Self::V) -> Self::V {
        vaddq_f32(a, b)
    }
    #[inline(always)]
    unsafe fn load2(p: *const f32) -> (Self::V, Self::V) {
        let x = vld2q_f32(p);
        (x.0, x.1)
    }
    #[inline(always)]
    unsafe fn store2(p: *mut f32, re: Self::V, im: Self::V) {
        vst2q_f32(p, float32x4x2_t(re, im))
    }
}
