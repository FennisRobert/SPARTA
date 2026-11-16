//! x86-64: AVX2 with FMA (four `f64` or eight `f32` lanes, 16 registers) and
//! AVX-512F (eight `f64` or sixteen `f32` lanes, 32 registers).
//!
//! These are only ever inlined into functions compiled with
//! `#[target_feature(enable = "avx2,fma")]` or `"avx512f"`, chosen at run
//! time after `is_x86_feature_detected!` confirmed the CPU has them.
use super::Simd;
use std::arch::x86_64::*;

#[derive(Clone, Copy)]
pub(crate) struct Avx2F64;
#[derive(Clone, Copy)]
pub(crate) struct Avx2F32;
#[derive(Clone, Copy)]
pub(crate) struct Avx512F64;
#[derive(Clone, Copy)]
pub(crate) struct Avx512F32;

impl Simd for Avx2F64 {
    type R = f64;
    type V = __m256d;
    const L: usize = 4;
    #[inline(always)]
    unsafe fn zero() -> Self::V {
        _mm256_setzero_pd()
    }
    #[inline(always)]
    unsafe fn splat(x: f64) -> Self::V {
        _mm256_set1_pd(x)
    }
    #[inline(always)]
    unsafe fn load(p: *const f64) -> Self::V {
        _mm256_loadu_pd(p)
    }
    #[inline(always)]
    unsafe fn store(p: *mut f64, v: Self::V) {
        _mm256_storeu_pd(p, v)
    }
    #[inline(always)]
    unsafe fn fma(a: Self::V, b: Self::V, c: Self::V) -> Self::V {
        _mm256_fmadd_pd(a, b, c)
    }
    #[inline(always)]
    unsafe fn fnma(a: Self::V, b: Self::V, c: Self::V) -> Self::V {
        _mm256_fnmadd_pd(a, b, c)
    }
    #[inline(always)]
    unsafe fn mul(a: Self::V, b: Self::V) -> Self::V {
        _mm256_mul_pd(a, b)
    }
    #[inline(always)]
    unsafe fn add(a: Self::V, b: Self::V) -> Self::V {
        _mm256_add_pd(a, b)
    }
    /// `[r0 i0 r1 i1] [r2 i2 r3 i3]`: the in-lane unpacks give `[r0 r2 r1 r3]`
    /// and `[i0 i2 i1 i3]`, and one cross-lane permute puts them in order.
    #[inline(always)]
    unsafe fn load2(p: *const f64) -> (Self::V, Self::V) {
        let (x0, x1) = (_mm256_loadu_pd(p), _mm256_loadu_pd(p.add(4)));
        let re = _mm256_permute4x64_pd::<0xD8>(_mm256_unpacklo_pd(x0, x1));
        let im = _mm256_permute4x64_pd::<0xD8>(_mm256_unpackhi_pd(x0, x1));
        (re, im)
    }
    /// The inverse of `load2`.
    #[inline(always)]
    unsafe fn store2(p: *mut f64, re: Self::V, im: Self::V) {
        let (re, im) = (_mm256_permute4x64_pd::<0xD8>(re), _mm256_permute4x64_pd::<0xD8>(im));
        _mm256_storeu_pd(p, _mm256_unpacklo_pd(re, im));
        _mm256_storeu_pd(p.add(4), _mm256_unpackhi_pd(re, im));
    }
}

impl Simd for Avx2F32 {
    type R = f32;
    type V = __m256;
    const L: usize = 8;
    #[inline(always)]
    unsafe fn zero() -> Self::V {
        _mm256_setzero_ps()
    }
    #[inline(always)]
    unsafe fn splat(x: f32) -> Self::V {
        _mm256_set1_ps(x)
    }
    #[inline(always)]
    unsafe fn load(p: *const f32) -> Self::V {
        _mm256_loadu_ps(p)
    }
    #[inline(always)]
    unsafe fn store(p: *mut f32, v: Self::V) {
        _mm256_storeu_ps(p, v)
    }
    #[inline(always)]
    unsafe fn fma(a: Self::V, b: Self::V, c: Self::V) -> Self::V {
        _mm256_fmadd_ps(a, b, c)
    }
    #[inline(always)]
    unsafe fn fnma(a: Self::V, b: Self::V, c: Self::V) -> Self::V {
        _mm256_fnmadd_ps(a, b, c)
    }
    #[inline(always)]
    unsafe fn mul(a: Self::V, b: Self::V) -> Self::V {
        _mm256_mul_ps(a, b)
    }
    #[inline(always)]
    unsafe fn add(a: Self::V, b: Self::V) -> Self::V {
        _mm256_add_ps(a, b)
    }
}

impl Simd for Avx512F64 {
    type R = f64;
    type V = __m512d;
    const L: usize = 8;
    #[inline(always)]
    unsafe fn zero() -> Self::V {
        _mm512_setzero_pd()
    }
    #[inline(always)]
    unsafe fn splat(x: f64) -> Self::V {
        _mm512_set1_pd(x)
    }
    #[inline(always)]
    unsafe fn load(p: *const f64) -> Self::V {
        _mm512_loadu_pd(p)
    }
    #[inline(always)]
    unsafe fn store(p: *mut f64, v: Self::V) {
        _mm512_storeu_pd(p, v)
    }
    #[inline(always)]
    unsafe fn fma(a: Self::V, b: Self::V, c: Self::V) -> Self::V {
        _mm512_fmadd_pd(a, b, c)
    }
    #[inline(always)]
    unsafe fn fnma(a: Self::V, b: Self::V, c: Self::V) -> Self::V {
        _mm512_fnmadd_pd(a, b, c)
    }
    #[inline(always)]
    unsafe fn mul(a: Self::V, b: Self::V) -> Self::V {
        _mm512_mul_pd(a, b)
    }
    #[inline(always)]
    unsafe fn add(a: Self::V, b: Self::V) -> Self::V {
        _mm512_add_pd(a, b)
    }
}

impl Simd for Avx512F32 {
    type R = f32;
    type V = __m512;
    const L: usize = 16;
    #[inline(always)]
    unsafe fn zero() -> Self::V {
        _mm512_setzero_ps()
    }
    #[inline(always)]
    unsafe fn splat(x: f32) -> Self::V {
        _mm512_set1_ps(x)
    }
    #[inline(always)]
    unsafe fn load(p: *const f32) -> Self::V {
        _mm512_loadu_ps(p)
    }
    #[inline(always)]
    unsafe fn store(p: *mut f32, v: Self::V) {
        _mm512_storeu_ps(p, v)
    }
    #[inline(always)]
    unsafe fn fma(a: Self::V, b: Self::V, c: Self::V) -> Self::V {
        _mm512_fmadd_ps(a, b, c)
    }
    #[inline(always)]
    unsafe fn fnma(a: Self::V, b: Self::V, c: Self::V) -> Self::V {
        _mm512_fnmadd_ps(a, b, c)
    }
    #[inline(always)]
    unsafe fn mul(a: Self::V, b: Self::V) -> Self::V {
        _mm512_mul_ps(a, b)
    }
    #[inline(always)]
    unsafe fn add(a: Self::V, b: Self::V) -> Self::V {
        _mm512_add_ps(a, b)
    }
}
