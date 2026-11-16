//! The four BLAS value types and the real type underneath each.
use num_complex::Complex;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// `f32` or `f64`: the arithmetic the kernels run in.
pub trait Real:
    Copy
    + Send
    + Sync
    + Debug
    + Default
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + 'static
{
    const ZERO: Self;
    const ONE: Self;
    /// The smallest normal number: below it a reciprocal overflows.
    const MIN_NORMAL: Self;
    fn abs(self) -> Self;
    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
}

macro_rules! real {
    ($t:ty) => {
        impl Real for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            const MIN_NORMAL: Self = <$t>::MIN_POSITIVE;
            #[inline(always)]
            fn abs(self) -> Self {
                <$t>::abs(self)
            }
            #[inline(always)]
            fn from_f64(x: f64) -> Self {
                x as $t
            }
            #[inline(always)]
            fn to_f64(self) -> f64 {
                self as f64
            }
        }
    };
}
real!(f32);
real!(f64);

/// One of `f32`, `f64`, `Complex<f32>` and `Complex<f64>`.
///
/// Complex values are stored interleaved (`re`, `im`), which is the layout of
/// `num_complex::Complex` and of C99 and Fortran complex numbers.
pub trait Element:
    Copy
    + Send
    + Sync
    + Debug
    + Default
    + PartialEq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + 'static
{
    type Real: Real + crate::gemm::Configs;
    /// Whether the value is two reals.
    const COMPLEX: bool;
    /// This type's tile sizes and kernel on `isa`.
    #[doc(hidden)]
    fn config(isa: crate::Isa) -> crate::gemm::Config<Self::Real> {
        use crate::gemm::Configs;
        if Self::COMPLEX {
            Self::Real::complex_config(isa)
        } else {
            Self::Real::real_config(isa)
        }
    }
    fn new(re: Self::Real, im: Self::Real) -> Self;
    fn re(self) -> Self::Real;
    fn im(self) -> Self::Real;
    /// `|re| + |im|`, the magnitude LAPACK's pivot search compares.
    #[inline(always)]
    fn abs1(self) -> Self::Real {
        self.re().abs() + self.im().abs()
    }
    #[inline(always)]
    fn zero() -> Self {
        Self::new(Self::Real::ZERO, Self::Real::ZERO)
    }
    #[inline(always)]
    fn one() -> Self {
        Self::new(Self::Real::ONE, Self::Real::ZERO)
    }
}

macro_rules! real_element {
    ($t:ty) => {
        impl Element for $t {
            type Real = $t;
            const COMPLEX: bool = false;
            #[inline(always)]
            fn new(re: $t, _: $t) -> Self {
                re
            }
            #[inline(always)]
            fn re(self) -> $t {
                self
            }
            #[inline(always)]
            fn im(self) -> $t {
                0.0
            }
        }
    };
}
real_element!(f32);
real_element!(f64);

macro_rules! complex_element {
    ($t:ty) => {
        impl Element for Complex<$t> {
            type Real = $t;
            const COMPLEX: bool = true;
            #[inline(always)]
            fn new(re: $t, im: $t) -> Self {
                Complex::new(re, im)
            }
            #[inline(always)]
            fn re(self) -> $t {
                self.re
            }
            #[inline(always)]
            fn im(self) -> $t {
                self.im
            }
        }
    };
}
complex_element!(f32);
complex_element!(f64);
