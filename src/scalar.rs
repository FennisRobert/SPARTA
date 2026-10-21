//! The four value types SPARTA factors: `f32`, `f64`, `Complex32`, `Complex64`.
//!
//! [`Scalar`] is what the public API is generic over, so that
//! `Solver::factor_values(&values)` picks the right numerics from the slice's
//! element type. Inside the CPU backend it supplies the handful of operations
//! that differ between real and complex arithmetic; everything else is plain
//! operators on the concrete type each copy of the backend is compiled for.
use num_complex::{Complex32, Complex64};
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for num_complex::Complex32 {}
    impl Sealed for num_complex::Complex64 {}
}

/// Which of the four value types a [`Scalar`] is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScalarKind {
    F32,
    F64,
    Complex32,
    Complex64,
}

impl ScalarKind {
    /// Whether the type carries an imaginary part.
    pub const fn is_complex(self) -> bool {
        matches!(self, Self::Complex32 | Self::Complex64)
    }
    /// Whether the type is double precision.
    pub const fn is_double(self) -> bool {
        matches!(self, Self::F64 | Self::Complex64)
    }
    /// The BLAS prefix of the type: `s`, `d`, `c` or `z`.
    pub const fn blas(self) -> char {
        match self {
            Self::F32 => 's',
            Self::F64 => 'd',
            Self::Complex32 => 'c',
            Self::Complex64 => 'z',
        }
    }
}

impl std::fmt::Display for ScalarKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::F32 => "float32",
            Self::F64 => "float64",
            Self::Complex32 => "complex64",
            Self::Complex64 => "complex128",
        })
    }
}

/// A value type SPARTA can factor. Implemented for `f32`, `f64`, `Complex32`
/// and `Complex64` only.
pub trait Scalar:
    sealed::Sealed
    + Copy
    + Debug
    + Default
    + PartialEq
    + Send
    + Sync
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
{
    /// The real type of the same precision.
    type Real: Copy + Debug + PartialOrd + Send + Sync + 'static;
    const KIND: ScalarKind;
    /// The real number `r`.
    fn from_re(r: Self::Real) -> Self;
    /// The real part.
    fn re(self) -> Self::Real;
    /// The imaginary part, zero for a real type.
    fn im(self) -> Self::Real;
    /// The modulus: `|x|` for a real type.
    fn norm(self) -> Self::Real;
    /// Whether every component is finite.
    fn finite(self) -> bool;
    /// `self / b` by the scaled component formula the GPU kernels use, which
    /// for a real type is plain division.
    fn scaled_div(self, b: Self) -> Self;
    /// The value as complex128, exactly.
    fn to_c64(self) -> Complex64;
    /// `v` rounded to this type; the imaginary part is dropped for a real type.
    fn from_c64(v: Complex64) -> Self;
    /// The real part as `f64`, for routines that take real scalars by value.
    fn re_f64(self) -> f64;
}

macro_rules! real_scalar {
    ($t:ty, $kind:ident) => {
        impl Scalar for $t {
            type Real = $t;
            const KIND: ScalarKind = ScalarKind::$kind;
            #[inline(always)]
            fn from_re(r: $t) -> Self {
                r
            }
            #[inline(always)]
            fn re(self) -> $t {
                self
            }
            #[inline(always)]
            fn im(self) -> $t {
                0.0
            }
            #[inline(always)]
            fn norm(self) -> $t {
                self.abs()
            }
            #[inline(always)]
            fn finite(self) -> bool {
                self.is_finite()
            }
            #[inline(always)]
            fn scaled_div(self, b: Self) -> Self {
                self / b
            }
            #[inline(always)]
            fn to_c64(self) -> Complex64 {
                Complex64::new(self as f64, 0.0)
            }
            #[inline(always)]
            fn from_c64(v: Complex64) -> Self {
                v.re as $t
            }
            #[inline(always)]
            fn re_f64(self) -> f64 {
                self as f64
            }
        }
    };
}
real_scalar!(f32, F32);
real_scalar!(f64, F64);

macro_rules! complex_scalar {
    ($t:ty, $r:ty, $kind:ident) => {
        impl Scalar for $t {
            type Real = $r;
            const KIND: ScalarKind = ScalarKind::$kind;
            #[inline(always)]
            fn from_re(r: $r) -> Self {
                <$t>::new(r, 0.0)
            }
            #[inline(always)]
            fn re(self) -> $r {
                self.re
            }
            #[inline(always)]
            fn im(self) -> $r {
                self.im
            }
            #[inline(always)]
            fn norm(self) -> $r {
                self.re.hypot(self.im)
            }
            #[inline(always)]
            fn finite(self) -> bool {
                self.re.is_finite() && self.im.is_finite()
            }
            #[inline(always)]
            fn scaled_div(self, b: Self) -> Self {
                let a = self;
                if b.re.abs() >= b.im.abs() {
                    let r = b.im / b.re;
                    let d = b.re + b.im * r;
                    <$t>::new((a.re + a.im * r) / d, (a.im - a.re * r) / d)
                } else {
                    let r = b.re / b.im;
                    let d = b.im + b.re * r;
                    <$t>::new((a.re * r + a.im) / d, (a.im * r - a.re) / d)
                }
            }
            #[inline(always)]
            fn to_c64(self) -> Complex64 {
                Complex64::new(self.re as f64, self.im as f64)
            }
            #[inline(always)]
            fn from_c64(v: Complex64) -> Self {
                <$t>::new(v.re as $r, v.im as $r)
            }
            #[inline(always)]
            fn re_f64(self) -> f64 {
                self.re as f64
            }
        }
    };
}
complex_scalar!(Complex32, f32, Complex32);
complex_scalar!(Complex64, f64, Complex64);
