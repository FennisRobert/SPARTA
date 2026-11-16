//! Which instruction set the kernels run on, chosen once at run time.
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::OnceLock;

/// The kernel family in use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Isa {
    /// Portable Rust, one value at a time; any CPU.
    Scalar,
    /// aarch64 Advanced SIMD, 128-bit.
    Neon,
    /// x86-64 AVX2 with FMA, 256-bit.
    Avx2,
    /// x86-64 AVX-512F, 512-bit.
    Avx512,
    /// Arm SME2 (Scalable Matrix Extension) at a 512-bit streaming vector
    /// length: Apple M4 and later.
    Sme512,
}

impl Isa {
    /// The name `SPARTA_KERNEL` accepts and [`describe`] prints.
    pub fn name(self) -> &'static str {
        match self {
            Isa::Scalar => "scalar",
            Isa::Neon => "neon",
            Isa::Avx2 => "avx2",
            Isa::Avx512 => "avx512",
            Isa::Sme512 => "sme512",
        }
    }

    fn code(self) -> u8 {
        self as u8 + 1
    }

    fn from_code(code: u8) -> Option<Isa> {
        [Isa::Scalar, Isa::Neon, Isa::Avx2, Isa::Avx512, Isa::Sme512].get(code.checked_sub(1)? as usize).copied()
    }

    /// Whether this CPU can run the family.
    pub fn supported(self) -> bool {
        match self {
            Isa::Scalar => true,
            Isa::Neon => cfg!(target_arch = "aarch64"),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => is_x86_feature_detected!("avx512f"),
            #[cfg(not(target_arch = "x86_64"))]
            Isa::Avx2 | Isa::Avx512 => false,
            #[cfg(target_arch = "aarch64")]
            Isa::Sme512 => crate::sme512::supported(),
            #[cfg(not(target_arch = "aarch64"))]
            Isa::Sme512 => false,
        }
    }

    /// Every family this CPU can run, best first.
    pub fn available() -> Vec<Isa> {
        [Isa::Sme512, Isa::Avx512, Isa::Avx2, Isa::Neon, Isa::Scalar].into_iter().filter(|i| i.supported()).collect()
    }
}

/// A family forced by [`force`], or zero.
static FORCED: AtomicU8 = AtomicU8::new(0);

/// The best family this CPU runs, or the one `SPARTA_KERNEL` names when the
/// CPU runs it.
fn detected() -> Isa {
    static DETECTED: OnceLock<Isa> = OnceLock::new();
    *DETECTED.get_or_init(|| {
        let best = Isa::available()[0];
        match std::env::var("SPARTA_KERNEL") {
            Ok(name) if !name.is_empty() => match Isa::available().into_iter().find(|i| i.name() == name) {
                Some(isa) => isa,
                None => {
                    eprintln!("SPARTA: SPARTA_KERNEL={name} is not available on this CPU; using {}", best.name());
                    best
                }
            },
            _ => best,
        }
    })
}

/// The family the next call runs on.
pub fn current() -> Isa {
    Isa::from_code(FORCED.load(Ordering::Relaxed)).unwrap_or_else(detected)
}

/// Run every later call on `isa`, or go back to the detected family with
/// `None`. For tests and benchmarks: it is process-wide.
///
/// - `isa`: the family; refused when this CPU cannot run it.
pub fn force(isa: Option<Isa>) -> Result<(), String> {
    match isa {
        Some(isa) if !isa.supported() => Err(format!("{} is not available on this CPU", isa.name())),
        Some(isa) => {
            FORCED.store(isa.code(), Ordering::Relaxed);
            Ok(())
        }
        None => {
            FORCED.store(0, Ordering::Relaxed);
            Ok(())
        }
    }
}

/// Define `fn $name` that runs `$body` compiled for the kernel family in use.
///
/// The crate is compiled for the target's baseline, which on x86-64 is SSE2:
/// only the microkernels carry AVX2 or AVX-512 themselves. This compiles a
/// generic `$body`, which must be `#[inline(always)]`, twice more on x86-64,
/// with AVX2 and with AVX-512 enabled, and picks the copy matching
/// [`current`]. Elsewhere it is a plain call. Used where it was measured to
/// matter: the LDLᵀ panel. The packing and the small triangular solves
/// showed no change on Zen 4.
macro_rules! multiversion {
    ($(#[$m:meta])* $vis:vis unsafe fn $name:ident <$T:ident : $B:path> ($($a:ident : $t:ty),* $(,)?) $(-> $r:ty)? => $body:ident) => {
        $(#[$m])*
        #[allow(clippy::too_many_arguments)]
        $vis unsafe fn $name<$T: $B>($($a: $t),*) $(-> $r)? {
            #[cfg(target_arch = "x86_64")]
            {
                #[target_feature(enable = "avx512f,avx2,fma")]
                #[allow(clippy::too_many_arguments)]
                unsafe fn avx512<$T: $B>($($a: $t),*) $(-> $r)? {
                    $body::<$T>($($a),*)
                }
                #[target_feature(enable = "avx2,fma")]
                #[allow(clippy::too_many_arguments)]
                unsafe fn avx2<$T: $B>($($a: $t),*) $(-> $r)? {
                    $body::<$T>($($a),*)
                }
                match $crate::isa::current() {
                    $crate::isa::Isa::Avx512 => return avx512::<$T>($($a),*),
                    $crate::isa::Isa::Avx2 => return avx2::<$T>($($a),*),
                    _ => {}
                }
            }
            $body::<$T>($($a),*)
        }
    };
}
pub(crate) use multiversion;
