//! Kernels for Arm's Scalable Matrix Extension (SME2) at a 512-bit streaming
//! vector length: today, the outer-product unit of Apple M4 and later.
//!
//! SME is part of the Arm architecture, not one vendor's, and these kernels
//! are plain instructions that run on any operating system. What ties them to
//! a CPU is the vector length: SME lets each implementation choose 128 to 2048
//! bits, and every tile size, offset and loop count here is written for 512.
//! A CPU with another length would get its own folder (`sme256/`, ...).
//!
//! The matrix unit is a separate engine: the core enters streaming mode
//! (`smstart`), feeds it whole-vector loads and outer products that
//! accumulate into the ZA tile array, and leaves (`smstop`). Ordinary NEON
//! code cannot run in streaming mode, so each kernel is one `asm!` block
//! from `smstart` to `smstop`, and nothing the compiler generates runs in
//! between.
//!
//! The kernels assume the 512-bit streaming vector length of Apple's unit;
//! [`supported`] checks it, together with the SME2 and `f64` outer-product
//! features they use.
use crate::gemm::Block;

pub(crate) mod c;
pub(crate) mod d;
pub(crate) mod pack;
pub(crate) mod s;
pub(crate) mod z;

/// Whether this CPU runs these kernels: SME2, `f64` outer products, and a
/// 512-bit streaming vector length. Detected on macOS; other systems report
/// false until their feature query is added, and use NEON.
pub(crate) fn supported() -> bool {
    #[cfg(target_os = "macos")]
    {
        sysctl("hw.optional.arm.FEAT_SME2") && sysctl("hw.optional.arm.FEAT_SME_F64F64") && unsafe { svl_bytes() } == 64
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

#[cfg(target_os = "macos")]
fn sysctl(name: &str) -> bool {
    extern "C" {
        fn sysctlbyname(name: *const std::ffi::c_char, old: *mut std::ffi::c_void, len: *mut usize, new: *mut std::ffi::c_void, newlen: usize) -> i32;
    }
    let name = std::ffi::CString::new(name).unwrap();
    let (mut value, mut len) = (0i32, std::mem::size_of::<i32>());
    // SAFETY: a four-byte integer query into a four-byte buffer.
    let ok = unsafe { sysctlbyname(name.as_ptr(), (&mut value as *mut i32).cast(), &mut len, std::ptr::null_mut(), 0) } == 0;
    ok && value != 0
}

/// The streaming vector length in bytes. Readable outside streaming mode.
///
/// # Safety
/// The CPU must have SME.
#[cfg(target_os = "macos")]
unsafe fn svl_bytes() -> u64 {
    let n: u64;
    std::arch::asm!(".arch_extension sme", "rdsvl {n}, #1", n = out(reg) n, options(nomem, nostack, preserves_flags));
    n
}

/// The `f64` macro-kernel: every 32 × 16 tile of the block.
///
/// A C that is not contiguous down its columns goes through a scratch tile.
pub(crate) unsafe fn macro_d(b: &Block<f64>) {
    let (mr, nr) = (d::MR, d::NR);
    for j0 in (0..b.nc).step_by(nr) {
        for i0 in (0..b.mc).step_by(mr) {
            let (m, n) = (mr.min(b.mc - i0), nr.min(b.nc - j0));
            let (ap, bp) = (b.ap.add(i0 * b.kc), b.bp.add(j0 * b.kc));
            let c = b.c.offset(i0 as isize * b.rs + j0 as isize * b.cs);
            let unit = (b.alpha.0 == 1.0 || b.alpha.0 == -1.0) && (b.beta.0 == 0.0 || b.beta.0 == 1.0);
            if b.rs == 1 && unit {
                d::tile_unit(b.kc, ap, bp, c, b.cs, m, n, b.alpha.0 < 0.0, b.beta.0 == 1.0);
            } else if b.rs == 1 {
                d::tile(b.kc, ap, bp, c, b.cs, m, n, b.alpha.0, b.beta.0);
            } else {
                let mut tile = [0.0f64; 32 * 16];
                d::tile(b.kc, ap, bp, tile.as_mut_ptr(), 32, m, n, b.alpha.0, 0.0);
                for j in 0..n {
                    for i in 0..m {
                        let p = c.offset(i as isize * b.rs + j as isize * b.cs);
                        let v = tile[j * 32 + i];
                        *p = if b.beta.0 == 0.0 { v } else { v + b.beta.0 * *p };
                    }
                }
            }
        }
    }
}

/// The `Complex<f64>` macro-kernel: every 16 × 16 tile of the block, from
/// split-packed A and B.
pub(crate) unsafe fn macro_z(b: &Block<f64>) {
    let (mr, nr) = (z::MR, z::NR);
    for j0 in (0..b.nc).step_by(nr) {
        for i0 in (0..b.mc).step_by(mr) {
            let (m, n) = (mr.min(b.mc - i0), nr.min(b.nc - j0));
            let (ap, bp) = (b.ap.add(2 * i0 * b.kc), b.bp.add(2 * j0 * b.kc));
            let c = b.c.offset(2 * (i0 as isize * b.rs + j0 as isize * b.cs));
            let real_alpha = b.alpha.1 == 0.0 && (b.alpha.0 == 1.0 || b.alpha.0 == -1.0);
            let unit_beta = b.beta.1 == 0.0 && (b.beta.0 == 0.0 || b.beta.0 == 1.0);
            if b.rs == 1 && real_alpha && unit_beta {
                z::tile_unit(b.kc, ap, bp, c, b.cs, m, n, b.alpha.0 < 0.0, b.beta.0 == 1.0);
            } else if b.rs == 1 {
                z::tile(b.kc, ap, bp, c, b.cs, m, n, b.alpha, b.beta);
            } else {
                let mut tile = [0.0f64; 2 * 16 * 16];
                z::tile(b.kc, ap, bp, tile.as_mut_ptr(), 16, m, n, b.alpha, (0.0, 0.0));
                for j in 0..n {
                    for i in 0..m {
                        let p = c.offset(2 * (i as isize * b.rs + j as isize * b.cs));
                        let (vr, vi) = (tile[2 * (j * 16 + i)], tile[2 * (j * 16 + i) + 1]);
                        let (cr, ci) = (*p, *p.add(1));
                        let (br, bi) = b.beta;
                        *p = vr + if b.beta == (0.0, 0.0) { 0.0 } else { br * cr - bi * ci };
                        *p.add(1) = vi + if b.beta == (0.0, 0.0) { 0.0 } else { br * ci + bi * cr };
                    }
                }
            }
        }
    }
}

/// The `f32` macro-kernel: every 32 × 32 tile of the block.
pub(crate) unsafe fn macro_s(b: &Block<f32>) {
    let (mr, nr) = (s::MR, s::NR);
    for j0 in (0..b.nc).step_by(nr) {
        for i0 in (0..b.mc).step_by(mr) {
            let (m, n) = (mr.min(b.mc - i0), nr.min(b.nc - j0));
            let (ap, bp) = (b.ap.add(i0 * b.kc), b.bp.add(j0 * b.kc));
            let c = b.c.offset(i0 as isize * b.rs + j0 as isize * b.cs);
            let unit = (b.alpha.0 == 1.0 || b.alpha.0 == -1.0) && (b.beta.0 == 0.0 || b.beta.0 == 1.0);
            if b.rs == 1 && unit {
                s::tile_unit(b.kc, ap, bp, c, b.cs, m, n, b.alpha.0 < 0.0, b.beta.0 == 1.0);
            } else if b.rs == 1 {
                s::tile(b.kc, ap, bp, c, b.cs, m, n, b.alpha.0, b.beta.0);
            } else {
                let mut tile = [0.0f32; 32 * 32];
                s::tile(b.kc, ap, bp, tile.as_mut_ptr(), 32, m, n, b.alpha.0, 0.0);
                for j in 0..n {
                    for i in 0..m {
                        let p = c.offset(i as isize * b.rs + j as isize * b.cs);
                        let v = tile[j * 32 + i];
                        *p = if b.beta.0 == 0.0 { v } else { v + b.beta.0 * *p };
                    }
                }
            }
        }
    }
}

/// The `Complex<f32>` macro-kernel: every 32 × 16 tile of the block, from
/// split-packed A and B.
pub(crate) unsafe fn macro_c(b: &Block<f32>) {
    let (mr, nr) = (c::MR, c::NR);
    for j0 in (0..b.nc).step_by(nr) {
        for i0 in (0..b.mc).step_by(mr) {
            let (m, n) = (mr.min(b.mc - i0), nr.min(b.nc - j0));
            let (ap, bp) = (b.ap.add(2 * i0 * b.kc), b.bp.add(2 * j0 * b.kc));
            let cp = b.c.offset(2 * (i0 as isize * b.rs + j0 as isize * b.cs));
            let real_alpha = b.alpha.1 == 0.0 && (b.alpha.0 == 1.0 || b.alpha.0 == -1.0);
            let unit_beta = b.beta.1 == 0.0 && (b.beta.0 == 0.0 || b.beta.0 == 1.0);
            if b.rs == 1 && real_alpha && unit_beta {
                c::tile_unit(b.kc, ap, bp, cp, b.cs, m, n, b.alpha.0 < 0.0, b.beta.0 == 1.0);
            } else if b.rs == 1 {
                c::tile(b.kc, ap, bp, cp, b.cs, m, n, b.alpha, b.beta);
            } else {
                let mut tile = [0.0f32; 2 * 32 * 16];
                c::tile(b.kc, ap, bp, tile.as_mut_ptr(), 32, m, n, b.alpha, (0.0, 0.0));
                let beta_zero = b.beta == (0.0, 0.0);
                for j in 0..n {
                    for i in 0..m {
                        let p = cp.offset(2 * (i as isize * b.rs + j as isize * b.cs));
                        let (vr, vi) = (tile[2 * (j * 32 + i)], tile[2 * (j * 32 + i) + 1]);
                        let (cr, ci) = (*p, *p.add(1));
                        let (br, bi) = b.beta;
                        *p = vr + if beta_zero { 0.0 } else { br * cr - bi * ci };
                        *p.add(1) = vi + if beta_zero { 0.0 } else { br * ci + bi * cr };
                    }
                }
            }
        }
    }
}
