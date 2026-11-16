//! The FMA ceiling of one core and of the machine: independent fused
//! multiply-adds on registers only, no memory traffic. The number a kernel's
//! GFLOP/s is a fraction of.
//!
//!     cargo run --release -p sparta-blas --example peak
fn main() {
    // Best of several: the first runs while the core is still ramping its clock.
    let one = (0..5).map(|_| peak_f64()).fold(0.0, f64::max);
    let threads = rayon::current_num_threads();
    let start = std::time::Instant::now();
    let total: f64 = (0..3).map(|_| rayon::broadcast(|_| peak_f64()).iter().sum::<f64>()).fold(0.0, f64::max);
    let _ = start;
    println!("NEON f64 FMA peak: {one:.1} GFLOP/s on one core, {total:.1} GFLOP/s on {threads} threads");
    #[cfg(target_os = "macos")]
    sme::report(threads);
}

/// The matrix unit's outer-product ceiling (SME, 512-bit vectors), where the
/// CPU has one: independent `fmopa` into all of ZA, no memory traffic.
#[cfg(target_os = "macos")]
mod sme {
    use std::time::Instant;

    /// GFLOP/s of `iters` rounds of outer products of `bits`-bit floats.
    fn run(f64_: bool, iters: u64) -> f64 {
        let t = Instant::now();
        // smstart/smstop zero every vector and predicate register: all are
        // declared clobbered.
        macro_rules! body {
            ($($line:literal),*) => {
                unsafe {
                    std::arch::asm!(
                        ".arch_extension sme", ".arch_extension sme-f64f64",
                        "smstart", "zero {{za}}", "ptrue p0.d",
                        "fdup z0.d, #1.0", "fdup z1.d, #1.0", "fdup z2.d, #1.0", "fdup z3.d, #1.0",
                        "2:", $($line,)* "subs x9, x9, #1", "b.ne 2b", "smstop",
                        inout("x9") iters => _,
                        out("v8") _, out("v9") _, out("v10") _, out("v11") _,
                        out("v12") _, out("v13") _, out("v14") _, out("v15") _,
                        clobber_abi("C"),
                    )
                }
            };
        }
        if f64_ {
            body!(
                "fmopa za0.d, p0/m, p0/m, z0.d, z2.d", "fmopa za1.d, p0/m, p0/m, z0.d, z3.d",
                "fmopa za2.d, p0/m, p0/m, z1.d, z2.d", "fmopa za3.d, p0/m, p0/m, z1.d, z3.d",
                "fmopa za4.d, p0/m, p0/m, z0.d, z2.d", "fmopa za5.d, p0/m, p0/m, z0.d, z3.d",
                "fmopa za6.d, p0/m, p0/m, z1.d, z2.d", "fmopa za7.d, p0/m, p0/m, z1.d, z3.d"
            );
            (iters * 8 * 64 * 2) as f64 / t.elapsed().as_secs_f64() / 1e9
        } else {
            body!(
                "fmopa za0.s, p0/m, p0/m, z0.s, z2.s", "fmopa za1.s, p0/m, p0/m, z0.s, z3.s",
                "fmopa za2.s, p0/m, p0/m, z1.s, z2.s", "fmopa za3.s, p0/m, p0/m, z1.s, z3.s"
            );
            (iters * 4 * 256 * 2) as f64 / t.elapsed().as_secs_f64() / 1e9
        }
    }

    pub fn report(threads: usize) {
        let has = |name: &str| {
            let out = std::process::Command::new("sysctl").args(["-n", name]).output();
            out.map(|o| String::from_utf8_lossy(&o.stdout).trim() == "1").unwrap_or(false)
        };
        if !(has("hw.optional.arm.FEAT_SME") && has("hw.optional.arm.FEAT_SME_F64F64")) {
            return;
        }
        for (label, f64_) in [("f64", true), ("f32", false)] {
            let one = (0..5).map(|_| run(f64_, 10_000_000)).fold(0.0, f64::max);
            let all: f64 = (0..3)
                .map(|_| rayon::broadcast(|_| run(f64_, 5_000_000)).iter().sum::<f64>())
                .fold(0.0, f64::max);
            println!("SME  {label} outer-product peak: {one:.0} GFLOP/s on one thread, {all:.0} GFLOP/s on {threads} threads");
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn peak_f64() -> f64 {
    // Written in assembly so nothing but 24 independent FMAs per iteration
    // and the loop counter remain.
    let iters = 100_000_000u64;
    let t = std::time::Instant::now();
    unsafe {
        std::arch::asm!(
            "movi v30.2d, #0", "movi v31.2d, #0",
            "2:",
            "fmla v0.2d, v30.2d, v31.2d", "fmla v1.2d, v30.2d, v31.2d", "fmla v2.2d, v30.2d, v31.2d", "fmla v3.2d, v30.2d, v31.2d", "fmla v4.2d, v30.2d, v31.2d", "fmla v5.2d, v30.2d, v31.2d", "fmla v6.2d, v30.2d, v31.2d", "fmla v7.2d, v30.2d, v31.2d", "fmla v8.2d, v30.2d, v31.2d", "fmla v9.2d, v30.2d, v31.2d", "fmla v10.2d, v30.2d, v31.2d", "fmla v11.2d, v30.2d, v31.2d", "fmla v12.2d, v30.2d, v31.2d", "fmla v13.2d, v30.2d, v31.2d", "fmla v14.2d, v30.2d, v31.2d", "fmla v15.2d, v30.2d, v31.2d", "fmla v16.2d, v30.2d, v31.2d", "fmla v17.2d, v30.2d, v31.2d", "fmla v18.2d, v30.2d, v31.2d", "fmla v19.2d, v30.2d, v31.2d", "fmla v20.2d, v30.2d, v31.2d", "fmla v21.2d, v30.2d, v31.2d", "fmla v22.2d, v30.2d, v31.2d", "fmla v23.2d, v30.2d, v31.2d",
            "subs {n}, {n}, #1",
            "b.ne 2b",
            n = inout(reg) iters => _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _,
            out("v6") _, out("v7") _, out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _, out("v16") _, out("v17") _,
            out("v18") _, out("v19") _, out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v30") _, out("v31") _,
        );
    }
    let s = t.elapsed().as_secs_f64();
    (iters * 24 * 2 * 2) as f64 / s / 1e9
}

#[cfg(target_arch = "x86_64")]
fn peak_f64() -> f64 {
    #[target_feature(enable = "avx2,fma")]
    unsafe fn run() -> f64 {
        use std::arch::x86_64::*;
        let iters = 50_000_000u64;
        let mut acc = [_mm256_setzero_pd(); 12];
        let (a, b) = (_mm256_set1_pd(1.0000001), _mm256_set1_pd(0.9999999));
        let t = std::time::Instant::now();
        for _ in 0..iters {
            for v in acc.iter_mut() {
                *v = _mm256_fmadd_pd(a, b, *v);
            }
        }
        let s = t.elapsed().as_secs_f64();
        std::hint::black_box(&acc);
        (iters * 12 * 4 * 2) as f64 / s / 1e9
    }
    unsafe { run() }
}
