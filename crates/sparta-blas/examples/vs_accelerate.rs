//! GFLOP/s of SpartaBLAS against Accelerate (its `$NEWLAPACK` routines, as
//! SPARTA calls them), single-threaded and on the whole machine.
//!
//!     cargo run --release -p sparta-blas --example vs_accelerate
#[cfg(target_os = "macos")]
fn main() {
    mac::main()
}
#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("Accelerate is macOS only");
}

#[cfg(target_os = "macos")]
mod mac {
    use sparta_blas::{abi, threading, Complex, Threading};
    use std::ffi::{c_char, c_void};
    use std::time::Instant;

    #[link(name = "Accelerate", kind = "framework")]
    extern "C" {
        fn BLASSetThreading(t: u32) -> i32;
    }
    extern "C" {
        fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    }
    type Zgemm = unsafe extern "C" fn(i32, i32, i32, i32, i32, i32, *const c_void, *const c_void, i32, *const c_void, i32, *const c_void, *mut c_void, i32);
    type Dgemm = unsafe extern "C" fn(i32, i32, i32, i32, i32, i32, f64, *const c_void, i32, *const c_void, i32, f64, *mut c_void, i32);
    type Trsm = unsafe extern "C" fn(i32, i32, i32, i32, i32, i32, i32, *const c_void, *const c_void, i32, *mut c_void, i32);
    type Getrf = unsafe extern "C" fn(*const i32, *const i32, *mut c_void, *const i32, *mut i32, *mut i32);

    fn sym<T>(name: &str) -> T {
        let c = std::ffi::CString::new(name).unwrap();
        let p = unsafe { dlsym(-2isize as *mut c_void, c.as_ptr()) };
        assert!(!p.is_null(), "{name} not found");
        unsafe { std::mem::transmute_copy(&p) }
    }

    /// Best seconds per call over repeats lasting at least ~0.3 s.
    fn time(mut f: impl FnMut()) -> f64 {
        f();
        let mut best = f64::MAX;
        let start = Instant::now();
        let mut reps = 0;
        while reps < 3 || start.elapsed().as_secs_f64() < 0.3 {
            let t = Instant::now();
            f();
            best = best.min(t.elapsed().as_secs_f64());
            reps += 1;
        }
        best
    }

    fn fill(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
            })
            .collect()
    }

    pub fn main() {
        let acc_z: Zgemm = sym("cblas_zgemm$NEWLAPACK");
        // Accelerate's `$NEWLAPACK` cgemm loses accuracy (see src/blas.rs history); the legacy one is what SPARTA used.
        let acc_c: Zgemm = sym("cblas_cgemm");
        type Sgemm = unsafe extern "C" fn(i32, i32, i32, i32, i32, i32, f32, *const c_void, i32, *const c_void, i32, f32, *mut c_void, i32);
        let acc_s: Sgemm = sym("cblas_sgemm$NEWLAPACK");
        let acc_d: Dgemm = sym("cblas_dgemm$NEWLAPACK");
        let acc_zt: Trsm = sym("cblas_ztrsm$NEWLAPACK");
        let acc_zgetrf: Getrf = sym("zgetrf$NEWLAPACK");
        println!("{}", sparta_blas::describe());
        // (m, n, k): squares, and the shapes of a front's Schur update:
        // tall and wide with a short depth.
        let shapes = [(128, 128, 32), (200, 200, 48), (300, 300, 64), (150, 400, 100), (256, 256, 256), (512, 512, 512), (1024, 1024, 1024), (2000, 2000, 64), (2000, 2000, 192), (600, 600, 96), (3000, 3000, 400)];
        const RM: i32 = abi::ROW_MAJOR;
        const NT: i32 = abi::NO_TRANS;
        const T: i32 = abi::TRANS;
        for (label, accelerate_threads, ours) in [("1 thread", 1u32, Threading::Caller), ("machine", 0, Threading::Machine)] {
            unsafe { BLASSetThreading(accelerate_threads) };
            threading::set(ours);
            println!("\n== {label}\n{:<18} {:>10} {:>10} {:>7}   {:>10} {:>10} {:>7}", "C -= A B^T", "z ours", "z Accel", "ratio", "d ours", "d Accel", "ratio");
            for &(m, n, k) in &shapes {
                let a = fill(2 * m * k, 1);
                let b = fill(2 * n * k, 2);
                let mut c = fill(2 * m * n, 3);
                let (minus, one) = ([-1.0f64, 0.0], [1.0f64, 0.0]);
                let (mi, ki, ni) = (m as i32, k as i32, n as i32);
                let zflops = 8.0 * (m * n * k) as f64 / 1e9;
                let dflops = 2.0 * (m * n * k) as f64 / 1e9;
                let mut run = |f: &mut dyn FnMut(*mut c_void)| time(|| f(c.as_mut_ptr().cast()));
                let zo = run(&mut |c| unsafe { abi::zgemm(RM, NT, T, mi, ni, ki, minus.as_ptr().cast(), a.as_ptr().cast(), ki, b.as_ptr().cast(), ki, one.as_ptr().cast(), c, ni) });
                let za = run(&mut |c| unsafe { acc_z(RM, NT, T, mi, ni, ki, minus.as_ptr().cast(), a.as_ptr().cast(), ki, b.as_ptr().cast(), ki, one.as_ptr().cast(), c, ni) });
                let d_o = run(&mut |c| unsafe { abi::dgemm(RM, NT, T, mi, ni, ki, -1.0, a.as_ptr().cast(), ki, b.as_ptr().cast(), ki, 1.0, c, ni) });
                let da = run(&mut |c| unsafe { acc_d(RM, NT, T, mi, ni, ki, -1.0, a.as_ptr().cast(), ki, b.as_ptr().cast(), ki, 1.0, c, ni) });
                println!(
                    "{:<18} {:>10.1} {:>10.1} {:>7.2}   {:>10.1} {:>10.1} {:>7.2}",
                    format!("{m}x{n}x{k}"),
                    zflops / zo,
                    zflops / za,
                    za / zo,
                    dflops / d_o,
                    dflops / da,
                    da / d_o
                );
            }
            println!("{:<18} {:>10} {:>10} {:>7}   {:>10} {:>10} {:>7}", "single precision", "c ours", "c Accel", "ratio", "s ours", "s Accel", "ratio");
            for &(m, n, k) in &shapes {
                let a: Vec<f32> = fill(2 * m * k, 1).iter().map(|&x| x as f32).collect();
                let b: Vec<f32> = fill(2 * n * k, 2).iter().map(|&x| x as f32).collect();
                let mut c: Vec<f32> = fill(2 * m * n, 3).iter().map(|&x| x as f32).collect();
                let (minus, one) = ([-1.0f32, 0.0], [1.0f32, 0.0]);
                let (mi, ki, ni) = (m as i32, k as i32, n as i32);
                let cflops = 8.0 * (m * n * k) as f64 / 1e9;
                let sflops = 2.0 * (m * n * k) as f64 / 1e9;
                let mut run = |f: &mut dyn FnMut(*mut c_void)| time(|| f(c.as_mut_ptr().cast()));
                let co = run(&mut |c| unsafe { abi::cgemm(RM, NT, T, mi, ni, ki, minus.as_ptr().cast(), a.as_ptr().cast(), ki, b.as_ptr().cast(), ki, one.as_ptr().cast(), c, ni) });
                let ca = run(&mut |c| unsafe { acc_c(RM, NT, T, mi, ni, ki, minus.as_ptr().cast(), a.as_ptr().cast(), ki, b.as_ptr().cast(), ki, one.as_ptr().cast(), c, ni) });
                let so = run(&mut |c| unsafe { abi::sgemm(RM, NT, T, mi, ni, ki, -1.0, a.as_ptr().cast(), ki, b.as_ptr().cast(), ki, 1.0, c, ni) });
                let sa = run(&mut |c| unsafe { acc_s(RM, NT, T, mi, ni, ki, -1.0, a.as_ptr().cast(), ki, b.as_ptr().cast(), ki, 1.0, c, ni) });
                println!("{:<18} {:>10.1} {:>10.1} {:>7.2}   {:>10.1} {:>10.1} {:>7.2}", format!("{m}x{n}x{k}"), cflops / co, cflops / ca, ca / co, sflops / so, sflops / sa, sa / so);
            }
            println!("{:<18} {:>10} {:>10} {:>7}", "ztrsm R/UnitUpper", "ours", "Accel", "ratio");
            for &(m, n) in &[(300usize, 8usize), (300, 16), (300, 32), (500, 48), (500, 64), (800, 100), (2000, 500), (2680, 996), (800, 256)] {
                let mut t = fill(2 * n * n, 4);
                for i in 0..n {
                    t[2 * (i * n + i)] += n as f64;
                }
                let b0 = fill(2 * m * n, 5);
                let mut b = b0.clone();
                let one = [1.0f64, 0.0];
                let (mi, ni) = (m as i32, n as i32);
                let flops = 4.0 * (m * n * n) as f64 / 1e9;
                let o = time(|| { b.copy_from_slice(&b0); unsafe { abi::ztrsm(RM, abi::RIGHT, abi::UPPER, NT, abi::UNIT, mi, ni, one.as_ptr().cast(), t.as_ptr().cast(), ni, b.as_mut_ptr().cast(), ni) } });
                let a_ = time(|| { b.copy_from_slice(&b0); unsafe { acc_zt(RM, abi::RIGHT, abi::UPPER, NT, abi::UNIT, mi, ni, one.as_ptr().cast(), t.as_ptr().cast(), ni, b.as_mut_ptr().cast(), ni) } });
                println!("{:<18} {:>10.1} {:>10.1} {:>7.2}", format!("{m}x{n}"), flops / o, flops / a_, a_ / o);
            }
            println!("{:<18} {:>10} {:>10} {:>7}", "zgetrf", "ours", "Accel", "ratio");
            for &n in &[256usize, 700, 1286, 2000] {
                let a0 = fill(2 * n * n, 6);
                let mut a = a0.clone();
                let mut ipiv = vec![0i32; n];
                let (ni, mut info) = (n as i32, 0i32);
                let flops = 8.0 / 3.0 * (n * n * n) as f64 / 1e9;
                let o = time(|| { a.copy_from_slice(&a0); unsafe { abi::zgetrf(&ni, &ni, a.as_mut_ptr().cast(), &ni, ipiv.as_mut_ptr(), &mut info) } });
                let a_ = time(|| { a.copy_from_slice(&a0); unsafe { acc_zgetrf(&ni, &ni, a.as_mut_ptr().cast(), &ni, ipiv.as_mut_ptr(), &mut info) } });
                let _ = Complex::new(0.0, 0.0);
                println!("{:<18} {:>10.1} {:>10.1} {:>7.2}", format!("{n}"), flops / o, flops / a_, a_ / o);
            }
        }
    }
}
