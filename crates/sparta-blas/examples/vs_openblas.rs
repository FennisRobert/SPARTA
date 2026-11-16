//! GFLOP/s of SpartaBLAS against an OpenBLAS loaded at run time, single
//! threaded and on the whole machine, on the shapes SPARTA's CPU
//! factorization calls.
//!
//!     OPENBLAS_LIB=/path/to/libopenblas.so cargo run --release -p sparta-blas --example vs_openblas
//!
//! Both plain symbols (`cblas_zgemm`) and the `scipy_` prefix of the
//! `scipy-openblas32` wheel are accepted. SpartaBLAS runs on the Rayon global
//! pool, OpenBLAS on its own threads, both sized to every core.
#[cfg(unix)]
fn main() {
    unix::main()
}
#[cfg(not(unix))]
fn main() {
    eprintln!("vs_openblas loads OpenBLAS with dlopen, Unix only");
}

#[cfg(unix)]
mod unix {
    use sparta_blas::{abi, threading, Threading};
    use std::ffi::{c_char, c_int, c_void, CString};
    use std::time::Instant;

    extern "C" {
        fn dlopen(path: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    }
    type Zgemm = unsafe extern "C" fn(i32, i32, i32, i32, i32, i32, *const c_void, *const c_void, i32, *const c_void, i32, *const c_void, *mut c_void, i32);
    type Dgemm = unsafe extern "C" fn(i32, i32, i32, i32, i32, i32, f64, *const c_void, i32, *const c_void, i32, f64, *mut c_void, i32);
    type Trsm = unsafe extern "C" fn(i32, i32, i32, i32, i32, i32, i32, *const c_void, *const c_void, i32, *mut c_void, i32);
    type Sytrf = unsafe extern "C" fn(*const u8, *const i32, *mut c_void, *const i32, *mut i32, *mut c_void, *const i32, *mut i32);
    type SetThreads = unsafe extern "C" fn(c_int);

    struct Lib(*mut c_void);
    impl Lib {
        fn sym<T>(&self, name: &str) -> T {
            for prefix in ["", "scipy_"] {
                let c = CString::new(format!("{prefix}{name}")).unwrap();
                let p = unsafe { dlsym(self.0, c.as_ptr()) };
                if !p.is_null() {
                    return unsafe { std::mem::transmute_copy(&p) };
                }
            }
            panic!("{name} not found in OPENBLAS_LIB");
        }
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
        let path = std::env::var("OPENBLAS_LIB").expect("set OPENBLAS_LIB to the OpenBLAS shared library");
        let cpath = CString::new(path.clone()).unwrap();
        let lib = Lib(unsafe { dlopen(cpath.as_ptr(), 2) });
        assert!(!lib.0.is_null(), "cannot open {path}");
        let ob_z: Zgemm = lib.sym("cblas_zgemm");
        let ob_d: Dgemm = lib.sym("cblas_dgemm");
        let ob_zt: Trsm = lib.sym("cblas_ztrsm");
        let ob_zs: Sytrf = lib.sym("zsytrf_");
        let ob_threads: SetThreads = lib.sym("openblas_set_num_threads");
        let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
        let only = std::env::args().nth(1).unwrap_or_default();
        println!("{} against {path}, {cores} cores", sparta_blas::describe());
        // (m, n, k): squares, and the shapes of a front's Schur update:
        // tall and wide with a short depth.
        let shapes = [(128, 128, 32), (200, 200, 48), (300, 300, 64), (150, 400, 100), (256, 256, 256), (512, 512, 512), (1024, 1024, 1024), (2000, 2000, 64), (2000, 2000, 192), (600, 600, 96), (3000, 3000, 400)];
        const RM: i32 = abi::ROW_MAJOR;
        const NT: i32 = abi::NO_TRANS;
        const T: i32 = abi::TRANS;
        for (label, ob_n, ours) in [("1 thread", 1, Threading::Caller), ("machine", cores as c_int, Threading::Machine)] {
            if !only.is_empty() && only != label.split(' ').next().unwrap() {
                continue;
            }
            unsafe { ob_threads(ob_n) };
            threading::set(ours);
            println!("\n== {label}\n{:<18} {:>10} {:>10} {:>7}   {:>10} {:>10} {:>7}", "C -= A B^T", "z ours", "z OpenBLAS", "ratio", "d ours", "d OpenBLAS", "ratio");
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
                let zb = run(&mut |c| unsafe { ob_z(RM, NT, T, mi, ni, ki, minus.as_ptr().cast(), a.as_ptr().cast(), ki, b.as_ptr().cast(), ki, one.as_ptr().cast(), c, ni) });
                let d_o = run(&mut |c| unsafe { abi::dgemm(RM, NT, T, mi, ni, ki, -1.0, a.as_ptr().cast(), ki, b.as_ptr().cast(), ki, 1.0, c, ni) });
                let db = run(&mut |c| unsafe { ob_d(RM, NT, T, mi, ni, ki, -1.0, a.as_ptr().cast(), ki, b.as_ptr().cast(), ki, 1.0, c, ni) });
                println!(
                    "{:<18} {:>10.1} {:>10.1} {:>7.2}   {:>10.1} {:>10.1} {:>7.2}",
                    format!("{m}x{n}x{k}"),
                    zflops / zo,
                    zflops / zb,
                    zb / zo,
                    dflops / d_o,
                    dflops / db,
                    db / d_o
                );
            }
            println!("{:<18} {:>10} {:>10} {:>7}", "ztrsm R/UnitUpper", "ours", "OpenBLAS", "ratio");
            for &(m, n) in &[(300usize, 8usize), (300, 32), (500, 64), (800, 100), (2000, 500), (2680, 996), (800, 256)] {
                let mut t = fill(2 * n * n, 4);
                for i in 0..n {
                    t[2 * (i * n + i)] += n as f64;
                }
                let b0 = fill(2 * m * n, 5);
                let mut b = b0.clone();
                let one = [1.0f64, 0.0];
                let (mi, ni) = (m as i32, n as i32);
                let flops = 4.0 * (m * n * n) as f64 / 1e9;
                let o = time(|| {
                    b.copy_from_slice(&b0);
                    unsafe { abi::ztrsm(RM, abi::RIGHT, abi::UPPER, NT, abi::UNIT, mi, ni, one.as_ptr().cast(), t.as_ptr().cast(), ni, b.as_mut_ptr().cast(), ni) }
                });
                let x = time(|| {
                    b.copy_from_slice(&b0);
                    unsafe { ob_zt(RM, abi::RIGHT, abi::UPPER, NT, abi::UNIT, mi, ni, one.as_ptr().cast(), t.as_ptr().cast(), ni, b.as_mut_ptr().cast(), ni) }
                });
                println!("{:<18} {:>10.1} {:>10.1} {:>7.2}", format!("{m}x{n}"), flops / o, flops / x, x / o);
            }
            println!("{:<18} {:>10} {:>10} {:>7}", "zsytrf L", "ours", "OpenBLAS", "ratio");
            for &n in &[128usize, 256, 512, 704, 1286] {
                let mut a0 = fill(2 * n * n, 6);
                for i in 0..n {
                    for j in 0..i {
                        let (x, y) = (a0[2 * (i * n + j)], a0[2 * (i * n + j) + 1]);
                        a0[2 * (j * n + i)] = x;
                        a0[2 * (j * n + i) + 1] = y;
                    }
                }
                let mut a = a0.clone();
                let mut ipiv = vec![0i32; n];
                let (ni, mut info) = (n as i32, 0i32);
                let lwork = (64 * n) as i32;
                let mut work = vec![0.0f64; 2 * 64 * n];
                let flops = 8.0 / 3.0 * (n * n * n) as f64 / 1e9;
                let o = time(|| {
                    a.copy_from_slice(&a0);
                    unsafe { abi::zsytrf(b"L".as_ptr(), &ni, a.as_mut_ptr().cast(), &ni, ipiv.as_mut_ptr(), work.as_mut_ptr().cast(), &lwork, &mut info) }
                });
                let x = time(|| {
                    a.copy_from_slice(&a0);
                    unsafe { ob_zs(b"L".as_ptr(), &ni, a.as_mut_ptr().cast(), &ni, ipiv.as_mut_ptr(), work.as_mut_ptr().cast(), &lwork, &mut info) }
                });
                println!("{:<18} {:>10.1} {:>10.1} {:>7.2}", format!("{n}"), flops / o, flops / x, x / o);
            }
        }
    }
}
