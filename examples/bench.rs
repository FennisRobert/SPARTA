//! End-to-end SPARTA benchmark on real assembled systems, with no Python in
//! the measured path.
//!
//!     cargo run --release --example bench -- test_matrices/em_m1_sym*.sparta
//!
//! The pattern is analyzed once; every later snapshot reuses that plan through
//! `refactor`, which is how an EMerge frequency sweep actually calls the solver.
//! `--force-cpu` selects end-to-end complex128 CPU numerics.
//! Throughput is reported in MDOF/s, millions of unknowns per second: `fact`
//! over the device (or CPU) factorization time, `step` over one sweep step's
//! factor and solve wall time.
//! Options: `--max-front N`, `--skip-solve`, and `--curl-curl K` to generate a
//! synthetic `K^3` curl-curl system instead of reading files.
mod common;

use common::{gib, relative_difference, System};
use sparta::{Analysis, Complex32, Kind, Solver};
use std::time::Instant;

fn to_single(values: &[common::C64]) -> Vec<Complex32> {
    values
        .iter()
        .map(|v| Complex32::new(v.re as f32, v.im as f32))
        .collect()
}

fn main() {
    let mut max_front = usize::MAX;
    let mut skip_solve = false;
    let mut force_cpu = false;
    let mut paths = Vec::new();
    let mut generate = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--max-front" => {
                max_front = args
                    .next()
                    .expect("--max-front needs a value")
                    .parse()
                    .unwrap()
            }
            "--force-cpu" | "--force_cpu" => force_cpu = true,
            "--skip-solve" => skip_solve = true,
            "--curl-curl" => generate.push(
                args.next()
                    .expect("--curl-curl needs a grid size")
                    .parse()
                    .unwrap(),
            ),
            _ => paths.push(arg),
        }
    }
    if paths.is_empty() && generate.is_empty() {
        eprintln!("usage: bench [--max-front N] [--skip-solve] [--force-cpu] [--curl-curl K] FILE...");
        std::process::exit(2);
    }

    let clock = Instant::now();
    let mut systems: Vec<System> = generate.into_iter().map(System::curl_curl).collect();
    systems.extend(
        paths
            .iter()
            .map(|p| System::load(p).unwrap_or_else(|e| panic!("{e}"))),
    );
    // The container's storage flag picks the factorization: a lower triangle is
    // complex-symmetric and takes LDL^T, a full matrix takes LU. Mixing the two
    // in one run would analyze one pattern and factor another.
    if systems.iter().any(|s| s.full) != systems.iter().all(|s| s.full) {
        eprintln!("symmetric and unsymmetric systems cannot share one analysis");
        std::process::exit(2);
    }
    let unsymmetric = systems[0].full;
    println!(
        "loaded {} system(s) in {:.3}s: n={} nnz({})={}",
        systems.len(),
        clock.elapsed().as_secs_f64(),
        systems[0].n,
        if unsymmetric { "full" } else { "lower" },
        systems[0].values.len()
    );

    let head = &systems[0];
    let clock = Instant::now();
    let analyze = if unsymmetric {
        Analysis::lu
    } else {
        Analysis::ldlt
    };
    let config = sparta::Config {
        force_cpu,
        max_front: (max_front != usize::MAX).then_some(max_front),
        ..sparta::Config::default()
    };
    let plan = analyze(head.n, &head.col_ptr, &head.row_idx, &config)
        .unwrap_or_else(|e| panic!("analyze failed: {e}"));
    let analyze_seconds = clock.elapsed().as_secs_f64();
    println!("ordering {:?}", config.ordering);
    println!(
        "path     {}",
        match plan.kind() {
            Kind::Ldlt => "LDL^T, complex-symmetric",
            Kind::Lu => "LU, partial pivoting",
        }
    );
    println!(
        "analyze  {analyze_seconds:7.3}s  supernodes={}  levels={}  max_front={}  large={}",
        plan.supernodes(),
        plan.levels(),
        plan.max_front_size(),
        plan.large_fronts()
    );
    let resident: usize = plan.front_bytes();
    // `total` is every device buffer, not just these two: see
    // `Analysis::device_bytes`. The descriptors and the assembly map are worth
    // an eighth of a large solve on their own.
    println!(
        "memory   workspace={:.3} GiB  factors={:.3} GiB (U {:.3})  total={:.3} GiB",
        gib(resident),
        gib(plan.factor_bytes()),
        gib(plan.upper_bytes()),
        gib(plan.device_bytes())
    );

    if force_cpu {
        println!("backend  CPU complex128 ({})", env!("SPARTA_DENSE_LIBRARY"));
    } else {
        println!("backend  {} complex64", sparta::Device::COMPILED.name());
    }
    if !force_cpu && !sparta::gpu_available() {
        println!("\nno GPU device: analysis only");
        return;
    }
    let clock = Instant::now();
    let mut solver = Solver::new(plan).unwrap_or_else(|e| panic!("solver allocation failed: {e}"));
    println!("allocate {:7.3}s", clock.elapsed().as_secs_f64());

    println!(
        "\n{:<14} {:>7} {:>9} {:>9} {:>9} {:>11} {:>11} {:>7} {:>7}",
        "snapshot",
        "k0",
        "factor",
        if force_cpu { "CPU" } else { "device" },
        "solve",
        "residual",
        "vs ref",
        "fact",
        "step"
    );
    let mdof = |seconds: f64| head.n as f64 / seconds / 1e6;
    let (mut factor_total, mut solve_total) = (0.0, 0.0);
    for system in &systems {
        assert_eq!(
            system.n, head.n,
            "{}: pattern differs from the analyzed one",
            system.name
        );
        assert_eq!(
            system.row_idx, head.row_idx,
            "{}: pattern differs",
            system.name
        );
        assert_eq!(
            system.col_ptr, head.col_ptr,
            "{}: column pointers differ",
            system.name
        );
        let values = if force_cpu {
            Vec::new()
        } else {
            to_single(&system.values)
        };

        let clock = Instant::now();
        let result = if force_cpu {
            solver.factor_c128(&system.values)
        } else {
            solver.factor(&values)
        };
        if let Err(e) = result {
            println!(
                "{:<14} {:>7.3}  factorization failed: {e}",
                system.name, system.k0
            );
            continue;
        }
        let factor_seconds = clock.elapsed().as_secs_f64();
        let device_seconds = solver.factor_seconds();
        factor_total += factor_seconds;
        if solver.perturbed_pivots() != 0 {
            println!(
                "  note: {} pivot(s) perturbed; these factors are of a slightly \
                 perturbed matrix and should drive refinement, not a bare solve",
                solver.perturbed_pivots()
            );
        }

        if skip_solve {
            println!(
                "{:<14} {:>7.3} {:>9.4} {:>9.4} {:>9} {:>11} {:>11}",
                system.name, system.k0, factor_seconds, device_seconds, "-", "-", "-"
            );
            continue;
        }
        let clock = Instant::now();
        let wide = if force_cpu {
            solver
                .solve_c128(system.rhs_port(0))
                .unwrap_or_else(|e| panic!("solve: {e}"))
        } else {
            solver
                .solve(&to_single(system.rhs_port(0)))
                .unwrap_or_else(|e| panic!("solve: {e}"))
                .iter()
                .map(|v| common::C64::new(v.re as f64, v.im as f64))
                .collect()
        };
        let solve_seconds = clock.elapsed().as_secs_f64();
        solve_total += solve_seconds;

        // Port 0 only: `reference` spans every port, and comparing a
        // single-port solution against the whole block would normalise by the
        // wrong vector and understate the error.
        let against_reference = system
            .reference_port(0)
            .map_or(f64::NAN, |r| relative_difference(&wide, r));
        println!(
            "{:<14} {:>7.3} {:>9.4} {:>9.4} {:>9.4} {:>11.3e} {:>11.3e} {:>7.2} {:>7.2}",
            system.name,
            system.k0,
            factor_seconds,
            device_seconds,
            solve_seconds,
            system.relative_residual(&wide, 0),
            against_reference,
            mdof(device_seconds),
            mdof(factor_seconds + solve_seconds)
        );
    }
    println!(
        "\ntotals   analyze {analyze_seconds:.3}s + factor {factor_total:.3}s + solve {solve_total:.3}s \
         = {:.3}s over {} snapshot(s)",
        analyze_seconds + factor_total + solve_total,
        systems.len()
    );
}
