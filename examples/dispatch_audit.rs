//! Audits how much of the factorization's dispatched work is useful.
//!
//!     cargo run --release --example dispatch_audit -- test_matrices/em_m1_sym1.sparta
//!
//! Strip tiles are scheduled as a column-ordered suffix, so every group that is
//! dispatched can write; the example asserts that. Schur tiles are not: the
//! mandatory end-of-level drain covers every tile of the level whether or not it
//! owes anything, so its dispatched and writing counts differ by design.
//!
//! Counts follow the plan's `schur_batch`, the drain cadence the bridge
//! dispatches at. The plan is built from `Config::default()`, which reads
//! `SPARTA_SCHUR_BATCH`, so set that variable to see what a given cadence
//! actually launches.
#[cfg(any(sparta_metal, sparta_vulkan))]
mod common;

#[cfg(any(sparta_metal, sparta_vulkan))]
use common::System;
#[cfg(any(sparta_metal, sparta_vulkan))]
use sparta::{Analysis, Kind};

/// The audit counts GPU dispatches, which only a build with a GPU backend plans.
#[cfg(not(any(sparta_metal, sparta_vulkan)))]
fn main() {
    eprintln!("dispatch_audit needs a GPU backend: build on Apple Silicon or with --features vulkan");
    std::process::exit(2);
}

#[cfg(any(sparta_metal, sparta_vulkan))]
fn main() {
    let mut systems: Vec<System> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--curl-curl" => systems.push(System::curl_curl(
                args.next()
                    .expect("--curl-curl needs a grid size")
                    .parse()
                    .unwrap(),
            )),
            path => systems.push(System::load(path).unwrap_or_else(|e| panic!("{e}"))),
        }
    }
    for system in &systems {
        // The container's storage flag picks the factorization, so an
        // unsymmetric file audits the LU schedule: the same levels and panels
        // over close to twice the tiles.
        let analyze = if system.full {
            Analysis::lu
        } else {
            Analysis::ldlt
        };
        let config = sparta::Config::default();
        let plan = analyze(system.n, &system.col_ptr, &system.row_idx, &config).unwrap();
        let (children, cleared, extended) = plan.assembly_profile();
        println!("\nassembly");
        let listed: Vec<String> = children
            .iter()
            .enumerate()
            .filter(|(_, fronts)| **fronts > 0)
            .map(|(count, fronts)| format!("{count}:{fronts}"))
            .collect();
        println!("  children per front   {}", listed.join("  "));
        println!(
            "  front clear          {cleared:>14} entries  {:.2} GB written",
            cleared as f64 * 8.0 / 1e9
        );
        println!(
            "  child extend-add     {extended:>14} entries  {:.2} GB moved (24 B/entry)",
            extended as f64 * 24.0 / 1e9
        );
        let (wide, inside, below, rows_below) = plan.panel_profile();
        let total = inside + below;
        println!("\npanel factorization");
        println!("  wide fronts          {wide}");
        if total > 0 {
            println!(
                "  work in fully summed rows  {inside:>14}  {:.1}%",
                100.0 * inside as f64 / total as f64
            );
            println!(
                "  work in rows below it      {below:>14}  {:.1}%   over {rows_below} rows",
                100.0 * below as f64 / total as f64
            );
        }
        let audit = plan.dispatch_audit();
        println!(
            "{} (n={}, {})",
            system.name,
            system.n,
            match plan.kind() {
                Kind::Ldlt => "LDL^T",
                Kind::Lu => "LU",
            }
        );
        println!(
            "  supernodes={}  levels={}  small={}  large={}",
            plan.supernodes(),
            plan.levels(),
            plan.supernodes() - plan.large_fronts(),
            plan.large_fronts()
        );
        println!(
            "  dispatches={}  command buffers={}  panel threadgroups={}",
            audit.dispatches, audit.command_buffers, audit.panel_groups
        );
        println!(
            "  update_strip groups dispatched={:<12} can write={:<12} ({:.1}% launch-only)",
            audit.strip_groups,
            audit.useful_strip_groups,
            100.0 * (1.0 - audit.useful_strip_groups as f64 / audit.strip_groups as f64)
        );
        assert_eq!(
            audit.strip_groups, audit.useful_strip_groups,
            "strip schedule dispatched groups that cannot write"
        );
        println!(
            "  update_schur groups dispatched={:<12} with pending work={:<12} \
             (schur_batch={})",
            audit.schur_groups,
            audit.writing_schur_groups,
            match config.schur_batch {
                0 => "0 (final-only)".to_string(),
                batch => batch.to_string(),
            }
        );
        println!(
            "  small fronts={} (worst {} rows on one thread)  fixed serial ops={:.1}M  \
             trailing-update work={:.1}M complex FMA",
            audit.small_fronts,
            audit.worst_small_front,
            audit.small_fixed_ops as f64 / 1e6,
            audit.update_flops as f64 / 1e6
        );
        let total = (audit.pivot_block_flops + audit.schur_flops) as f64;
        println!(
            "  blocking split: fully-summed block {:.1}M FMA ({:.1}%) + \
             Schur complement {:.1}M FMA ({:.1}%)",
            audit.pivot_block_flops as f64 / 1e6,
            100.0 * audit.pivot_block_flops as f64 / total,
            audit.schur_flops as f64 / 1e6,
            100.0 * audit.schur_flops as f64 / total
        );
        println!("    panel width={}", plan.panel_width());
        // A front whose fully summed block is narrow could factor every one of
        // its pivots inside a single threadgroup, with no dispatch per pivot.
        println!("  fully-summed width  fronts   pivot steps   pivot-block work");
        let mut bounds = vec![(0usize, 16usize), (16, 32), (32, 64), (64, usize::MAX)];
        bounds.reverse();
        for (lo, hi) in bounds.into_iter().rev() {
            let (mut fronts, mut steps, mut work) = (0usize, 0usize, 0usize);
            for d in plan.large_nodes() {
                let (nr, nc) = (d.0, d.1);
                if nc > lo && nc <= hi {
                    fronts += 1;
                    steps += nc;
                    // Same region the audit's pivot_block_flops counts: the
                    // lower strip for LDL^T, the whole cross for LU.
                    work += (0..nc)
                        .map(|k| match plan.kind() {
                            Kind::Ldlt => ((k + 1)..nc).map(|j| nr - j).sum::<usize>(),
                            Kind::Lu => (nc - k - 1) * (nr - k - 1) + (nc - k - 1) * (nr - nc),
                        })
                        .sum::<usize>();
                }
            }
            let label = if hi == usize::MAX {
                format!("nc > {lo}")
            } else {
                format!("{lo} < nc <= {hi}")
            };
            println!(
                "  {label:<18} {fronts:<8} {steps:<13} {:.1}M FMA",
                work as f64 / 1e6
            );
        }
    }
}
