//! A/B ordering latency and exact symbolic fill; no GPU required.
//! cargo run --release --example ordering_bench -- --curl-curl 16 --threads 1,4 --repeat 3
//! cargo run --release --example ordering_bench -- test_matrices/em_m1_sym1.sparta
mod common;
use sparta::symbolic::{
    column_counts::column_counts_gnp, elimination_tree::EliminationTree, permute_pattern,
    symmetric_pattern,
};
use sparta::scalpel;
use std::time::Instant;

/// Parses workloads, thread counts and repetitions, then prints median timing and fill.
fn main() {
    let mut args = std::env::args().skip(1);
    let mut systems = Vec::new();
    let mut threads = vec![1, 4];
    let mut repeat = 3;
    let mut options = scalpel::Options::default();
    let mut geo = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--curl-curl" => systems.push(common::System::curl_curl(
                args.next().unwrap().parse().unwrap(),
            )),
            "--threads" => {
                threads = args
                    .next()
                    .unwrap()
                    .split(',')
                    .map(|s| s.parse().unwrap())
                    .collect()
            }
            "--repeat" => repeat = args.next().unwrap().parse().unwrap(),
            "--coarsen" => options.coarsen_threshold = args.next().unwrap().parse().unwrap(),
            "--half-levels" => options.half_levels = true,
            "--wide" => options.wide_cycles = args.next().unwrap().parse().unwrap(),
            "--balance-quality" => options.balance_quality = true,
            "--imbalance" => options.imbalance = args.next().unwrap().parse().unwrap(),
            "--one-sided" => options.one_sided_refinement = true,
            "--full-trials" => options.refine_all_candidates = true,
            "--fm" => options.fm_passes = args.next().unwrap().parse().unwrap(),
            "--trials" => options.separator_trials = args.next().unwrap().parse().unwrap(),
            "--geo" => geo = true,
            "--spectral" => options.spectral_candidates = args.next().unwrap().parse().unwrap(),
            "--fine-selection" => options.fine_selection = true,
            "--leaf" => options.leaf_size = args.next().unwrap().parse().unwrap(),
            _ => systems.push(common::System::load(&arg).unwrap()),
        }
    }
    assert!(repeat > 0 && threads.iter().all(|&n| n > 0));
    if systems.is_empty() {
        systems.push(common::System::curl_curl(12));
    }
    println!("matrix,backend,threads,n,median_ms,nnz_L,work_proxy,perm_hash");
    for s in systems {
        // Union A and A^T, including unsymmetric systems, without squaring A.
        let mut cols = vec![Vec::new(); s.n];
        for j in 0..s.n {
            for &i in &s.row_idx[s.col_ptr[j]..s.col_ptr[j + 1]] {
                cols[i.min(j)].push(i.max(j));
            }
        }
        let mut ptr = vec![0];
        let mut rows = Vec::new();
        for col in &mut cols {
            col.sort_unstable();
            col.dedup();
            rows.extend_from_slice(col);
            ptr.push(rows.len());
        }
        let full = symmetric_pattern(s.n, &ptr, &rows);
        if geo && s.name.starts_with("curl_curl_") {
            let k = ((s.n / 3) as f64).cbrt().round() as usize;
            let nodes = k * k * k;
            let at = |d: usize, axis: usize| (d % nodes / [1, k, k * k][axis]) % k;
            fn nd(dofs: Vec<usize>, at: &dyn Fn(usize, usize) -> usize, nodes: usize, out: &mut Vec<usize>, leaf: usize) {
                if dofs.len() <= leaf {
                    out.extend(dofs);
                    return;
                }
                let (mut axis, mut span) = (0, 0);
                for a in 0..3 {
                    let lo = dofs.iter().map(|&d| at(d, a)).min().unwrap();
                    let hi = dofs.iter().map(|&d| at(d, a)).max().unwrap();
                    if hi - lo > span { span = hi - lo; axis = a; }
                }
                if span < 2 { out.extend(dofs); return; }
                let lo = dofs.iter().map(|&d| at(d, axis)).min().unwrap();
                let m = lo + span / 2;
                let (mut l, mut r, mut sep) = (vec![], vec![], vec![]);
                for d in dofs {
                    let c = at(d, axis);
                    if c == m || (d / nodes == axis && c == m + 1) { sep.push(d) } else if c < m { l.push(d) } else { r.push(d) }
                }
                nd(l, at, nodes, out, leaf);
                nd(r, at, nodes, out, leaf);
                out.extend(sep);
            }
            let leaf: usize = std::env::var("GEOLEAF").ok().map(|v| v.parse().unwrap()).unwrap_or(24);
            let mut perm = Vec::new();
            nd((0..s.n).collect(), &at, nodes, &mut perm, leaf);
            let p = permute_pattern(&full, &perm);
            let e = EliminationTree::from_pattern(&p);
            let counts = column_counts_gnp(&p, &e);
            let nnz: usize = counts.iter().sum();
            let work: u128 = counts.iter().map(|&n| (n as u128) * (n as u128)).sum();
            println!("{},GEOMETRIC,0,{},0,{},{}", s.name, s.n, nnz, work);
        }
        for count in threads.iter().copied() {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(count.max(1))
                .build()
                .unwrap();
            let run = || pool.install(|| scalpel::order(&full.col_ptr, &full.row_idx, &options).unwrap());
            let mut perm = run(); // warm allocator, workers and code before timed samples
            let mut samples = Vec::new();
            for _ in 0..repeat {
                let t = Instant::now();
                perm = run();
                samples.push(t.elapsed().as_secs_f64() * 1000.0);
            }
            samples.sort_by(f64::total_cmp);
            let p = permute_pattern(&full, &perm);
            let e = EliminationTree::from_pattern(&p);
            let counts = column_counts_gnp(&p, &e);
            let nnz: usize = counts.iter().sum();
            let work: u128 = counts.iter().map(|&n| (n as u128) * (n as u128)).sum();
            // FNV-1a over the permutation, to check that a change keeps the output identical.
            let hash = perm.iter().fold(0xcbf29ce484222325u64, |h, &v| {
                (h ^ v as u64).wrapping_mul(0x100000001b3)
            });
            println!(
                "{},SCALPEL,{},{},{:.3},{},{},{:016x}",
                s.name,
                count.max(1),
                s.n,
                samples[repeat / 2],
                nnz,
                work,
                hash
            );
        }
    }
}
