use scalpel::{order, Options};

/// Builds sorted undirected CSR on `n` vertices from predicate `edge`.
fn graph(n: usize, edge: impl Fn(usize, usize) -> bool) -> (Vec<usize>, Vec<usize>) {
    let mut p = vec![0];
    let mut a = Vec::new();
    for v in 0..n {
        for u in 0..n {
            if edge(v, u) {
                a.push(u);
            }
        }
        p.push(a.len());
    }
    (p, a)
}
/// Checks bijection and determinism across one and four workers for CSR `p`/`a`.
fn check(p: &[usize], a: &[usize]) {
    let o = Options {
        leaf_size: 16,
        parallel_threshold: 32,
        coarsen_threshold: 32,
        ..Options::default()
    };
    let mut previous = None;
    for threads in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let result = pool.install(|| order(p, a, &o).unwrap());
        let mut sorted = result.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..p.len() - 1).collect::<Vec<_>>());
        if let Some(ref prev) = previous {
            assert_eq!(&result, prev);
        }
        previous = Some(result);
    }
}
#[test]
/// Covers empty, isolated, path, clique, hub and disconnected inputs.
fn topology_and_parallel_determinism() {
    for n in [0, 1, 2, 17, 129, 257] {
        for kind in 0..5 {
            let (p, a) = graph(n, |v, u| match kind {
                0 => v == u,
                1 => v.abs_diff(u) == 1,
                2 => true,
                3 => v == 0 || u == 0,
                _ => v / 13 == u / 13 && v.abs_diff(u) == 1,
            });
            check(&p, &a);
        }
    }
}
#[test]
/// Varies sparse non-mesh graphs without relying on external random libraries.
fn random_graphs() {
    for seed in 1..12u64 {
        let (p, a) = graph(173, |v, u| {
            let (x, y) = (v.min(u) as u64, v.max(u) as u64);
            v != u
                && (x.wrapping_mul(6364136223846793005)
                    ^ y.wrapping_mul(1442695040888963407)
                    ^ seed)
                    .count_ones()
                    % 11
                    == 0
        });
        check(&p, &a);
    }
}
#[test]
/// Rejects malformed pointers, invalid indices, duplicates and directed edges.
fn invalid_graphs_return_errors() {
    for (p, a) in [
        (Vec::<usize>::new(), Vec::<usize>::new()),
        (vec![1], vec![]),
        (vec![0, 2, 1], vec![0]),
        (vec![0, 1], vec![1]),
        (vec![0, 2, 2], vec![1, 1]),
        (vec![0, 1, 1], vec![1]),
    ] {
        assert!(order(&p, &a, &Options::default()).is_err());
    }
    assert!(order::<usize>(
        &[0],
        &[],
        &Options {
            leaf_size: 257,
            ..Options::default()
        }
    )
    .is_err());
    assert!(order::<usize>(
        &[0],
        &[],
        &Options {
            imbalance: f64::NAN,
            ..Options::default()
        }
    )
    .is_err());
}
#[test]
/// A single leaf's minimum fill avoids fill on a path regardless of vertex labels.
fn leaf_minimum_degree_preserves_path() {
    let n = 100;
    let (p, a) = graph(n, |v, u| ((v * 37) % n).abs_diff((u * 37) % n) == 1);
    let o = Options {
        leaf_size: 128,
        ..Options::default()
    };
    let perm = order(&p, &a, &o).unwrap();
    let mut alive = vec![true; n];
    for v in perm {
        assert!(a[p[v]..p[v + 1]].iter().filter(|&&u| alive[u]).count() <= 1);
        alive[v] = false;
    }
}

#[test]
/// Compressed, variably weighted mesh blocks preserve all original columns.
fn weighted_twins_and_option_variants() {
    let mut block = Vec::new();
    for b in 0..144 {
        block.extend(std::iter::repeat_n(b, 1 + b % 3));
    }
    let (p, a) = graph(block.len(), |v, u| {
        let (v, u) = (block[v], block[u]);
        v == u || (v / 12 == u / 12 && v.abs_diff(u) == 1) || v.abs_diff(u) == 12
    });
    check(&p, &a);
    for compress in [false, true] {
        for coarsen_threshold in [0, 32, 512] {
            let o = Options {
                compress,
                coarsen_threshold,
                leaf_size: 32,
                parallel_threshold: 32,
                ..Options::default()
            };
            let mut result = order(&p, &a, &o).unwrap();
            result.sort_unstable();
            assert_eq!(result, (0..block.len()).collect::<Vec<_>>());
        }
    }
}

#[test]
/// Full-candidate and one-sided refinement retain deterministic valid permutations.
fn refinement_modes_are_deterministic() {
    let (p, a) = graph(400, |v, u| {
        (v / 20 == u / 20 && v.abs_diff(u) == 1) || v.abs_diff(u) == 20
    });
    for one_sided_refinement in [false, true] {
        let o = Options {
            one_sided_refinement,
            refine_all_candidates: true,
            leaf_size: 32,
            parallel_threshold: 64,
            coarsen_threshold: 64,
            ..Options::default()
        };
        let outputs: Vec<_> = [1, 4]
            .into_iter()
            .map(|threads| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap()
                    .install(|| order(&p, &a, &o).unwrap())
            })
            .collect();
        assert_eq!(outputs[0], outputs[1]);
        let mut sorted = outputs[0].clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..400).collect::<Vec<_>>());
    }
}
