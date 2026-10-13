use sparta::Config;
use sparta::{Analysis, Complex32, OrderingMethod, Solver};

#[test]
/// SCALPEL produces usable LDLT/LU plans and accurate GPU solutions.
fn scalpel_factors_and_solves() {
    // A 2D mesh forces recursion rather than exercising only leaf ordering.
    let k = 20;
    let n = k * k;
    for method in [OrderingMethod::Scalpel] {
        for lu in [false, true] {
            let mut ptr = vec![0];
            let mut rows = Vec::new();
            let mut values = Vec::new();
            let mut rhs = vec![Complex32::new(0.0, 0.0); n];
            for j in 0..n {
                for i in 0..n {
                    let edge = (i / k == j / k && i.abs_diff(j) == 1) || i.abs_diff(j) == k;
                    if (i == j || edge) && (lu || i >= j) {
                        let v = if i == j {
                            Complex32::new(6.0, 0.2)
                        } else {
                            Complex32::new(-1.0, if lu && i < j { 0.1 } else { 0.0 })
                        };
                        rows.push(i);
                        values.push(v);
                        rhs[i] += v;
                        if !lu && i != j {
                            rhs[j] += v;
                        }
                    }
                }
                ptr.push(rows.len());
            }
            let build = if lu { Analysis::lu } else { Analysis::ldlt };
            let config = Config {
                ordering: method,
                ..Config::default()
            };
            let plan = build(n, &ptr, &rows, &config).unwrap();
            assert_eq!(plan.dimension(), n);
            // A one-row front limit cannot hold any real front.
            let tight = Config {
                max_front: Some(1),
                ..config.clone()
            };
            assert!(build(n, &ptr, &rows, &tight).is_err());
            if sparta::gpu_available() {
                let mut solver = Solver::new(plan).unwrap();
                solver.factor(&values).unwrap();
                let x = solver.solve(&rhs).unwrap();
                assert!(x
                    .iter()
                    .all(|v| (*v - Complex32::new(1.0, 0.0)).norm() < 2e-4));
            }
        }
    }
}

#[test]
/// SCALPEL is the default: explicit SCALPEL is byte-identical to the default
/// analysis path.
fn default_is_scalpel() {
    let ptr = [0, 2, 4, 5];
    let rows = [0, 1, 1, 2, 2];
    assert_eq!(Config::default().ordering, OrderingMethod::Scalpel);
    let a = Analysis::ldlt(3, &ptr, &rows, &front(99)).unwrap();
    let b = Analysis::ldlt(
        3,
        &ptr,
        &rows,
        &Config {
            ordering: OrderingMethod::Scalpel,
            ..front(99)
        },
    )
    .unwrap();
    assert_eq!(a.digest(), b.digest());
    assert!(Analysis::ldlt(
        2,
        &[0, 1, 1],
        &[2],
        &Config {
            ordering: OrderingMethod::Scalpel,
            ..front(99)
        }
    )
    .is_err());
}

/// A default configuration with the front limit set to `rows`.
fn front(rows: usize) -> Config {
    Config {
        max_front: Some(rows),
        ..Config::default()
    }
}
