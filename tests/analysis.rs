use sparta::Analysis;
use sparta::Config;
#[test]
fn rejects_malformed_csc_without_panicking() {
    for (n, p, r) in [
        (2, vec![0, 3, 2], vec![0, 1]),
        (2, vec![0, 1, 2], vec![0, 0]),
        (2, vec![0, 2, 2], vec![1, 1]),
        (2, vec![0, 0, 1], vec![2]),
        (0, vec![0], vec![]),
    ] {
        assert!(Analysis::ldlt(n, &p, &r, &front(128)).is_err());
    }
}
#[test]
fn builds_children_first_forest_and_enforces_front_limit() {
    let n = 32;
    let mut ptr = vec![0];
    let mut rows = Vec::new();
    for j in 0..n {
        rows.push(j);
        if j + 1 < n {
            rows.push(j + 1);
        }
        ptr.push(rows.len());
    }
    let plan = Analysis::ldlt(n, &ptr, &rows, &front(128)).unwrap();
    assert_eq!(plan.dimension(), n);
    assert!(plan.supernodes() > 1);
    assert!(plan.levels() > 1);
    assert!(plan.front_bytes() > 0);
    assert!(plan.factor_bytes() > 0);
    assert!(Analysis::ldlt(n, &ptr, &rows, &front(1)).is_err());
}

#[test]
fn dense_two_by_two_has_a_complete_pivot_panel() {
    // Missing diagonal entries must not prevent a structurally complete panel.
    let p = Analysis::ldlt(2, &[0, 1, 1], &[1], &front(128)).unwrap();
    assert_eq!(p.supernodes(), 1);
    assert_eq!(p.levels(), 1);
}

#[test]
fn reported_286_row_front_is_admitted_to_parallel_path() {
    let n = 286;
    let mut ptr = vec![0];
    let mut rows = Vec::new();
    for col in 0..n {
        rows.extend(col..n);
        ptr.push(rows.len());
    }
    let plan = Analysis::ldlt(n, &ptr, &rows, &front(4096)).unwrap();
    assert_eq!(plan.max_front_size(), n);
    assert_eq!(plan.large_fronts(), 1);
    assert!(Analysis::ldlt(n, &ptr, &rows, &front(256)).is_err());
    // Front height is no longer capped by the analyzer; only an explicit
    // caller limit rejects a front, and zero is not a usable limit.
    assert!(Analysis::ldlt(n, &ptr, &rows, &front(4097)).is_ok());
    assert!(Analysis::ldlt(n, &ptr, &rows, &front(usize::MAX)).is_ok());
    assert!(Analysis::ldlt(n, &ptr, &rows, &front(0)).is_err());
}

#[test]
fn one_dense_front_is_stored_once_in_workspace_and_factor_store() {
    // A single dense front has nothing to reuse, so both regions are exact.
    let n = 300;
    let mut ptr = vec![0];
    let mut rows = Vec::new();
    for col in 0..n {
        rows.extend(col..n);
        ptr.push(rows.len());
    }
    let plan = Analysis::ldlt(n, &ptr, &rows, &front(usize::MAX)).unwrap();
    assert_eq!(plan.supernodes(), 1);
    assert_eq!(plan.front_bytes(), n * n * 8);
    assert_eq!(plan.factor_bytes(), n * n * 8);
}

/// A default configuration with the front limit set to `rows`.
fn front(rows: usize) -> Config {
    Config {
        max_front: (rows != usize::MAX).then_some(rows),
        ..Config::default()
    }
}
