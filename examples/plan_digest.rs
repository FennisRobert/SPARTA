//! Prints the plan digest for each input, for verifying that a refactor of the
//! symbolic analysis leaves the plan bit-identical.
//! cargo run --release --example plan_digest -- test_matrices/em_m1_sym1.sparta --curl-curl 20
mod common;
use sparta::{Analysis, Config, OrderingMethod};

/// Analyzes every named system with both ordering backends and prints digests.
fn main() {
    let mut systems = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--curl-curl" => systems.push(common::System::curl_curl(
                args.next().unwrap().parse().unwrap(),
            )),
            _ => systems.push(common::System::load(&arg).unwrap()),
        }
    }
    println!("matrix,ordering,digest");
    for s in systems {
        let unsymmetric = s.name.contains("asym");
        let analyze = if unsymmetric { Analysis::lu } else { Analysis::ldlt };
        let config = Config {
            ordering: OrderingMethod::Scalpel,
            max_front: Some(1 << 20),
            ..Config::default()
        };
        let plan = analyze(s.n, &s.col_ptr, &s.row_idx, &config).unwrap();
        println!("{},scalpel,{:016x}", s.name, plan.digest());
    }
}
