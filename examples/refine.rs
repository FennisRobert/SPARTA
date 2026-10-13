//! Mixed-precision iterative refinement on top of the single-precision factor.
//!
//!     cargo run --release --example refine -- .local/M3_7.sparta
//!
//! The factorization stays in complex64 on the GPU. Only the residual and the
//! running solution are held in complex128 on the CPU, which costs one sparse
//! matrix-vector product per step against a factor that took far longer to
//! build. Each step's right-hand side is scaled to unit norm before it is
//! narrowed to complex64, so a residual far below f32's range still carries its
//! significant digits into the solve.
mod common;

use common::{norm, C64};
use sparta::{Analysis, Complex32, Solver};

fn main() {
    let steps: usize = 4;
    for path in std::env::args().skip(1) {
        let system = common::System::load(&path).unwrap_or_else(|e| panic!("{e}"));
        let plan = Analysis::ldlt(
            system.n,
            &system.col_ptr,
            &system.row_idx,
            &sparta::Config::default(),
        )
        .unwrap_or_else(|e| panic!("analyze: {e}"));
        let mut solver = Solver::new(plan).unwrap_or_else(|e| panic!("allocate: {e}"));
        let values: Vec<Complex32> = system
            .values
            .iter()
            .map(|v| Complex32::new(v.re as f32, v.im as f32))
            .collect();
        if let Err(e) = solver.factor(&values) {
            println!("{:<10} factorization failed: {e}", system.name);
            continue;
        }
        let bnorm = norm(system.rhs_port(0));
        let mut x = vec![C64::new(0.0, 0.0); system.n];
        let mut residual = system.rhs_port(0).to_vec();
        print!("{:<10}", system.name);
        for _ in 0..=steps {
            let scale = norm(&residual);
            if scale == 0.0 {
                break;
            }
            let rhs: Vec<Complex32> = residual
                .iter()
                .map(|v| Complex32::new((v.re / scale) as f32, (v.im / scale) as f32))
                .collect();
            let step = match solver.solve(&rhs) {
                Ok(s) => s,
                Err(e) => {
                    print!("  solve failed: {e}");
                    break;
                }
            };
            for (xi, d) in x.iter_mut().zip(&step) {
                *xi += C64::new(d.re as f64, d.im as f64) * scale;
            }
            let ax = system.multiply(&x);
            for (r, (a, b)) in residual.iter_mut().zip(ax.iter().zip(system.rhs_port(0))) {
                *r = b - a;
            }
            print!("  {:.3e}", norm(&residual) / bnorm);
        }
        println!();
    }
}
