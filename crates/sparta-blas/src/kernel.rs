//! The microkernels: one `MR × NR` tile of `C = alpha·A·B + beta·C` from
//! packed slivers of A and B, with the tile held in registers.
//!
//! `MR = MV · L`: the tile is `MV` vectors tall and `NR` columns wide. Each
//! step of the `k` loop loads one column of the A sliver as `MV` vectors,
//! broadcasts each of the `NR` B values, and does `MV · NR` fused multiply-adds.
//!
//! Complex values use split packing: an A sliver holds, per `k`, the `MR` real
//! parts and then the `MR` imaginary parts, and the tile keeps real and
//! imaginary accumulators apart. A complex multiply-add is then four real
//! FMAs on whole vectors, with no shuffles in the loop; the interleaved layout
//! of C is undone only once per tile, in the epilogue.
use crate::element::Real;
use crate::simd::Simd;

/// `for $i in 0..$n { $body }` written out 16 times, each copy guarded by
/// `$i < $n`. `$n` is a const generic, so the guards fold away and the loop is
/// unrolled whatever the compiler's heuristics would decide: a rolled loop
/// over the accumulators makes it keep them in memory, which costs a store per
/// FMA. Every tile here is at most 16 vectors in either direction.
macro_rules! unroll {
    ($i:ident < $n:ident => $body:block) => {
        unroll!(@ $i $n $body; 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
    };
    (@ $i:ident $n:ident $body:block; $($k:literal)*) => {
        $( { const $i: usize = $k; if $i < $n $body } )*
    };
}

/// Room for the largest tile any kernel uses, in reals: AVX-512 `f32`,
/// 32 × 12 real or 16 × 12 complex.
const MAX_TILE: usize = 512;

/// Where a tile goes: `m × n` values of C at `c`, strides in elements (one
/// complex value is one element).
#[derive(Clone, Copy)]
pub(crate) struct Target<R> {
    pub c: *mut R,
    pub rs: isize,
    pub cs: isize,
    pub m: usize,
    pub n: usize,
}

/// One real tile: `C[0..m, 0..n] = alpha·(A·B) + beta·C`.
///
/// - `kc`: the depth of the product.
/// - `a`: the packed A sliver, `kc` columns of `MR` reals.
/// - `b`: the packed B sliver, `kc` rows of `NR` reals.
/// - `t`: the target; `m ≤ MR` and `n ≤ NR`. With `beta == 0` C is only
///   written, so whatever it held (NaN included) is overwritten.
#[inline(always)]
pub(crate) unsafe fn real<S: Simd, const MV: usize, const NR: usize>(
    kc: usize,
    a: *const S::R,
    b: *const S::R,
    t: Target<S::R>,
    alpha: S::R,
    beta: S::R,
) {
    let l = S::L;
    let mut acc = [[S::zero(); NR]; MV];
    let (mut a, mut b) = (a, b);
    for _ in 0..kc {
        let mut av = [S::zero(); MV];
        unroll!(I < MV => { av[I] = S::load(a.add(I * l)); });
        unroll!(J < NR => {
            let bv = S::splat(*b.add(J));
            unroll!(I < MV => { acc[I][J] = S::fma(av[I], bv, acc[I][J]); });
        });
        a = a.add(MV * l);
        b = b.add(NR);
    }

    // A full tile on a C contiguous down its columns goes straight from the
    // accumulators. Every index into `acc` is a constant of the unrolled
    // loops, so the compiler keeps it in registers; a loop variable would make
    // it keep `acc` in memory and store every accumulator on every step of
    // the loop above. Writing through `tile` instead cost a store and a load
    // per vector, which at the short depths of a front's update (32 to 64)
    // is a large share of the tile. On Zen 4 (AVX-512, one thread) it took
    // `dgemm` at depths 32 to 64 from 0.77-0.88 times OpenBLAS's speed to
    // 0.90-0.99, and `zgemm` on every shape tried from 0.89-1.04 to 0.99-1.07.
    let mr = MV * l;
    let va = S::splat(alpha);
    if t.m == mr && t.n == NR && t.rs == 1 {
        if beta == S::R::ZERO {
            unroll!(J < NR => {
                let col = t.c.offset(J as isize * t.cs);
                unroll!(I < MV => { S::store(col.add(I * l), S::mul(acc[I][J], va)); });
            });
        } else if beta == S::R::ONE {
            unroll!(J < NR => {
                let col = t.c.offset(J as isize * t.cs);
                unroll!(I < MV => {
                    let p = col.add(I * l);
                    S::store(p, S::fma(acc[I][J], va, S::load(p)));
                });
            });
        } else {
            let vb = S::splat(beta);
            unroll!(J < NR => {
                let col = t.c.offset(J as isize * t.cs);
                unroll!(I < MV => {
                    let p = col.add(I * l);
                    S::store(p, S::fma(acc[I][J], va, S::mul(S::load(p), vb)));
                });
            });
        }
        return;
    }
    // Otherwise the accumulators go to memory once, with stores the compiler
    // unrolls fully, and everything after reads them back from `tile`.
    let mut tile = [S::R::ZERO; MAX_TILE];
    unroll!(J < NR => {
        unroll!(I < MV => { S::store(tile.as_mut_ptr().add(J * mr + I * l), acc[I][J]); });
    });
    let tile = tile.as_ptr();
    // A partial tile, or C not contiguous down its columns.
    for j in 0..t.n {
        for i in 0..t.m {
            let p = t.c.offset(i as isize * t.rs + j as isize * t.cs);
            let v = alpha * *tile.add(j * mr + i);
            *p = if beta == S::R::ZERO { v } else { v + beta * *p };
        }
    }
}

/// One complex tile, as [`real`] with complex `alpha` and `beta` given as
/// (re, im) pairs.
///
/// - `a`: per `k`, `MR` real parts then `MR` imaginary parts.
/// - `b`: per `k`, `NR` interleaved (re, im) pairs.
/// - `t.c`: points at reals; the value at row `i`, column `j` starts at real
///   `2·(i·rs + j·cs)`.
#[inline(always)]
pub(crate) unsafe fn complex<S: Simd, const MV: usize, const NR: usize>(
    kc: usize,
    a: *const S::R,
    b: *const S::R,
    t: Target<S::R>,
    alpha: (S::R, S::R),
    beta: (S::R, S::R),
) {
    let l = S::L;
    let mr = MV * l;
    let mut re = [[S::zero(); NR]; MV];
    let mut im = [[S::zero(); NR]; MV];
    let (mut a, mut b) = (a, b);
    for _ in 0..kc {
        let mut ar = [S::zero(); MV];
        let mut ai = [S::zero(); MV];
        unroll!(I < MV => {
            ar[I] = S::load(a.add(I * l));
            ai[I] = S::load(a.add(mr + I * l));
        });
        unroll!(J < NR => {
            let br = S::splat(*b.add(2 * J));
            let bi = S::splat(*b.add(2 * J + 1));
            unroll!(I < MV => {
                re[I][J] = S::fma(ar[I], br, re[I][J]);
                im[I][J] = S::fma(ar[I], bi, im[I][J]);
                re[I][J] = S::fnma(ai[I], bi, re[I][J]);
                im[I][J] = S::fma(ai[I], br, im[I][J]);
            });
        });
        a = a.add(2 * mr);
        b = b.add(2 * NR);
    }

    // As in `real`: a full tile goes straight from the accumulators, with
    // constant indices only.
    let zero = S::R::ZERO;
    let beta_zero = beta.0 == zero && beta.1 == zero;
    if t.m == mr && t.n == NR && t.rs == 1 {
        let (ar, ai) = (S::splat(alpha.0), S::splat(alpha.1));
        let (br, bi) = (S::splat(beta.0), S::splat(beta.1));
        let alpha_real = alpha.1 == zero;
        let beta_one = beta.0 == S::R::ONE && beta.1 == zero;
        unroll!(J < NR => {
            let col = t.c.offset(2 * J as isize * t.cs);
            unroll!(I < MV => {
                let p = col.add(2 * I * l);
                let (r, m) = (re[I][J], im[I][J]);
                let (mut vr, mut vi) = if alpha_real {
                    (S::mul(r, ar), S::mul(m, ar))
                } else {
                    (S::fnma(m, ai, S::mul(r, ar)), S::fma(r, ai, S::mul(m, ar)))
                };
                if !beta_zero {
                    let (cr, ci) = S::load2(p);
                    if beta_one {
                        vr = S::add(vr, cr);
                        vi = S::add(vi, ci);
                    } else {
                        vr = S::fnma(ci, bi, S::fma(cr, br, vr));
                        vi = S::fma(cr, bi, S::fma(ci, br, vi));
                    }
                }
                S::store2(p, vr, vi);
            });
        });
        return;
    }
    // Otherwise through memory, real parts then imaginary parts per column.
    let mut tile = [S::R::ZERO; MAX_TILE];
    unroll!(J < NR => {
        unroll!(I < MV => {
            S::store(tile.as_mut_ptr().add(2 * J * mr + I * l), re[I][J]);
            S::store(tile.as_mut_ptr().add(2 * J * mr + mr + I * l), im[I][J]);
        });
    });
    let tile = tile.as_ptr();
    for j in 0..t.n {
        for i in 0..t.m {
            let p = t.c.offset(2 * (i as isize * t.rs + j as isize * t.cs));
            let (r, m) = (*tile.add(2 * j * mr + i), *tile.add(2 * j * mr + mr + i));
            let (vr, vi) = (alpha.0 * r - alpha.1 * m, alpha.0 * m + alpha.1 * r);
            if beta_zero {
                *p = vr;
                *p.add(1) = vi;
            } else {
                let (cr, ci) = (*p, *p.add(1));
                *p = vr + beta.0 * cr - beta.1 * ci;
                *p.add(1) = vi + beta.0 * ci + beta.1 * cr;
            }
        }
    }
}
