//! `Complex<f64>` GEMM tile on the SME matrix unit.
//!
//! One tile is 16 × 16 complex values of C. Real and imaginary parts are kept
//! apart, in A, in B and in ZA, so a complex multiply-add is four real outer
//! products and nothing else:
//!
//! ```text
//! re += br ⊗ ar − bi ⊗ ai        im += br ⊗ ai + bi ⊗ ar
//! ```
//!
//! ZA holds 2 × 2 blocks of 8 × 8, real and imaginary: `za{4h + 2q}` is the
//! real part of columns `8h ..`, rows `8q ..`, and `za{4h + 2q + 1}` its
//! imaginary part. As in `d.rs` the tiles are stored transposed, so a
//! horizontal slice is one column of C.
//!
//! Packed layouts (see `pack.rs`): per depth, 16 real parts then 16
//! imaginary parts, for A and for B. One step of the depth loop is two
//! four-register loads and sixteen outer products.
use std::arch::asm;

pub(crate) const MR: usize = 16;
pub(crate) const NR: usize = 16;

/// The depth loop, adding (`fmopa`) or subtracting (`fmops`) the product. Its
/// imaginary-part `bi ⊗ ai` term takes the opposite operation, `$neg`.
macro_rules! depth {
    ($add:literal, $neg:literal) => {
        concat!(
            "ld1d {{z0.d-z3.d}}, pn8/z, [x0]\n",
            "ld1d {{z4.d-z7.d}}, pn8/z, [x1]\n",
            "add x0, x0, #256\n",
            "add x1, x1, #256\n",
            // h = 0 (B columns 0-7: br z4, bi z6), q = 0 (A rows 0-7: ar z0, ai z2)
            $add, " za0.d, p0/m, p0/m, z4.d, z0.d\n",
            $neg, " za0.d, p0/m, p0/m, z6.d, z2.d\n",
            $add, " za1.d, p0/m, p0/m, z4.d, z2.d\n",
            $add, " za1.d, p0/m, p0/m, z6.d, z0.d\n",
            // h = 0, q = 1 (ar z1, ai z3)
            $add, " za2.d, p0/m, p0/m, z4.d, z1.d\n",
            $neg, " za2.d, p0/m, p0/m, z6.d, z3.d\n",
            $add, " za3.d, p0/m, p0/m, z4.d, z3.d\n",
            $add, " za3.d, p0/m, p0/m, z6.d, z1.d\n",
            // h = 1 (br z5, bi z7), q = 0
            $add, " za4.d, p0/m, p0/m, z5.d, z0.d\n",
            $neg, " za4.d, p0/m, p0/m, z7.d, z2.d\n",
            $add, " za5.d, p0/m, p0/m, z5.d, z2.d\n",
            $add, " za5.d, p0/m, p0/m, z7.d, z0.d\n",
            // h = 1, q = 1
            $add, " za6.d, p0/m, p0/m, z5.d, z1.d\n",
            $neg, " za6.d, p0/m, p0/m, z7.d, z3.d\n",
            $add, " za7.d, p0/m, p0/m, z5.d, z3.d\n",
            $add, " za7.d, p0/m, p0/m, z7.d, z1.d\n",
            "subs x3, x3, #1\n",
        )
    };
}

/// Row masks p1 (rows 0-7) and p2 (rows 8-15) from x5 = rows that exist;
/// x20 = 16, the offset in doubles of rows 8.. within a column.
macro_rules! masks {
    () => {
        concat!(
            "mov x10, #0\n",
            "mov x11, #8\n",
            "whilelt p1.d, x10, x5\n",
            "whilelt p2.d, x11, x5\n",
            "mov x20, #16\n",
        )
    };
}

/// For each column j < n (x6), with x13 at the column and w12 its slice in
/// half h: run `$body` with `$t0..$t3` naming the half's four tiles.
macro_rules! each_column {
    ($body:ident, $l0:literal, $l1:literal, $l2:literal) => {
        concat!(
            "mov x13, x2\n",
            "cmp x6, #8\n",
            "mov x15, #8\n",
            "csel x15, x6, x15, lo\n",
            "mov w12, #0\n",
            $l0, ":\n",
            $body!("za0h", "za1h", "za2h", "za3h"),
            "add x13, x13, x4\n",
            "add w12, w12, #1\n",
            "cmp x12, x15\n",
            "b.lo ", $l0, "b\n",
            "subs x15, x6, #8\n",
            "b.le ", $l2, "f\n",
            "mov w12, #0\n",
            $l1, ":\n",
            $body!("za4h", "za5h", "za6h", "za7h"),
            "add x13, x13, x4\n",
            "add w12, w12, #1\n",
            "cmp x12, x15\n",
            "b.lo ", $l1, "b\n",
            $l2, ":\n",
        )
    };
}

/// C's column into ZA: interleaved loads split re/im, then vector-to-slice moves.
macro_rules! load_column {
    ($t0:literal, $t1:literal, $t2:literal, $t3:literal) => {
        concat!(
            "ld2d {{z8.d, z9.d}}, p1/z, [x13]\n",
            "ld2d {{z10.d, z11.d}}, p2/z, [x13, x20, lsl #3]\n",
            "mova ", $t0, ".d[w12, 0], p0/m, z8.d\n",
            "mova ", $t1, ".d[w12, 0], p0/m, z9.d\n",
            "mova ", $t2, ".d[w12, 0], p0/m, z10.d\n",
            "mova ", $t3, ".d[w12, 0], p0/m, z11.d\n",
        )
    };
}

/// ZA's column back to C, interleaving re/im on the store.
macro_rules! store_column {
    ($t0:literal, $t1:literal, $t2:literal, $t3:literal) => {
        concat!(
            "mova z8.d, p0/m, ", $t0, ".d[w12, 0]\n",
            "mova z9.d, p0/m, ", $t1, ".d[w12, 0]\n",
            "mova z10.d, p0/m, ", $t2, ".d[w12, 0]\n",
            "mova z11.d, p0/m, ", $t3, ".d[w12, 0]\n",
            "st2d {{z8.d, z9.d}}, p1, [x13]\n",
            "st2d {{z10.d, z11.d}}, p2, [x13, x20, lsl #3]\n",
        )
    };
}

/// General alpha and beta: `C = alpha·acc + beta·C` with z24/z25 = alpha
/// (re, im), z26/z27 = beta, x9 = 0 when beta is zero.
macro_rules! scale_column {
    ($t0:literal, $t1:literal, $t2:literal, $t3:literal) => {
        concat!(
            "mova z8.d, p0/m, ", $t0, ".d[w12, 0]\n",
            "mova z9.d, p0/m, ", $t1, ".d[w12, 0]\n",
            "mova z10.d, p0/m, ", $t2, ".d[w12, 0]\n",
            "mova z11.d, p0/m, ", $t3, ".d[w12, 0]\n",
            // alpha · (z8 + i z9) -> z12, z13 ; alpha · (z10 + i z11) -> z14, z15
            "fmul z12.d, z8.d, z24.d\n",
            "fmls z12.d, p0/m, z9.d, z25.d\n",
            "fmul z13.d, z9.d, z24.d\n",
            "fmla z13.d, p0/m, z8.d, z25.d\n",
            "fmul z14.d, z10.d, z24.d\n",
            "fmls z14.d, p0/m, z11.d, z25.d\n",
            "fmul z15.d, z11.d, z24.d\n",
            "fmla z15.d, p0/m, z10.d, z25.d\n",
            "cbz x9, 5f\n",
            "ld2d {{z16.d, z17.d}}, p1/z, [x13]\n",
            "ld2d {{z18.d, z19.d}}, p2/z, [x13, x20, lsl #3]\n",
            "fmla z12.d, p0/m, z16.d, z26.d\n",
            "fmls z12.d, p0/m, z17.d, z27.d\n",
            "fmla z13.d, p0/m, z17.d, z26.d\n",
            "fmla z13.d, p0/m, z16.d, z27.d\n",
            "fmla z14.d, p0/m, z18.d, z26.d\n",
            "fmls z14.d, p0/m, z19.d, z27.d\n",
            "fmla z15.d, p0/m, z19.d, z26.d\n",
            "fmla z15.d, p0/m, z18.d, z27.d\n",
            "5:\n",
            "st2d {{z12.d, z13.d}}, p1, [x13]\n",
            "st2d {{z14.d, z15.d}}, p2, [x13, x20, lsl #3]\n",
        )
    };
}

/// `C[0..m, 0..n] = alpha·(A·B) + beta·C` for one 16 × 16 complex tile.
///
/// - `kc`: depth, at least 1.
/// - `ap`, `bp`: split-packed slivers (per depth, 16 re then 16 im).
/// - `c`: the tile's first value, as reals (re, im interleaved); C is
///   contiguous down its columns and `cs` is the column stride in complex
///   elements.
/// - `m, n ≤ 16`: the part that exists.
///
/// # Safety
/// As `d::tile`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn tile(kc: usize, ap: *const f64, bp: *const f64, c: *mut f64, cs: isize, m: usize, n: usize, alpha: (f64, f64), beta: (f64, f64)) {
    let beta_zero = beta == (0.0, 0.0);
    asm!(
        ".arch_extension sme",
        ".arch_extension sme2",
        ".arch_extension sme-f64f64",
        "smstart",
        "ptrue p0.d",
        "ptrue pn8.d",
        "zero {{za}}",
        "1:",
        depth!("fmopa", "fmops"),
        "b.ne 1b",
        masks!(),
        "dup z24.d, x7",
        "dup z25.d, x8",
        "dup z26.d, x14",
        "dup z27.d, x16",
        each_column!(scale_column, "7", "8", "9"),
        "smstop",
        inout("x0") ap => _,
        inout("x1") bp => _,
        inout("x2") c => _,
        inout("x3") kc => _,
        inout("x4") cs * 16 => _,
        inout("x5") m => _,
        inout("x6") n => _,
        inout("x7") alpha.0.to_bits() => _,
        inout("x8") alpha.1.to_bits() => _,
        inout("x9") (!beta_zero) as u64 => _,
        inout("x14") beta.0.to_bits() => _,
        inout("x16") beta.1.to_bits() => _,
        out("x10") _,
        out("x11") _,
        out("x12") _,
        out("x13") _,
        out("x15") _,
        out("x20") _,
        out("v8") _,
        out("v9") _,
        out("v10") _,
        out("v11") _,
        out("v12") _,
        out("v13") _,
        out("v14") _,
        out("v15") _,
        clobber_abi("C"),
    );
}

/// [`tile`] for `alpha = ±1` and `beta ∈ {0, 1}`: C goes into ZA and back
/// with no arithmetic of its own, as `d::tile_unit`.
///
/// # Safety
/// As [`tile`].
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn tile_unit(kc: usize, ap: *const f64, bp: *const f64, c: *mut f64, cs: isize, m: usize, n: usize, subtract: bool, accumulate: bool) {
    asm!(
        ".arch_extension sme",
        ".arch_extension sme2",
        ".arch_extension sme-f64f64",
        "smstart",
        "ptrue p0.d",
        "ptrue pn8.d",
        masks!(),
        "zero {{za}}",
        "cbz x9, 20f",
        each_column!(load_column, "11", "12", "13"),
        "20:",
        "cbnz x7, 30f",
        "1:",
        depth!("fmopa", "fmops"),
        "b.ne 1b",
        "b 40f",
        "30:",
        depth!("fmops", "fmopa"),
        "b.ne 30b",
        "40:",
        each_column!(store_column, "14", "15", "16"),
        "smstop",
        inout("x0") ap => _,
        inout("x1") bp => _,
        inout("x2") c => _,
        inout("x3") kc => _,
        inout("x4") cs * 16 => _,
        inout("x5") m => _,
        inout("x6") n => _,
        inout("x7") subtract as u64 => _,
        inout("x9") accumulate as u64 => _,
        out("x10") _,
        out("x11") _,
        out("x12") _,
        out("x13") _,
        out("x15") _,
        out("x20") _,
        out("v8") _,
        out("v9") _,
        out("v10") _,
        out("v11") _,
        out("v12") _,
        out("v13") _,
        out("v14") _,
        out("v15") _,
        clobber_abi("C"),
    );
}
