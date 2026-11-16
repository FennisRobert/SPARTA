//! `Complex<f32>` GEMM tile on the SME matrix unit.
//!
//! One tile is `MR = 32` by `NR = 16` complex values of C. Real and imaginary
//! parts are kept apart as in `z.rs`, in the four 32-bit ZA tiles: `za{2q}`
//! is the real part of rows `16q ..`, `za{2q + 1}` its imaginary part, all 16
//! columns, stored transposed (a horizontal slice is one column of C). Each
//! depth step is two loads (four A vectors, two B vectors) and eight outer
//! products.
use std::arch::asm;

pub(crate) const MR: usize = 32;
pub(crate) const NR: usize = 16;

macro_rules! depth {
    ($add:literal, $neg:literal) => {
        concat!(
            "ld1w {{z0.s-z3.s}}, pn8/z, [x0]\n",
            "ld1w {{z4.s-z5.s}}, pn8/z, [x1]\n",
            "add x0, x0, #256\n",
            "add x1, x1, #128\n",
            // rows 0-15: ar z0, ai z2 ; B: br z4, bi z5
            $add, " za0.s, p0/m, p0/m, z4.s, z0.s\n",
            $neg, " za0.s, p0/m, p0/m, z5.s, z2.s\n",
            $add, " za1.s, p0/m, p0/m, z4.s, z2.s\n",
            $add, " za1.s, p0/m, p0/m, z5.s, z0.s\n",
            // rows 16-31: ar z1, ai z3
            $add, " za2.s, p0/m, p0/m, z4.s, z1.s\n",
            $neg, " za2.s, p0/m, p0/m, z5.s, z3.s\n",
            $add, " za3.s, p0/m, p0/m, z4.s, z3.s\n",
            $add, " za3.s, p0/m, p0/m, z5.s, z1.s\n",
            "subs x3, x3, #1\n",
        )
    };
}

/// Row masks p1 (rows 0-15) and p2 (16-31) of x5; x20 = 32, the offset in
/// floats of row 16 within a column.
macro_rules! masks {
    () => {
        concat!(
            "mov x10, #0\n",
            "mov x11, #16\n",
            "whilelt p1.s, x10, x5\n",
            "whilelt p2.s, x11, x5\n",
            "mov x20, #32\n",
        )
    };
}

/// For each column j < n (x6, at most 16): x13 at it, w12 its slice.
macro_rules! each_column {
    ($body:ident, $l0:literal) => {
        concat!(
            "mov x13, x2\n",
            "mov w12, #0\n",
            $l0, ":\n",
            $body!(),
            "add x13, x13, x4\n",
            "add w12, w12, #1\n",
            "cmp x12, x6\n",
            "b.lo ", $l0, "b\n",
        )
    };
}

macro_rules! load_column {
    () => {
        concat!(
            "ld2w {{z8.s, z9.s}}, p1/z, [x13]\n",
            "ld2w {{z10.s, z11.s}}, p2/z, [x13, x20, lsl #2]\n",
            "mova za0h.s[w12, 0], p0/m, z8.s\n",
            "mova za1h.s[w12, 0], p0/m, z9.s\n",
            "mova za2h.s[w12, 0], p0/m, z10.s\n",
            "mova za3h.s[w12, 0], p0/m, z11.s\n",
        )
    };
}

macro_rules! store_column {
    () => {
        concat!(
            "mova z8.s, p0/m, za0h.s[w12, 0]\n",
            "mova z9.s, p0/m, za1h.s[w12, 0]\n",
            "mova z10.s, p0/m, za2h.s[w12, 0]\n",
            "mova z11.s, p0/m, za3h.s[w12, 0]\n",
            "st2w {{z8.s, z9.s}}, p1, [x13]\n",
            "st2w {{z10.s, z11.s}}, p2, [x13, x20, lsl #2]\n",
        )
    };
}

/// General alpha (z24, z25) and beta (z26, z27), x9 = 0 when beta is zero.
macro_rules! scale_column {
    () => {
        concat!(
            "mova z8.s, p0/m, za0h.s[w12, 0]\n",
            "mova z9.s, p0/m, za1h.s[w12, 0]\n",
            "mova z10.s, p0/m, za2h.s[w12, 0]\n",
            "mova z11.s, p0/m, za3h.s[w12, 0]\n",
            "fmul z12.s, z8.s, z24.s\n",
            "fmls z12.s, p0/m, z9.s, z25.s\n",
            "fmul z13.s, z9.s, z24.s\n",
            "fmla z13.s, p0/m, z8.s, z25.s\n",
            "fmul z14.s, z10.s, z24.s\n",
            "fmls z14.s, p0/m, z11.s, z25.s\n",
            "fmul z15.s, z11.s, z24.s\n",
            "fmla z15.s, p0/m, z10.s, z25.s\n",
            "cbz x9, 5f\n",
            "ld2w {{z16.s, z17.s}}, p1/z, [x13]\n",
            "ld2w {{z18.s, z19.s}}, p2/z, [x13, x20, lsl #2]\n",
            "fmla z12.s, p0/m, z16.s, z26.s\n",
            "fmls z12.s, p0/m, z17.s, z27.s\n",
            "fmla z13.s, p0/m, z17.s, z26.s\n",
            "fmla z13.s, p0/m, z16.s, z27.s\n",
            "fmla z14.s, p0/m, z18.s, z26.s\n",
            "fmls z14.s, p0/m, z19.s, z27.s\n",
            "fmla z15.s, p0/m, z19.s, z26.s\n",
            "fmla z15.s, p0/m, z18.s, z27.s\n",
            "5:\n",
            "st2w {{z12.s, z13.s}}, p1, [x13]\n",
            "st2w {{z14.s, z15.s}}, p2, [x13, x20, lsl #2]\n",
        )
    };
}

/// `C[0..m, 0..n] = alpha·(A·B) + beta·C` for one 32 × 16 complex tile;
/// arguments as `z::tile` (`c` as reals, `cs` in complex elements).
///
/// # Safety
/// As `d::tile`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn tile(kc: usize, ap: *const f32, bp: *const f32, c: *mut f32, cs: isize, m: usize, n: usize, alpha: (f32, f32), beta: (f32, f32)) {
    let beta_zero = beta == (0.0, 0.0);
    asm!(
        ".arch_extension sme",
        ".arch_extension sme2",
        "smstart",
        "ptrue p0.s",
        "ptrue pn8.s",
        "zero {{za}}",
        "1:",
        depth!("fmopa", "fmops"),
        "b.ne 1b",
        masks!(),
        "dup z24.s, w7",
        "dup z25.s, w8",
        "dup z26.s, w14",
        "dup z27.s, w16",
        each_column!(scale_column, "7"),
        "smstop",
        inout("x0") ap => _,
        inout("x1") bp => _,
        inout("x2") c => _,
        inout("x3") kc => _,
        inout("x4") cs * 8 => _,
        inout("x5") m => _,
        inout("x6") n => _,
        inout("x7") alpha.0.to_bits() as u64 => _,
        inout("x8") alpha.1.to_bits() as u64 => _,
        inout("x9") (!beta_zero) as u64 => _,
        inout("x14") beta.0.to_bits() as u64 => _,
        inout("x16") beta.1.to_bits() as u64 => _,
        out("x10") _,
        out("x11") _,
        out("x12") _,
        out("x13") _,
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

/// [`tile`] for `alpha = ±1`, `beta ∈ {0, 1}`.
///
/// # Safety
/// As `d::tile`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn tile_unit(kc: usize, ap: *const f32, bp: *const f32, c: *mut f32, cs: isize, m: usize, n: usize, subtract: bool, accumulate: bool) {
    asm!(
        ".arch_extension sme",
        ".arch_extension sme2",
        "smstart",
        "ptrue p0.s",
        "ptrue pn8.s",
        masks!(),
        "zero {{za}}",
        "cbz x9, 20f",
        each_column!(load_column, "11"),
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
        each_column!(store_column, "14"),
        "smstop",
        inout("x0") ap => _,
        inout("x1") bp => _,
        inout("x2") c => _,
        inout("x3") kc => _,
        inout("x4") cs * 8 => _,
        inout("x5") m => _,
        inout("x6") n => _,
        inout("x7") subtract as u64 => _,
        inout("x9") accumulate as u64 => _,
        out("x10") _,
        out("x11") _,
        out("x12") _,
        out("x13") _,
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
