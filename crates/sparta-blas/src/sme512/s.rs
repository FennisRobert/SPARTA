//! `f32` GEMM tile on the SME matrix unit.
//!
//! One tile is `MR = 32` rows by `NR = 32` columns of C in the four 32-bit
//! ZA tiles (16 × 16 floats each at the 512-bit vector length). Tile
//! `za{2h + q}` holds columns `16h ..` and rows `16q ..`, stored transposed as
//! in `d.rs`: a horizontal slice is one column of C. Each depth step is two
//! two-register loads and four outer products of 256 multiply-adds each.
use std::arch::asm;

pub(crate) const MR: usize = 32;
pub(crate) const NR: usize = 32;

macro_rules! depth {
    ($op:literal) => {
        concat!(
            "ld1w {{z0.s-z1.s}}, pn8/z, [x0]\n",
            "ld1w {{z4.s-z5.s}}, pn8/z, [x1]\n",
            "add x0, x0, #128\n",
            "add x1, x1, #128\n",
            $op, " za0.s, p0/m, p0/m, z4.s, z0.s\n",
            $op, " za1.s, p0/m, p0/m, z4.s, z1.s\n",
            $op, " za2.s, p0/m, p0/m, z5.s, z0.s\n",
            $op, " za3.s, p0/m, p0/m, z5.s, z1.s\n",
            "subs x3, x3, #1\n",
        )
    };
}

/// For each column j < n (x6): x13 at the column, w12 its slice in half h,
/// `$body` given the half's two tiles (rows 0-15, rows 16-31).
macro_rules! each_column {
    ($body:ident, $l0:literal, $l1:literal, $l2:literal) => {
        concat!(
            "mov x13, x2\n",
            "cmp x6, #16\n",
            "mov x15, #16\n",
            "csel x15, x6, x15, lo\n",
            "mov w12, #0\n",
            $l0, ":\n",
            $body!("za0h", "za1h"),
            "add x13, x13, x4\n",
            "add w12, w12, #1\n",
            "cmp x12, x15\n",
            "b.lo ", $l0, "b\n",
            "subs x15, x6, #16\n",
            "b.le ", $l2, "f\n",
            "mov w12, #0\n",
            $l1, ":\n",
            $body!("za2h", "za3h"),
            "add x13, x13, x4\n",
            "add w12, w12, #1\n",
            "cmp x12, x15\n",
            "b.lo ", $l1, "b\n",
            $l2, ":\n",
        )
    };
}

macro_rules! load_column {
    ($t0:literal, $t1:literal) => {
        concat!(
            "ld1w {{", $t0, ".s[w12, 0]}}, p1/z, [x13]\n",
            "ld1w {{", $t1, ".s[w12, 0]}}, p2/z, [x13, x20, lsl #2]\n",
        )
    };
}

macro_rules! store_column {
    ($t0:literal, $t1:literal) => {
        concat!(
            "st1w {{", $t0, ".s[w12, 0]}}, p1, [x13]\n",
            "st1w {{", $t1, ".s[w12, 0]}}, p2, [x13, x20, lsl #2]\n",
        )
    };
}

/// `C = alpha·acc + beta·C` for one column: z28 = alpha, z29 = beta, x9 the
/// beta case (0 zero, 1 one, 2 other).
macro_rules! scale_column {
    ($t0:literal, $t1:literal) => {
        concat!(
            "mova z0.s, p0/m, ", $t0, ".s[w12, 0]\n",
            "mova z1.s, p0/m, ", $t1, ".s[w12, 0]\n",
            "fmul z0.s, z0.s, z28.s\n",
            "fmul z1.s, z1.s, z28.s\n",
            "cbz x9, 5f\n",
            "ld1w z8.s, p1/z, [x13]\n",
            "ld1w z9.s, p2/z, [x13, x20, lsl #2]\n",
            "cmp x9, #1\n",
            "b.ne 6f\n",
            "fadd z0.s, z0.s, z8.s\n",
            "fadd z1.s, z1.s, z9.s\n",
            "b 5f\n",
            "6:\n",
            "fmla z0.s, p0/m, z8.s, z29.s\n",
            "fmla z1.s, p0/m, z9.s, z29.s\n",
            "5:\n",
            "st1w z0.s, p1, [x13]\n",
            "st1w z1.s, p2, [x13, x20, lsl #2]\n",
        )
    };
}

/// Row masks: p1 rows 0-15, p2 rows 16-31 of x5; x20 = 16.
macro_rules! masks {
    () => {
        concat!(
            "mov x10, #0\n",
            "mov x20, #16\n",
            "whilelt p1.s, x10, x5\n",
            "whilelt p2.s, x20, x5\n",
        )
    };
}

/// `C[0..m, 0..n] = alpha·(A·B) + beta·C` for one 32 × 32 tile; arguments as
/// `d::tile`.
///
/// # Safety
/// As `d::tile`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn tile(kc: usize, ap: *const f32, bp: *const f32, c: *mut f32, cs: isize, m: usize, n: usize, alpha: f32, beta: f32) {
    let mode: u64 = if beta == 0.0 {
        0
    } else if beta == 1.0 {
        1
    } else {
        2
    };
    asm!(
        ".arch_extension sme",
        ".arch_extension sme2",
        "smstart",
        "ptrue p0.s",
        "ptrue pn8.s",
        "zero {{za}}",
        "1:",
        depth!("fmopa"),
        "b.ne 1b",
        masks!(),
        "dup z28.s, w7",
        "dup z29.s, w8",
        each_column!(scale_column, "7", "8", "9"),
        "smstop",
        inout("x0") ap => _,
        inout("x1") bp => _,
        inout("x2") c => _,
        inout("x3") kc => _,
        inout("x4") cs * 4 => _,
        inout("x5") m => _,
        inout("x6") n => _,
        inout("x7") alpha.to_bits() as u64 => _,
        inout("x8") beta.to_bits() as u64 => _,
        inout("x9") mode => _,
        out("x10") _,
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

/// [`tile`] for `alpha = ±1`, `beta ∈ {0, 1}`: C straight into ZA and back.
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
        each_column!(load_column, "11", "12", "13"),
        "20:",
        "cbnz x7, 30f",
        "1:",
        depth!("fmopa"),
        "b.ne 1b",
        "b 40f",
        "30:",
        depth!("fmops"),
        "b.ne 30b",
        "40:",
        each_column!(store_column, "14", "15", "16"),
        "smstop",
        inout("x0") ap => _,
        inout("x1") bp => _,
        inout("x2") c => _,
        inout("x3") kc => _,
        inout("x4") cs * 4 => _,
        inout("x5") m => _,
        inout("x6") n => _,
        inout("x7") subtract as u64 => _,
        inout("x9") accumulate as u64 => _,
        out("x10") _,
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
