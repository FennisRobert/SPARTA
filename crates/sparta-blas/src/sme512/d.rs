//! `f64` GEMM tile on the SME matrix unit (Apple M4 and later).
//!
//! One tile is `MR = 32` rows by `NR = 16` columns of C, held in all eight
//! 64-bit ZA tiles (8 × 8 doubles each at the 512-bit vector length). Tile
//! `za{4·h + q}` holds columns `8h .. 8h + 8` and rows `8q .. 8q + 8`, stored
//! transposed: its horizontal slice `s` is column `8h + s`, rows running along
//! the slice. That makes every slice one contiguous column of C.
//!
//! Each step of the depth loop loads the A column (32 doubles, one SME2
//! four-register load) and the B row (16 doubles, a two-register load) and
//! issues eight outer products: 512 multiply-adds for six loads.
use std::arch::asm;

pub(crate) const MR: usize = 32;
pub(crate) const NR: usize = 16;

/// `C[0..m, 0..n] = alpha·(A·B) + beta·C` for one tile.
///
/// - `kc`: depth, at least 1.
/// - `ap`: `kc` columns of 32 doubles (packed, zero-padded).
/// - `bp`: `kc` rows of 16 doubles (packed, zero-padded).
/// - `c`: the tile's first element; C is contiguous down its columns, and
///   `cs` is the column stride in elements.
/// - `m ≤ 32`, `n ≤ 16`: the part of the tile that exists. With `beta == 0`
///   C is only written.
///
/// # Safety
/// The packed buffers must hold `kc` full slivers, and `c` the `m × n` block.
/// The CPU must have SME2 and SME_F64F64 at a 512-bit streaming vector length.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn tile(kc: usize, ap: *const f64, bp: *const f64, c: *mut f64, cs: isize, m: usize, n: usize, alpha: f64, beta: f64) {
    let mode: u64 = if beta == 0.0 {
        0
    } else if beta == 1.0 {
        1
    } else {
        2
    };
    // One column of the tile is in z0..z3 (rows 0-7, 8-15, 16-23, 24-31),
    // x13 points at it, p1..p4 mask the rows that exist.
    macro_rules! column {
        () => {
            concat!(
                "fmul z0.d, z0.d, z28.d\n",
                "fmul z1.d, z1.d, z28.d\n",
                "fmul z2.d, z2.d, z28.d\n",
                "fmul z3.d, z3.d, z28.d\n",
                "cbz x9, 5f\n",
                "ld1d z8.d, p1/z, [x13]\n",
                "ld1d z9.d, p2/z, [x13, #1, mul vl]\n",
                "ld1d z10.d, p3/z, [x13, #2, mul vl]\n",
                "ld1d z11.d, p4/z, [x13, #3, mul vl]\n",
                "cmp x9, #1\n",
                "b.ne 6f\n",
                "fadd z0.d, z0.d, z8.d\n",
                "fadd z1.d, z1.d, z9.d\n",
                "fadd z2.d, z2.d, z10.d\n",
                "fadd z3.d, z3.d, z11.d\n",
                "b 5f\n",
                "6:\n",
                "fmla z0.d, p0/m, z8.d, z29.d\n",
                "fmla z1.d, p0/m, z9.d, z29.d\n",
                "fmla z2.d, p0/m, z10.d, z29.d\n",
                "fmla z3.d, p0/m, z11.d, z29.d\n",
                "5:\n",
                "st1d z0.d, p1, [x13]\n",
                "st1d z1.d, p2, [x13, #1, mul vl]\n",
                "st1d z2.d, p3, [x13, #2, mul vl]\n",
                "st1d z3.d, p4, [x13, #3, mul vl]\n",
                "add x13, x13, x4\n",
            )
        };
    }
    asm!(
        ".arch_extension sme",
        ".arch_extension sme2",
        ".arch_extension sme-f64f64",
        "smstart",
        "zero {{za}}",
        "ptrue p0.d",
        "ptrue pn8.d",
        // The depth loop.
        "1:",
        "ld1d {{z0.d-z3.d}}, pn8/z, [x0]",
        "ld1d {{z4.d-z5.d}}, pn8/z, [x1]",
        "add x0, x0, #256",
        "add x1, x1, #128",
        "fmopa za0.d, p0/m, p0/m, z4.d, z0.d",
        "fmopa za1.d, p0/m, p0/m, z4.d, z1.d",
        "fmopa za2.d, p0/m, p0/m, z4.d, z2.d",
        "fmopa za3.d, p0/m, p0/m, z4.d, z3.d",
        "fmopa za4.d, p0/m, p0/m, z5.d, z0.d",
        "fmopa za5.d, p0/m, p0/m, z5.d, z1.d",
        "fmopa za6.d, p0/m, p0/m, z5.d, z2.d",
        "fmopa za7.d, p0/m, p0/m, z5.d, z3.d",
        "subs x3, x3, #1",
        "b.ne 1b",
        // Write-out: row masks, alpha and beta.
        "mov x10, #0",
        "whilelt p1.d, x10, x5",
        "mov x10, #8",
        "whilelt p2.d, x10, x5",
        "mov x10, #16",
        "whilelt p3.d, x10, x5",
        "mov x10, #24",
        "whilelt p4.d, x10, x5",
        "dup z28.d, x7",
        "dup z29.d, x8",
        "mov x13, x2",
        // Columns 0..min(n, 8): slices of za0..za3.
        "cmp x6, #8",
        "mov x15, #8",
        "csel x15, x6, x15, lo",
        "mov w12, #0",
        "7:",
        "mova z0.d, p0/m, za0h.d[w12, 0]",
        "mova z1.d, p0/m, za1h.d[w12, 0]",
        "mova z2.d, p0/m, za2h.d[w12, 0]",
        "mova z3.d, p0/m, za3h.d[w12, 0]",
        column!(),
        "add w12, w12, #1",
        "cmp x12, x15",
        "b.lo 7b",
        // Columns 8..n: slices of za4..za7.
        "subs x15, x6, #8",
        "b.le 10f",
        "mov w12, #0",
        "9:",
        "mova z0.d, p0/m, za4h.d[w12, 0]",
        "mova z1.d, p0/m, za5h.d[w12, 0]",
        "mova z2.d, p0/m, za6h.d[w12, 0]",
        "mova z3.d, p0/m, za7h.d[w12, 0]",
        column!(),
        "add w12, w12, #1",
        "cmp x12, x15",
        "b.lo 9b",
        "10:",
        "smstop",
        inout("x0") ap => _,
        inout("x1") bp => _,
        inout("x2") c => _,
        inout("x3") kc => _,
        inout("x4") cs * 8 => _,
        inout("x5") m => _,
        inout("x6") n => _,
        inout("x7") alpha.to_bits() => _,
        inout("x8") beta.to_bits() => _,
        inout("x9") mode => _,
        out("x10") _,
        out("x12") _,
        out("x13") _,
        out("x15") _,
        // Entering and leaving streaming mode zeroes every vector and
        // predicate register, so all of them are clobbered, including the
        // callee-saved v8..v15 that `clobber_abi` leaves out.
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

/// [`tile`] for `alpha = ±1` and `beta ∈ {0, 1}`, the products a factorization
/// makes (`C -= A·B`, `C = -A·B`), with no write-out arithmetic: C is loaded
/// straight into ZA (or ZA zeroed), the outer products add or subtract into
/// it, and ZA is stored straight back.
///
/// - `subtract`: `alpha = -1` (else `+1`).
/// - `accumulate`: `beta = 1` (else `0`).
///
/// # Safety
/// As [`tile`]; C must be contiguous down its columns.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn tile_unit(kc: usize, ap: *const f64, bp: *const f64, c: *mut f64, cs: isize, m: usize, n: usize, subtract: bool, accumulate: bool) {
    // Move C's columns between memory and ZA slices: `$op` is ld1d or st1d,
    // `$z` the zeroing qualifier loads need.
    macro_rules! columns {
        ($op:literal, $z:literal, $l0:literal, $l1:literal) => {
            concat!(
                "mov x13, x2\n",
                "cmp x6, #8\n",
                "mov x15, #8\n",
                "csel x15, x6, x15, lo\n",
                "mov w12, #0\n",
                $l0, ":\n",
                $op, " {{za0h.d[w12, 0]}}, p1", $z, ", [x13]\n",
                $op, " {{za1h.d[w12, 0]}}, p2", $z, ", [x13, x20, lsl #3]\n",
                $op, " {{za2h.d[w12, 0]}}, p3", $z, ", [x13, x21, lsl #3]\n",
                $op, " {{za3h.d[w12, 0]}}, p4", $z, ", [x13, x22, lsl #3]\n",
                "add x13, x13, x4\n",
                "add w12, w12, #1\n",
                "cmp x12, x15\n",
                "b.lo ", $l0, "b\n",
                "subs x15, x6, #8\n",
                "b.le ", $l1, "f\n",
                "mov w12, #0\n",
                "8:\n",
                $op, " {{za4h.d[w12, 0]}}, p1", $z, ", [x13]\n",
                $op, " {{za5h.d[w12, 0]}}, p2", $z, ", [x13, x20, lsl #3]\n",
                $op, " {{za6h.d[w12, 0]}}, p3", $z, ", [x13, x21, lsl #3]\n",
                $op, " {{za7h.d[w12, 0]}}, p4", $z, ", [x13, x22, lsl #3]\n",
                "add x13, x13, x4\n",
                "add w12, w12, #1\n",
                "cmp x12, x15\n",
                "b.lo 8b\n",
                $l1, ":\n",
            )
        };
    }
    macro_rules! depth {
        ($op:literal) => {
            concat!(
                "ld1d {{z0.d-z3.d}}, pn8/z, [x0]\n",
                "ld1d {{z4.d-z5.d}}, pn8/z, [x1]\n",
                "add x0, x0, #256\n",
                "add x1, x1, #128\n",
                $op, " za0.d, p0/m, p0/m, z4.d, z0.d\n",
                $op, " za1.d, p0/m, p0/m, z4.d, z1.d\n",
                $op, " za2.d, p0/m, p0/m, z4.d, z2.d\n",
                $op, " za3.d, p0/m, p0/m, z4.d, z3.d\n",
                $op, " za4.d, p0/m, p0/m, z5.d, z0.d\n",
                $op, " za5.d, p0/m, p0/m, z5.d, z1.d\n",
                $op, " za6.d, p0/m, p0/m, z5.d, z2.d\n",
                $op, " za7.d, p0/m, p0/m, z5.d, z3.d\n",
                "subs x3, x3, #1\n",
            )
        };
    }
    asm!(
        ".arch_extension sme",
        ".arch_extension sme2",
        ".arch_extension sme-f64f64",
        "smstart",
        "ptrue p0.d",
        "ptrue pn8.d",
        "mov x20, #8",
        "mov x21, #16",
        "mov x22, #24",
        "mov x10, #0",
        "whilelt p1.d, x10, x5",
        "whilelt p2.d, x20, x5",
        "whilelt p3.d, x21, x5",
        "whilelt p4.d, x22, x5",
        "zero {{za}}",
        "cbz x9, 20f",
        columns!("ld1d", "/z", "7", "9"),
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
        // p1..p4 mask the rows that exist; columns beyond n are not stored.
        columns!("st1d", "", "17", "19"),
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
        out("x12") _,
        out("x13") _,
        out("x15") _,
        out("x20") _,
        out("x21") _,
        out("x22") _,
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
