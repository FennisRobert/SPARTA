//! Transposing packs on the matrix unit.
//!
//! A packed sliver is `kc` steps of `w` values: for each depth `p`, the `w`
//! lines' values at `p`, contiguous. When each line is contiguous along the
//! depth instead (the rows of a row-major A, the columns of B in `A·Bᵀ`),
//! packing is a transpose. ZA does it in hardware: eight lines go into the
//! horizontal slices of one tile, and its vertical slices come out as eight
//! depths of eight lines each.
use std::arch::asm;

/// Groups of eight lines handed to one streaming-mode call.
const GROUPS: usize = 64;

/// Pack `lines` lines of `kc` doubles into slivers of width `w`, zero-padded,
/// when the lines are contiguous along the depth. Returns false, having done
/// nothing, for any other layout.
///
/// - `p`: line 0 at depth 0; line `l` starts at `p + l·ls`, depth `d` is
///   at `+ d·ks`.
/// - `dst`: `ceil(lines / w)` slivers of `kc · w` doubles.
///
/// # Safety
/// The source must hold the block, and `dst` the packed slivers. SME
/// support at a 512-bit vector length is the caller's to check.
pub(crate) unsafe fn transpose_d(lines: usize, kc: usize, p: *const f64, ls: isize, ks: isize, w: usize, dst: *mut f64) -> bool {
    if ks != 1 || !w.is_multiple_of(8) || kc == 0 || lines == 0 {
        return false;
    }
    // Zero the padding lines of the last sliver once; the kernel writes only
    // the groups that hold lines.
    let slivers = lines.div_ceil(w);
    let used = lines.div_ceil(8) * 8;
    if used < slivers * w {
        let last = dst.add((slivers - 1) * w * kc);
        let first_pad = used - (slivers - 1) * w;
        for d in 0..kc {
            for i in first_pad..w {
                *last.add(d * w + i) = 0.0;
            }
        }
    }
    let groups = lines.div_ceil(8);
    let mut src = [std::ptr::null::<f64>(); GROUPS];
    let mut out = [std::ptr::null_mut::<f64>(); GROUPS];
    let mut valid = [0u64; GROUPS];
    for g0 in (0..groups).step_by(GROUPS) {
        let count = GROUPS.min(groups - g0);
        for k in 0..count {
            let g = g0 + k;
            let line = 8 * g;
            src[k] = p.offset(line as isize * ls);
            out[k] = dst.add((line / w) * w * kc + line % w);
            valid[k] = (lines - line).min(8) as u64;
        }
        group_call(count, kc, &src, &out, &valid, ls, w);
    }
    true
}

/// One streaming-mode pass over `count` groups of up to eight lines.
unsafe fn group_call(count: usize, kc: usize, src: &[*const f64; GROUPS], out: &[*mut f64; GROUPS], valid: &[u64; GROUPS], ls: isize, w: usize) {
    asm!(
        ".arch_extension sme",
        "smstart",
        "ptrue p0.d",
        "mov x9, #0",
        // Per group: x7 = lines at depth k0, x8 = packed output at depth k0,
        // x5 = lines that exist.
        "1:",
        "ldr x7, [x10, x9, lsl #3]",
        "ldr x8, [x11, x9, lsl #3]",
        "ldr x5, [x13, x9, lsl #3]",
        "mov x6, #0",
        // Per block of eight depths.
        "2:",
        "whilelt p1.d, x6, x2",
        "zero {{za0.d}}",
        "mov w12, #0",
        "mov x14, x7",
        "3:",
        "ld1d {{za0h.d[w12, 0]}}, p1/z, [x14]",
        "add x14, x14, x1",
        "add w12, w12, #1",
        "cmp x12, x5",
        "b.lo 3b",
        // Depths in this block: min(8, kc - k0).
        "sub x15, x2, x6",
        "mov x16, #8",
        "cmp x15, x16",
        "csel x15, x15, x16, lo",
        "mov w12, #0",
        "mov x14, x8",
        "4:",
        "st1d {{za0v.d[w12, 0]}}, p0, [x14]",
        "add x14, x14, x4",
        "add w12, w12, #1",
        "cmp x12, x15",
        "b.lo 4b",
        "add x6, x6, #8",
        "add x7, x7, #64",
        "add x8, x8, x4, lsl #3",
        "cmp x6, x2",
        "b.lo 2b",
        "add x9, x9, #1",
        "cmp x9, x3",
        "b.lo 1b",
        "smstop",
        inout("x1") ls * 8 => _,
        inout("x2") kc => _,
        inout("x3") count => _,
        inout("x4") w * 8 => _,
        inout("x10") src.as_ptr() => _,
        inout("x11") out.as_ptr() => _,
        inout("x13") valid.as_ptr() => _,
        out("x5") _,
        out("x6") _,
        out("x7") _,
        out("x8") _,
        out("x9") _,
        out("x12") _,
        out("x14") _,
        out("x15") _,
        out("x16") _,
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

/// Split-pack `lines` lines of `kc` complex values into slivers of width `w`:
/// per depth, the `w` real parts then the `w` imaginary parts, zero-padded.
/// Every layout is handled (always returns true): lines contiguous along the
/// depth go through ZA, where the vertical slices of an interleaved row
/// chunk alternate real and imaginary parts; anything else is copied
/// directly.
///
/// - `p`: line 0 at depth 0, as reals; strides `ls` and `ks` in complex
///   elements.
///
/// # Safety
/// As [`transpose_d`].
pub(crate) unsafe fn split_z(lines: usize, kc: usize, p: *const f64, ls: isize, ks: isize, w: usize, dst: *mut f64) -> bool {
    let slivers = lines.div_ceil(w);
    if ks != 1 || !w.is_multiple_of(8) {
        for s in 0..slivers {
            let out = dst.add(s * 2 * w * kc);
            for d in 0..kc {
                for i in 0..w {
                    let l = s * w + i;
                    let (re, im) = if l < lines {
                        let v = p.offset(2 * (l as isize * ls + d as isize * ks));
                        (*v, *v.add(1))
                    } else {
                        (0.0, 0.0)
                    };
                    *out.add(d * 2 * w + i) = re;
                    *out.add(d * 2 * w + w + i) = im;
                }
            }
        }
        return true;
    }
    let used = lines.div_ceil(8) * 8;
    if used < slivers * w {
        let last = dst.add((slivers - 1) * 2 * w * kc);
        let first_pad = used - (slivers - 1) * w;
        for d in 0..kc {
            for i in first_pad..w {
                *last.add(d * 2 * w + i) = 0.0;
                *last.add(d * 2 * w + w + i) = 0.0;
            }
        }
    }
    let groups = lines.div_ceil(8);
    let mut src = [std::ptr::null::<f64>(); GROUPS];
    let mut out = [std::ptr::null_mut::<f64>(); GROUPS];
    let mut valid = [0u64; GROUPS];
    for g0 in (0..groups).step_by(GROUPS) {
        let count = GROUPS.min(groups - g0);
        for k in 0..count {
            let line = 8 * (g0 + k);
            src[k] = p.offset(2 * line as isize * ls);
            out[k] = dst.add((line / w) * 2 * w * kc + line % w);
            valid[k] = (lines - line).min(8) as u64;
        }
        asm!(
            ".arch_extension sme",
            "smstart",
            "ptrue p0.d",
            "lsl x2, x2, #1",
            "mov x9, #0",
            "1:",
            "ldr x7, [x10, x9, lsl #3]",
            "ldr x8, [x11, x9, lsl #3]",
            "ldr x5, [x13, x9, lsl #3]",
            "mov x6, #0",
            // x6 counts reals along the line: 8 reals = 4 complex depths.
            "2:",
            "whilelt p1.d, x6, x2",
            "zero {{za0.d}}",
            "mov w12, #0",
            "mov x14, x7",
            "3:",
            "ld1d {{za0h.d[w12, 0]}}, p1/z, [x14]",
            "add x14, x14, x1",
            "add w12, w12, #1",
            "cmp x12, x5",
            "b.lo 3b",
            // Depths in this block: min(4, (2kc - x6) / 2).
            "sub x15, x2, x6",
            "lsr x15, x15, #1",
            "mov x16, #4",
            "cmp x15, x16",
            "csel x15, x15, x16, lo",
            "mov w12, #0",
            "mov x14, x8",
            "mov x16, #0",
            "4:",
            "st1d {{za0v.d[w12, 0]}}, p0, [x14]",
            "st1d {{za0v.d[w12, 1]}}, p0, [x14, x17, lsl #3]",
            "add x14, x14, x4",
            "add w12, w12, #2",
            "add x16, x16, #1",
            "cmp x16, x15",
            "b.lo 4b",
            "add x6, x6, #8",
            "add x7, x7, #64",
            "add x8, x8, x4, lsl #2",
            "cmp x6, x2",
            "b.lo 2b",
            "add x9, x9, #1",
            "cmp x9, x3",
            "b.lo 1b",
            "smstop",
            inout("x1") ls * 16 => _,
            inout("x2") kc => _,
            inout("x3") count => _,
            inout("x4") w * 16 => _,
            inout("x17") w => _,
            inout("x10") src.as_ptr() => _,
            inout("x11") out.as_ptr() => _,
            inout("x13") valid.as_ptr() => _,
            out("x5") _,
            out("x6") _,
            out("x7") _,
            out("x8") _,
            out("x9") _,
            out("x12") _,
            out("x14") _,
            out("x15") _,
            out("x16") _,
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
    true
}

/// [`transpose_d`] for `f32`: sixteen lines per 32-bit ZA tile, sixteen
/// depths per block.
///
/// # Safety
/// As [`transpose_d`].
pub(crate) unsafe fn transpose_s(lines: usize, kc: usize, p: *const f32, ls: isize, ks: isize, w: usize, dst: *mut f32) -> bool {
    if ks != 1 || !w.is_multiple_of(16) || kc == 0 || lines == 0 {
        return false;
    }
    let slivers = lines.div_ceil(w);
    let used = lines.div_ceil(16) * 16;
    if used < slivers * w {
        let last = dst.add((slivers - 1) * w * kc);
        let first_pad = used - (slivers - 1) * w;
        for d in 0..kc {
            for i in first_pad..w {
                *last.add(d * w + i) = 0.0;
            }
        }
    }
    let groups = lines.div_ceil(16);
    let mut src = [std::ptr::null::<f32>(); GROUPS];
    let mut out = [std::ptr::null_mut::<f32>(); GROUPS];
    let mut valid = [0u64; GROUPS];
    for g0 in (0..groups).step_by(GROUPS) {
        let count = GROUPS.min(groups - g0);
        for k in 0..count {
            let line = 16 * (g0 + k);
            src[k] = p.offset(line as isize * ls);
            out[k] = dst.add((line / w) * w * kc + line % w);
            valid[k] = (lines - line).min(16) as u64;
        }
        asm!(
            ".arch_extension sme",
            "smstart",
            "ptrue p0.s",
            "mov x9, #0",
            "1:",
            "ldr x7, [x10, x9, lsl #3]",
            "ldr x8, [x11, x9, lsl #3]",
            "ldr x5, [x13, x9, lsl #3]",
            "mov x6, #0",
            "2:",
            "whilelt p1.s, x6, x2",
            "zero {{za0.s}}",
            "mov w12, #0",
            "mov x14, x7",
            "3:",
            "ld1w {{za0h.s[w12, 0]}}, p1/z, [x14]",
            "add x14, x14, x1",
            "add w12, w12, #1",
            "cmp x12, x5",
            "b.lo 3b",
            "sub x15, x2, x6",
            "mov x16, #16",
            "cmp x15, x16",
            "csel x15, x15, x16, lo",
            "mov w12, #0",
            "mov x14, x8",
            "4:",
            "st1w {{za0v.s[w12, 0]}}, p0, [x14]",
            "add x14, x14, x4",
            "add w12, w12, #1",
            "cmp x12, x15",
            "b.lo 4b",
            "add x6, x6, #16",
            "add x7, x7, #64",
            "add x8, x8, x4, lsl #4",
            "cmp x6, x2",
            "b.lo 2b",
            "add x9, x9, #1",
            "cmp x9, x3",
            "b.lo 1b",
            "smstop",
            inout("x1") ls * 4 => _,
            inout("x2") kc => _,
            inout("x3") count => _,
            inout("x4") w * 4 => _,
            inout("x10") src.as_ptr() => _,
            inout("x11") out.as_ptr() => _,
            inout("x13") valid.as_ptr() => _,
            out("x5") _,
            out("x6") _,
            out("x7") _,
            out("x8") _,
            out("x9") _,
            out("x12") _,
            out("x14") _,
            out("x15") _,
            out("x16") _,
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
    true
}

/// [`split_z`] for `Complex<f32>`: sixteen lines per tile, eight complex
/// depths (sixteen floats) per block, real parts in the even vertical slices
/// and imaginary parts in the odd ones. Every layout is handled.
///
/// # Safety
/// As [`transpose_d`].
pub(crate) unsafe fn split_c(lines: usize, kc: usize, p: *const f32, ls: isize, ks: isize, w: usize, dst: *mut f32) -> bool {
    let slivers = lines.div_ceil(w);
    if ks != 1 || !w.is_multiple_of(16) {
        for s in 0..slivers {
            let out = dst.add(s * 2 * w * kc);
            for d in 0..kc {
                for i in 0..w {
                    let l = s * w + i;
                    let (re, im) = if l < lines {
                        let v = p.offset(2 * (l as isize * ls + d as isize * ks));
                        (*v, *v.add(1))
                    } else {
                        (0.0, 0.0)
                    };
                    *out.add(d * 2 * w + i) = re;
                    *out.add(d * 2 * w + w + i) = im;
                }
            }
        }
        return true;
    }
    let used = lines.div_ceil(16) * 16;
    if used < slivers * w {
        let last = dst.add((slivers - 1) * 2 * w * kc);
        let first_pad = used - (slivers - 1) * w;
        for d in 0..kc {
            for i in first_pad..w {
                *last.add(d * 2 * w + i) = 0.0;
                *last.add(d * 2 * w + w + i) = 0.0;
            }
        }
    }
    let groups = lines.div_ceil(16);
    let mut src = [std::ptr::null::<f32>(); GROUPS];
    let mut out = [std::ptr::null_mut::<f32>(); GROUPS];
    let mut valid = [0u64; GROUPS];
    for g0 in (0..groups).step_by(GROUPS) {
        let count = GROUPS.min(groups - g0);
        for k in 0..count {
            let line = 16 * (g0 + k);
            src[k] = p.offset(2 * line as isize * ls);
            out[k] = dst.add((line / w) * 2 * w * kc + line % w);
            valid[k] = (lines - line).min(16) as u64;
        }
        asm!(
            ".arch_extension sme",
            "smstart",
            "ptrue p0.s",
            "lsl x2, x2, #1",
            "mov x9, #0",
            "1:",
            "ldr x7, [x10, x9, lsl #3]",
            "ldr x8, [x11, x9, lsl #3]",
            "ldr x5, [x13, x9, lsl #3]",
            "mov x6, #0",
            // x6 counts floats along the line: 16 floats = 8 complex depths.
            "2:",
            "whilelt p1.s, x6, x2",
            "zero {{za0.s}}",
            "mov w12, #0",
            "mov x14, x7",
            "3:",
            "ld1w {{za0h.s[w12, 0]}}, p1/z, [x14]",
            "add x14, x14, x1",
            "add w12, w12, #1",
            "cmp x12, x5",
            "b.lo 3b",
            "sub x15, x2, x6",
            "lsr x15, x15, #1",
            "mov x16, #8",
            "cmp x15, x16",
            "csel x15, x15, x16, lo",
            "mov w12, #0",
            "mov x14, x8",
            "mov x16, #0",
            "4:",
            "st1w {{za0v.s[w12, 0]}}, p0, [x14]",
            "st1w {{za0v.s[w12, 1]}}, p0, [x14, x17, lsl #2]",
            "add x14, x14, x4",
            "add w12, w12, #2",
            "add x16, x16, #1",
            "cmp x16, x15",
            "b.lo 4b",
            "add x6, x6, #16",
            "add x7, x7, #64",
            "add x8, x8, x4, lsl #3",
            "cmp x6, x2",
            "b.lo 2b",
            "add x9, x9, #1",
            "cmp x9, x3",
            "b.lo 1b",
            "smstop",
            inout("x1") ls * 8 => _,
            inout("x2") kc => _,
            inout("x3") count => _,
            inout("x4") w * 8 => _,
            inout("x17") w => _,
            inout("x10") src.as_ptr() => _,
            inout("x11") out.as_ptr() => _,
            inout("x13") valid.as_ptr() => _,
            out("x5") _,
            out("x6") _,
            out("x7") _,
            out("x8") _,
            out("x9") _,
            out("x12") _,
            out("x14") _,
            out("x15") _,
            out("x16") _,
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
    true
}
