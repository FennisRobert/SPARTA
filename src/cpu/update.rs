//! Dense CPU updates for the panel path. Large products go to whichever BLAS
//! `build.rs` linked; smaller ones, and builds with no BLAS at all, use the
//! Rust loops below.
use super::Scalar;
use super::{dense, factor::Front, C};
use crate::Kind;

/// Subtract the completed panel `[start,end)` from the remaining square front.
/// Packs L and D*Lᵀ (LDLᵀ) or U (LU), always using ordinary transpose.
pub(super) fn trailing(front: &mut Front<'_>, start: usize, end: usize) {
    let (m, width) = (front.m, end - start);
    let tail = m - end;
    if tail == 0 {
        return;
    }
    let mut left = vec![C::from_re(0.0); tail * width];
    let mut right = left.clone();
    for i in end..m {
        let mut k = start;
        while k < end {
            let at = (i - end) * width + k - start;
            let l = front.a[i * m + k];
            left[at] = l;
            if front.kind == Kind::Lu {
                right[at] = front.a[k * m + i];
                k += 1;
            } else if front.two[k] {
                let r = front.a[i * m + k + 1];
                left[at + 1] = r;
                right[at] = l * front.d[k] + r * front.sub[k];
                right[at + 1] = l * front.sub[k] + r * front.d[k + 1];
                k += 2;
            } else {
                right[at] = l * front.d[k];
                k += 1;
            }
        }
    }
    if dense::available() && tail >= front.config.cpu_blas_min_rows {
        // The target is `tail` rows of stride `m` starting at `(end, end)`,
        // and both packed operands are separate allocations.
        let trailing = &mut front.a[end * m + end..];
        dense::gemm_nt(tail, tail, width, &left, width, &right, width, trailing, m);
        return;
    }
    for base in (0..tail).step_by(front.config.cpu_block_rows) {
        for i in base..(base + front.config.cpu_block_rows).min(tail) {
            for j in 0..tail {
                let mut v = front.a[(end + i) * m + end + j];
                for k in 0..width {
                    v -= left[i * width + k] * right[j * width + k];
                }
                front.a[(end + i) * m + end + j] = v;
            }
        }
    }
}
