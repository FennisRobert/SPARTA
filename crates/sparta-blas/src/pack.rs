//! Packing: copy a block of A or B into the order its microkernel reads.
//!
//! Packing is where layout disappears. Row-major or column-major, transposed
//! or not, and negative strides (reversed triangles in `trsm`) all arrive here
//! as a pair of element strides, and leave as contiguous slivers. Partial
//! slivers are zero-padded, so the kernel always runs full tiles; only its
//! store is clipped.
use crate::element::{Element, Real};

/// A strided matrix view: element `(i, j)` is at `p + i·rs + j·cs`.
#[derive(Clone, Copy)]
pub(crate) struct View<T> {
    pub p: *const T,
    pub rs: isize,
    pub cs: isize,
}

impl<T> View<T> {
    /// Element `(i, j)`.
    #[inline(always)]
    pub unsafe fn at(self, i: usize, j: usize) -> *const T {
        self.p.offset(i as isize * self.rs + j as isize * self.cs)
    }
}

/// Pack the `mc × kc` block of A at `a` into slivers of `mr` rows.
///
/// Real values: per sliver, `kc` columns of `mr` values. Complex values (split
/// packing): per sliver and column, `mr` real parts then `mr` imaginary parts.
/// `dst` holds `ceil(mc / mr) · mr · kc` elements' worth of reals.
pub(crate) unsafe fn pack_a<T: Element>(mc: usize, kc: usize, a: View<T>, mr: usize, dst: *mut T::Real) {
    let mut d = dst;
    for i0 in (0..mc).step_by(mr) {
        let rows = mr.min(mc - i0);
        if !T::COMPLEX {
            if a.rs == 1 {
                // Columns contiguous: one copy per column of the sliver.
                for p in 0..kc {
                    let src = a.at(i0, p) as *const T::Real;
                    std::ptr::copy_nonoverlapping(src, d, rows);
                    for i in rows..mr {
                        *d.add(i) = T::Real::ZERO;
                    }
                    d = d.add(mr);
                }
            } else {
                for p in 0..kc {
                    for i in 0..mr {
                        *d.add(i) = if i < rows { (*a.at(i0 + i, p)).re() } else { T::Real::ZERO };
                    }
                    d = d.add(mr);
                }
            }
        } else {
            for p in 0..kc {
                for i in 0..mr {
                    let v = if i < rows { *a.at(i0 + i, p) } else { T::zero() };
                    *d.add(i) = v.re();
                    *d.add(mr + i) = v.im();
                }
                d = d.add(2 * mr);
            }
        }
    }
}

/// Pack the `kc × nc` block of B at `b` into slivers of `nr` columns: per
/// sliver, `kc` rows of `nr` values, complex ones interleaved.
pub(crate) unsafe fn pack_b<T: Element>(kc: usize, nc: usize, b: View<T>, nr: usize, dst: *mut T::Real) {
    let mut d = dst as *mut T;
    for j0 in (0..nc).step_by(nr) {
        let cols = nr.min(nc - j0);
        if b.cs == 1 {
            for p in 0..kc {
                std::ptr::copy_nonoverlapping(b.at(p, j0), d, cols);
                for j in cols..nr {
                    *d.add(j) = T::zero();
                }
                d = d.add(nr);
            }
        } else {
            for p in 0..kc {
                for j in 0..nr {
                    *d.add(j) = if j < cols { *b.at(p, j0 + j) } else { T::zero() };
                }
                d = d.add(nr);
            }
        }
    }
}

/// Reals needed to pack an `rows × kc` block in slivers of `r`.
pub(crate) fn packed_len<T: Element>(rows: usize, kc: usize, r: usize) -> usize {
    rows.div_ceil(r) * r * kc * if T::COMPLEX { 2 } else { 1 }
}
