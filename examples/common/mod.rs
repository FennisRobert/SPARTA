//! Loader for SPARTA's native benchmark format (see `tools/npz_to_sparta.py`).
#![allow(dead_code)]
use std::io::Read;

pub type C64 = num_complex::Complex<f64>;

const MAGIC_V1: &[u8; 8] = b"SPARTA01";
const MAGIC_V2: &[u8; 8] = b"SPARTA02";

/// One assembled system in sorted CSC, its right-hand sides, and optionally the
/// reference solutions it shipped with.
///
/// `SPARTA01` files are always one port and always the lower triangle of a
/// complex-symmetric matrix. `SPARTA02` adds a port count and a storage flag, so
/// it can also carry the unsymmetric periodic-array models, which have to keep
/// both triangles and which SPARTA cannot factor until there is an LU path.
pub struct System {
    pub name: String,
    pub n: usize,
    /// Excitation ports: right-hand side columns sharing one matrix.
    pub nrhs: usize,
    pub k0: f64,
    /// True when `values` is the whole matrix, false when it is the lower
    /// triangle of a complex-symmetric one. Everything that assumes symmetry --
    /// `multiply`, `Analysis::ldlt` -- is only valid when this is false.
    pub full: bool,
    pub col_ptr: Vec<usize>,
    pub row_idx: Vec<usize>,
    pub values: Vec<C64>,
    /// Port-major: `nrhs` consecutive blocks of `n`.
    pub rhs: Vec<C64>,
    pub reference: Option<Vec<C64>>,
}

struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8], String> {
        let end = self.at.checked_add(len).ok_or("length overflow")?;
        let slice = self.bytes.get(self.at..end).ok_or("file ends early")?;
        self.at = end;
        Ok(slice)
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f64(&mut self) -> Result<f64, String> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn indices(&mut self, count: usize) -> Result<Vec<usize>, String> {
        Ok(self
            .take(count * 8)?
            .chunks_exact(8)
            .map(|w| u64::from_le_bytes(w.try_into().unwrap()) as usize)
            .collect())
    }
    fn complex(&mut self, count: usize) -> Result<Vec<C64>, String> {
        Ok(self
            .take(count * 16)?
            .chunks_exact(16)
            .map(|w| {
                C64::new(
                    f64::from_le_bytes(w[..8].try_into().unwrap()),
                    f64::from_le_bytes(w[8..].try_into().unwrap()),
                )
            })
            .collect())
    }
}

impl System {
    /// A synthetic 3D time-harmonic curl-curl system on a `k^3` grid: the same
    /// indefinite, complex-symmetric, gradient-near-null-space operator class as
    /// the EMerge snapshots, but generated at any size for scaling runs.
    pub fn curl_curl(k: usize) -> Self {
        let (n, col_ptr, row_idx, values) = sparta_matgen::curl_curl(&[k, k, k], 1.0, 0.1);
        // A manufactured solution gives the run something to score against.
        let mut system = Self {
            name: format!("curl_curl_{k}^3"),
            n,
            nrhs: 1,
            k0: 1.0,
            full: false,
            col_ptr,
            row_idx,
            values,
            rhs: Vec::new(),
            reference: None,
        };
        let exact: Vec<C64> = (0..n)
            .map(|i| C64::new((i % 17) as f64 * 0.1 - 0.8, (i % 7) as f64 * 0.2 - 0.6))
            .collect();
        system.rhs = system.multiply(&exact);
        system.reference = Some(exact);
        system
    }

    pub fn load(path: &str) -> Result<Self, String> {
        let mut bytes = Vec::new();
        std::fs::File::open(path)
            .map_err(|e| format!("{path}: {e}"))?
            .read_to_end(&mut bytes)
            .map_err(|e| format!("{path}: {e}"))?;
        let mut cursor = Cursor {
            bytes: &bytes,
            at: 0,
        };
        let version = match cursor.take(8)? {
            m if m == MAGIC_V1 => 1,
            m if m == MAGIC_V2 => 2,
            _ => return Err(format!("{path}: not a SPARTA benchmark file")),
        };
        let (n, nnz) = (cursor.u64()? as usize, cursor.u64()? as usize);
        // v1 had no port count and no storage flag: one port, lower triangle.
        let nrhs = if version >= 2 {
            cursor.u64()? as usize
        } else {
            1
        };
        let flags = cursor.u64()?;
        let k0 = cursor.f64()?;
        let col_ptr = cursor.indices(n + 1)?;
        let row_idx = cursor.indices(nnz)?;
        let values = cursor.complex(nnz)?;
        let rhs = cursor.complex(n * nrhs)?;
        let reference = if flags & 1 != 0 {
            Some(cursor.complex(n * nrhs)?)
        } else {
            None
        };
        let full = flags & 2 != 0;
        if col_ptr[n] != nnz {
            return Err(format!("{path}: column pointers do not end at nnz"));
        }
        let name = std::path::Path::new(path)
            .file_stem()
            .map_or_else(|| path.to_string(), |s| s.to_string_lossy().into_owned());
        Ok(Self {
            name,
            n,
            nrhs,
            k0,
            full,
            col_ptr,
            row_idx,
            values,
            rhs,
            reference,
        })
    }

    /// Right-hand side for one excitation port.
    pub fn rhs_port(&self, port: usize) -> &[C64] {
        &self.rhs[port * self.n..(port + 1) * self.n]
    }
    /// Reference solution for one port, when the file shipped one.
    pub fn reference_port(&self, port: usize) -> Option<&[C64]> {
        self.reference
            .as_ref()
            .map(|r| &r[port * self.n..(port + 1) * self.n])
    }

    /// The same matrix with both triangles stored, for factoring a symmetric
    /// system with LU. A system that is already full is returned as it is.
    #[allow(dead_code)]
    pub fn mirrored(self) -> Self {
        if self.full {
            return self;
        }
        let mut columns: Vec<Vec<(usize, C64)>> = vec![Vec::new(); self.n];
        for j in 0..self.n {
            for k in self.col_ptr[j]..self.col_ptr[j + 1] {
                let (i, a) = (self.row_idx[k], self.values[k]);
                columns[j].push((i, a));
                if i != j {
                    columns[i].push((j, a));
                }
            }
        }
        let (mut col_ptr, mut row_idx, mut values) = (vec![0], Vec::new(), Vec::new());
        for mut column in columns {
            column.sort_unstable_by_key(|e| e.0);
            row_idx.extend(column.iter().map(|e| e.0));
            values.extend(column.iter().map(|e| e.1));
            col_ptr.push(row_idx.len());
        }
        Self { full: true, col_ptr, row_idx, values, ..self }
    }

    /// `A x` in double precision. Used to score the single-precision GPU solve.
    ///
    /// A symmetric system stores only its lower triangle, so every off-diagonal
    /// entry is applied twice; a full one is applied as written.
    pub fn multiply(&self, x: &[C64]) -> Vec<C64> {
        let mut y = vec![C64::new(0.0, 0.0); self.n];
        for j in 0..self.n {
            for k in self.col_ptr[j]..self.col_ptr[j + 1] {
                let (i, a) = (self.row_idx[k], self.values[k]);
                y[i] += a * x[j];
                if !self.full && i != j {
                    y[j] += a * x[i];
                }
            }
        }
        y
    }

    pub fn relative_residual(&self, x: &[C64], port: usize) -> f64 {
        let y = self.multiply(x);
        let b = self.rhs_port(port);
        norm(&y.iter().zip(b).map(|(a, b)| a - b).collect::<Vec<_>>()) / norm(b)
    }
}

pub fn norm(v: &[C64]) -> f64 {
    v.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt()
}

pub fn relative_difference(a: &[C64], b: &[C64]) -> f64 {
    norm(&a.iter().zip(b).map(|(x, y)| x - y).collect::<Vec<_>>()) / norm(b)
}

pub fn gib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}
