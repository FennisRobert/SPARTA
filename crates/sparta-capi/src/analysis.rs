//! Step 1 of the C API: turning a matrix pattern into a plan.

use crate::config;
use crate::status::{fail, from_error, guard, guard_value, Status};
use sparta::Analysis;

/// Mirrors `sparta_matrix` in `sparta.h`.
const MATRIX_SYMMETRIC: i32 = 0;
const MATRIX_GENERAL: i32 = 1;

/// What a `sparta_analysis *` points at.
///
/// The nonzero count travels with the plan because every later `values` array
/// has to be that long, and the C caller is told the number rather than left to
/// recompute it.
pub struct Plan {
    pub analysis: Analysis,
    pub rows: i64,
    pub nonzeros: i64,
}

/// Reads a caller's column offsets, checking each one before it is used to
/// index anything. Returns the offsets and the nonzero count they imply.
///
/// # Safety
/// `col_ptr` must point to `n + 1` readable `int64_t`.
unsafe fn read_offsets(col_ptr: *const i64, n: usize) -> Result<(Vec<usize>, usize), Status> {
    let declared = unsafe { std::slice::from_raw_parts(col_ptr, n + 1) };
    if declared[0] != 0 {
        return Err(fail(Status::Invalid, "col_ptr[0] must be 0"));
    }
    if declared.windows(2).any(|w| w[0] > w[1]) {
        return Err(fail(
            Status::Invalid,
            "col_ptr must not decrease: entry j + 1 comes after entry j",
        ));
    }
    Ok((
        declared.iter().map(|&v| v as usize).collect(),
        declared[n] as usize,
    ))
}

/// Plans the computation for a matrix pattern.
#[no_mangle]
pub unsafe extern "C" fn sparta_analyze(
    n: i64,
    col_ptr: *const i64,
    row_index: *const i64,
    kind: i32,
    config: *const config::CConfig,
    out: *mut *mut Plan,
) -> Status {
    guard(|| {
        let Some(out) = (unsafe { out.as_mut() }) else {
            return fail(Status::Invalid, "sparta_analyze needs somewhere to put the plan");
        };
        *out = std::ptr::null_mut();
        if n <= 0 {
            return fail(Status::Invalid, format!("n must be at least 1, not {n}"));
        }
        if col_ptr.is_null() || row_index.is_null() {
            return fail(Status::Invalid, "col_ptr and row_index must not be null");
        }
        let settings = match unsafe { config::read(config) } {
            Ok(settings) => settings,
            Err(status) => return status,
        };

        let n = n as usize;
        let (ptr, nonzeros) = match unsafe { read_offsets(col_ptr, n) } {
            Ok(offsets) => offsets,
            Err(status) => return status,
        };
        let rows: Vec<usize> = unsafe { std::slice::from_raw_parts(row_index, nonzeros) }
            .iter()
            .map(|&v| v as usize)
            .collect();

        let analysis = match kind {
            MATRIX_SYMMETRIC => Analysis::ldlt(n, &ptr, &rows, &settings),
            MATRIX_GENERAL => Analysis::lu(n, &ptr, &rows, &settings),
            other => {
                return fail(
                    Status::Invalid,
                    format!("kind must be SPARTA_SYMMETRIC or SPARTA_GENERAL, not {other}"),
                )
            }
        };
        match analysis {
            Ok(analysis) => {
                *out = Box::into_raw(Box::new(Plan {
                    analysis,
                    rows: n as i64,
                    nonzeros: nonzeros as i64,
                }));
                Status::Ok
            }
            Err(error) => from_error(error),
        }
    })
}

/// Borrows a plan handle, treating null as absent rather than as a fault.
///
/// # Safety
/// `plan` must be null or a live handle from `sparta_analyze`.
unsafe fn borrow<'a>(plan: *const Plan) -> Option<&'a Plan> {
    unsafe { plan.as_ref() }
}

/// Returns the number of rows the plan was built for.
#[no_mangle]
pub unsafe extern "C" fn sparta_analysis_rows(plan: *const Plan) -> i64 {
    guard_value(0, || unsafe { borrow(plan) }.map_or(0, |plan| plan.rows))
}

/// Returns the number of nonzeros each `values` array must hold.
#[no_mangle]
pub unsafe extern "C" fn sparta_analysis_nonzeros(plan: *const Plan) -> i64 {
    guard_value(0, || unsafe { borrow(plan) }.map_or(0, |plan| plan.nonzeros))
}

/// Returns how many entries the factors hold per nonzero of the matrix.
#[no_mangle]
pub unsafe extern "C" fn sparta_analysis_fill_ratio(plan: *const Plan) -> f64 {
    guard_value(0.0, || {
        unsafe { borrow(plan) }.map_or(0.0, |plan| plan.analysis.fill_ratio())
    })
}

/// Returns the GPU memory the factors will occupy, in bytes.
#[no_mangle]
pub unsafe extern "C" fn sparta_analysis_factor_bytes(plan: *const Plan) -> usize {
    guard_value(0, || {
        unsafe { borrow(plan) }.map_or(0, |plan| plan.analysis.factor_bytes())
    })
}

/// Releases a plan.
#[no_mangle]
pub unsafe extern "C" fn sparta_analysis_free(plan: *mut Plan) {
    guard_value((), || {
        if !plan.is_null() {
            drop(unsafe { Box::from_raw(plan) });
        }
    });
}
