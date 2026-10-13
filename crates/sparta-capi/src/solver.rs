//! Steps 2 and 3 of the C API: factoring on the GPU and solving with the result.

use crate::analysis::Plan;
use crate::status::{fail, from_error, guard, guard_value, Status};
use sparta::{Complex32, Solver};

/// `sparta_complex` and [`Complex32`] are both two floats in declaration order,
/// so a caller's array is read in place rather than converted.
type Complex = Complex32;

/// What a `sparta_solver *` points at.
///
/// The dimensions are copied out of the plan the solver swallowed, so that the
/// accessors and the length checks still have them.
pub struct Session {
    solver: Solver,
    rows: usize,
    nonzeros: usize,
}

/// Reserves GPU memory for a plan and prepares the GPU kernels.
#[no_mangle]
pub unsafe extern "C" fn sparta_solver_create(
    plan: *mut *mut Plan,
    out: *mut *mut Session,
) -> Status {
    guard(|| {
        let Some(out) = (unsafe { out.as_mut() }) else {
            return fail(Status::Invalid, "sparta_solver_create needs somewhere to put the solver");
        };
        *out = std::ptr::null_mut();
        let Some(slot) = (unsafe { plan.as_mut() }) else {
            return fail(Status::Invalid, "sparta_solver_create needs a plan to take over");
        };
        if slot.is_null() {
            return fail(Status::Invalid, "the plan is null; it may already have been used");
        }
        // Taken whatever happens: `Solver::new` owns the plan from here, and
        // clearing the caller's handle now is what stops a second use of it.
        let taken = *unsafe { Box::from_raw(std::mem::replace(slot, std::ptr::null_mut())) };
        match Solver::new(taken.analysis) {
            Ok(solver) => {
                *out = Box::into_raw(Box::new(Session {
                    solver,
                    rows: taken.rows as usize,
                    nonzeros: taken.nonzeros as usize,
                }));
                Status::Ok
            }
            Err(error) => from_error(error),
        }
    })
}

/// Borrows a live solver, or explains why the handle cannot be used.
///
/// # Safety
/// `session` must be null or a live handle from `sparta_solver_create`.
unsafe fn borrow_mut<'a>(session: *mut Session, what: &str) -> Result<&'a mut Session, Status> {
    unsafe { session.as_mut() }
        .ok_or_else(|| fail(Status::Invalid, format!("{what} needs a solver")))
}

/// Factors a matrix on the GPU.
#[no_mangle]
pub unsafe extern "C" fn sparta_factor(session: *mut Session, values: *const Complex) -> Status {
    guard(|| {
        let session = match unsafe { borrow_mut(session, "sparta_factor") } {
            Ok(session) => session,
            Err(status) => return status,
        };
        if values.is_null() {
            return fail(Status::Invalid, "sparta_factor needs a values array");
        }
        let values = unsafe { std::slice::from_raw_parts(values, session.nonzeros) };
        match session.solver.factor(values) {
            Ok(()) => Status::Ok,
            Err(error) => from_error(error),
        }
    })
}

/// Solves for one right-hand side.
#[no_mangle]
pub unsafe extern "C" fn sparta_solve(
    session: *mut Session,
    b: *const Complex,
    x: *mut Complex,
) -> Status {
    unsafe { sparta_solve_many(session, b, x, 1) }
}

/// Solves for `count` right-hand sides at once.
#[no_mangle]
pub unsafe extern "C" fn sparta_solve_many(
    session: *mut Session,
    b: *const Complex,
    x: *mut Complex,
    count: i64,
) -> Status {
    guard(|| {
        let session = match unsafe { borrow_mut(session, "sparta_solve") } {
            Ok(session) => session,
            Err(status) => return status,
        };
        if b.is_null() || x.is_null() {
            return fail(Status::Invalid, "sparta_solve needs both a right-hand side and room for the answer");
        }
        if count < 1 {
            return fail(
                Status::Invalid,
                format!("count must be at least 1, not {count}"),
            );
        }
        let wanted = session.rows * count as usize;
        let b = unsafe { std::slice::from_raw_parts(b, wanted) };
        match session.solver.solve_many(b, count as usize) {
            // Copied rather than solved in place, which is what lets `x` be `b`.
            Ok(answer) => {
                unsafe { std::ptr::copy_nonoverlapping(answer.as_ptr(), x, wanted) };
                Status::Ok
            }
            Err(error) => from_error(error),
        }
    })
}

/// Returns the number of rows the solver was built for.
#[no_mangle]
pub unsafe extern "C" fn sparta_solver_rows(session: *const Session) -> i64 {
    guard_value(0, || {
        unsafe { session.as_ref() }.map_or(0, |session| session.rows as i64)
    })
}

/// Returns the number of nonzeros each `values` array must hold.
#[no_mangle]
pub unsafe extern "C" fn sparta_solver_nonzeros(session: *const Session) -> i64 {
    guard_value(0, || {
        unsafe { session.as_ref() }.map_or(0, |session| session.nonzeros as i64)
    })
}

/// Returns how long the GPU spent on the last factorization, in seconds.
#[no_mangle]
pub unsafe extern "C" fn sparta_factor_seconds(session: *const Session) -> f64 {
    guard_value(0.0, || {
        unsafe { session.as_ref() }.map_or(0.0, |s| s.solver.factor_device_seconds())
    })
}

/// Returns how many pivots the last factorization had to nudge away from zero.
#[no_mangle]
pub unsafe extern "C" fn sparta_perturbed_pivots(session: *const Session) -> u64 {
    guard_value(0, || {
        unsafe { session.as_ref() }.map_or(0, |s| s.solver.perturbed_pivots())
    })
}

/// Releases a solver and the GPU memory it holds.
#[no_mangle]
pub unsafe extern "C" fn sparta_solver_free(session: *mut Session) {
    guard_value((), || {
        if !session.is_null() {
            let mut session = unsafe { Box::from_raw(session) };
            // Hands the GPU buffers back now rather than at the end of the
            // drop, so a caller building the next solver does not hold two.
            session.solver.destroy();
        }
    });
}
