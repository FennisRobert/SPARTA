//! Steps 2 and 3 of the C API: factoring on the GPU and solving with the result.

use crate::analysis::Plan;
use crate::status::{fail, from_error, guard, guard_value, Status};
use sparta::{Complex32, Scalar, Solver};

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

/// Factors `values` of any value type the GPU takes; `what` names the entry
/// point in error messages.
///
/// # Safety
/// `session` must be null or live, and `values` null or `nonzeros` long.
unsafe fn factor<T: Scalar>(session: *mut Session, values: *const T, what: &str) -> Status {
    guard(|| {
        let session = match unsafe { borrow_mut(session, what) } {
            Ok(session) => session,
            Err(status) => return status,
        };
        if values.is_null() {
            return fail(Status::Invalid, format!("{what} needs a values array"));
        }
        let values = unsafe { std::slice::from_raw_parts(values, session.nonzeros) };
        match session.solver.factor_values(values) {
            Ok(()) => Status::Ok,
            Err(error) => from_error(error),
        }
    })
}

/// Solves `count` right-hand sides of any value type; `what` names the entry
/// point in error messages.
///
/// # Safety
/// `session` must be null or live; `b` and `x` null or `rows * count` long.
unsafe fn solve_many<T: Scalar>(session: *mut Session, b: *const T, x: *mut T, count: i64, what: &str) -> Status {
    guard(|| {
        let session = match unsafe { borrow_mut(session, what) } {
            Ok(session) => session,
            Err(status) => return status,
        };
        if b.is_null() || x.is_null() {
            return fail(Status::Invalid, format!("{what} needs both a right-hand side and room for the answer"));
        }
        if count < 1 {
            return fail(
                Status::Invalid,
                format!("count must be at least 1, not {count}"),
            );
        }
        let wanted = session.rows * count as usize;
        let b = unsafe { std::slice::from_raw_parts(b, wanted) };
        match session.solver.solve_values(b, count as usize) {
            // Copied rather than solved in place, which is what lets `x` be `b`.
            Ok(answer) => {
                unsafe { std::ptr::copy_nonoverlapping(answer.as_ptr(), x, wanted) };
                Status::Ok
            }
            Err(error) => from_error(error),
        }
    })
}

/// Factors a complex matrix on the GPU.
#[no_mangle]
pub unsafe extern "C" fn sparta_factor(session: *mut Session, values: *const Complex) -> Status {
    unsafe { factor(session, values, "sparta_factor") }
}

/// Factors a real matrix on the GPU, with real arithmetic.
#[no_mangle]
pub unsafe extern "C" fn sparta_factor_real(session: *mut Session, values: *const f32) -> Status {
    unsafe { factor(session, values, "sparta_factor_real") }
}

/// Solves for one complex right-hand side.
#[no_mangle]
pub unsafe extern "C" fn sparta_solve(
    session: *mut Session,
    b: *const Complex,
    x: *mut Complex,
) -> Status {
    unsafe { solve_many(session, b, x, 1, "sparta_solve") }
}

/// Solves for `count` complex right-hand sides at once.
#[no_mangle]
pub unsafe extern "C" fn sparta_solve_many(
    session: *mut Session,
    b: *const Complex,
    x: *mut Complex,
    count: i64,
) -> Status {
    unsafe { solve_many(session, b, x, count, "sparta_solve_many") }
}

/// Solves for one real right-hand side against real factors.
#[no_mangle]
pub unsafe extern "C" fn sparta_solve_real(session: *mut Session, b: *const f32, x: *mut f32) -> Status {
    unsafe { solve_many(session, b, x, 1, "sparta_solve_real") }
}

/// Solves for `count` real right-hand sides at once against real factors.
#[no_mangle]
pub unsafe extern "C" fn sparta_solve_many_real(
    session: *mut Session,
    b: *const f32,
    x: *mut f32,
    count: i64,
) -> Status {
    unsafe { solve_many(session, b, x, count, "sparta_solve_many_real") }
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
