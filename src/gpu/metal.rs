//! Apple Metal backend: the Objective-C bridge in `src/metal.m` and the
//! kernels in `src/kernels/*.metal`.
//!
//! One of the three backend modules behind `gpu::backend`; see `mod.rs` for
//! the interface all three implement.
use super::{Analysis, Complex32, Error, Result};
use std::{ffi::c_void, ptr::NonNull};
extern "C" {
    fn sparta_v1_available() -> i32;
    fn sparta_v1_create(
        data: *const *const c_void,
        lengths: *const usize,
        levels: *const u32,
        level_count: u32,
        steps: *const u32,
        step_count: u32,
        nodes: u32,
        n: u32,
        lu: u32,
        schur_batch: u32,
        panel_threads: u32,
        solve_threads: u32,
        lu_diag_threshold: f64,
        geometry: *const u32,
        error: *mut u8,
        cap: usize,
    ) -> *mut c_void;
    fn sparta_v1_destroy(context: *mut c_void);
    fn sparta_v1_set_factor_plan(
        context: *mut c_void,
        ids: *const c_void,
        ids_bytes: usize,
        tiles: *const c_void,
        tiles_bytes: usize,
        schur: *const c_void,
        schur_bytes: usize,
        levels: *const u32,
        level_count: u32,
        steps: *const u32,
        step_count: u32,
        error: *mut u8,
        cap: usize,
    ) -> i32;
    fn sparta_v1_buffer(context: *mut c_void, index: u32, length: *mut usize) -> *mut c_void;
    fn sparta_v1_factor_start(
        context: *mut c_void,
        input: *const c_void,
        first: u32,
        last: u32,
        error: *mut u8,
        cap: usize,
    ) -> i32;
    fn sparta_v1_factor_finish(
        context: *mut c_void,
        status: *mut u32,
        perturbed: *mut u64,
        seconds: *mut f64,
        error: *mut u8,
        cap: usize,
    ) -> i32;
    fn sparta_v1_pivot_range(context: *mut c_void, lo: *mut f64, hi: *mut f64) -> i32;
    fn sparta_v1_run(
        context: *mut c_void,
        factoring: i32,
        input: *const c_void,
        output: *mut c_void,
        nrhs: u32,
        status: *mut u32,
        perturbed: *mut u64,
        seconds: *mut f64,
        error: *mut u8,
        cap: usize,
    ) -> i32;
}
pub(crate) fn available() -> bool {
    unsafe { sparta_v1_available() != 0 }
}
fn message(bytes: &[u8]) -> Error {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    Error::Device(String::from_utf8_lossy(&bytes[..end]).into_owned())
}
// Deliberately !Send/!Sync. All mutations require exclusive Rust ownership;
// the Python class is likewise thread-affine. No borrowed input survives FFI.
pub(crate) struct Context(Option<NonNull<c_void>>);
impl Context {
    pub(super) fn new(plan: &Analysis) -> Result<Self> {
        let dispatch = plan.dispatch_plan();
        let mut data = [std::ptr::null::<c_void>(); 21];
        let mut lengths = [0usize; 21];
        macro_rules! buffer {
            ($i:expr, $slice:expr) => {{
                data[$i] = $slice.as_ptr().cast();
                lengths[$i] = std::mem::size_of_val($slice.as_slice());
            }};
        }
        buffer!(0, plan.nodes);
        buffer!(1, plan.rows);
        buffer!(2, plan.children);
        buffer!(3, plan.maps);
        buffer!(4, plan.entries);
        buffer!(5, dispatch.ids);
        buffer!(16, dispatch.tiles);
        buffer!(18, dispatch.schur);
        lengths[15] = plan.nodes.len() * std::mem::size_of::<crate::analysis::FrontState>();
        lengths[6] = plan.nnz * 8;
        lengths[7] = plan.elements * 8;
        lengths[8] = plan.n * 8;
        lengths[9] = plan.n * 8;
        lengths[10] = plan.n * 4;
        lengths[11] = plan.n * 4;
        lengths[12] = plan.nodes.len() * 4;
        lengths[13] = plan.n * 8;
        lengths[17] = plan.factor_elements * 8;
        lengths[19] = plan.panel_elements * 8;
        // The LU U12 store. Zero on the LDL^T path, where the bridge binds
        // the four-byte stub every unused buffer gets.
        lengths[20] = plan.upper_elements * 8;
        // Sizes are bounded by the device's own reported limits inside
        // create; only host-side arithmetic overflow is rejected here.
        lengths
            .iter()
            .try_fold(0usize, |a, &b| a.checked_add(b.max(4)))
            .ok_or_else(|| Error::Unsupported("buffer size overflow".into()))?;
        let mut error = [0u8; 4096];
        // SAFETY: all descriptor arrays are validated by Analysis and copied by
        // create before return; size/alignment match the repr(C) definitions.
        // Order matches the macro table in `sparta_v1_create`.
        let geometry = [
            plan.config.panel,
            plan.config.schur_block,
            plan.config.strip_block,
            plan.config.pivot_threads,
            plan.config.schur_chunk,
        ];
        let raw = unsafe {
            sparta_v1_create(
                data.as_ptr(),
                lengths.as_ptr(),
                dispatch.levels.as_ptr().cast(),
                dispatch.levels.len() as u32,
                dispatch.steps.as_ptr().cast(),
                dispatch.steps.len() as u32,
                plan.nodes.len() as u32,
                plan.n as u32,
                u32::from(plan.kind == crate::Kind::Lu),
                plan.config.schur_batch,
                plan.config.panel_threads,
                plan.config.solve_threads,
                plan.config.lu_diag_threshold,
                geometry.as_ptr(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        NonNull::new(raw)
            .map(|handle| Self(Some(handle)))
            .ok_or_else(|| message(&error))
    }
    pub(super) fn run(
        &mut self,
        values: &[Complex32],
        output: Option<&mut [Complex32]>,
        nrhs: u32,
        status: &mut [u32],
        perturbed: &mut u64,
    ) -> Result<f64> {
        let factoring = output.is_none();
        let out = output.map_or(std::ptr::null_mut(), |s| s.as_mut_ptr().cast());
        let mut seconds = 0.0;
        let mut error = [0u8; 4096];
        // SAFETY: Solver checks input/output sizes against its immutable plan;
        // run completes the command before these slices or context are reused.
        let handle = self
            .0
            .ok_or_else(|| Error::Invalid("this solver's GPU storage was released".into()))?;
        let ok = unsafe {
            sparta_v1_run(
                handle.as_ptr(),
                i32::from(factoring),
                values.as_ptr().cast(),
                out,
                nrhs,
                status.as_mut_ptr(),
                perturbed,
                &mut seconds,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if ok == 0 {
            Err(message(&error))
        } else {
            Ok(seconds)
        }
    }
}
/// The hybrid factorization's side of the bridge: see `hybrid.rs`.
impl Context {
    fn handle(&self) -> Result<NonNull<c_void>> {
        self.0
            .ok_or_else(|| Error::Invalid("this solver's GPU storage was released".into()))
    }
    /// Install the dispatch tables of the GPU's share of a hybrid factorization.
    pub(super) fn set_factor_plan(&mut self, plan: &crate::analysis::DispatchPlan) -> Result<()> {
        let handle = self.handle()?;
        let mut error = [0u8; 4096];
        // SAFETY: the slices outlive the call, which copies them.
        let ok = unsafe {
            sparta_v1_set_factor_plan(
                handle.as_ptr(),
                plan.ids.as_ptr().cast(),
                std::mem::size_of_val(plan.ids.as_slice()),
                plan.tiles.as_ptr().cast(),
                std::mem::size_of_val(plan.tiles.as_slice()),
                plan.schur.as_ptr().cast(),
                std::mem::size_of_val(plan.schur.as_slice()),
                plan.levels.as_ptr().cast(),
                plan.levels.len() as u32,
                plan.steps.as_ptr().cast(),
                plan.steps.len() as u32,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if ok == 0 { Err(message(&error)) } else { Ok(()) }
    }
    /// Host view of device buffer `index` as elements of `T`.
    ///
    /// Distinct indices are distinct allocations, so views of different
    /// buffers may be held at once.
    ///
    /// # Safety
    /// No command in flight may touch the elements the caller writes, `T`
    /// must match the buffer's element type, and no two live views may be of
    /// the same buffer.
    #[allow(clippy::mut_from_ref)]
    pub(super) unsafe fn buffer<T>(&self, index: u32) -> Result<&mut [T]> {
        let handle = self.handle()?;
        let mut length = 0usize;
        let data = sparta_v1_buffer(handle.as_ptr(), index, &mut length);
        if data.is_null() {
            return Err(Error::Invalid(format!("no device buffer {index}")));
        }
        Ok(std::slice::from_raw_parts_mut(data.cast(), length / std::mem::size_of::<T>()))
    }
    /// Commit levels `[first, last)` of the GPU's share without waiting;
    /// `values` starts a new factorization.
    pub(super) fn factor_start(&mut self, values: Option<&[Complex32]>, first: u32, last: u32) -> Result<()> {
        let handle = self.handle()?;
        let mut error = [0u8; 4096];
        let input = values.map_or(std::ptr::null(), |v| v.as_ptr().cast());
        // SAFETY: `values` is copied before the call returns.
        let ok = unsafe {
            sparta_v1_factor_start(handle.as_ptr(), input, first, last, error.as_mut_ptr(), error.len())
        };
        if ok == 0 { Err(message(&error)) } else { Ok(()) }
    }
    /// Wait for the levels started so far; returns their device seconds.
    pub(super) fn factor_finish(&mut self, status: &mut [u32], perturbed: &mut u64) -> Result<f64> {
        let handle = self.handle()?;
        let (mut error, mut seconds) = ([0u8; 4096], 0.0);
        // SAFETY: `status` holds one word per node, as the context requires.
        let ok = unsafe {
            sparta_v1_factor_finish(
                handle.as_ptr(),
                status.as_mut_ptr(),
                perturbed,
                &mut seconds,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if ok == 0 { Err(message(&error)) } else { Ok(seconds) }
    }
}
impl Context {
    /// Smallest and largest pivot magnitude, or None when unavailable.
    pub(super) fn pivot_range(&self) -> Option<(f64, f64)> {
        let handle = self.0?;
        let (mut lo, mut hi) = (0.0, 0.0);
        // SAFETY: the handle is live, and the scan only reads buffers the
        // context owns. Returns 0 without writing when it has nothing.
        let ok = unsafe { sparta_v1_pivot_range(handle.as_ptr(), &mut lo, &mut hi) };
        (ok != 0).then_some((lo, hi))
    }
    /// Release the Metal context and every buffer it owns, now.
    ///
    /// Idempotent: the handle is taken, so a later drop does nothing.
    pub(super) fn release(&mut self) {
        if let Some(handle) = self.0.take() {
            unsafe { sparta_v1_destroy(handle.as_ptr()) };
        }
    }
}
impl Drop for Context {
    fn drop(&mut self) {
        self.release();
    }
}
