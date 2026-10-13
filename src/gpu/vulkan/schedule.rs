//! The dispatch schedule: the Vulkan form of `sparta_v1_run` in
//! `src/metal.m`, which is its specification.
//!
//! The level order, the dispatch sequence within a level, the slice counts,
//! the solve widths and the command split every 64 panel passes are the same
//! as Metal's; the comments there say why each is what it is. What differs is
//! ordering. Metal's serial compute encoder orders every dispatch after the one
//! before it; Vulkan orders nothing, so every dispatch here is followed by a
//! compute-to-compute memory barrier. The schedule is one dependency chain, so
//! no barrier can be dropped without an argument and a validation run.
use super::device::Gpu;
use super::pipelines::Pipelines;
use crate::analysis::{DispatchLevel, StepCounts};
use crate::Result;
use ash::vk;

/// Workgroups `zero_fronts` and `store_factors` aim to put in flight per level
/// (SPARTA_ZERO_GROUPS in metal.m).
const ZERO_GROUPS: u32 = 512;
/// The same target for `assemble_children`, and its slice ceiling.
const CHILD_GROUPS: u32 = 512;
const CHILD_SLICES: u32 = 32;
/// The same for `factor_panel_tail`, and the rows it stages at a time, which
/// is also its width (TAIL_ROWS in the kernel).
const TAIL_GROUPS: u32 = 512;
/// Workgroups `factor_narrow_tail` aims for per level, and its rows per
/// workgroup pass (NARROW_TAIL_ROWS in the kernel). A level of narrow fronts
/// holds hundreds to thousands of them, so the target is higher than the wide
/// tail's; a slice past a front's last row exits at once.
const NARROW_TAIL_GROUPS: u32 = 8192;
const NARROW_TAIL_ROWS: u32 = 32;
const TAIL_SLICES: u32 = 64;
pub(super) const TAIL_ROWS: u32 = 64;
/// The two solve widths, and the level size past which a level stays narrow.
const SOLVE_NARROW: u32 = 256;
const SOLVE_WIDE: u32 = 512;
const SOLVE_CROWD: u32 = 8;
/// Panel passes per command buffer, bounding each submission's run time.
const PASSES_PER_COMMAND: u32 = 64;

/// The fixed facts of one context that the schedule reads.
pub(super) struct Shape<'a> {
    pub levels: &'a [DispatchLevel],
    pub steps: &'a [StepCounts],
    pub lu: bool,
    pub nodes: u32,
    pub n: u32,
    pub schur_batch: u32,
    pub panel_threads: u32,
    pub solve_threads: u32,
    pub pivot_threads: u32,
    /// The widest workgroup the device runs: Metal's `min(1024, pipeline max)`.
    pub wide: u32,
}

/// Timestamps around every dispatch, for `SPARTA_PROFILE` (see `profile`
/// in mod.rs). A mark with no kernel opens a command buffer: it is the
/// baseline for the buffer's first dispatch, so the time between two
/// submissions is billed to no kernel.
pub(super) struct Marks {
    pub pool: vk::QueryPool,
    pub capacity: u32,
    /// `(query, kernel, workgroups)` per mark.
    pub marks: Vec<(u32, Option<&'static str>, u32)>,
}

impl Marks {
    /// Writes the next timestamp at `stage` and remembers what it closes.
    fn mark(
        &mut self,
        gpu: &Gpu,
        command: vk::CommandBuffer,
        stage: vk::PipelineStageFlags,
        kernel: Option<&'static str>,
        groups: u32,
    ) {
        let query = self.marks.len() as u32;
        if query >= self.capacity {
            return;
        }
        // SAFETY: recording into a command buffer of this device; the query
        // was reset at the start of the run.
        unsafe {
            gpu.device
                .cmd_write_timestamp(command, stage, self.pool, query)
        };
        self.marks.push((query, kernel, groups));
    }
}

/// Records dispatches into a growing list of command buffers.
pub(super) struct Recorder<'a> {
    pub gpu: &'a Gpu,
    pub pipelines: &'a mut Pipelines,
    pub pool: vk::CommandPool,
    pub set: vk::DescriptorSet,
    pub commands: Vec<vk::CommandBuffer>,
    pub marks: Option<Marks>,
    bound: vk::Pipeline,
}

impl<'a> Recorder<'a> {
    /// Starts the first command buffer.
    pub fn new(
        gpu: &'a Gpu,
        pipelines: &'a mut Pipelines,
        pool: vk::CommandPool,
        set: vk::DescriptorSet,
        marks: Option<Marks>,
    ) -> Result<Self> {
        let mut recorder = Self {
            gpu,
            pipelines,
            pool,
            set,
            commands: Vec::new(),
            marks,
            bound: vk::Pipeline::null(),
        };
        recorder.begin()?;
        Ok(recorder)
    }

    /// The command buffer being recorded.
    pub fn current(&self) -> vk::CommandBuffer {
        self.commands[self.commands.len() - 1]
    }

    /// Opens a new command buffer and binds the descriptor set to it. Its first
    /// barrier (in `split`) orders it after the buffers submitted before it.
    fn begin(&mut self) -> Result<()> {
        let info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let device = &self.gpu.device;
        // SAFETY: the pool belongs to this device and is reset only after the
        // run's fence has signalled.
        unsafe {
            let command = device
                .allocate_command_buffers(&info)
                .map_err(|e| crate::Error::Device(format!("command buffer: {e}")))?[0];
            device
                .begin_command_buffer(
                    command,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|e| crate::Error::Device(format!("begin command buffer: {e}")))?;
            device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.pipelines.layout,
                0,
                &[self.set],
                &[],
            );
            self.commands.push(command);
            if let Some(marks) = &mut self.marks {
                marks.mark(
                    self.gpu,
                    command,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    None,
                    0,
                );
            }
        }
        self.bound = vk::Pipeline::null();
        Ok(())
    }

    /// Closes the current command buffer and opens the next.
    fn split(&mut self) -> Result<()> {
        self.end()?;
        self.begin()?;
        self.barrier();
        Ok(())
    }

    /// Ends the current command buffer.
    pub fn end(&mut self) -> Result<()> {
        // SAFETY: the buffer is in the recording state.
        unsafe { self.gpu.device.end_command_buffer(self.current()) }
            .map_err(|e| crate::Error::Device(format!("end command buffer: {e}")))
    }

    /// Makes every compute write so far visible to every later compute access.
    fn barrier(&self) {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
        // SAFETY: recording into the current command buffer.
        unsafe {
            self.gpu.device.cmd_pipeline_barrier(
                self.current(),
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
    }

    /// One dispatch of `kernel` at workgroup shape `shape`, `groups` workgroups,
    /// with push constants `range`, followed by the barrier that orders it
    /// before everything recorded after it. Nothing is recorded for zero groups.
    fn dispatch(
        &mut self,
        kernel: &'static str,
        shape: (u32, u32),
        groups: (u32, u32),
        range: [u32; 4],
    ) -> Result<()> {
        if groups.0 == 0 || groups.1 == 0 {
            return Ok(());
        }
        let pipeline = self.pipelines.get(self.gpu, kernel, shape.0, shape.1)?;
        let command = self.current();
        let device = &self.gpu.device;
        let bytes: Vec<u8> = range.iter().flat_map(|v| v.to_ne_bytes()).collect();
        // SAFETY: recording into the current command buffer with objects of
        // this device.
        unsafe {
            if pipeline != self.bound {
                device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, pipeline);
                self.bound = pipeline;
            }
            device.cmd_push_constants(
                command,
                self.pipelines.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &bytes,
            );
            device.cmd_dispatch(command, groups.0, groups.1, 1);
        }
        self.barrier();
        if let Some(marks) = &mut self.marks {
            marks.mark(
                self.gpu,
                command,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                Some(kernel),
                groups.0 * groups.1,
            );
        }
        Ok(())
    }
}

/// `min(ceil(target / fronts), ceil(rows^2 / 1024))`, at least 1: the slice
/// count of `zero_fronts` and `store_factors` for a level (see metal.m).
fn zero_splits(fronts: u32, max_rows: u32) -> u32 {
    let splits = if fronts >= ZERO_GROUPS {
        1
    } else {
        ZERO_GROUPS.div_ceil(fronts)
    };
    let useful = ((u64::from(max_rows) * u64::from(max_rows)).div_ceil(1024) as u32).max(1);
    splits.min(useful)
}

/// Slices per front for a level of `fronts`: `ceil(target / fronts)` capped.
fn capped_splits(fronts: u32, target: u32, cap: u32) -> u32 {
    let splits = if fronts >= target {
        1
    } else {
        target.div_ceil(fronts)
    };
    splits.min(cap)
}

/// Width of the panel head on `level`. LDL^T sweeps only the fully summed
/// block, so its head is as wide as the level's widest one, rounded to 32 and
/// floored at 64; LU spans every row and takes `panel_threads`.
fn head_width(level: &DispatchLevel, s: &Shape) -> u32 {
    if s.lu {
        s.panel_threads
    } else {
        s.panel_threads
            .min(64u32.max(level.max_cols.div_ceil(32) * 32))
    }
    .min(s.wide)
}

/// Solve width of `level`: the wide pipeline for tall fronts on a level too
/// small to fill the device on front count alone (see solve_wide in metal.m).
fn solve_width(level: &DispatchLevel, s: &Shape) -> u32 {
    let fronts = level.small_count + level.narrow_count + level.large_count;
    let wide = if s.solve_threads != 0 {
        s.solve_threads > SOLVE_NARROW
    } else {
        level.max_rows > SOLVE_NARROW && fronts <= SOLVE_CROWD
    };
    (if wide { SOLVE_WIDE } else { SOLVE_NARROW }).min(s.wide)
}

/// Every `(kernel, x, y)` pipeline a factorization and solve of this shape
/// dispatch, so a context builds them all up front as Metal does and a kernel
/// that does not build fails at creation, not mid-factorization.
pub(super) fn pipelines(s: &Shape) -> Vec<(&'static str, u32, u32)> {
    let k = |ldlt: &'static str, lu_name: &'static str| if s.lu { lu_name } else { ldlt };
    let mut out = vec![
        (k("factor_nodes", "factor_nodes_lu"), 32, 1),
        ("zero_fronts", s.wide, 1),
        (k("initialize_large", "initialize_large_lu"), s.wide, 1),
        ("assemble_children", s.wide, 1),
        (k("factor_narrow", "factor_narrow_lu"), s.pivot_threads, 1),
        (k("update_schur", "update_schur_lu"), 16, 16),
        ("advance_panel", 32, 1),
        (k("factor_panel_tail", "factor_panel_tail_lu"), TAIL_ROWS, 1),
        (k("update_strip", "update_strip_lu"), 16, 16),
        (k("store_factors", "store_factors_lu"), s.wide, 1),
    ];
    if !s.lu {
        out.push(("solve_diagonal", 32, 1));
    }
    out.push((
        k("factor_narrow_tail", "factor_narrow_tail_lu"),
        NARROW_TAIL_ROWS,
        1,
    ));
    for level in s.levels {
        out.push((
            k("factor_panel", "factor_panel_lu"),
            head_width(level, s),
            1,
        ));
        let width = solve_width(level, s);
        out.push((k("solve_forward", "solve_forward_lu"), width, 1));
        out.push((k("solve_backward", "solve_backward_lu"), width, 1));
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Records a whole factorization: every level, leaves first.
pub(super) fn factor(r: &mut Recorder, s: &Shape) -> Result<()> {
    let wide = s.wide;
    let k = |ldlt: &'static str, lu_name: &'static str| if s.lu { lu_name } else { ldlt };
    let mut passes = 0u32;
    for level in s.levels {
        let small = level.small_count;
        r.dispatch(
            k("factor_nodes", "factor_nodes_lu"),
            (32, 1),
            (small.div_ceil(32), 1),
            [level.small_start, small, 0, 0],
        )?;
        let fronts = level.narrow_count + level.large_count;
        if fronts > 0 {
            let splits = zero_splits(fronts, level.max_rows);
            r.dispatch(
                "zero_fronts",
                (wide, 1),
                (fronts * splits, 1),
                [level.narrow_start, fronts, splits, 0],
            )?;
        }
        r.dispatch(
            k("initialize_large", "initialize_large_lu"),
            (wide, 1),
            (fronts, 1),
            [level.narrow_start, fronts, 0, 0],
        )?;
        if fronts > 0 {
            let splits = capped_splits(fronts, CHILD_GROUPS, CHILD_SLICES);
            for ordinal in 0..level.max_children {
                r.dispatch(
                    "assemble_children",
                    (wide, 1),
                    (fronts * splits, 1),
                    [level.narrow_start, fronts, splits, ordinal],
                )?;
            }
        }
        r.dispatch(
            k("factor_narrow", "factor_narrow_lu"),
            (s.pivot_threads, 1),
            (level.narrow_count, 1),
            [level.narrow_start, level.narrow_count, 0, 0],
        )?;
        if level.narrow_count > 0 {
            let splits = capped_splits(level.narrow_count, NARROW_TAIL_GROUPS, TAIL_SLICES)
                .min(level.max_rows.div_ceil(NARROW_TAIL_ROWS).max(1));
            r.dispatch(
                k("factor_narrow_tail", "factor_narrow_tail_lu"),
                (NARROW_TAIL_ROWS, 1),
                (level.narrow_count * splits, 1),
                [level.narrow_start, level.narrow_count, splits, 0],
            )?;
        }
        r.dispatch(
            k("update_schur", "update_schur_lu"),
            (16, 16),
            (level.narrow_schur_count, 1),
            [level.narrow_schur_start, level.narrow_schur_count, 0, 0],
        )?;
        r.dispatch(
            "advance_panel",
            (32, 1),
            (level.narrow_count.div_ceil(32), 1),
            [level.narrow_start, level.narrow_count, 0, 0],
        )?;
        for q in 0..level.steps {
            let live = s.steps[(level.step_start + q) as usize];
            if live.nodes == 0 {
                break;
            }
            r.dispatch(
                k("factor_panel", "factor_panel_lu"),
                (head_width(level, s), 1),
                (live.nodes, 1),
                [level.large_start, live.nodes, 0, 0],
            )?;
            let splits = capped_splits(live.nodes, TAIL_GROUPS, TAIL_SLICES);
            r.dispatch(
                k("factor_panel_tail", "factor_panel_tail_lu"),
                (TAIL_ROWS, 1),
                (live.nodes * splits, 1),
                [level.large_start, live.nodes, splits, 0],
            )?;
            r.dispatch(
                k("update_strip", "update_strip_lu"),
                (16, 16),
                (live.tiles, 1),
                [level.tile_start + live.tile_start, live.tiles, 0, 0],
            )?;
            if s.schur_batch != 0 && (q + 1) % s.schur_batch == 0 {
                r.dispatch(
                    k("update_schur", "update_schur_lu"),
                    (16, 16),
                    (live.schur, 1),
                    [level.schur_start, live.schur, 0, 0],
                )?;
                r.dispatch(
                    "advance_panel",
                    (32, 1),
                    (live.nodes.div_ceil(32), 1),
                    [level.large_start, live.nodes, 0, 0],
                )?;
            }
            passes += 1;
            if passes == PASSES_PER_COMMAND {
                r.split()?;
                passes = 0;
            }
        }
        // The mandatory final drain over every Schur tile of the level.
        r.dispatch(
            k("update_schur", "update_schur_lu"),
            (16, 16),
            (level.schur_count, 1),
            [level.schur_start, level.schur_count, 0, 0],
        )?;
        r.dispatch(
            "advance_panel",
            (32, 1),
            (level.large_count.div_ceil(32), 1),
            [level.large_start, level.large_count, 0, 0],
        )?;
        let fronts = level.small_count + level.narrow_count + level.large_count;
        if fronts > 0 {
            let splits = zero_splits(fronts, level.max_rows);
            r.dispatch(
                k("store_factors", "store_factors_lu"),
                (wide, 1),
                (fronts * splits, 1),
                [level.small_start, fronts, splits, 0],
            )?;
        }
    }
    Ok(())
}

/// Records a solve of `nrhs` right-hand sides: forward leaves first, the block
/// diagonal on LDL^T, backward roots first.
pub(super) fn solve(r: &mut Recorder, s: &Shape, nrhs: u32) -> Result<()> {
    let (forward, backward) = if s.lu {
        ("solve_forward_lu", "solve_backward_lu")
    } else {
        ("solve_forward", "solve_backward")
    };
    for level in s.levels {
        let fronts = level.small_count + level.narrow_count + level.large_count;
        r.dispatch(
            forward,
            (solve_width(level, s), 1),
            (fronts, nrhs),
            [level.small_start, fronts, s.n, nrhs],
        )?;
    }
    if !s.lu {
        r.dispatch(
            "solve_diagonal",
            (32, 1),
            (s.nodes, nrhs),
            [0, s.nodes, s.n, nrhs],
        )?;
    }
    for level in s.levels.iter().rev() {
        let fronts = level.small_count + level.narrow_count + level.large_count;
        r.dispatch(
            backward,
            (solve_width(level, s), 1),
            (fronts, nrhs),
            [level.small_start, fronts, s.n, nrhs],
        )?;
    }
    Ok(())
}
