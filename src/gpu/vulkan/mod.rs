//! Vulkan compute backend, for AMD and NVIDIA.
//!
//! One of the three backend modules behind `gpu::backend`; see `gpu/mod.rs`
//! for the interface all three implement. `src/metal.m` is the specification
//! for this module and `src/kernels/*.metal` for the kernels in
//! `src/kernels/vulkan/`, whose SPIR-V `build.rs` embeds.
//!
//! - `device`: the instance, the device and its buffers.
//! - `pipelines`: one pipeline per kernel and workgroup shape.
//! - `schedule`: the dispatch sequence of a factorization and a solve.
//!
//! This file owns the buffers of one plan and moves data across the bus. Every
//! plan buffer is device-local; the host reaches them through two staging
//! buffers, one for what goes up and one for what comes back, and every run is
//! one queue submission that ends in those copies. A discrete GPU does not
//! share memory with the host as Apple silicon does, so the pivot data
//! `pivot_range` scans is copied back at the end of each factorization, while
//! it is small, rather than on request.
//!
//! `SPARTA_VULKAN_DEVICE` picks the device when several qualify: an index, or
//! a part of its name. Without it the first discrete GPU is used.
mod device;
mod pipelines;
mod schedule;

use super::{Analysis, Complex32, Error, Result};
use ash::vk;
use device::{Buffer, Gpu};
use pipelines::{Geometry, Pipelines, BINDINGS};

/// Whether a usable Vulkan compute device is visible to this process.
pub(crate) fn available() -> bool {
    static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVAILABLE.get_or_init(device::any_device)
}

/// Buffer slots of a plan, in Metal's numbering (slot 14 is the push-constant
/// range and has no buffer).
const SLOTS: usize = 21;

/// Everything one plan owns on the device. Deliberately !Send/!Sync, as the
/// Metal context is.
struct Inner {
    gpu: Gpu,
    pipelines: Pipelines,
    buffers: Vec<Option<Buffer>>,
    descriptors: vk::DescriptorPool,
    set: vk::DescriptorSet,
    commands: vk::CommandPool,
    fence: vk::Fence,
    timestamps: vk::QueryPool,
    upload: Staging,
    download: Staging,
    levels: Vec<crate::analysis::DispatchLevel>,
    steps: Vec<crate::analysis::StepCounts>,
    lu: bool,
    nodes: u32,
    n: u32,
    nnz: usize,
    schur_batch: u32,
    panel_threads: u32,
    solve_threads: u32,
    pivot_threads: u32,
    wide: u32,
    rhs_capacity: u32,
    /// `(loffset, nr, nc)` per node: where the LU pivots sit in the factor
    /// store. The plan's own node table is dropped once the solver exists.
    pivot_sites: Vec<(u32, u32, u32)>,
    /// The latest factorization's pivot range, scanned when it finished.
    pivots: Option<(f64, f64)>,
}

/// A host-visible buffer, mapped for the context's lifetime.
struct Staging {
    buffer: Option<Buffer>,
    mapped: *mut u8,
}

pub(crate) struct Context(Option<Box<Inner>>);

/// The bytes of a slice of plain `repr(C)` records.
fn bytes<T: Copy>(values: &[T]) -> &[u8] {
    // SAFETY: T is a plain-old-data record (u32 fields or complex floats), so
    // every byte of the slice is initialized.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

impl Context {
    /// Allocates the plan's buffers on the device and copies its descriptors
    /// there.
    pub(super) fn new(plan: &Analysis) -> Result<Self> {
        let gpu = Gpu::new()?;
        let dispatch = plan.dispatch_plan();
        let config = &plan.config;
        let limits = &gpu.limits;
        let wide = 1024
            .min(limits.max_compute_work_group_invocations)
            .min(limits.max_compute_work_group_size[0]);
        let geometry = Geometry {
            panel: config.panel,
            schur_block: config.schur_block,
            strip_block: config.strip_block,
            pivot_threads: config.pivot_threads,
            schur_chunk: config.schur_chunk,
            tail_rows: schedule::TAIL_ROWS,
            lu_diag_threshold: config.lu_diag_threshold as f32,
            max_simds: wide.div_ceil(gpu.min_subgroup),
            narrow_cols: config.narrow_cols,
        };
        if !(0.0..=1.0).contains(&config.lu_diag_threshold) {
            return Err(Error::Invalid(
                "lu_diag_threshold must be between 0 and 1".into(),
            ));
        }
        let pipelines = Pipelines::new(&gpu, geometry)?;
        let mut inner = Box::new(Inner {
            gpu,
            pipelines,
            buffers: (0..SLOTS).map(|_| None).collect(),
            descriptors: vk::DescriptorPool::null(),
            set: vk::DescriptorSet::null(),
            commands: vk::CommandPool::null(),
            fence: vk::Fence::null(),
            timestamps: vk::QueryPool::null(),
            upload: Staging {
                buffer: None,
                mapped: std::ptr::null_mut(),
            },
            download: Staging {
                buffer: None,
                mapped: std::ptr::null_mut(),
            },
            levels: dispatch.levels.clone(),
            steps: dispatch.steps.clone(),
            lu: plan.kind == crate::Kind::Lu,
            nodes: plan.nodes.len() as u32,
            n: plan.n as u32,
            nnz: plan.nnz,
            schur_batch: config.schur_batch,
            panel_threads: if config.panel_threads != 0 {
                config.panel_threads
            } else {
                512
            },
            solve_threads: config.solve_threads,
            pivot_threads: config.pivot_threads,
            wide,
            rhs_capacity: 1,
            pivot_sites: plan.nodes.iter().map(|n| (n.loffset, n.nr, n.nc)).collect(),
            pivots: None,
        });
        // From here on a failure drops `inner`, which releases whatever it holds.
        inner.create(plan, &dispatch)?;
        for (kernel, x, y) in schedule::pipelines(&inner.shape()) {
            inner.pipelines.get(&inner.gpu, kernel, x, y)?;
        }
        Ok(Self(Some(inner)))
    }

    pub(super) fn run(
        &mut self,
        values: &[Complex32],
        output: Option<&mut [Complex32]>,
        nrhs: u32,
        status: &mut [u32],
        perturbed: &mut u64,
    ) -> Result<f64> {
        let inner = self
            .0
            .as_mut()
            .ok_or_else(|| Error::Invalid("this solver's GPU storage was released".into()))?;
        inner.run(values, output, nrhs, status, perturbed)
    }

    /// Smallest and largest pivot magnitude of the latest factorization.
    pub(super) fn pivot_range(&self) -> Option<(f64, f64)> {
        self.0.as_ref()?.pivots
    }

    /// Releases the device and every buffer, now. Idempotent.
    pub(super) fn release(&mut self) {
        self.0.take();
    }
}

impl Inner {
    /// Creates the buffers, uploads the descriptors, and creates the
    /// descriptor set, command pool, fence and timestamp queries.
    fn create(&mut self, plan: &Analysis, dispatch: &crate::analysis::DispatchPlan) -> Result<()> {
        let node_state = std::mem::size_of::<crate::analysis::FrontState>();
        let mut data: [Option<&[u8]>; SLOTS] = [None; SLOTS];
        data[0] = Some(bytes(&plan.nodes));
        data[1] = Some(bytes(&plan.rows));
        data[2] = Some(bytes(&plan.children));
        data[3] = Some(bytes(&plan.maps));
        data[4] = Some(bytes(&plan.entries));
        data[5] = Some(bytes(&dispatch.ids));
        data[16] = Some(bytes(&dispatch.tiles));
        data[18] = Some(bytes(&dispatch.schur));
        let mut lengths = [0u64; SLOTS];
        for (length, data) in lengths.iter_mut().zip(&data) {
            *length = data.map_or(0, |d| d.len() as u64);
        }
        let n = plan.n as u64;
        lengths[6] = plan.nnz as u64 * 8;
        lengths[7] = plan.elements as u64 * 8;
        lengths[8] = n * 8;
        lengths[9] = n * 8;
        lengths[10] = n * 4;
        lengths[11] = n * 4;
        lengths[12] = u64::from(self.nodes) * 4;
        lengths[13] = n * 8;
        lengths[15] = u64::from(self.nodes) * node_state as u64;
        lengths[17] = plan.factor_elements as u64 * 8;
        lengths[19] = plan.panel_elements as u64 * 8;
        lengths[20] = plan.upper_elements as u64 * 8;
        let total: u64 = lengths.iter().map(|&l| l.max(4)).sum();
        let local = self.gpu.local_bytes();
        if local > 0 && total > local {
            return Err(Error::Unsupported(format!(
                "factorization needs {:.2} GiB of GPU buffers, {} has {:.2} GiB",
                total as f64 / device::GIB,
                self.gpu.name,
                local as f64 / device::GIB
            )));
        }
        for &slot in BINDINGS.iter() {
            let what = format!("buffer {slot}");
            self.buffers[slot as usize] = Some(self.gpu.buffer(
                lengths[slot as usize],
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                &what,
            )?);
        }
        self.create_objects()?;
        self.write_descriptors();
        // Upload the descriptor arrays and clear every other buffer, in one
        // submission: Vulkan memory starts undefined, where Metal's does not.
        let largest = data
            .iter()
            .flatten()
            .map(|d| d.len() as u64)
            .max()
            .unwrap_or(0);
        self.stage(true, largest)?;
        let pool = self.commands;
        let command = self.one_shot(pool)?;
        for slot in BINDINGS.iter().map(|&b| b as usize) {
            // SAFETY: recording into a fresh command buffer.
            unsafe {
                self.gpu
                    .device
                    .cmd_fill_buffer(command, self.buffer(slot), 0, vk::WHOLE_SIZE, 0)
            };
        }
        self.submit(&[command])?;
        // One descriptor array at a time through the upload staging buffer.
        for (slot, bytes) in data.iter().enumerate() {
            let Some(bytes) = *bytes else { continue };
            if bytes.is_empty() {
                continue;
            }
            // SAFETY: the staging buffer holds at least `largest` bytes and no
            // command reads it (the previous submission has completed).
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.upload.mapped, bytes.len())
            };
            let command = self.one_shot(pool)?;
            let region = vk::BufferCopy::default().size(bytes.len() as u64);
            let (src, dst) = (self.upload_buffer(), self.buffer(slot));
            // SAFETY: recording a copy between two live buffers of this device.
            unsafe {
                self.gpu
                    .device
                    .cmd_copy_buffer(command, src, dst, &[region])
            };
            self.submit(&[command])?;
        }
        Ok(())
    }

    /// The descriptor pool and set, the command pool, the fence and the
    /// timestamp query pool.
    fn create_objects(&mut self) -> Result<()> {
        let device = &self.gpu.device;
        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(BINDINGS.len() as u32)];
        // SAFETY: every create info outlives its call; the objects are
        // destroyed in `Drop`.
        unsafe {
            self.descriptors = device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(&sizes),
                    None,
                )
                .map_err(|e| Error::Device(format!("descriptor pool: {e}")))?;
            let layouts = [self.pipelines.set_layout];
            self.set = device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(self.descriptors)
                        .set_layouts(&layouts),
                )
                .map_err(|e| Error::Device(format!("descriptor set: {e}")))?[0];
            self.commands = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(self.gpu.queue_family)
                        .flags(vk::CommandPoolCreateFlags::TRANSIENT),
                    None,
                )
                .map_err(|e| Error::Device(format!("command pool: {e}")))?;
            self.fence = device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(|e| Error::Device(format!("fence: {e}")))?;
            self.timestamps = device
                .create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(2),
                    None,
                )
                .map_err(|e| Error::Device(format!("timestamp queries: {e}")))?;
        }
        Ok(())
    }

    /// Points every binding of the set at its buffer.
    fn write_descriptors(&self) {
        let infos: Vec<[vk::DescriptorBufferInfo; 1]> = BINDINGS
            .iter()
            .map(|&b| {
                [vk::DescriptorBufferInfo::default()
                    .buffer(self.buffer(b as usize))
                    .offset(0)
                    .range(vk::WHOLE_SIZE)]
            })
            .collect();
        let writes: Vec<_> = BINDINGS
            .iter()
            .zip(&infos)
            .map(|(&b, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(self.set)
                    .dst_binding(b)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(info)
            })
            .collect();
        // SAFETY: the set is not in use by any pending command.
        unsafe { self.gpu.device.update_descriptor_sets(&writes, &[]) };
    }

    fn buffer(&self, slot: usize) -> vk::Buffer {
        self.buffers[slot]
            .as_ref()
            .map_or(vk::Buffer::null(), |b| b.buffer)
    }
    fn upload_buffer(&self) -> vk::Buffer {
        self.upload
            .buffer
            .as_ref()
            .map_or(vk::Buffer::null(), |b| b.buffer)
    }
    fn download_buffer(&self) -> vk::Buffer {
        self.download
            .buffer
            .as_ref()
            .map_or(vk::Buffer::null(), |b| b.buffer)
    }

    /// Grows the upload (`up`) or download staging buffer to `bytes`. The
    /// download buffer prefers host-cached memory, which the host reads fast.
    fn stage(&mut self, up: bool, bytes: u64) -> Result<()> {
        let staging = if up {
            &mut self.upload
        } else {
            &mut self.download
        };
        if staging.buffer.as_ref().is_some_and(|b| b.size >= bytes) {
            return Ok(());
        }
        if let Some(old) = staging.buffer.take() {
            self.gpu.free(&old);
        }
        let flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let cached = flags | vk::MemoryPropertyFlags::HOST_CACHED;
        let buffer = match up {
            false => self
                .gpu
                .buffer(bytes, cached, "staging buffer")
                .or_else(|_| self.gpu.buffer(bytes, flags, "staging buffer"))?,
            true => self.gpu.buffer(bytes, flags, "staging buffer")?,
        };
        // SAFETY: host-visible memory bound to this buffer, mapped whole.
        let mapped = unsafe {
            self.gpu.device.map_memory(
                buffer.memory,
                0,
                vk::WHOLE_SIZE,
                vk::MemoryMapFlags::empty(),
            )
        }
        .map_err(|e| Error::Device(format!("mapping a staging buffer: {e}")))?;
        let staging = if up {
            &mut self.upload
        } else {
            &mut self.download
        };
        staging.buffer = Some(buffer);
        staging.mapped = mapped.cast();
        Ok(())
    }

    /// A command buffer from `pool`, begun for one submission.
    fn one_shot(&self, pool: vk::CommandPool) -> Result<vk::CommandBuffer> {
        let info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: the pool belongs to this device.
        unsafe {
            let command = self
                .gpu
                .device
                .allocate_command_buffers(&info)
                .map_err(|e| Error::Device(format!("command buffer: {e}")))?[0];
            self.gpu
                .device
                .begin_command_buffer(
                    command,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|e| Error::Device(format!("begin command buffer: {e}")))?;
            Ok(command)
        }
    }

    /// A barrier between transfer and compute work, or transfer and host.
    fn transfer_barrier(
        &self,
        command: vk::CommandBuffer,
        from: vk::AccessFlags,
        to: vk::AccessFlags,
    ) {
        let stage = |access: vk::AccessFlags| {
            if access.intersects(vk::AccessFlags::HOST_READ | vk::AccessFlags::HOST_WRITE) {
                vk::PipelineStageFlags::HOST
            } else if access
                .intersects(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            {
                vk::PipelineStageFlags::COMPUTE_SHADER
            } else {
                vk::PipelineStageFlags::TRANSFER
            }
        };
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(from)
            .dst_access_mask(to);
        // SAFETY: recording into a command buffer of this device.
        unsafe {
            self.gpu.device.cmd_pipeline_barrier(
                command,
                stage(from),
                stage(to),
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
    }

    /// Ends and submits `commands` (ending the last one), waits for them, and
    /// frees every command buffer of the pool.
    fn submit(&self, commands: &[vk::CommandBuffer]) -> Result<()> {
        let device = &self.gpu.device;
        // SAFETY: the buffers were recorded on this device's pool; the fence is
        // unsignalled; the pool is reset only once the fence has signalled.
        unsafe {
            if let Some(&last) = commands.last() {
                device
                    .end_command_buffer(last)
                    .map_err(|e| Error::Device(format!("end command buffer: {e}")))?;
            }
            // One submission per command buffer, as Metal commits each of its
            // command buffers: a single submission of a whole factorization
            // (seconds of work on an integrated GPU) stalled the AMD driver,
            // with the CPU spinning and the GPU idle. Queue order and the
            // barrier each buffer opens with keep them in sequence.
            for (i, command) in commands.iter().enumerate() {
                let one = [*command];
                let submit = [vk::SubmitInfo::default().command_buffers(&one)];
                let fence = if i + 1 == commands.len() {
                    self.fence
                } else {
                    vk::Fence::null()
                };
                device
                    .queue_submit(self.gpu.queue, &submit, fence)
                    .map_err(|e| Error::Device(format!("queue submit: {e}")))?;
            }
            let waited = device.wait_for_fences(&[self.fence], true, u64::MAX);
            device
                .reset_fences(&[self.fence])
                .map_err(|e| Error::Device(format!("fence reset: {e}")))?;
            device
                .reset_command_pool(self.commands, vk::CommandPoolResetFlags::empty())
                .map_err(|e| Error::Device(format!("command pool reset: {e}")))?;
            waited.map_err(|e| Error::Device(format!("GPU execution failed: {e}")))
        }
    }

    /// One factorization (`output` is None) or one solve of `nrhs` columns.
    fn run(
        &mut self,
        values: &[Complex32],
        output: Option<&mut [Complex32]>,
        nrhs: u32,
        status: &mut [u32],
        perturbed: &mut u64,
    ) -> Result<f64> {
        let factoring = output.is_none();
        let nrhs = if factoring { 1 } else { nrhs };
        if nrhs == 0 {
            return Err(Error::Invalid(
                "at least one right-hand side required".into(),
            ));
        }
        if !factoring && nrhs > self.rhs_capacity {
            self.grow_rhs(nrhs)?;
        }
        let (n, nodes) = (u64::from(self.n), u64::from(self.nodes));
        let up_bytes = if factoring {
            self.nnz as u64 * 8
        } else {
            n * u64::from(nrhs) * 8
        };
        self.stage(true, up_bytes)?;
        // Readback layout: status, front states, then the pivot data.
        let state_bytes = nodes * std::mem::size_of::<crate::analysis::FrontState>() as u64;
        let down_bytes = if factoring {
            nodes * 4 + state_bytes + if self.lu { n * 8 } else { n * 20 }
        } else {
            n * u64::from(nrhs) * 8
        };
        self.stage(false, down_bytes)?;
        let input = bytes(values);
        // SAFETY: the upload buffer holds `up_bytes` and no pending command
        // reads it; `input` is exactly that long, as Solver checked.
        unsafe {
            std::ptr::copy_nonoverlapping(
                input.as_ptr(),
                self.upload.mapped,
                input.len().min(up_bytes as usize),
            )
        };
        let marks = if crate::profile::enabled() { Some(self.marks()?) } else { None };
        let (commands, marks) = self.record(factoring, nrhs, up_bytes, state_bytes, marks)?;
        let submitted = self.submit(&commands);
        if let Some(marks) = marks {
            if submitted.is_ok() {
                self.report(&marks);
            }
            // SAFETY: the run has completed; the pool is used by no command.
            unsafe { self.gpu.device.destroy_query_pool(marks.pool, None) };
        }
        submitted?;
        let seconds = self.device_seconds()?;
        let down = self.download.mapped;
        if factoring {
            // SAFETY: the download buffer was filled by the completed run.
            unsafe {
                let words = std::slice::from_raw_parts(down.cast::<u32>(), self.nodes as usize);
                let count = status.len().min(words.len());
                status[..count].copy_from_slice(&words[..count]);
                let states = std::slice::from_raw_parts(
                    down.add(nodes as usize * 4).cast::<u32>(),
                    self.nodes as usize * 5,
                );
                *perturbed = states.chunks(5).map(|s| u64::from(s[4])).sum();
                let pivots = down.add((nodes * 4 + state_bytes) as usize);
                self.pivots = if self.lu {
                    lu_pivot_range(std::slice::from_raw_parts(
                        pivots.cast::<f32>(),
                        self.n as usize * 2,
                    ))
                } else {
                    let n = self.n as usize;
                    ldlt_pivot_range(
                        std::slice::from_raw_parts(pivots.cast::<f32>(), 2 * n),
                        std::slice::from_raw_parts(pivots.add(8 * n).cast::<f32>(), 2 * n),
                        std::slice::from_raw_parts(pivots.add(16 * n).cast::<u32>(), n),
                    )
                };
            }
        } else if let Some(out) = output {
            // SAFETY: as above; `out` holds n * nrhs values, as Solver checked.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    down,
                    out.as_mut_ptr().cast::<u8>(),
                    (n * u64::from(nrhs) * 8) as usize,
                )
            };
        }
        Ok(seconds)
    }

    /// Records the whole run: upload, schedule, readback, with a timestamp on
    /// either side. Returns the command buffers in submission order.
    fn record(
        &mut self,
        factoring: bool,
        nrhs: u32,
        up_bytes: u64,
        state_bytes: u64,
        marks: Option<schedule::Marks>,
    ) -> Result<(Vec<vk::CommandBuffer>, Option<schedule::Marks>)> {
        let (upload, download) = (self.upload_buffer(), self.download_buffer());
        let target = self.buffer(if factoring { 6 } else { 13 });
        let (status, state) = (self.buffer(12), self.buffer(15));
        let pivot_sources = [
            self.buffer(8),
            self.buffer(9),
            self.buffer(11),
            self.buffer(17),
        ];
        let n = u64::from(self.n);
        let nodes = u64::from(self.nodes);
        let sites: Vec<vk::BufferCopy> = if factoring && self.lu {
            let base = nodes * 4 + state_bytes;
            let mut at = 0u64;
            let mut regions = Vec::with_capacity(self.n as usize);
            for &(loffset, nr, nc) in &self.pivot_sites {
                for k in 0..u64::from(nc) {
                    let element = u64::from(loffset) + k * u64::from(nr) + k;
                    regions.push(
                        vk::BufferCopy::default()
                            .src_offset(element * 8)
                            .dst_offset(base + at * 8)
                            .size(8),
                    );
                    at += 1;
                }
            }
            regions
        } else {
            Vec::new()
        };
        let timestamps = self.timestamps;
        let first = self.one_shot(self.commands)?;
        // SAFETY: recording into a fresh command buffer; every handle belongs
        // to this device.
        unsafe {
            let device = &self.gpu.device;
            device.cmd_reset_query_pool(first, timestamps, 0, 2);
            if let Some(marks) = &marks {
                device.cmd_reset_query_pool(first, marks.pool, 0, marks.capacity);
            }
            device.cmd_write_timestamp(first, vk::PipelineStageFlags::TOP_OF_PIPE, timestamps, 0);
            if up_bytes > 0 {
                let region = vk::BufferCopy::default().size(up_bytes);
                device.cmd_copy_buffer(first, upload, target, &[region]);
            }
        }
        self.transfer_barrier(
            first,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        );
        self.end_command(first)?;
        let (set, pool) = (self.set, self.commands);
        // Built from the fields, not through shape(), so the recorder can
        // borrow the pipelines mutably at the same time.
        let shape = schedule::Shape {
            levels: &self.levels,
            steps: &self.steps,
            lu: self.lu,
            nodes: self.nodes,
            n: self.n,
            schur_batch: self.schur_batch,
            panel_threads: self.panel_threads,
            solve_threads: self.solve_threads,
            pivot_threads: self.pivot_threads,
            wide: self.wide,
        };
        let mut recorder =
            schedule::Recorder::new(&self.gpu, &mut self.pipelines, pool, set, marks)?;
        if factoring {
            schedule::factor(&mut recorder, &shape)?;
        } else {
            schedule::solve(&mut recorder, &shape, nrhs)?;
        }
        let last = recorder.current();
        let mut commands = vec![first];
        commands.extend(recorder.commands.iter().copied());
        let marks = recorder.marks.take();
        drop(recorder);
        self.transfer_barrier(
            last,
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
        );
        // SAFETY: as above.
        unsafe {
            let device = &self.gpu.device;
            let copy = |src: vk::Buffer, from: u64, to: u64, size: u64| {
                device.cmd_copy_buffer(
                    last,
                    src,
                    download,
                    &[vk::BufferCopy::default()
                        .src_offset(from)
                        .dst_offset(to)
                        .size(size)],
                )
            };
            if factoring {
                copy(status, 0, 0, nodes * 4);
                copy(state, 0, nodes * 4, state_bytes);
                let base = nodes * 4 + state_bytes;
                if self.lu {
                    if !sites.is_empty() {
                        device.cmd_copy_buffer(last, pivot_sources[3], download, &sites);
                    }
                } else {
                    copy(pivot_sources[0], 0, base, n * 8);
                    copy(pivot_sources[1], 0, base + n * 8, n * 8);
                    copy(pivot_sources[2], 0, base + n * 16, n * 4);
                }
            } else {
                copy(target, 0, 0, n * u64::from(nrhs) * 8);
            }
        }
        self.transfer_barrier(
            last,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::HOST_READ,
        );
        // SAFETY: as above.
        unsafe {
            self.gpu.device.cmd_write_timestamp(
                last,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                timestamps,
                1,
            );
        }
        Ok((commands, marks))
    }

    /// A timestamp pool large enough to mark every dispatch of a run: a bound
    /// from the plan's level and step counts, plus a mark per command buffer.
    fn marks(&self) -> Result<schedule::Marks> {
        let dispatches: u32 = self
            .levels
            .iter()
            .map(|l| 16 + l.max_children + 5 * l.steps + l.steps / 32)
            .sum();
        let capacity = dispatches + 64;
        // SAFETY: the create info outlives the call; the pool is destroyed
        // after the run.
        let pool = unsafe {
            self.gpu.device.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(capacity),
                None,
            )
        }
        .map_err(|e| Error::Device(format!("profiling queries: {e}")))?;
        Ok(schedule::Marks {
            pool,
            capacity,
            marks: Vec::new(),
        })
    }

    /// Prints the device time per kernel to stderr, as SPARTA_PROFILE does on
    /// Metal: each dispatch is billed from the previous mark to its own.
    fn report(&self, marks: &schedule::Marks) {
        let count = marks.marks.len();
        let mut ticks = vec![0u64; count];
        // SAFETY: every query up to `count` was written by the completed run.
        let read = unsafe {
            self.gpu.device.get_query_pool_results(
                marks.pool,
                0,
                &mut ticks,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )
        };
        if read.is_err() {
            return;
        }
        let period = f64::from(self.gpu.limits.timestamp_period) * 1e-9;
        let mut classes: Vec<(&'static str, f64, u32)> = Vec::new();
        let mut previous = 0u64;
        // SPARTA_PROFILE=2 also lists every dispatch, as on Metal: whether a
        // kernel is slow because it is big or because it is starved of
        // workgroups near the root of the tree.
        let every = std::env::var("SPARTA_PROFILE").is_ok_and(|v| v.trim() == "2");
        if every {
            eprintln!("\nSPARTA_PROFILE_DISPATCH  seq,kernel,groups,seconds");
        }
        for (&(query, kernel, groups), &tick) in marks.marks.iter().zip(&ticks) {
            if let Some(kernel) = kernel {
                let seconds = tick.saturating_sub(previous) as f64 * period;
                if every {
                    eprintln!("D,{query},{kernel},{groups},{seconds:.6}");
                }
                match classes.iter_mut().find(|c| c.0 == kernel) {
                    Some(class) => {
                        class.1 += seconds;
                        class.2 += 1;
                    }
                    None => classes.push((kernel, seconds, 1)),
                }
            }
            previous = tick;
        }
        let total: f64 = classes.iter().map(|c| c.1).sum();
        classes.sort_by(|a, b| b.1.total_cmp(&a.1));
        eprintln!("\nSPARTA_PROFILE  {} ({} dispatches)", self.gpu.name, count);
        eprintln!(
            "  {:<22} {:>10} {:>8} {:>10}",
            "kernel", "seconds", "share", "dispatches"
        );
        for (kernel, seconds, calls) in &classes {
            eprintln!(
                "  {kernel:<22} {seconds:>10.4} {:>7.1}% {calls:>10}",
                if total > 0.0 {
                    100.0 * seconds / total
                } else {
                    0.0
                }
            );
        }
        eprintln!("  {:<22} {total:>10.4}", "sum of kernels");
    }

    /// The schedule's view of this context.
    fn shape(&self) -> schedule::Shape<'_> {
        schedule::Shape {
            levels: &self.levels,
            steps: &self.steps,
            lu: self.lu,
            nodes: self.nodes,
            n: self.n,
            schur_batch: self.schur_batch,
            panel_threads: self.panel_threads,
            solve_threads: self.solve_threads,
            pivot_threads: self.pivot_threads,
            wide: self.wide,
        }
    }

    fn end_command(&self, command: vk::CommandBuffer) -> Result<()> {
        // SAFETY: the buffer is in the recording state.
        unsafe { self.gpu.device.end_command_buffer(command) }
            .map_err(|e| Error::Device(format!("end command buffer: {e}")))
    }

    /// Seconds between the run's two timestamps.
    fn device_seconds(&self) -> Result<f64> {
        let mut ticks = [0u64; 2];
        // SAFETY: both queries were written by the completed run.
        unsafe {
            self.gpu.device.get_query_pool_results(
                self.timestamps,
                0,
                &mut ticks,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )
        }
        .map_err(|e| Error::Device(format!("reading timestamps: {e}")))?;
        let period = f64::from(self.gpu.limits.timestamp_period);
        Ok(ticks[1].saturating_sub(ticks[0]) as f64 * period * 1e-9)
    }

    /// Grows the solution block (and the LU solve scratch) to `nrhs` columns.
    fn grow_rhs(&mut self, nrhs: u32) -> Result<()> {
        let bytes = u64::from(self.n) * u64::from(nrhs) * 8;
        let slots: &[usize] = if self.lu { &[13, 8] } else { &[13] };
        for &slot in slots {
            let grown = self.gpu.buffer(
                bytes,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                "the solution block",
            )?;
            if let Some(old) = self.buffers[slot].replace(grown) {
                self.gpu.free(&old);
            }
        }
        self.write_descriptors();
        self.rhs_capacity = nrhs;
        Ok(())
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        let device = &self.gpu.device;
        // SAFETY: every run waits for its fence, so nothing is pending; each
        // handle is destroyed once, before the device (dropped with `gpu`).
        unsafe {
            let _ = device.device_wait_idle();
            if self.timestamps != vk::QueryPool::null() {
                device.destroy_query_pool(self.timestamps, None);
            }
            if self.fence != vk::Fence::null() {
                device.destroy_fence(self.fence, None);
            }
            if self.commands != vk::CommandPool::null() {
                device.destroy_command_pool(self.commands, None);
            }
            if self.descriptors != vk::DescriptorPool::null() {
                device.destroy_descriptor_pool(self.descriptors, None);
            }
        }
        for buffer in self.buffers.iter().flatten() {
            self.gpu.free(buffer);
        }
        for staging in [&self.upload, &self.download] {
            if let Some(buffer) = &staging.buffer {
                self.gpu.free(buffer);
            }
        }
        self.pipelines.destroy(&self.gpu);
    }
}

/// LDL^T pivot range from the block diagonal: 1x1 pivots by magnitude, 2x2
/// blocks by their determinant relative to their largest entry (see
/// `sparta_v1_pivot_range` in metal.m).
fn ldlt_pivot_range(d: &[f32], sub: &[f32], two: &[u32]) -> Option<(f64, f64)> {
    let n = two.len();
    let (mut small, mut large) = (f64::INFINITY, 0.0f64);
    let hypot = |a: f32, b: f32| f64::from(a).hypot(f64::from(b));
    let mut k = 0;
    while k < n {
        let m = if two[k] != 0 && k + 1 < n {
            let (ar, ai) = (f64::from(d[2 * k]), f64::from(d[2 * k + 1]));
            let (br, bi) = (f64::from(sub[2 * k]), f64::from(sub[2 * k + 1]));
            let (cr, ci) = (f64::from(d[2 * k + 2]), f64::from(d[2 * k + 3]));
            let dr = ar * cr - ai * ci - (br * br - bi * bi);
            let di = ar * ci + ai * cr - 2.0 * br * bi;
            let top = ar.hypot(ai).max(br.hypot(bi)).max(cr.hypot(ci));
            k += 2;
            if top > 0.0 {
                dr.hypot(di) / top
            } else {
                0.0
            }
        } else {
            let m = hypot(d[2 * k], d[2 * k + 1]);
            k += 1;
            m
        };
        if m.is_finite() {
            small = small.min(m);
            large = large.max(m);
        }
    }
    (small.is_finite() && large > 0.0).then_some((small, large))
}

/// LU pivot range from U's diagonal, gathered in node order.
fn lu_pivot_range(pivots: &[f32]) -> Option<(f64, f64)> {
    let (mut small, mut large) = (f64::INFINITY, 0.0f64);
    for p in pivots.chunks(2) {
        let m = f64::from(p[0]).hypot(f64::from(p[1]));
        if m.is_finite() {
            small = small.min(m);
            large = large.max(m);
        }
    }
    (small.is_finite() && large > 0.0).then_some((small, large))
}
