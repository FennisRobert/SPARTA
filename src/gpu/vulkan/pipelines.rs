//! The kernels as pipelines: one SPIR-V module per kernel, and one pipeline
//! per kernel and workgroup shape.
//!
//! Metal reads a kernel's width at run time from [[threads_per_threadgroup]],
//! so one pipeline serves every width the host dispatches. A Vulkan pipeline
//! fixes its workgroup size, so a kernel dispatched at several widths --
//! `factor_panel` follows each level's widest fully summed block -- gets a
//! pipeline per width, built the first time that width is asked for.
//!
//! The frontal geometry reaches every pipeline as specialization constants,
//! in the order `src/kernels/vulkan/common.comp` declares them. They come from
//! the plan's `Config`, the values Metal receives as preprocessor macros, so
//! the plan and the kernels cannot disagree.
use super::device::Gpu;
use crate::{Error, Result};
use ash::vk;
use std::collections::HashMap;

include!(concat!(env!("OUT_DIR"), "/vulkan_shaders.rs"));

/// Specialization constants 0..=7 and 10 of common.comp, fixed for a
/// context; 8 and 9 are the workgroup shape of each pipeline.
#[derive(Clone, Copy)]
pub(super) struct Geometry {
    pub panel: u32,
    pub schur_block: u32,
    pub strip_block: u32,
    pub pivot_threads: u32,
    pub schur_chunk: u32,
    pub tail_rows: u32,
    pub lu_diag_threshold: f32,
    pub max_simds: u32,
    pub narrow_cols: u32,
}

/// Every pipeline a context has built, and the modules they come from.
pub(super) struct Pipelines {
    pub set_layout: vk::DescriptorSetLayout,
    pub layout: vk::PipelineLayout,
    modules: HashMap<&'static str, vk::ShaderModule>,
    built: HashMap<(&'static str, u32, u32), vk::Pipeline>,
    geometry: Geometry,
    /// Use the float-atomic builds of the kernels that have one.
    atomic_float: bool,
}

/// Descriptor bindings 0..=20 except 14, which is the push-constant range.
pub(super) const BINDINGS: [u32; 20] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20,
];

impl Pipelines {
    /// The descriptor and pipeline layouts every kernel shares: twenty storage
    /// buffers and a 16-byte push-constant range.
    pub fn new(gpu: &Gpu, geometry: Geometry) -> Result<Self> {
        let bindings: Vec<_> = BINDINGS
            .iter()
            .map(|&b| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(16)];
        // SAFETY: the create infos outlive the calls; the layouts are
        // destroyed in `destroy`.
        unsafe {
            let set_layout = gpu
                .device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .map_err(|e| Error::Device(format!("descriptor set layout: {e}")))?;
            let sets = [set_layout];
            let layout = gpu
                .device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&sets)
                        .push_constant_ranges(&push),
                    None,
                )
                .map_err(|e| Error::Device(format!("pipeline layout: {e}")))?;
            Ok(Self {
                set_layout,
                layout,
                modules: HashMap::new(),
                built: HashMap::new(),
                geometry,
                atomic_float: gpu.atomic_float,
            })
        }
    }

    /// The pipeline of `kernel` at workgroup shape `x` by `y`, built on first use.
    ///
    /// Returns `Error::Device` naming the kernel and shape when the driver
    /// refuses it, which on a valid plan means the shape exceeds a device limit.
    pub fn get(&mut self, gpu: &Gpu, kernel: &'static str, x: u32, y: u32) -> Result<vk::Pipeline> {
        if let Some(&pipeline) = self.built.get(&(kernel, x, y)) {
            return Ok(pipeline);
        }
        let module = self.module(gpu, kernel)?;
        let g = self.geometry;
        let values: [u32; 11] = [
            g.panel,
            g.schur_block,
            g.strip_block,
            g.pivot_threads,
            g.schur_chunk,
            g.tail_rows,
            g.lu_diag_threshold.to_bits(),
            g.max_simds,
            x,
            y,
            g.narrow_cols,
        ];
        let entries: Vec<_> = (0..values.len() as u32)
            .map(|i| vk::SpecializationMapEntry {
                constant_id: i,
                offset: i * 4,
                size: 4,
            })
            .collect();
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let special = vk::SpecializationInfo::default()
            .map_entries(&entries)
            .data(&data);
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main")
            .specialization_info(&special);
        let info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(self.layout);
        // SAFETY: module, layout and the create info are valid for the call.
        let pipeline = unsafe {
            gpu.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
                .map_err(|(_, e)| {
                    Error::Device(format!(
                        "Vulkan pipeline {kernel} at {x}x{y} failed: {e}; check the geometry \
                         settings against this device's limits"
                    ))
                })?[0]
        };
        self.built.insert((kernel, x, y), pipeline);
        Ok(pipeline)
    }

    /// The shader module of `kernel`, created on first use from the embedded
    /// SPIR-V (its float-atomic build when the device has float atomics).
    fn module(&mut self, gpu: &Gpu, kernel: &'static str) -> Result<vk::ShaderModule> {
        if let Some(&module) = self.modules.get(kernel) {
            return Ok(module);
        }
        let atomic = format!("{kernel}+atomic");
        let lookup = |name: &str| SHADERS.iter().find(|(n, _)| *n == name).map(|(_, b)| *b);
        let bytes = (if self.atomic_float {
            lookup(&atomic)
        } else {
            None
        })
        .or_else(|| lookup(kernel))
        .ok_or_else(|| Error::Device(format!("no SPIR-V module for kernel {kernel}")))?;
        let words = ash::util::read_spv(&mut std::io::Cursor::new(bytes))
            .map_err(|e| Error::Device(format!("reading SPIR-V of {kernel}: {e}")))?;
        // SAFETY: valid SPIR-V from build.rs; destroyed in `destroy`.
        let module = unsafe {
            gpu.device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
        }
        .map_err(|e| Error::Device(format!("shader module {kernel}: {e}")))?;
        self.modules.insert(kernel, module);
        Ok(module)
    }

    /// Destroys every pipeline, module and layout.
    pub fn destroy(&mut self, gpu: &Gpu) {
        // SAFETY: the caller guarantees no pending command uses them.
        unsafe {
            for (_, pipeline) in self.built.drain() {
                gpu.device.destroy_pipeline(pipeline, None);
            }
            for (_, module) in self.modules.drain() {
                gpu.device.destroy_shader_module(module, None);
            }
            gpu.device.destroy_pipeline_layout(self.layout, None);
            gpu.device
                .destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}
