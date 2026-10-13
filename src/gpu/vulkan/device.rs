//! The Vulkan instance, the chosen physical device and its logical device,
//! and device-local buffers.
//!
//! One `Gpu` per solver context, as Metal takes the system device per context:
//! Vulkan needs no process-wide state, and a context that owns its device can
//! be released without coordinating with any other.
use crate::{Error, Result};
use ash::vk;
use std::ffi::CStr;

/// The device a context runs on, with everything the rest of the backend
/// reads about it.
pub(super) struct Gpu {
    /// Keeps the Vulkan loader library alive for as long as the handles below.
    _entry: ash::Entry,
    pub instance: ash::Instance,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub queue_family: u32,
    pub limits: vk::PhysicalDeviceLimits,
    pub memory: vk::PhysicalDeviceMemoryProperties,
    /// Largest single buffer the device accepts, in bytes.
    pub max_buffer: u64,
    /// Smallest subgroup the device may run a compute pipeline at.
    pub min_subgroup: u32,
    /// Whether the device adds floats atomically in storage buffers, which
    /// selects the float-atomic build of the forward solve kernels.
    pub atomic_float: bool,
    pub name: String,
}

/// A device-local buffer and the memory behind it.
pub(super) struct Buffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    /// Bytes, as requested (at least 4).
    pub size: u64,
}

/// What a physical device offers that this backend depends on.
struct Candidate {
    physical: vk::PhysicalDevice,
    properties: vk::PhysicalDeviceProperties,
    queue_family: u32,
    atomic_float: bool,
    min_subgroup: u32,
    max_buffer: u64,
}

/// The environment variable that picks the device: an index into the list of
/// suitable devices, or a case-insensitive part of its name ("nvidia", "amd").
pub(super) const DEVICE_VARIABLE: &str = "SPARTA_VULKAN_DEVICE";

impl Gpu {
    /// Loads Vulkan, picks a device and creates the logical device with the
    /// features the kernels need.
    ///
    /// Returns `Error::Device` when there is no loader, no Vulkan 1.3 device
    /// with subgroup arithmetic in compute shaders, or `SPARTA_VULKAN_DEVICE`
    /// names none of them.
    pub fn new() -> Result<Self> {
        // SAFETY: loading the system Vulkan loader; its entry points are only
        // called through ash's typed wrappers below.
        let entry = unsafe { ash::Entry::load() }
            .map_err(|e| Error::Device(format!("no Vulkan loader: {e}")))?;
        let app = vk::ApplicationInfo::default()
            .application_name(c"sparta")
            .api_version(vk::API_VERSION_1_3);
        // SAFETY: the create info outlives the call.
        let instance = unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app),
                None,
            )
        }
        .map_err(|e| Error::Device(format!("creating a Vulkan instance: {e}")))?;
        match Self::open(entry.clone(), instance.clone()) {
            Ok(gpu) => Ok(gpu),
            Err(e) => {
                // SAFETY: nothing created from the instance survives a failed open.
                unsafe { instance.destroy_instance(None) };
                Err(e)
            }
        }
    }

    /// Picks the device and creates the logical device on `instance`.
    fn open(entry: ash::Entry, instance: ash::Instance) -> Result<Self> {
        let candidates = candidates(&instance)?;
        let chosen = choose(&candidates)?;
        let priorities = [1.0f32];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(chosen.queue_family)
            .queue_priorities(&priorities)];
        let mut atomic = vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default()
            .shader_buffer_float32_atomic_add(chosen.atomic_float);
        let extensions = [ash::ext::shader_atomic_float::NAME.as_ptr()];
        // maintenance4 (core in 1.3) is what lets a workgroup size come from a
        // specialization constant (LocalSizeId), which every kernel uses.
        let mut core = vk::PhysicalDeviceVulkan13Features::default().maintenance4(true);
        // The kernels use the Vulkan memory model: workgroup-scoped coherence
        // and explicit barrier semantics (see common.comp).
        let mut model = vk::PhysicalDeviceVulkan12Features::default()
            .vulkan_memory_model(true)
            .vulkan_memory_model_device_scope(true);
        let mut info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .push_next(&mut core)
            .push_next(&mut model);
        if chosen.atomic_float {
            info = info
                .enabled_extension_names(&extensions)
                .push_next(&mut atomic);
        }
        // SAFETY: the physical device came from this instance; the create info
        // and everything it points to outlive the call.
        let device = unsafe { instance.create_device(chosen.physical, &info, None) }
            .map_err(|e| Error::Device(format!("creating the Vulkan device: {e}")))?;
        // SAFETY: family and index were requested in the create info above.
        let queue = unsafe { device.get_device_queue(chosen.queue_family, 0) };
        // SAFETY: a valid physical device of this instance.
        let memory = unsafe { instance.get_physical_device_memory_properties(chosen.physical) };
        let name = chosen
            .properties
            .device_name_as_c_str()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        Ok(Self {
            _entry: entry,
            instance,
            device,
            queue,
            queue_family: chosen.queue_family,
            limits: chosen.properties.limits,
            memory,
            max_buffer: chosen.max_buffer,
            min_subgroup: chosen.min_subgroup,
            atomic_float: chosen.atomic_float,
            name,
        })
    }

    /// A memory type with all of `wanted` among those `bits` allows.
    pub fn memory_type(&self, bits: u32, wanted: vk::MemoryPropertyFlags) -> Option<u32> {
        (0..self.memory.memory_type_count).find(|&i| {
            bits & (1 << i) != 0
                && self.memory.memory_types[i as usize]
                    .property_flags
                    .contains(wanted)
        })
    }

    /// Creates a buffer of `size` bytes (at least 4) in memory with `wanted`
    /// properties, usable as storage and as a copy source and destination.
    ///
    /// `what` names the buffer in the error when the device refuses it.
    pub fn buffer(&self, size: u64, wanted: vk::MemoryPropertyFlags, what: &str) -> Result<Buffer> {
        let size = size.max(4);
        if size > self.max_buffer {
            return Err(Error::Unsupported(format!(
                "{what} needs {:.2} GiB, the largest buffer {} accepts is {:.2} GiB",
                size as f64 / GIB,
                self.name,
                self.max_buffer as f64 / GIB
            )));
        }
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let device = &self.device;
        // SAFETY: the create info is valid for the call; every object created
        // here is destroyed on the error paths below or owned by the result.
        unsafe {
            let buffer = device
                .create_buffer(&info, None)
                .map_err(|e| Error::Device(format!("creating {what}: {e}")))?;
            let needs = device.get_buffer_memory_requirements(buffer);
            let Some(kind) = self.memory_type(needs.memory_type_bits, wanted) else {
                device.destroy_buffer(buffer, None);
                return Err(Error::Device(format!("no suitable memory type for {what}")));
            };
            let allocate = vk::MemoryAllocateInfo::default()
                .allocation_size(needs.size)
                .memory_type_index(kind);
            let memory = match device.allocate_memory(&allocate, None) {
                Ok(memory) => memory,
                Err(e) => {
                    device.destroy_buffer(buffer, None);
                    return Err(Error::Device(format!(
                        "allocating {:.2} GiB for {what}: {e}",
                        size as f64 / GIB
                    )));
                }
            };
            if let Err(e) = device.bind_buffer_memory(buffer, memory, 0) {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
                return Err(Error::Device(format!("binding memory for {what}: {e}")));
            }
            Ok(Buffer {
                buffer,
                memory,
                size,
            })
        }
    }

    /// Destroys `buffer` and frees its memory.
    pub fn free(&self, buffer: &Buffer) {
        // SAFETY: the caller guarantees no pending command uses the buffer.
        unsafe {
            self.device.destroy_buffer(buffer.buffer, None);
            self.device.free_memory(buffer.memory, None);
        }
    }

    /// Bytes of device-local memory: the largest device-local heap.
    pub fn local_bytes(&self) -> u64 {
        let heaps = &self.memory.memory_heaps[..self.memory.memory_heap_count as usize];
        heaps
            .iter()
            .filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|h| h.size)
            .max()
            .unwrap_or(0)
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        // SAFETY: the owner destroys every child object before the Gpu.
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

pub(super) const GIB: f64 = 1073741824.0;

/// Every device the backend can run on, in the loader's order.
///
/// Suitable means Vulkan 1.3, a queue family with compute, and basic and
/// arithmetic subgroup operations in compute shaders.
fn candidates(instance: &ash::Instance) -> Result<Vec<Candidate>> {
    // SAFETY: plain queries on a live instance.
    let physicals = unsafe { instance.enumerate_physical_devices() }
        .map_err(|e| Error::Device(format!("listing Vulkan devices: {e}")))?;
    let mut out = Vec::new();
    for physical in physicals {
        // SAFETY: as above, for a device of this instance; the chained
        // structures outlive the calls that fill them.
        unsafe {
            let properties = instance.get_physical_device_properties(physical);
            if properties.api_version < vk::API_VERSION_1_3 {
                continue;
            }
            let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
            let mut subgroup_size = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
            let mut maintenance = vk::PhysicalDeviceMaintenance4Properties::default();
            let mut all = vk::PhysicalDeviceProperties2::default()
                .push_next(&mut subgroup)
                .push_next(&mut subgroup_size)
                .push_next(&mut maintenance);
            instance.get_physical_device_properties2(physical, &mut all);
            let operations = vk::SubgroupFeatureFlags::BASIC | vk::SubgroupFeatureFlags::ARITHMETIC;
            if !subgroup
                .supported_stages
                .contains(vk::ShaderStageFlags::COMPUTE)
                || !subgroup.supported_operations.contains(operations)
            {
                continue;
            }
            let mut atomic = vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default();
            let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut atomic);
            instance.get_physical_device_features2(physical, &mut features);
            let families = instance.get_physical_device_queue_family_properties(physical);
            let Some(queue_family) = families
                .iter()
                .position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
            else {
                continue;
            };
            let max_buffer = maintenance
                .max_buffer_size
                .min(u64::from(properties.limits.max_storage_buffer_range));
            out.push(Candidate {
                physical,
                properties,
                queue_family: queue_family as u32,
                atomic_float: atomic.shader_buffer_float32_atomic_add == vk::TRUE,
                min_subgroup: subgroup_size.min_subgroup_size.max(1),
                max_buffer,
            });
        }
    }
    Ok(out)
}

/// The device `SPARTA_VULKAN_DEVICE` names, else the first discrete GPU, else
/// the first suitable device.
fn choose(candidates: &[Candidate]) -> Result<&Candidate> {
    let name = |c: &Candidate| {
        c.properties
            .device_name_as_c_str()
            .map(CStr::to_string_lossy)
            .unwrap_or_default()
            .into_owned()
    };
    if candidates.is_empty() {
        return Err(Error::Device(
            "no Vulkan 1.3 device with subgroup arithmetic in compute shaders".into(),
        ));
    }
    if let Ok(wanted) = std::env::var(DEVICE_VARIABLE) {
        let wanted = wanted.trim().to_lowercase();
        let found = match wanted.parse::<usize>() {
            Ok(index) => candidates.get(index),
            Err(_) => candidates
                .iter()
                .find(|c| name(c).to_lowercase().contains(&wanted)),
        };
        return found.ok_or_else(|| {
            let names: Vec<String> = candidates.iter().map(name).collect();
            Error::Device(format!(
                "{DEVICE_VARIABLE}={wanted} names no suitable device; these are: {}",
                names.join(", ")
            ))
        });
    }
    Ok(candidates
        .iter()
        .find(|c| c.properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU)
        .unwrap_or(&candidates[0]))
}

/// Whether any suitable device exists. Loads Vulkan and lists devices, so the
/// caller caches the answer.
pub(super) fn any_device() -> bool {
    // SAFETY: as in `Gpu::new`.
    let Ok(entry) = (unsafe { ash::Entry::load() }) else {
        return false;
    };
    let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
    // SAFETY: the create info outlives the call; the instance is destroyed
    // before returning.
    unsafe {
        let Ok(instance) = entry.create_instance(
            &vk::InstanceCreateInfo::default().application_info(&app),
            None,
        ) else {
            return false;
        };
        let found = candidates(&instance)
            .map(|c| choose(&c).is_ok())
            .unwrap_or(false);
        instance.destroy_instance(None);
        found
    }
}
