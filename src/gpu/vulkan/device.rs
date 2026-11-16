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
    /// Largest buffer a descriptor binding can cover, in bytes
    /// (`maxStorageBufferRange`, 4 GiB on NVIDIA).
    pub max_bound: u64,
    /// Largest buffer the kernels reach by address, in bytes: the device's
    /// largest buffer and allocation, and 2^31 complex64 elements, the most
    /// the kernels' signed 32-bit indices reach.
    pub max_addressed: u64,
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
    /// Whether this is a portability driver offering `VK_KHR_portability_subset`.
    portability: bool,
    min_subgroup: u32,
    max_bound: u64,
    max_addressed: u64,
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
        let entry = unsafe { ash::Entry::load() }.map_err(|e| Error::Device(no_loader(&e)))?;
        let app = vk::ApplicationInfo::default()
            .application_name(c"sparta")
            .api_version(vk::API_VERSION_1_3);
        let instance = create_instance(&entry, &app).map_err(|e| Error::Device(no_instance(e)))?;
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
        let (candidates, rejected) = candidates(&instance)?;
        let chosen = choose(&candidates, &rejected)?;
        let priorities = [1.0f32];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(chosen.queue_family)
            .queue_priorities(&priorities)];
        let mut atomic = vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default()
            .shader_buffer_float32_atomic_add(chosen.atomic_float);
        let mut extensions = Vec::new();
        if chosen.atomic_float {
            extensions.push(ash::ext::shader_atomic_float::NAME.as_ptr());
        }
        // A portability driver (MoltenVK) requires its subset extension
        // enabled whenever it offers it.
        if chosen.portability {
            extensions.push(ash::khr::portability_subset::NAME.as_ptr());
        }
        // maintenance4 (core in 1.3) is what lets a workgroup size come from a
        // specialization constant (LocalSizeId), which every kernel uses.
        let mut core = vk::PhysicalDeviceVulkan13Features::default().maintenance4(true);
        // The kernels use the Vulkan memory model: workgroup-scoped coherence
        // and explicit barrier semantics (see common.comp).
        // Buffer device addresses (required in 1.3) reach the arrays larger
        // than a binding can cover; see common.comp.
        let mut model = vk::PhysicalDeviceVulkan12Features::default()
            .vulkan_memory_model(true)
            .vulkan_memory_model_device_scope(true)
            .buffer_device_address(true);
        let mut info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&extensions)
            .push_next(&mut core)
            .push_next(&mut model);
        if chosen.atomic_float {
            info = info.push_next(&mut atomic);
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
            max_bound: chosen.max_bound,
            max_addressed: chosen.max_addressed,
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
    /// An `addressed` buffer is reached by the kernels through its device
    /// address ([`Gpu::address`]) instead of a binding, which lifts the 4 GiB
    /// binding window.
    ///
    /// `what` names the buffer in the error when the device refuses it.
    pub fn buffer(
        &self,
        size: u64,
        wanted: vk::MemoryPropertyFlags,
        what: &str,
        addressed: bool,
    ) -> Result<Buffer> {
        let size = size.max(4);
        let limit = if addressed { self.max_addressed } else { self.max_bound };
        if size > limit {
            return Err(Error::Unsupported(format!(
                "{what} needs {:.2} GiB, the largest buffer {} accepts is {:.2} GiB",
                size as f64 / GIB,
                self.name,
                limit as f64 / GIB
            )));
        }
        let mut usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        if addressed {
            usage |= vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        }
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
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
            let mut flags = vk::MemoryAllocateFlagsInfo::default()
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let mut allocate = vk::MemoryAllocateInfo::default()
                .allocation_size(needs.size)
                .memory_type_index(kind);
            if addressed {
                allocate = allocate.push_next(&mut flags);
            }
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

    /// The device address of a buffer created with `addressed`, which the
    /// kernels receive in their push constants.
    pub fn address(&self, buffer: &Buffer) -> u64 {
        // SAFETY: a live buffer of this device created with
        // SHADER_DEVICE_ADDRESS usage on memory allocated with DEVICE_ADDRESS.
        unsafe {
            self.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(buffer.buffer),
            )
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

/// Create an instance for `app`. On Apple the only Vulkan driver is MoltenVK,
/// a portability driver the loader lists only to an instance that enables
/// `VK_KHR_portability_enumeration`; elsewhere no extension is requested.
fn create_instance(entry: &ash::Entry, app: &vk::ApplicationInfo) -> ash::prelude::VkResult<ash::Instance> {
    let portability = [ash::khr::portability_enumeration::NAME.as_ptr()];
    let mut info = vk::InstanceCreateInfo::default().application_info(app);
    if cfg!(target_vendor = "apple") {
        info = info
            .enabled_extension_names(&portability)
            .flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR);
    }
    // SAFETY: the create info and everything it points to outlive the call.
    unsafe { entry.create_instance(&info, None) }
}

/// Every device the backend can run on, in the loader's order.
///
/// Suitable means Vulkan 1.3 with buffer device addresses, a queue family with
/// compute, and basic and arithmetic subgroup operations in compute shaders.
/// Also returns, for every device that is not suitable, its name and why not.
fn candidates(instance: &ash::Instance) -> Result<(Vec<Candidate>, Vec<String>)> {
    // SAFETY: plain queries on a live instance.
    let physicals = unsafe { instance.enumerate_physical_devices() }
        .map_err(|e| Error::Device(format!("listing Vulkan devices: {e}")))?;
    let mut out = Vec::new();
    let mut rejected = Vec::new();
    for physical in physicals {
        // SAFETY: as above, for a device of this instance; the chained
        // structures outlive the calls that fill them.
        unsafe {
            let properties = instance.get_physical_device_properties(physical);
            let name = device_name(&properties);
            if properties.api_version < vk::API_VERSION_1_3 {
                rejected.push(format!(
                    "{name} (Vulkan {}.{}, needs 1.3)",
                    vk::api_version_major(properties.api_version),
                    vk::api_version_minor(properties.api_version)
                ));
                continue;
            }
            let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
            let mut subgroup_size = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
            let mut maintenance = vk::PhysicalDeviceMaintenance4Properties::default();
            let mut allocation = vk::PhysicalDeviceMaintenance3Properties::default();
            let mut all = vk::PhysicalDeviceProperties2::default()
                .push_next(&mut subgroup)
                .push_next(&mut subgroup_size)
                .push_next(&mut maintenance)
                .push_next(&mut allocation);
            instance.get_physical_device_properties2(physical, &mut all);
            let operations = vk::SubgroupFeatureFlags::BASIC | vk::SubgroupFeatureFlags::ARITHMETIC;
            if !subgroup
                .supported_stages
                .contains(vk::ShaderStageFlags::COMPUTE)
                || !subgroup.supported_operations.contains(operations)
            {
                rejected.push(format!("{name} (no subgroup arithmetic in compute shaders)"));
                continue;
            }
            let mut atomic = vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default();
            let mut core12 = vk::PhysicalDeviceVulkan12Features::default();
            let mut features = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut atomic)
                .push_next(&mut core12);
            instance.get_physical_device_features2(physical, &mut features);
            if core12.buffer_device_address != vk::TRUE {
                rejected.push(format!("{name} (no buffer device addresses)"));
                continue;
            }
            let families = instance.get_physical_device_queue_family_properties(physical);
            let Some(queue_family) = families
                .iter()
                .position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
            else {
                rejected.push(format!("{name} (no compute queue)"));
                continue;
            };
            let max_bound = maintenance
                .max_buffer_size
                .min(u64::from(properties.limits.max_storage_buffer_range));
            let max_addressed = maintenance
                .max_buffer_size
                .min(allocation.max_memory_allocation_size)
                .min(8 << 31);
            let portability = instance
                .enumerate_device_extension_properties(physical)
                .unwrap_or_default()
                .iter()
                .any(|e| e.extension_name_as_c_str() == Ok(ash::khr::portability_subset::NAME));
            out.push(Candidate {
                physical,
                properties,
                queue_family: queue_family as u32,
                atomic_float: atomic.shader_buffer_float32_atomic_add == vk::TRUE,
                portability,
                min_subgroup: subgroup_size.min_subgroup_size.max(1),
                max_bound,
                max_addressed,
            });
        }
    }
    Ok((out, rejected))
}

/// The device `SPARTA_VULKAN_DEVICE` names, else the first discrete GPU, else
/// the first suitable device.
///
/// With no suitable device, the error says why: no GPU driver behind the
/// loader, or the devices found and what each one lacks.
fn choose<'c>(candidates: &'c [Candidate], rejected: &[String]) -> Result<&'c Candidate> {
    let name = |c: &Candidate| device_name(&c.properties);
    if candidates.is_empty() && rejected.is_empty() {
        return Err(Error::Device(no_driver()));
    }
    if candidates.is_empty() {
        return Err(Error::Device(format!(
            "no Vulkan device SPARTA can use: {}. Update the GPU driver, or \
             select the CPU backend (force_cpu=True in Python)",
            rejected.join(", ")
        )));
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

/// The device a context would run on, or why there is none. Loads Vulkan and
/// lists devices, so the caller caches the answer.
pub(super) fn probe() -> std::result::Result<String, String> {
    // SAFETY: as in `Gpu::new`.
    let entry = unsafe { ash::Entry::load() }.map_err(|e| no_loader(&e))?;
    let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
    // SAFETY: the create info outlives the call; the instance is destroyed
    // before returning.
    unsafe {
        let instance = create_instance(&entry, &app).map_err(no_instance)?;
        let found = candidates(&instance).and_then(|(c, rejected)| {
            choose(&c, &rejected).map(|chosen| device_name(&chosen.properties))
        });
        instance.destroy_instance(None);
        found.map_err(|e| match e {
            Error::Device(message) => message,
            other => other.to_string(),
        })
    }
}

fn device_name(properties: &vk::PhysicalDeviceProperties) -> String {
    properties
        .device_name_as_c_str()
        .map(CStr::to_string_lossy)
        .unwrap_or_default()
        .into_owned()
}

/// The error when the Vulkan loader itself cannot be opened.
fn no_loader(error: &ash::LoadingError) -> String {
    format!("no Vulkan loader found ({error}). {}", driver_hint())
}

/// The error when no instance can be created. A loader with no GPU driver
/// behind it fails here, with `ERROR_INCOMPATIBLE_DRIVER`.
fn no_instance(error: vk::Result) -> String {
    if error == vk::Result::ERROR_INCOMPATIBLE_DRIVER {
        no_driver()
    } else {
        format!("creating a Vulkan instance: {error}")
    }
}

fn no_driver() -> String {
    format!("the Vulkan loader found no GPU driver. {}", driver_hint())
}

/// What to install when Vulkan has no driver for the GPU. On Windows the
/// loader and the driver come with the GPU driver. On Linux they are separate
/// packages, and the NVIDIA driver on a server or headless install often comes
/// without its Vulkan part: CUDA then works and Vulkan does not. When the
/// NVIDIA kernel module is loaded, its version names the package.
fn driver_hint() -> String {
    let fallback = "or select the CPU backend (force_cpu=True in Python)";
    if !cfg!(target_os = "linux") {
        return format!("Install a current GPU driver with Vulkan 1.3, {fallback}.");
    }
    let nvidia = std::fs::read_to_string("/proc/driver/nvidia/version")
        .ok()
        .and_then(|text| nvidia_version(&text));
    let mut hint = match nvidia {
        Some(version) => {
            let major = version.split('.').next().unwrap_or_default();
            format!(
                "The NVIDIA driver {version} is loaded, but its Vulkan part is \
                 missing, as on server and headless installs. On Ubuntu and \
                 Debian, install it with `sudo apt install libvulkan1 \
                 libnvidia-gl-{major}` (libnvidia-gl-{major}-server for the \
                 server driver), then check with `vulkaninfo --summary`, {fallback}."
            )
        }
        None => format!(
            "Install the Vulkan loader (libvulkan1 on Ubuntu and Debian, \
             vulkan-loader on Fedora) and the GPU's Vulkan driver, then check \
             with `vulkaninfo --summary`, {fallback}."
        ),
    };
    if std::path::Path::new("/.dockerenv").exists() {
        hint.push_str(
            " In a container, NVIDIA_DRIVER_CAPABILITIES must include graphics.",
        );
    }
    hint
}

/// The driver version in `/proc/driver/nvidia/version`: its first word made
/// of dot-separated numbers, such as "580.178.04".
fn nvidia_version(text: &str) -> Option<String> {
    text.split_whitespace()
        .find(|word| word.contains('.') && word.split('.').all(|p| p.parse::<u32>().is_ok()))
        .map(str::to_owned)
}

#[cfg(test)]
mod tests {
    use super::nvidia_version;

    #[test]
    fn reads_the_nvidia_driver_version() {
        let proprietary = "NVRM version: NVIDIA UNIX x86_64 Kernel Module  580.178.04  \
                           Release Build  (dvs-builder@U16-I3-B03-4-3)  Tue Oct  7 2025\n\
                           GCC version:  gcc version 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04)";
        assert_eq!(nvidia_version(proprietary).as_deref(), Some("580.178.04"));
        let open = "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  570.86.15  Release Build";
        assert_eq!(nvidia_version(open).as_deref(), Some("570.86.15"));
        assert_eq!(nvidia_version("no version here"), None);
    }
}
