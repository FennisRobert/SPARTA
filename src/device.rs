//! Which GPU backend a build targets, and the frontal geometry it wants.
//!
//! The numerics are the same on every device; what differs is the shape the
//! kernels are compiled in. A [`Device`] therefore carries two things: the
//! hardware limits [`Config::check`] must respect, and a [`Profile`] of
//! measured geometry that [`Config::default`] starts from.
//!
//! One build targets one GPU backend, chosen by `build.rs` from the `metal`
//! and `vulkan` Cargo features, or from the target OS when neither is given.
//! [`Device::COMPILED`] is that choice, and it is what `Config::default`
//! returns. A plan may still be *built* for another device, which is how the
//! geometry and memory estimates of one backend are compared against another
//! without that hardware, but `Solver::new` refuses to run it: the kernels it
//! would need are not in this binary.

/// A GPU backend, and the geometry its kernels are compiled in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    /// Apple Metal on Apple Silicon. `src/kernels/*.metal`, `src/metal.m`.
    Metal,
    /// Vulkan compute, for AMD and NVIDIA; the default on Windows and Linux.
    /// `src/kernels/vulkan/`, `src/gpu/vulkan/`. See
    /// https://github.com/FennisRobert/SPARTA/issues/31.
    Vulkan,
}

/// The frontal geometry and dispatch widths one device is tuned for.
///
/// Every field is a [`Config`](crate::Config) field. They live together here
/// so a device's numbers are read in one place and cannot be half-applied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Profile {
    pub panel: u32,
    pub schur_block: u32,
    pub strip_block: u32,
    pub narrow_cols: u32,
    pub pivot_threads: u32,
    pub schur_chunk: u32,
    pub panel_threads: u32,
    pub solve_threads: u32,
    pub schur_batch: u32,
    pub small_front_rows: u32,
}

impl Device {
    /// Every device, in the order a written template documents them.
    pub const ALL: [Device; 2] = [Device::Metal, Device::Vulkan];

    /// The backend this build compiled kernels for.
    pub const COMPILED: Device = if cfg!(sparta_vulkan) {
        Device::Vulkan
    } else {
        Device::Metal
    };

    /// The name used by [`Device::parse`] and reported by `sparta_device`.
    pub fn name(self) -> &'static str {
        match self {
            Device::Metal => "metal",
            Device::Vulkan => "vulkan",
        }
    }

    /// One line on what this device is.
    pub fn summary(self) -> &'static str {
        match self {
            Device::Metal => "Apple Silicon, complex64 through Metal",
            Device::Vulkan => "AMD and NVIDIA, complex64 through Vulkan compute",
        }
    }

    /// Parses a device name, case-insensitively.
    pub fn parse(name: &str) -> Option<Device> {
        let wanted = name.trim().to_ascii_lowercase();
        Device::ALL.into_iter().find(|d| d.name() == wanted)
    }

    /// Reports whether this build can run this device's kernels at all.
    ///
    /// False does not mean the machine lacks the hardware -- that is
    /// [`crate::gpu_available`] -- only that the kernels were not compiled in.
    pub fn compiled(self) -> bool {
        self == Device::COMPILED
    }

    /// Shared memory one workgroup may declare, in bytes.
    ///
    /// Metal's limit is a fixed 32 KiB. Vulkan's is per-device
    /// `maxComputeSharedMemorySize`, whose guaranteed minimum is 16 KiB; the
    /// smaller figure is used so a plan that passes `check` runs on any
    /// conformant implementation rather than only on the roomier ones.
    pub fn shared_bytes(self) -> u32 {
        match self {
            Device::Metal => 32 * 1024,
            Device::Vulkan => 16 * 1024,
        }
    }

    /// Threads that execute in lockstep, which cooperative widths must fill.
    ///
    /// 32 on Apple and on NVIDIA. AMD GCN is 64 and RDNA can be either, so
    /// Vulkan takes 64: a multiple of 64 is also a multiple of 32, and the
    /// reverse is what leaves half a wave idle.
    pub fn lane_width(self) -> u32 {
        match self {
            Device::Metal => 32,
            Device::Vulkan => 64,
        }
    }

    /// Largest workgroup this device's kernels may be dispatched with.
    ///
    /// Metal guarantees 1024. Vulkan's `maxComputeWorkGroupInvocations` floor
    /// is 128, but every desktop AMD and NVIDIA part reports at least 1024;
    /// 1024 is the cap enforced here and the profile stays well under it.
    pub fn max_group(self) -> u32 {
        1024
    }

    /// The geometry this device is tuned for.
    ///
    /// Metal's numbers are measured; see `documentation/sparta_backend.typ`
    /// and the comments in `src/kernels/*.metal`.
    ///
    /// **Vulkan's are not measured.** They are the widest shapes that fit the
    /// portable limits above, so a plan built with them is valid everywhere,
    /// and nothing more than that. Whoever implements the backend should treat
    /// every one as an open question and re-measure it; the numbers are here
    /// so the seam exists, not because they are right.
    pub(crate) fn profile(self) -> Profile {
        match self {
            Device::Metal => Profile {
                panel: 16,
                schur_block: 4,
                strip_block: 4,
                narrow_cols: 32,
                pivot_threads: 128,
                schur_chunk: 16,
                panel_threads: 512,
                solve_threads: 0,
                schur_batch: 0,
                small_front_rows: 32,
            },
            Device::Vulkan => Profile {
                panel: 16,
                schur_block: 4,
                strip_block: 4,
                narrow_cols: 32,
                // A multiple of 64, so an AMD wave64 is never half idle.
                pivot_threads: 128,
                // 2 * 64 * 16 complex floats is exactly Vulkan's guaranteed
                // 16 KiB, leaving no room for a driver that reserves any. Half
                // that is 8 KiB and comfortable on every implementation.
                schur_chunk: 8,
                panel_threads: 256,
                solve_threads: 0,
                schur_batch: 0,
                small_front_rows: 32,
            },
        }
    }
}

impl std::fmt::Display for Device {
    /// Writes the device's configuration-file name.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Config;

    #[test]
    /// Every device's own profile passes its own limits, and names round-trip.
    fn profiles_are_valid_and_names_round_trip() {
        for device in Device::ALL {
            let config = Config::for_device(device);
            assert_eq!(config.check(), Ok(()), "{device} profile is invalid");
            assert_eq!(config.device, device);
            assert_eq!(Device::parse(device.name()), Some(device));
            assert_eq!(Device::parse(&device.name().to_ascii_uppercase()), Some(device));
        }
        assert_eq!(Device::parse("cuda"), None);
    }

    #[test]
    /// The compiled-in device is what the plain defaults carry.
    fn defaults_follow_the_compiled_device() {
        assert_eq!(Config::default().device, Device::COMPILED);
        assert!(Device::COMPILED.compiled());
        assert_eq!(Config::default(), Config::for_device(Device::COMPILED));
    }

    #[test]
    /// A profile that overruns its device's shared memory is rejected there
    /// and may still be legal on a roomier one.
    fn shared_memory_limits_are_per_device() {
        // 2 * 64 * 32 complex floats is exactly Metal's 32 KiB and twice
        // Vulkan's guaranteed 16 KiB.
        let wide = |device: Device| Config {
            schur_chunk: 32,
            ..Config::for_device(device)
        };
        assert_eq!(wide(Device::Metal).check(), Ok(()));
        assert!(wide(Device::Vulkan).check().is_err());
    }
}
