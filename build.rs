//! Chooses the GPU backend and the CPU dense-kernel provider for this build,
//! and compiles whatever the choice needs.
//!
//! Two decisions, both made here so nothing downstream repeats them:
//!
//! - **GPU backend.** `--features metal` or `--features vulkan` pick one
//!   explicitly; with neither, the target OS decides. The choice is published
//!   as `cfg(sparta_metal)` / `cfg(sparta_vulkan)`, which `src/gpu/mod.rs`
//!   reads, and mirrored by `Device::COMPILED`. A build for a target that
//!   supports nothing compiles the symbolic analysis and the CPU backend, and
//!   reports no device.
//! - **CPU BLAS and LAPACK.** `src/cpu/dense.rs` calls plain CBLAS and
//!   f2c-style LAPACK, which Accelerate, OpenBLAS and MKL all export. The
//!   provider is published as `cfg(sparta_accelerate)` / `cfg(sparta_blas)`.

use std::{env, fs, path::PathBuf, process::Command};

/// `src/kernels/*.metal`, concatenated in this order to form the shader
/// source. Dependency order, not alphabetical.
const KERNELS: &[&str] = &[
    "common",
    "assembly",
    "ldlt_factor",
    "ldlt_update",
    "ldlt_solve",
    "lu_factor",
    "lu_update",
    "lu_solve",
];

/// `src/kernels/vulkan/*.comp` and the kernels each one holds. Every kernel is
/// its own SPIR-V module, compiled from its file with `-DKERNEL_<name>`.
const VULKAN_KERNELS: &[(&str, &[&str])] = &[
    (
        "assembly",
        &[
            "zero_fronts",
            "initialize_large",
            "initialize_large_lu",
            "assemble_children",
            "store_factors",
            "store_factors_lu",
            "advance_panel",
        ],
    ),
    (
        "ldlt_factor",
        &[
            "factor_nodes",
            "factor_narrow",
            "factor_narrow_tail",
            "factor_panel",
            "factor_panel_tail",
        ],
    ),
    ("ldlt_update", &["update_strip", "update_schur"]),
    (
        "ldlt_solve",
        &["solve_forward", "solve_backward", "solve_diagonal"],
    ),
    (
        "lu_factor",
        &[
            "factor_nodes_lu",
            "factor_narrow_lu",
            "factor_narrow_tail_lu",
            "factor_panel_lu",
            "factor_panel_tail_lu",
        ],
    ),
    ("lu_update", &["update_strip_lu", "update_schur_lu"]),
    ("lu_solve", &["solve_forward_lu", "solve_backward_lu"]),
];

/// Kernels that also get an `ATOMIC_FLOAT` build, for devices with float
/// atomic add on storage buffers. The plain build uses compare-and-swap.
const VULKAN_ATOMIC_KERNELS: &[&str] = &["solve_forward", "solve_forward_lu"];

/// Which GPU backend this build compiles kernels for.
#[derive(PartialEq)]
enum Backend {
    Metal,
    Vulkan,
    None,
}

/// Picks the GPU backend from the features, or from the target when none is given.
///
/// Asking for both is a configuration mistake rather than a merge of the two,
/// because exactly one set of kernels is compiled in.
fn backend(os: &str, arch: &str) -> Backend {
    let (metal, vulkan) = (
        env::var_os("CARGO_FEATURE_METAL").is_some(),
        env::var_os("CARGO_FEATURE_VULKAN").is_some(),
    );
    match (metal, vulkan) {
        (true, true) => panic!(
            "features `metal` and `vulkan` are mutually exclusive: one build \
             compiles one set of kernels. Pick one, or neither to let the \
             target decide."
        ),
        (true, false) => Backend::Metal,
        (false, true) => Backend::Vulkan,
        // Apple Silicon is the only target Metal runs on. Everything else that
        // has a GPU worth using has Vulkan, but the backend does not exist
        // yet, so the honest default there is no device rather than one that
        // fails at the first dispatch.
        (false, false) if os == "macos" && arch == "aarch64" => Backend::Metal,
        (false, false) => Backend::None,
    }
}

/// Emits the link directives for the CPU dense kernels.
///
/// Accelerate is the obvious choice on Apple and needs no external package.
/// Elsewhere the crate links a system CBLAS and LAPACK when `SPARTA_BLAS_LIB`
/// names the library (for example `openblas`, or `mkl_rt` for MKL);
/// `SPARTA_BLAS_DIR` adds a search path for one that is not on the default
/// one. When `SPARTA_BLAS_LIB` is unset nothing is linked and the portable
/// Rust loops are used. Both are read at build time and are the only way to redirect this,
/// because a wrong choice must fail to link rather than fail at runtime.
/// `SPARTA_DENSE_LIBRARY` carries the choice to every target for reporting.
fn dense_kernels(os: &str) {
    println!("cargo:rerun-if-env-changed=SPARTA_BLAS_LIB");
    println!("cargo:rerun-if-env-changed=SPARTA_BLAS_DIR");
    if os == "macos" || os == "ios" {
        println!("cargo:rustc-cfg=sparta_accelerate");
        println!("cargo:rustc-env=SPARTA_DENSE_LIBRARY=Accelerate");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        return;
    }
    // No system BLAS is guaranteed to exist off Apple, so a build that was not
    // told which one to use gets the portable Rust loops rather than a link
    // error: `dense::available()` reports false and every caller keeps to
    // `panel.rs`.
    let Ok(lib) = env::var("SPARTA_BLAS_LIB") else {
        println!("cargo:rustc-env=SPARTA_DENSE_LIBRARY=portable loops");
        return;
    };
    println!("cargo:rustc-env=SPARTA_DENSE_LIBRARY={lib}");
    if let Ok(dir) = env::var("SPARTA_BLAS_DIR") {
        println!("cargo:rustc-link-search=native={dir}");
    }
    println!("cargo:rustc-cfg=sparta_blas");
    for name in lib.split(',').filter(|n| !n.trim().is_empty()) {
        println!("cargo:rustc-link-lib={}", name.trim());
    }
}

/// Compiles the Metal shader source and the Objective-C bridge.
fn build_metal(out: &PathBuf) {
    // One Metal translation unit, concatenated in dependency order: `common`
    // defines everything the rest share, and each path's files follow it. The
    // split is for reading, not for compilation -- there are no include
    // guards and no forward declarations, so this order is load-bearing.
    let source: String = KERNELS
        .iter()
        .map(|file| {
            let path = format!("src/kernels/{file}.metal");
            fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
        })
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(out.join("shader.h"), format!("@{source:?}")).unwrap();
    assert!(Command::new("xcrun")
        .args([
            "clang",
            "-arch",
            "arm64",
            "-O3",
            "-fobjc-arc",
            "-fmodules",
            "-c",
            "src/metal.m",
            "-o"
        ])
        .arg(out.join("metal.o"))
        .arg("-I")
        .arg(out)
        .status()
        .unwrap()
        .success());
    assert!(Command::new("ar")
        .arg("crs")
        .arg(out.join("libsparta_gpu.a"))
        .arg(out.join("metal.o"))
        .status()
        .unwrap()
        .success());
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=sparta_gpu");
    for f in ["Foundation", "Metal"] {
        println!("cargo:rustc-link-lib=framework={f}");
    }
}

/// The GLSL compiler: `GLSLC` if set, else the Vulkan SDK's, else `glslc` on
/// the path.
fn glslc() -> PathBuf {
    println!("cargo:rerun-if-env-changed=GLSLC");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    if let Some(path) = env::var_os("GLSLC") {
        return path.into();
    }
    if let Some(sdk) = env::var_os("VULKAN_SDK") {
        for bin in ["Bin", "bin"] {
            let path = PathBuf::from(&sdk).join(bin).join("glslc");
            if path.with_extension("exe").exists() || path.exists() {
                return path;
            }
        }
    }
    "glslc".into()
}

/// Compiles every Vulkan kernel to SPIR-V and writes `vulkan_shaders.rs`, the
/// table `src/gpu/vulkan.rs` embeds: `(name, module)` per kernel, with
/// `name+atomic` for the float-atomic builds.
fn build_vulkan(out: &PathBuf) {
    let compiler = glslc();
    let mut table = String::from("pub(super) const SHADERS: &[(&str, &[u8])] = &[\n");
    let mut compile = |file: &str, kernel: &str, atomic: bool| {
        let name = if atomic {
            format!("{kernel}+atomic")
        } else {
            kernel.to_string()
        };
        let spv = out.join(format!("{}.spv", name.replace('+', "_")));
        let mut command = Command::new(&compiler);
        command
            .args([
                "-fshader-stage=compute",
                "--target-env=vulkan1.3",
                "-Werror",
            ])
            .arg(format!("-DKERNEL_{kernel}"))
            .arg(format!("src/kernels/vulkan/{file}.comp"))
            .arg("-o")
            .arg(&spv);
        if atomic {
            command.arg("-DATOMIC_FLOAT");
        }
        let status = command.status().unwrap_or_else(|e| {
            panic!(
                "running {}: {e}. The vulkan feature compiles its kernels with glslc: \
                 install the Vulkan SDK (or set GLSLC to a glslc binary)",
                compiler.display()
            )
        });
        assert!(status.success(), "glslc failed on {file}.comp for {name}");
        table.push_str(&format!(
            "    ({name:?}, include_bytes!({:?})),\n",
            spv.display().to_string()
        ));
    };
    for (file, kernels) in VULKAN_KERNELS {
        for kernel in *kernels {
            compile(file, kernel, false);
            if VULKAN_ATOMIC_KERNELS.contains(kernel) {
                compile(file, kernel, true);
            }
        }
    }
    table.push_str("];\n");
    fs::write(out.join("vulkan_shaders.rs"), table).unwrap();
}

fn main() {
    // Declared so a `cfg` the build script sets is not an unexpected-cfg warning.
    for flag in [
        "sparta_metal",
        "sparta_vulkan",
        "sparta_accelerate",
        "sparta_blas",
    ] {
        println!("cargo:rustc-check-cfg=cfg({flag})");
    }
    println!("cargo:rerun-if-changed=src/metal.m");
    for file in KERNELS {
        println!("cargo:rerun-if-changed=src/kernels/{file}.metal");
    }
    println!("cargo:rerun-if-changed=src/kernels/vulkan/common.comp");
    for (file, _) in VULKAN_KERNELS {
        println!("cargo:rerun-if-changed=src/kernels/vulkan/{file}.comp");
    }
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    dense_kernels(&os);

    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    match backend(&os, &arch) {
        Backend::Metal => {
            assert!(
                os == "macos" && arch == "aarch64",
                "the metal backend targets Apple Silicon only, not {os}-{arch}"
            );
            println!("cargo:rustc-cfg=sparta_metal");
            build_metal(&out);
        }
        Backend::Vulkan => {
            println!("cargo:rustc-cfg=sparta_vulkan");
            // SPIR-V embedded the way `shader.h` embeds the Metal source, so the
            // shipped library stays self-contained.
            build_vulkan(&out);
        }
        Backend::None => {}
    }
}
