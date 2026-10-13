#!/bin/sh
# Full validation: Rust unit and plan tests, then the GPU numerical suite.
# Build caches and the test virtualenv live under ignored .local/ so the source
# tree stays lean. Requires Apple Silicon with Metal for the Python tests; the
# Vulkan backend's Python run is https://github.com/FennisRobert/SPARTA/issues/36.
set -eu
cd "$(dirname "$0")"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$PWD/.local/target}"
export TMPDIR="$PWD/.local/tmp"
export CLANG_MODULE_CACHE_PATH="$PWD/.local/clang"
export PIP_CACHE_DIR="$PWD/.local/pip"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$CLANG_MODULE_CACHE_PATH"

cargo test --workspace "$@"
cargo build --release --examples "$@"
# The Vulkan backend must keep building, so that a change to the shared layers
# cannot quietly break it. Its kernels compile with glslc from the Vulkan SDK,
# so it is checked where that is installed; its GPU tests pass without doing
# anything when no Vulkan device is present.
if command -v glslc >/dev/null 2>&1 || [ -n "${VULKAN_SDK:-}${GLSLC:-}" ]; then
    cargo test --features vulkan --test shader "$@"
else
    printf 'Skipping the Vulkan backend: no glslc (install the Vulkan SDK).\n'
fi
# So must a build with no GPU backend, which every target other than Apple
# Silicon gets by default. Apple Silicon always picks Metal, so check it there
# for Intel macOS instead.
if [ "$(uname -s)-$(uname -m)" = "Darwin-arm64" ]; then
    rustup target add x86_64-apple-darwin >/dev/null
    cargo check --all-targets --target x86_64-apple-darwin "$@"
else
    cargo check --all-targets "$@"
fi

if [ ! -x .local/venv/bin/python ]; then
    "${PYTHON:-python3}" -m venv .local/venv
    .local/venv/bin/python -m pip install --quiet --upgrade pip
    .local/venv/bin/python -m pip install --quiet numpy scipy maturin
fi
export VIRTUAL_ENV="$PWD/.local/venv"

.local/venv/bin/maturin develop --release --manifest-path python/Cargo.toml "$@"
printf 'Running Metal correctness tests, including fronts through 4096 rows...\n'
SPARTA_REQUIRE_METAL=1 .local/venv/bin/python - <<'SPARTA_TEST_RUNNER'
import subprocess
import sys
with open('.local/validation.log', 'w') as log:
    process = subprocess.Popen(
        [sys.executable, '-u', '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py', '-v'],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        print(line, end='', flush=True)
        log.write(line)
        log.flush()
    sys.exit(process.wait())
SPARTA_TEST_RUNNER
printf 'Results saved in %s/.local/validation.log\n' "$PWD"
