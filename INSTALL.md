# Setup and Build

AMSLib is a CMake (>= 3.18, C++17) project. The build itself is
straightforward; the effort is in providing dependencies so that
`find_package` succeeds.

## Dependencies

**Always required:**

* HDF5 (C component)
* LibTorch (PyTorch C++ API)
* `nlohmann_json`
* `{fmt}`
* `tl::expected`
* A C++17 compiler and a threading library

`fmt` and `tl::expected` can fall back to CMake `FetchContent` if package
discovery fails. For network-free builds, provide local packages through the
hint variables below.

**Optional, enabled per build flag:**

| Dependency | Enabled by |
| --- | --- |
| MPI | `ENABLE_MPI` |
| CUDA (NVIDIA) | `ENABLE_CUDA` |
| HIP / ROCm (AMD) | `ENABLE_HIP` |
| Caliper | `ENABLE_CALIPER` |
| amqp-cpp, OpenSSL, libevent | `ENABLE_RMQ` |
| PerfFlowAspect | `ENABLE_PERFFLOWASPECT` |
| Python workflow drivers | `ENABLE_WORKFLOW` |
| Catch2 tests | `ENABLE_TESTS` |

`ENABLE_CUDA` and `ENABLE_HIP` are mutually exclusive.

## Build Options

| Option | Default | Description |
| --- | --- | --- |
| `ENABLE_MPI` | `OFF` | Enable MPI support. |
| `ENABLE_CUDA` | `OFF` | Enable CUDA support for NVIDIA GPUs. |
| `ENABLE_HIP` | `OFF` | Enable HIP support for AMD GPUs. |
| `ENABLE_CALIPER` | `OFF` | Enable Caliper profiling. |
| `ENABLE_TESTS` | `OFF` | Build the Catch2-based test suite. |
| `ENABLE_WORKFLOW` | `OFF` | Install the Python drivers used by the outer workflow. |
| `ENABLE_RMQ` | `OFF` | Enable the RabbitMQ database backend. |
| `ENABLE_PERFFLOWASPECT` | `OFF` | Enable PerfFlowAspect profiling. |
| `AMS_ENABLE_DEBUG` | `OFF` | Enable verbose AMS debug messages. |
| `AMS_INSTALL_FLUX_PYTHON` | `OFF` | Install the Python workflow package with the `flux-python` optional dependency when `ENABLE_WORKFLOW=On`. |
| `BUILD_SHARED_LIBS` | CMake default | Build shared libraries when `ON`; static when `OFF`. |
| `AMS_DEFER_STATIC_TPL_RESOLUTION` | `OFF` | Defer selected static TPL resolution to downstream final links when building shared AMS. |
| `AMS_PIP_INSTALL_ARGS` | empty | Extra arguments passed to `pip install` when `ENABLE_WORKFLOW=On`. |

When building shared libraries, use
`-DBUILD_SHARED_LIBS=On -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=On`. For a static
build, set both to `Off`.

## Dependency Location Hints

When a dependency is installed outside default search paths, point CMake at its
config package or add its prefix to `CMAKE_PREFIX_PATH`. On LC systems,
`scripts/gitlab/setup-env.sh` exports the `AMS_*` variables shown below.

| Package | CMake variable | LC export |
| --- | --- | --- |
| libTorch | `Torch_DIR` | `$AMS_TORCH_PATH` |
| HDF5 | `HDF5_DIR` | `$AMS_HDF5_PATH` |
| Caliper | `caliper_DIR` | `$AMS_CALIPER_PATH` |
| amqp-cpp | `amqpcpp_DIR` | `$AMS_AMQPCPP_PATH` |
| nlohmann_json | `nlohmann_json_DIR` | `$AMS_NLOHMANN_JSON_DIR` |
| fmt | `AMS_FMT_DIR` | `$AMS_FMT_DIR` |
| tl-expected | `tl-expected_DIR` | `$AMS_TL_EXPECTED_DIR` |
| Catch2, when `ENABLE_TESTS=On` | `AMS_CATCH2_DIR` | `$AMS_CATCH2_DIR` |
| CUDA arch | `CMAKE_CUDA_ARCHITECTURES` | `$AMS_CUDA_ARCH` |
| HIP arch | `CMAKE_HIP_ARCHITECTURES` | auto-detected or `$AMS_HIP_ARCH` |
| Zlib, when needed by static HDF5 | `ZLIB_ROOT` or `ZLIB_DIR` | `$AMS_ZLIB_PATH` |

## Spack and LC Setup

Before running Spack commands or sourcing the LC setup script in restricted
environments, keep Spack and XDG caches out of the home directory:

```bash
export SPACK_DISABLE_LOCAL_CONFIG=true
export SPACK_SKIP_MODULES=1
export SPACK_USER_CACHE_PATH=/tmp
export XDG_CACHE_HOME=/tmp
```

On LLNL Livermore Computing systems, source the repository setup script from
the repository root:

```bash
source scripts/gitlab/setup-env.sh
```

The script loads the appropriate compiler, MPI, and ROCm modules, activates the
AMS Spack environment, and exports dependency locations used by CMake.

### LC Workflow Python Environments

On LC systems, do not create AMS Workflow environments with plain
`python3 -m venv`. The AMS Spack environments use a Python external and LC
Flux provides the compatible `flux` Python bindings for the active Flux
installation. Create the venv through the repository helper so it links both
Spack Python packages and a venv-local shim for system Flux Python:

```bash
host=$(hostname)
host=${host//[0-9]/}
python3 scripts/make-spack-venv.py \
  --env "/usr/workspace/AMS/ams-spack-environments/1.1/${host}/" \
  --output "venv-${host}" \
  --with-system-flux-python
source "venv-${host}/bin/activate"
```

The helper writes system Flux metadata into the venv and warns on activation if the
active `flux version` differs from the recorded one. The shim exposes Flux
without putting unrelated LC Python packages ahead of AMS Spack packages.
Recreate the venv after system Flux changes. For LC workflow CMake builds, pass
`-DAMS_PIP_INSTALL_ARGS="--no-build-isolation"` so pip uses the prepared venv
instead of an isolated build environment. Do not enable
`AMS_INSTALL_FLUX_PYTHON` for this path; Flux Python is supplied by the
prepared system-backed venv, not by pip.

Design choices for system Flux Python:

- AMS does not install `flux-python` as a default Python dependency on LC.
  The Python bindings must match the active LC `flux` command and runtime,
  which can change independently of the AMS Spack environment.
- `scripts/make-spack-venv.py --with-system-flux-python` discovers the active
  Flux binding with the venv Python first, then falls back to `flux python`
  when Flux keeps its bindings outside the default Python import path.
- The helper writes `system-flux-python.pth` in the venv `site-packages` directory.
  This file points at a venv-local `system_flux_python` shim, not at the whole LC
  Python `site-packages` tree, so packages such as LC's system `numpy` do not
  shadow the AMS Spack Python packages.
- The helper writes `system-flux-python.json` beside the `.pth` file. This records
  `which flux`, `flux version`, the venv Python version, the `flux python`
  version, the original `flux.__file__`, the original system Flux Python path, and
  the shim path.
- The activation hook compares the current `flux version` against the recorded
  metadata and warns when they differ. Treat that warning as a signal to
  recreate the venv with `--with-system-flux-python`.

### Non-System Flux Workflow Installs

When building AMS Workflow in an environment that does not provide compatible
system Flux Python bindings, enable the workflow Flux extra:

```bash
cmake -S . -B build \
  -DENABLE_WORKFLOW=On \
  -DAMS_INSTALL_FLUX_PYTHON=On
```

With this flag, the `PyAMS` target runs pip against the generated package tree
as `pip install <build-dir>[flux]`, which installs the `flux-python` optional
dependency declared by `pyproject.toml`. Use this for container or non-LC
builds that rely on pip-managed Flux Python. Leave it off when using
`scripts/make-spack-venv.py --with-system-flux-python`.

## Convenience Configure Script

`scripts/ams-configure.sh` assembles a standard CMake command and maps LC
`AMS_*` exports to the current `-D*_DIR` hints:

```bash
scripts/ams-configure.sh
scripts/ams-configure.sh --mpi --rmq
scripts/ams-configure.sh --hip --mpi --caliper
scripts/ams-configure.sh --mpi --tests
scripts/ams-configure.sh --workflow --install-flux-python
scripts/ams-configure.sh --mpi --rmq --dry-run
```

When `--tests` is enabled and `AMS_CATCH2_DIR` is set, the helper forwards
`-DAMS_CATCH2_DIR="$AMS_CATCH2_DIR"` so LC builds use the Spack-provided
Catch2 package.

Manual CMake is still the clearest path when forcing LC Cray/ROCm compilers
such as `amdclang` and `amdclang++`, unless the helper script is extended with
compiler options.

## Manual CMake Installation

This representative LC command enables MPI, Caliper, RabbitMQ, and debug
messages with shared libraries:

```bash
export SPACK_DISABLE_LOCAL_CONFIG=true
export SPACK_SKIP_MODULES=1
export SPACK_USER_CACHE_PATH=/tmp
export XDG_CACHE_HOME=/tmp
source scripts/gitlab/setup-env.sh

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
<<<<<<< HEAD
  -DBUILD_SHARED_LIBS=On \
  -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=On \
  -DENABLE_MPI=On \
  -DENABLE_CALIPER=On \
  -DENABLE_RMQ=On \
  -DENABLE_WORKFLOW=On \
  -DAMS_PIP_INSTALL_ARGS="--no-build-isolation" \
  -DAMS_ENABLE_DEBUG=On \
  -DTorch_DIR="$AMS_TORCH_PATH" \
  -DHDF5_DIR="$AMS_HDF5_PATH" \
  -Dcaliper_DIR="$AMS_CALIPER_PATH" \
  -Damqpcpp_DIR="$AMS_AMQPCPP_PATH" \
  -DAMS_FMT_DIR="$AMS_FMT_DIR" \
  -Dnlohmann_json_DIR="$AMS_NLOHMANN_JSON_DIR" \
  -Dtl-expected_DIR="$AMS_TL_EXPECTED_DIR"
=======
  -DWITH_CUDA=On \
  -DUMPIRE_DIR=$AMS_UMPIRE_PATH \
  -DMFEM_DIR=$AMS_MFEM_PATH \
  -DWITH_FAISS=On \
  -DWITH_MPI=On \
  -DENABLE_TORCH=On \
  -DWITH_TESTS=Off \
  -DTorch_DIR=$AMS_TORCH_PATH \
  -DFAISS_DIR=$AMS_FAISS_PATH \
  -DAMS_CUDA_ARCH=${AMS_CUDA_ARCH} \
  -DWITH_AMS_DEBUG=On \
  ../
>>>>>>> 4598085 (WIP. Removing Torch from the main path (all tests green with WITH_TORCH=On except integration tests))

cmake --build build -j 6
cmake --install build
```

## Example Builds

### 1. Minimal CPU Build on LC

```bash
export SPACK_DISABLE_LOCAL_CONFIG=true
export SPACK_SKIP_MODULES=1
export SPACK_USER_CACHE_PATH=/tmp
export XDG_CACHE_HOME=/tmp
source scripts/gitlab/setup-env.sh

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=On \
  -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=On \
  -DTorch_DIR="$AMS_TORCH_PATH" \
  -DHDF5_DIR="$AMS_HDF5_PATH" \
  -DAMS_FMT_DIR="$AMS_FMT_DIR" \
  -Dnlohmann_json_DIR="$AMS_NLOHMANN_JSON_DIR" \
  -Dtl-expected_DIR="$AMS_TL_EXPECTED_DIR"
```

### 2. HIP / ROCm Build on Tuolumne or Tioga

Use `amdclang` and `amdclang++` on LC Cray/ROCm machines.

```bash
export SPACK_DISABLE_LOCAL_CONFIG=true
export SPACK_SKIP_MODULES=1
export SPACK_USER_CACHE_PATH=/tmp
export XDG_CACHE_HOME=/tmp
source scripts/gitlab/setup-env.sh

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=On \
  -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=On \
  -DCMAKE_C_COMPILER=amdclang \
  -DCMAKE_CXX_COMPILER=amdclang++ \
  -DENABLE_HIP=On \
  -DENABLE_MPI=On \
  -DENABLE_CALIPER=On \
  -DENABLE_RMQ=Off \
  -DENABLE_WORKFLOW=Off \
  -DENABLE_TESTS=Off \
  -DAMS_ENABLE_DEBUG=On \
  -DTorch_DIR="$AMS_TORCH_PATH" \
  -DHDF5_DIR="$AMS_HDF5_PATH" \
  -Dcaliper_DIR="$AMS_CALIPER_PATH" \
  -DAMS_FMT_DIR="$AMS_FMT_DIR" \
  -Dnlohmann_json_DIR="$AMS_NLOHMANN_JSON_DIR" \
  -Dtl-expected_DIR="$AMS_TL_EXPECTED_DIR"

cmake --build build -j
ctest --test-dir build -N
```

The Tuolumne validation used this shape with `ENABLE_TESTS=Off`; CTest
reported `Total Tests: 0`. To build the Catch2 tests on LC, switch to
`-DENABLE_TESTS=On` and add `-DAMS_CATCH2_DIR="$AMS_CATCH2_DIR"`.

### 3. CUDA + Caliper on LC

```bash
export SPACK_DISABLE_LOCAL_CONFIG=true
export SPACK_SKIP_MODULES=1
export SPACK_USER_CACHE_PATH=/tmp
export XDG_CACHE_HOME=/tmp
source scripts/gitlab/setup-env.sh

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=On \
  -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=On \
  -DENABLE_CUDA=On \
  -DCMAKE_CUDA_ARCHITECTURES="$AMS_CUDA_ARCH" \
  -DENABLE_CALIPER=On \
  -Dcaliper_DIR="$AMS_CALIPER_PATH" \
  -DTorch_DIR="$AMS_TORCH_PATH" \
  -DHDF5_DIR="$AMS_HDF5_PATH" \
  -DAMS_FMT_DIR="$AMS_FMT_DIR" \
  -Dnlohmann_json_DIR="$AMS_NLOHMANN_JSON_DIR" \
  -Dtl-expected_DIR="$AMS_TL_EXPECTED_DIR"
```

### 4. Minimal CPU Build with Manual Dependencies

```bash
export CMAKE_PREFIX_PATH=/opt/ams-deps:$CMAKE_PREFIX_PATH

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=On \
  -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=On \
  -DTorch_DIR=/opt/libtorch/share/cmake/Torch \
  -DHDF5_DIR=/opt/hdf5 \
  -DAMS_FMT_DIR=/opt/ams-deps/lib/cmake/fmt \
  -Dnlohmann_json_DIR=/opt/ams-deps/lib/cmake/nlohmann_json \
  -Dtl-expected_DIR=/opt/ams-deps/share/cmake/tl-expected
```

Add feature flags such as `-DENABLE_RMQ=On`, `-DENABLE_MPI=On`, or
`-DENABLE_HIP=On` and the corresponding package hints as needed.

## Tests and Catch2

`ENABLE_TESTS=On` enters `tests/AMSlib/CMakeLists.txt` and looks for a Catch2
CMake package. On LC systems, `scripts/gitlab/setup-env.sh` exports
`AMS_CATCH2_DIR` pointing at the Spack-provided package directory; pass it with
`-DAMS_CATCH2_DIR="$AMS_CATCH2_DIR"` to avoid network access.

If Catch2 package discovery fails and no `AMS_CATCH2_DIR` hint is provided,
CMake falls back to `FetchContent` from GitHub for Catch2 v3.11.0. For
network-free builds outside LC, provide a local/package Catch2 config directory
with `AMS_CATCH2_DIR`, or configure with `-DENABLE_TESTS=Off`.

## Build and Install

```bash
cmake --build build -j "$(nproc)"
cmake --install build
```
