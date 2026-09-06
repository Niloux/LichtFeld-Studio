# Standalone training build

Current milestones, the agreed contextcapture configuration and next steps are
recorded in [the train-only roadmap (中文)](train-only-roadmap.md).

`LFS_BUILD_PROFILE=train` builds `lfs-train` inside this fork. The default
`studio` profile retains the desktop application. Use separate build directories
for the two profiles; changing a configured directory's profile is rejected.

The training profile keeps the CUDA FastGS and gsplat backends, strategies,
losses, evaluation, COLMAP and NeRF-transform datasets, image caching/prefetch,
nvImageCodec, PLY import/export, checkpoints and the `.licht` training project
container. It does not configure the desktop renderer, editor, video libraries,
MCP/TCP, the embedded Python runtime, plugins or automatic prior generation.

OpenMesh data structures and general Tensor/neural arithmetic kernels remain in
this first version. MoGe model and weight-file implementations are excluded.
The codec build can use a host Python interpreter as a build tool; Python is not
a runtime dependency of `lfs-train`.

## Build

Install CUDA 12.8 or a compatible newer toolkit, CMake 3.30+, GCC/G++ 14 on Linux,
Ninja, Git, curl, zip/unzip, pkg-config, autoconf, automake, libtool and nasm.
Set `VCPKG_ROOT` to a bootstrapped vcpkg checkout. Manifest versions, overrides
and overlay ports remain pinned by this repository.

```sh
cmake -S . -B build/train -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 \
  -DCMAKE_CUDA_HOST_COMPILER=g++-14 \
  -DLFS_BUILD_PROFILE=train \
  -DLFS_TRAIN_PYTHON_LOSS=OFF -DLFS_TRAIN_PREPROCESS=OFF \
  -DBUILD_TESTS=OFF -DLFS_BUILD_TRAIN_TESTS=ON
cmake --build build/train --target lfs-train -j 4
```

For a build machine without a GPU, add `-DBUILD_CUDA_PTX_ONLY=ON` and, if needed,
`-DBUILD_CUDA_MIN_SM=86`. Training still requires a supported NVIDIA GPU and driver.

On systems with recent glibc, CUDA 12.8 can fail compiler detection because
`sinpi`/`cospi` (glibc 2.41+) and `rsqrt` (glibc 2.43+) declarations disagree on
`noexcept`. An optional local header overlay handles this specific failure:

```sh
python3 scripts/prepare_train_cuda_headers.py --output /tmp/lfs-cuda-include
# Add this option to the configure command above:
# -DCMAKE_CUDA_FLAGS=-I/tmp/lfs-cuda-include
```

The script checks the toolkit version and declaration counts and does not edit
system headers. Keep the overlay directory until the build is finished.

## Train and resume

```sh
env -u DISPLAY -u WAYLAND_DISPLAY build/train/lfs-train \
  -d /path/to/colmap-scene -o /path/to/run --iter 30000 --export ply

env -u DISPLAY -u WAYLAND_DISPLAY build/train/lfs-train \
  --resume /path/to/run/project.licht --iter 40000 --export ply
```

The training argument parser and parameter objects are shared with studio.
`--headless` is implicit. Existing strategies and optimization options remain
available. Only PLY final export is supported. `.resume` checkpoints are also
accepted by `--resume`.

Projects use the existing snapshot/recovery machinery, including the lifetime
of the source document and recovery lock. Mesh and sequence assets are rejected;
arbitrary desktop project compatibility is not promised. Pass `-o` when resuming
to save into a new output directory. Without `-o`, a resumed `.licht` remains the
save destination. PLY files contain the exported model, not optimizer state.

Automatic normal generation defaults to off. Provide existing mask/depth/normal
files when enabling the corresponding loss; missing required supervision is an
error (image alpha can supply a mask when enabled). Older saved configurations
that enable auto-generation must be overridden with `--no-normal-auto-generate`.

RGB-only training needs no depth or normal maps: both supervision options default
to off. For fisheye datasets, use `--gut` to retain the native camera model.

Python scripts, server/editor options and unsupported export formats are errors.
`LFS_TRAIN_PYTHON_LOSS=ON` and `LFS_TRAIN_PREPROCESS=ON` currently fail at configure
time: standalone integrations have not been implemented. To customize a loss in
this version, edit the existing C++/CUDA loss and gradient implementation in
`src/training`; returning a scalar through a control callback does not supply its
gradient to the rasterizer.

The CLI reports training, final export and asynchronous project-writer failures
through a nonzero exit code. SIGINT/SIGTERM request an orderly stop and return
128 plus the signal number after shutdown.

## Verify the build boundary

```sh
ctest --test-dir build/train --output-on-failure
python3 tools/check_train_build.py build/train --executable build/train/lfs-train
```

The audit checks actual compilation units, installed vcpkg packages and (on
Linux) resolved runtime dependencies. CMake also rejects forbidden desktop
and extra-format targets. These checks do not replace training quality or
performance regressions on a real dataset.

The train-only CI workflow configures and compiles Release and Debug without
installing desktop development packages. It runs the profile/build-graph
contracts; GPU training and driver-dependent runtime checks run separately.

For a small generated GPU regression (training, PLY, project resume and missing
supervision/capability errors), run:

```sh
python3 tools/train_smoke.py build/train/lfs-train --output /tmp/lfs-train-smoke
```

The output directory must be empty. Add `--jpeg` when Pillow is installed to
exercise JPEG inputs; the default uses generated PNG images.

Local validation on 2026-09-06 passed the three CTest checks, the compilation,
package and runtime-dependency audit, and all nine generated GPU smoke cases.
The contextcapture dataset (252 fisheye images, 25,094 initial points) completed
20 RGB-only steps with `--gut --max-width 512 --max-cap 100000 --sh-degree 0`,
exported PLY, and resumed to step 24 using the installed executable. This is
pipeline validation, not a quality or performance comparison with studio.
The Release build and install were tested locally; the new CI workflow and a
full studio/Debug build have not been run locally.

The smoke suite also covers 600-step SH-rest refinement and rejects error logs
even if the executable exits with code zero. A subsequent 1,000-step RGB/GUT run
on contextcapture verified Q16 workspace cleanup after densification: the model
grew from 25,094 to 35,186 Gaussians and saved/exported/exited without CUDA errors.

## Install

```sh
cmake --install build/train --prefix /path/to/lfs-train-install
```

The install includes the LFS shared libraries, OpenMesh data library,
nvImageCodec and its enabled extensions. CUDA runtime libraries and a compatible
NVIDIA driver must be available on the target machine. Use the same system ABI
as the build host; this is not a fully portable cross-distribution bundle.
