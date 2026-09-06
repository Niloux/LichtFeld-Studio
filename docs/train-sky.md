# Fixed Gaussian sky

`lfs-train --sky` trains a separate set of fixed sky Gaussians behind the
foreground. Positions form a Fibonacci upper hemisphere around the world origin
(+Z up). Isotropic scales are nearest-neighbor distances; rotations are identity
and opacity stays fixed. Only SH0 colors are optimized, initialized to white and
clamped to RGB [1/255, 1]. This matches the SH0 configuration used for the local
3dgeer workflow. Higher-order sky SH is not implemented; foreground SH settings
remain independent.

Sky Gaussians never enter the foreground strategy, densification count, scene
nodes, or PLY export. Their geometry, colors, Adam moments and step are persisted
in the `.licht` checkpoint. Sky supervision discourages foreground opacity in
sky pixels; it does not guarantee that all foreground floaters disappear.

## Dataset and command

The same LiDAR/sky setup is available as
[`configs/contextcapture_gaussian_sky.json`](../configs/contextcapture_gaussian_sky.json):
run `./build/train/lfs-train --config configs/contextcapture_gaussian_sky.json`
from the repository root. See [JSON configuration](train-only.md#json-training-configuration)
for overrides and path handling.

Keep ordinary validity masks separate from sky masks. White means sky and black
means foreground. Masks must match the original image dimensions. Channel zero
supplies coverage for grayscale/RGB masks. The directory mirrors the images:

```text
contextcapture/
  images/L/frame.JPG
  masks/L/frame.JPG.png
  sky_masks/L/frame.JPG.png
```

The sky loader tries `frame.JPG.png`, `frame.png`, then the original filename.
Relative sky directories resolve under the dataset root. All dataset cameras,
including held-out cameras, need matching sky masks. Ordinary `ignore` masks
should retain valid sky pixels so the sky can learn their appearance.

```bash
./build/train/lfs-train \
  -d /home/wuyou/3dgeer/data/contextcapture \
  --init /home/wuyou/3dgeer/data/contextcapture/lidar.ply \
  -o ./output/contextcapture_lidar_gaussian_sky \
  --gut -r 4 --max-width 0 --sh-degree 0 \
  --exposure-correction --mask-mode ignore \
  --iter 30000 --export ply \
  --sky --sky-mask-dir sky_masks --sky-num-points 100000
```

Use a new output directory when switching from cubemap training. Remove the old
`--sky-resolution`, `--sky-boundary-pixels` and `--sky-smoothness` options.

| Option | Default | Meaning |
| --- | --- | --- |
| `--sky` | off | Enable fixed Gaussian sky |
| `--sky-mask-dir` | `sky_masks` | Separate white-sky masks |
| `--sky-num-points` | 100000 | Fixed count, range 16–1000000 |
| `--sky-radius` | 10000 | Hemisphere radius in scene coordinates |
| `--sky-initial-opacity` | 0.7 | Fixed per-Gaussian opacity, strictly between 0 and 1 |
| `--sky-lr` | 0.0025 | SH0 Adam learning rate, epsilon 1e-15 |
| `--sky-alpha-weight` | 0.1 | Mean foreground opacity penalty over valid sky coverage |

## Rendering and losses

The separate sky renders over black. Foreground RGB is composed as
`foreground + (1 - foreground_alpha) * sky`. Exposure correction acts on that
composition and its gradients reach both models. Sky color receives the complete
valid photometric gradient multiplied by foreground transmittance, with no
extra sky-mask gate. The independent sky mask supplies only the foreground alpha
penalty. Area resampling preserves original mask coverage; there is no extra
mask erosion, boundary photometric downweighting or sky-specific densification
map suppression. The foreground strategy retains its usual behavior.

Sky uses the GUT world-space rasterizer with a far plane of 1e10, including native
fisheye models. Foreground rendering can use GUT or FastGS. Sky translation is
finite-distance world-space geometry, unlike the previous direction-only cube.
Exact nearest-neighbor scales and a deterministic zero angular phase are used;
3dgeer uses FLANN and a seeded random angular phase, so this is not a bitwise
replica of its point initialization.

The renderer reuses per-thread buffers. Sky forward results are copied before
foreground rendering. Sky backward rerenders after foreground backward consumes
its context, and preserves foreground images used by the strategy. This costs
an extra sky forward pass per iteration. The mask cache is bounded to 256 MiB of
CPU storage, except that a single unusually large mask can exceed that budget.

## Checkpoints and supported paths

```bash
./build/train/lfs-train \
  --resume ./output/contextcapture_lidar_gaussian_sky/project.licht \
  -o ./output/contextcapture_lidar_gaussian_sky_resumed \
  --iter 40000 --export ply
```

Gaussian sky uses component version 2 inside embedded checkpoint version 4.
Geometry and optimizer state survive resume; changing count, radius or fixed
opacity on resume is rejected. Cubemap component version 1 cannot be resumed as
Gaussian sky. Feature-off checkpoints retain version 3 and remain readable.

Supported paths include mask mode `none`/`ignore`, PPISP without controller,
exposure correction and bilateral grids. Orthographic cameras, pre-undistortion,
other background modes and PPISP controller training are rejected. Use `--gut`
for native fisheye images. Evaluation and timelapse renders include the sky;
foreground PLY and generic viewer renders do not carry this training background.
The dataset must use +Z as up and fit well inside the sky hemisphere radius.

## Validation

With `LFS_BUILD_TRAIN_TESTS=ON`, run the small GPU checks explicitly:

```bash
build/train/lfs-sky-test
python3 tools/train_sky_smoke.py build/train/lfs-train --output /tmp/lfs-gaussian-sky-smoke
```

The component test checks camera models, SH0 finite differences, fixed geometry,
original mask coverage, alpha loss, cache ownership, and Adam continuation.
The smoke harness checks both foreground renderers, exposure correction,
evaluation, checkpoint resume, foreground-only PLY counts, invalid geometry and
feature-off compatibility. These verify implementation behavior; full-scene
quality comparisons are performed separately.

Local validation (2026-09-06): the GPU component test, all nine integration
cases and three CTest checks passed. The actual 252-image fisheye dataset ran
five steps with 1,488,579 LiDAR foreground points and 100,000 sky Gaussians at
maximum image width 512. PLY retained 1,488,579 foreground points. Full-resolution
30,000-step quality validation and a complete Studio build were not run.
