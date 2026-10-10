# Fixed Gaussian sky

## Evaluate training views without holding out images

To train all views while exporting evaluation images and metrics, set
`dataset.use_test_split=false` and `optimization.enable_eval=true` in your JSON
configuration. In this mode `dataset.test_every=8` samples every eighth image
independently for each physical camera, in camera UID order. For example, a
252-image stereo dataset with 126 images per camera trains all 252 images and
evaluates 32 images (16 per camera).

These are **training-view metrics**, not held-out validation. The log and the
text report identify the evaluation split; each `eval_step_*` folder also has
an `evaluation_manifest.json` mapping images to filenames and camera IDs.
The new flag defaults to `true`, preserving existing held-out splits. It is
saved in configs and projects. Disabling `enable_eval` still trains all views
without evaluation. Use a separate output directory for each experiment.

## Sky training

`lfs-train --sky` trains a separate set of fixed sky Gaussians behind the
foreground. Positions form a Fibonacci upper hemisphere around the world origin
(+Z up). Isotropic scales are nearest-neighbor distances; rotations are identity
and opacity stays fixed. Only SH0 colors are optimized, initialized to white and
clamped to RGB [1/255, 1]. This matches the SH0 configuration used for the local
3dgeer workflow. Higher-order sky SH is not implemented; foreground SH settings
remain independent.

Sky Gaussians never enter the foreground strategy, densification count, scene
nodes, or foreground PLY. With `--export ply`, final export additionally writes
`sky.ply` in the output directory, using the standard Gaussian PLY fields for
positions, SH0 colors, opacity, scales, and rotations. The foreground file retains
its usual name and point count. Sky geometry, colors, Adam moments and step are
persisted in the `.licht` checkpoint. With mask mode `none`/`ignore`, sky training retains
the unmasked composition path and has no foreground-opacity penalty.

For joint foreground/sky supervision, use `sky_enabled=true`,
`mask_mode="segment_and_ignore"`, and `invert_masks=false`. The upstream bands
remain unchanged: 0–127 ignores the pixel, 128–250 segments it, and 251–255 keeps
it. Author dynamic objects as 0, sky as 128, and static scene content as 255.
The segment band's existing `mask_opacity_penalty_weight` penalizes accumulated
**foreground** alpha; it does not penalize the separate sky Gaussians. The power,
weight, and image-wide normalization of this penalty retain upstream semantics.
Without Gaussian sky enabled, `segment_and_ignore` also retains its upstream
behavior, including no RGB supervision in the segment band.

## Dataset and command

Configure your dataset, initialization PLY and sky parameters in a full JSON
parameter snapshot, then enable `optimization.sky_enabled`. The current
[`configs/展厅.json`](../configs/展厅.json) is an indoor example with sky disabled.
See [JSON configuration](train-only.md#json-training-configuration) for overrides
and path handling.

No sky masks are required for `none`/`ignore`. Joint supervision uses the existing
three-band mask files; loading preserves their labels rather than binarizing 128.

```bash
./build/train/lfs-train \
  -d /home/wuyou/3dgeer/data/contextcapture \
  --init /home/wuyou/3dgeer/data/contextcapture/lidar.ply \
  -o ./output/contextcapture_lidar_gaussian_sky \
  --gut -r 4 --max-width 0 --sh-degree 0 \
  --exposure-correction --mask-mode ignore \
  --iter 30000 --export ply \
  --sky --sky-num-points 100000
```

Use a new output directory when switching from cubemap training. Remove the old
`--sky-resolution`, `--sky-boundary-pixels` and `--sky-smoothness` options.

| Option | Default | Meaning |
| --- | --- | --- |
| `--sky` | off | Enable fixed Gaussian sky |
| `--sky-num-points` | 100000 | Fixed count, range 16–1000000 |
| `--sky-radius` | 10000 | Hemisphere radius in scene coordinates |
| `--sky-initial-opacity` | 0.7 | Fixed per-Gaussian opacity, strictly between 0 and 1 |
| `--sky-lr` | 0.0025 | SH0 Adam learning rate, epsilon 1e-15 |

## Rendering and losses

The separate sky renders over black. Foreground RGB is composed as
`foreground + (1 - foreground_alpha) * sky`. Exposure correction acts on that
composition and its gradients reach both models. Sky color receives the complete
valid photometric gradient multiplied by foreground transmittance. The foreground
strategy retains its usual behavior.

In joint mode, keep pixels retain this composite photometric loss and the original
foreground strategy. Segment pixels additionally supervise the independent sky
render with masked L1 RGB loss, normalized by the segment/ROI weights. This branch
uses the same exposure/PPISP/bilateral-grid order as the foreground and sends its
image gradient only to the sky. It is independent of foreground transmittance, so
initial foreground occlusion does not starve sky color learning. An image with no
segment pixels contributes zero sky-only loss.

Foreground RGB gradients are zeroed outside the keep band after appearance
backward, including any neighboring SSIM-window gradients. Foreground error maps
continue to use upstream keep-band masking. In gradient-norm densification, the
segment alpha derivative is excluded from the norm while still updating foreground
opacity and geometry. GUT replays backward for this norm only when no pixel-error
map is used; FastGS accumulates the separate norm derivative in its existing pass.
Sky count and geometry stay fixed in every mode. Sky mislabeled as keep can still
drive foreground reconstruction.

Sky uses the GUT world-space rasterizer with a far plane of 1e10, including native
fisheye models. Foreground rendering can use GUT or FastGS. Sky translation is
finite-distance world-space geometry, unlike the previous direction-only cube.
Exact nearest-neighbor scales and a deterministic zero angular phase are used;
3dgeer uses FLANN and a seeded random angular phase, so this is not a bitwise
replica of its point initialization.

The renderer reuses per-thread buffers. Sky forward results are copied before
foreground rendering. Sky backward rerenders after foreground backward consumes
its context, and preserves foreground images used by the strategy. This costs
an extra sky forward pass per iteration.

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

Supported paths include mask mode `none`/`ignore`/`segment_and_ignore`, PPISP without controller,
exposure correction and bilateral grids. Orthographic cameras, pre-undistortion,
other background modes and PPISP controller training are rejected. Use `--gut`
for native fisheye images. Evaluation and timelapse renders include the sky;
foreground PLY and generic viewer renders do not carry this training background.
The separate `sky.ply` can be loaded as a Gaussian model for inspection.
The dataset must use +Z as up and fit well inside the sky hemisphere radius.

## Validation

With `LFS_BUILD_TRAIN_TESTS=ON`, run the small GPU checks explicitly:

```bash
build/train/lfs-sky-test
python3 tools/train_sky_smoke.py build/train/lfs-train --output /tmp/lfs-gaussian-sky-smoke
```

The component test checks the upstream mask bands in byte/normalized-float form,
sky-only color gradients, foreground alpha penalties, gradient-norm densification
isolation in both renderers, shared appearance accumulation, camera models, SH0
finite differences, fixed geometry, and Adam continuation. The smoke harness checks both foreground renderers,
exposure correction, evaluation, checkpoint resume, foreground-only PLY counts,
invalid geometry and feature-off compatibility. These verify implementation
behavior; full-scene quality comparisons are performed separately.

Local validation (2026-09-06): the GPU component test, all nine integration
cases and three CTest checks passed. The actual 252-image fisheye dataset ran
five steps with 1,488,579 LiDAR foreground points and 100,000 sky Gaussians at
maximum image width 512. PLY retained 1,488,579 foreground points. Full-resolution
30,000-step quality validation and a complete Studio build were not run.

Local joint-mask validation (2026-10-10): `lfs-train` and the focused GPU checks
compiled; `lfs-sky-test` and `lfs-ppisp-test` passed. Four real fisheye views with
0/128/255 masks completed 32 steps using 100,000 uniformly sampled LiDAR
initialization points and the fixed 100,000-point sky. PPISP and the bilateral
grid were active from the first step. The run saved a `.licht` project and
exported exactly 100,000 foreground points, with no CUDA error in the log.
This is a reduced-input pipeline check; it does not establish full-scene floater
removal or a reconstruction-quality improvement.

Local sky PLY export validation (2026-10-10): the incremental `lfs-train` build
passed. Reloading the 32-step project with the same iteration limit exported
`sky.ply` with 100,000 Gaussians and all standard Gaussian attributes. Every
field was finite; positions retained the fixed +Z hemisphere and radius 10000.
The foreground PLY's binary vertex payload was identical to the original export.
