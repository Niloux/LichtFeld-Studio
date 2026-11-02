#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Score how thin and how well oriented a trained splat model is on surfaces known to be flat.

The LAS only picks the patches: 20 cm cells whose points fit a plane within a few millimeters.
On those patches the report gives, split into floor, ceiling and walls:

* thickness: along rays of the held-out views, the depth between 10% and 90% of the accumulated
  alpha, projected on the patch normal so that oblique views do not count extra.
* orientation: angle between each splat's thinnest axis and the patch normal.
* on_surface: share of the splats' weight within 5 mm and 10 mm of the patch plane.

Image quality comes from the metrics.csv next to the PLY. Needs numpy, scipy, opencv and torch
with CUDA. Example:

    python tools/splat_surface_quality.py --ply run/splat.ply --dataset work/scene \\
        --las scene.las.xyz.npy --out run/surface.json
    python tools/splat_surface_quality.py --selftest
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import splat_vs_las as sv
import splats_to_mesh as m

PATCH_SIZE_M = 0.2
PATCH_VOXEL_M = 0.015
PATCH_MAX_RMS_M = 0.003
PATCH_MIN_FILL = 0.6
MAX_RANGE_M = 8.0
MIN_COSINE = 0.5
OCCLUSION_TOLERANCE_M = 0.05
SURFACE_BAND_M = 0.15
LEVELS = (0.9, 0.1)
CLASSES = ("floor", "ceiling", "wall", "other")


def flat_patches(points, max_rms=PATCH_MAX_RMS_M, min_fill=PATCH_MIN_FILL):
    """Fits a plane per cell on voxel means and keeps the well-filled cells that are flat."""
    means = sv.voxel_means(points, PATCH_VOXEL_M)
    cell = np.floor(means / PATCH_SIZE_M).astype(np.int64)
    cell -= cell.min(axis=0)
    scale = cell.max(axis=0) + 1
    _, inverse, counts = np.unique(cell[:, 0] + scale[0] * (cell[:, 1] + scale[1] * cell[:, 2]), return_inverse=True, return_counts=True)
    size = counts.size
    center = np.stack([np.bincount(inverse, means[:, i], size) / counts for i in range(3)], axis=1)
    offset = means - center[inverse]
    covariance = np.zeros((size, 3, 3))
    for i in range(3):
        for j in range(i, 3):
            covariance[:, i, j] = covariance[:, j, i] = np.bincount(inverse, offset[:, i] * offset[:, j], size) / counts
    values, vectors = np.linalg.eigh(covariance)
    rms = np.sqrt(np.maximum(values[:, 0], 0))
    # A cell the plane crosses at an angle holds less than a full square, so the fill test stays loose.
    full = (PATCH_SIZE_M / PATCH_VOXEL_M) ** 2
    keep = (rms < max_rms) & (counts > min_fill * full)
    normal = vectors[keep, :, 0]
    horizontal = np.abs(normal[:, 2]) > 0.9
    split = np.median(center[keep][horizontal, 2]) if horizontal.any() else 0.0
    kind = np.full(keep.sum(), 3, dtype=np.int8)
    kind[horizontal & (center[keep][:, 2] <= split)] = 0
    kind[horizontal & (center[keep][:, 2] > split)] = 1
    kind[np.abs(normal[:, 2]) < 0.2] = 2
    return {"center": center[keep], "normal": normal, "kind": kind, "rms": rms[keep]}


def splat_report(ply, patches):
    tree = cKDTree(patches["center"], balanced_tree=False, compact_nodes=False)
    means = ply["means"].astype(np.float64)
    distance, nearest = tree.query(means, distance_upper_bound=PATCH_SIZE_M * 0.75, workers=-1)
    found = np.isfinite(distance)
    index, chosen = np.nonzero(found)[0], nearest[found]
    offset = ((means[index] - patches["center"][chosen]) * patches["normal"][chosen]).sum(axis=1)
    near = np.abs(offset) < SURFACE_BAND_M
    index, chosen, offset = index[near], chosen[near], offset[near]
    scales = ply["scales"][index]
    ordered = np.sort(scales, axis=1)
    weight = ply["opacity"][index] * ordered[:, 1] * ordered[:, 2]
    rotation = sv.quat_to_matrix(ply["quats"][index])
    axis = rotation[np.arange(index.size), :, np.argmin(scales, axis=1)]
    angle = np.degrees(np.arccos(np.clip(np.abs((axis * patches["normal"][chosen]).sum(axis=1)), 0, 1)))

    def weighted(values, q, subset):
        order = np.argsort(values[subset])
        cumulative = np.cumsum(weight[subset][order])
        return float(values[subset][order][np.searchsorted(cumulative, q * cumulative[-1])])

    report = {}
    for label, code in (("all", None), *zip(CLASSES, range(4))):
        subset = np.ones(index.size, dtype=bool) if code is None else patches["kind"][chosen] == code
        if not subset.any():
            continue
        share = lambda band: float(weight[subset & (np.abs(offset) < band)].sum() / weight[subset].sum())
        report[label] = {"normal_deg_p50": weighted(angle, 0.5, subset), "normal_deg_p90": weighted(angle, 0.9, subset),
                         "on_surface_5mm": share(0.005), "on_surface_10mm": share(0.01), "splats": int(subset.sum())}
    return report


def thickness_report(ply_path, dataset, patches, device, per_view=4000):
    root = Path(dataset)
    cameras = sv.read_cameras(root / "sparse/0/cameras.bin")
    held_out = sv.read_images(root / "sparse/0/images.bin")[::8]
    splats = m.load_splats(ply_path, device)
    center = torch.as_tensor(patches["center"], device=device, dtype=torch.float32)
    normal = torch.as_tensor(patches["normal"], device=device, dtype=torch.float32)
    kind = torch.as_tensor(patches["kind"], device=device)
    lenses = {}
    spans = {code: [] for code in range(4)}
    bias = {code: [] for code in range(4)}
    hits = 0
    for image in held_out:
        lens, _, mask, R, t, prior = m.load_view(root, image, cameras, lenses, device)
        p = center @ R.T + t
        uv, r, _ = lens.project(p)
        px, py = torch.floor(uv[:, 0]).long(), torch.floor(uv[:, 1]).long()
        ok = lens.in_field_of_view(p) & (r < MAX_RANGE_M) & (px >= 0) & (px < lens.width) & (py >= 0) & (py < lens.height)
        ids = torch.nonzero(ok).squeeze(1)
        if ids.numel() == 0:
            continue
        lidar = torch.as_tensor(prior.astype(np.float32) / 1000.0, device=device)
        cosine = ((center[ids] + (t @ R)) * normal[ids]).sum(dim=1).abs() / r[ids]
        qx, qy = px[ids], py[ids]
        clear = mask[qy, qx] & (lidar[qy, qx] > 0) & ((lidar[qy, qx] - r[ids]).abs() < OCCLUSION_TOLERANCE_M) & (cosine > MIN_COSINE)
        ids, qx, qy, cosine = ids[clear], qx[clear], qy[clear], cosine[clear]
        if ids.numel() == 0:
            continue
        rendered = sv.render_depths(splats, lens, image["R"], image["t"], kinds=("tilted",), levels=LEVELS)
        front, back = rendered[f"tilted_t{LEVELS[0]}"][qy, qx], rendered[f"tilted_t{LEVELS[1]}"][qy, qx]
        median = rendered["tilted_median"][qy, qx]
        valid = torch.isfinite(front) & torch.isfinite(back)
        span = (back - front)[valid] * cosine[valid]
        codes = kind[ids[valid]]
        error = (median - lidar[qy, qx])[valid]
        hits += int(valid.sum())
        for code in range(4):
            spans[code].append(span[codes == code].cpu().numpy())
            bias[code].append(error[codes == code].cpu().numpy())

    report = {"samples": hits, "views": len(held_out)}
    every = np.concatenate([np.concatenate(v) for v in spans.values() if v]) if hits else np.array([])
    if every.size:
        report["all"] = {"thickness_mm_p50": float(np.median(every) * 1000), "thickness_mm_p90": float(np.percentile(every, 90) * 1000)}
    for code, label in enumerate(CLASSES):
        values = np.concatenate(spans[code]) if spans[code] else np.array([])
        if values.size == 0:
            continue
        errors = np.concatenate(bias[code])
        report[label] = {"thickness_mm_p50": float(np.median(values) * 1000), "thickness_mm_p90": float(np.percentile(values, 90) * 1000),
                         "median_depth_vs_lidar_mm_p50": float(np.median(np.abs(errors)) * 1000), "samples": int(values.size)}
    return report


def image_report(ply_path):
    path = Path(ply_path).parent / "metrics.csv"
    if not path.exists():
        return None
    row = list(csv.DictReader(open(path)))[-1]
    return {key: float(row[key]) for key in ("iteration", "psnr", "ssim", "lpips")}


def selftest():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)

    # Patches: a flat floor and wall are found, a rough slab is not.
    floor = np.stack([rng.uniform(0, 2, 400_000), rng.uniform(0, 2, 400_000), rng.normal(0, 0.001, 400_000)], axis=1)
    wall = np.stack([rng.uniform(0, 2, 400_000), np.full(400_000, 3.0) + rng.normal(0, 0.001, 400_000), rng.uniform(0, 2, 400_000)], axis=1)
    rough = np.stack([rng.uniform(5, 7, 400_000), rng.uniform(0, 2, 400_000), rng.normal(0, 0.03, 400_000)], axis=1)
    patches = flat_patches(np.concatenate([floor, wall, rough]))
    centers = patches["center"]
    assert not np.any(centers[:, 0] > 4.5), "a rough surface must not make a flat patch"
    assert np.sum(patches["kind"] == 2) > 20 and np.sum(np.abs(patches["normal"][:, 2]) > 0.9) > 20, "must find the wall and the floor"
    assert np.abs(np.abs(patches["normal"][patches["kind"] == 2, 1]) - 1).max() < 0.01, "wall normals must point along y"

    # Orientation and on-surface share: splats lying on a plane against ones standing on it and floating 3 cm off.
    def splats_on_floor(z_offset, rotation_about_x):
        count = 20000
        c, s = np.cos(rotation_about_x / 2), np.sin(rotation_about_x / 2)
        return {"means": np.stack([rng.uniform(0.2, 1.8, count), rng.uniform(0.2, 1.8, count), np.full(count, z_offset)], axis=1).astype(np.float32),
                "opacity": np.full(count, 0.9, dtype=np.float32), "scales": np.tile(np.array([0.01, 0.01, 0.0005], dtype=np.float32), (count, 1)),
                "quats": np.tile(np.array([c, s, 0, 0], dtype=np.float32), (count, 1))}

    flat = splat_report(splats_on_floor(0.0, 0.0), patches)["floor"]
    tilted = splat_report(splats_on_floor(0.0, np.radians(30)), patches)["floor"]
    floating = splat_report(splats_on_floor(0.03, 0.0), patches)["floor"]
    print(f"  orientation {flat['normal_deg_p50']:.1f} and {tilted['normal_deg_p50']:.1f} deg, on surface {flat['on_surface_5mm']:.2f} and {floating['on_surface_5mm']:.2f}")
    assert flat["normal_deg_p50"] < 0.5 and abs(tilted["normal_deg_p50"] - 30) < 0.5, "angle to the patch normal"
    assert flat["on_surface_5mm"] > 0.99 and floating["on_surface_5mm"] < 0.01 and floating["on_surface_10mm"] < 0.01, "on-surface share"

    # Thickness: one opaque layer against two half-transparent layers 5 cm apart.
    lens = sv.Lens({"model": sv.PINHOLE, "width": 200, "height": 200, "params": np.array([200.0, 200.0, 100.0, 100.0])}, 200, 200, device)
    grid = torch.stack(torch.meshgrid(torch.arange(-0.6, 0.6, 0.01), torch.arange(-0.6, 0.6, 0.01), indexing="ij"), dim=-1).reshape(-1, 2).to(device)

    def wall_of(depths, opacity):
        means = torch.cat([torch.cat([grid, torch.full((grid.shape[0], 1), d, device=device)], dim=1) for d in depths])
        count = means.shape[0]
        return {"means": means, "opacity": torch.full((count,), opacity, device=device), "scales": torch.tensor([0.01, 0.01, 0.0005], device=device).repeat(count, 1),
                "rotations": torch.eye(3, device=device).repeat(count, 1, 1)}

    spans = []
    for splats in (wall_of([1.0], 0.95), wall_of([1.0, 1.05], 0.3)):
        out = sv.render_depths(splats, lens, np.eye(3), np.zeros(3), kinds=("tilted",), levels=LEVELS)
        span = (out["tilted_t0.1"] - out["tilted_t0.9"])[60:140, 60:140]
        spans.append(torch.nanmedian(span).item())
    print(f"  thickness: one layer {spans[0] * 1000:.1f} mm, two layers 5 cm apart {spans[1] * 1000:.1f} mm")
    assert spans[0] < 0.002, "one layer has no depth"
    assert 0.045 < spans[1] < 0.055, "two layers span their gap"
    print("selftest passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ply")
    parser.add_argument("--dataset")
    parser.add_argument("--las", help="xyz as a float64 .npy of shape [N, 3]")
    parser.add_argument("--patches", help="cache of the flat patches, so models are compared on the same ones")
    parser.add_argument("--patch-max-rms", type=float, default=PATCH_MAX_RMS_M, help="meters; a noisier or sparser scan needs more")
    parser.add_argument("--patch-min-fill", type=float, default=PATCH_MIN_FILL)
    parser.add_argument("--out")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        with torch.no_grad():
            selftest()
        return
    if not (args.ply and args.dataset and args.las):
        parser.error("--ply, --dataset and --las are required")
    if args.patches and Path(args.patches).exists():
        patches = dict(np.load(args.patches))
    else:
        patches = flat_patches(np.load(args.las), args.patch_max_rms, args.patch_min_fill)
        if args.patches:
            np.savez(args.patches, **patches)
    counts = {label: int((patches["kind"] == code).sum()) for code, label in enumerate(CLASSES)}
    with torch.no_grad():
        report = {"ply": args.ply, "patches": counts, "image": image_report(args.ply),
                  "splats": splat_report(sv.read_ply(args.ply), patches),
                  "rays": thickness_report(args.ply, args.dataset, patches, "cuda")}
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
