#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Score the geometry of a trained splat PLY against a LAS scan and a GeosunPre delivery.

Three independent reports:

* legacy:  per-splat distance to a plane fitted on the LAS, weighted by opacity x area.
* scan:    depth rendered from the delivery's cameras, compared to its LiDAR depth maps,
           for each combination of per-splat depth (center or tilt-corrected) and pixel
           depth (expected or transmittance median).
* surface: F-scores on points back-projected from those renders. The "virtual scan" numbers
           use every pixel; the "uniform" numbers keep one point per voxel on both sides, so
           they do not depend on how densely a surface happens to be sampled. Unlike the
           depth report these are not limited to pixels where the depth map has a value.

The tilt-corrected depth follows RaDe-GS (arXiv 2406.01467): the peak of the Gaussian along
the ray, under a local affine camera. Only supports COLMAP OPENCV_FISHEYE and PINHOLE cameras.

Needs numpy, scipy, opencv and torch with CUDA. Example:

    python tools/splat_vs_las.py --ply splat.ply --dataset work/scene \\
        --las scene.las.xyz.npy --out metrics.json
    python tools/splat_vs_las.py --selftest
"""

import argparse
import json
import struct
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree

FISHEYE = 5
PINHOLE = 1
CAMERA_PARAM_COUNT = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}
TAUS_M = (0.01, 0.02, 0.05)
MIN_RANGE_M = 0.05
MAX_HALF_WIDTH_PX = 256
MIN_FRAGMENT_ALPHA = 0.01
FRAGMENT_CHUNK = 20_000_000
BAND_ROWS = 64
DEPTH_METHODS = ("center_expected", "center_median", "tilted_expected", "tilted_median")


def read_ply(path):
    with open(path, "rb") as f:
        count = 0
        props = []
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            if line.startswith("format") and "binary_little_endian" not in line:
                raise ValueError(f"{path}: only binary little endian PLY is supported")
            if line.startswith("element vertex"):
                count = int(line.split()[2])
            elif line.startswith("property"):
                _, kind, name = line.split()
                props.append((name, {"float": "<f4", "double": "<f8", "uchar": "u1"}[kind]))
            elif line == "end_header":
                break
        data = np.fromfile(f, dtype=np.dtype(props), count=count)
    q = np.stack([data[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float32)
    return {
        "means": np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32),
        "opacity": 1.0 / (1.0 + np.exp(-data["opacity"].astype(np.float32))),
        "scales": np.exp(np.stack([data[f"scale_{i}"] for i in range(3)], axis=1).astype(np.float32)),
        "quats": q / np.linalg.norm(q, axis=1, keepdims=True),
    }


def read_cameras(path):
    cameras = {}
    with open(path, "rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            camera_id, model = struct.unpack("<ii", f.read(8))
            width, height = struct.unpack("<QQ", f.read(16))
            count = CAMERA_PARAM_COUNT[model]
            params = np.array(struct.unpack(f"<{count}d", f.read(8 * count)))
            if model not in (FISHEYE, PINHOLE):
                raise ValueError(f"camera {camera_id}: unsupported COLMAP model {model}")
            cameras[camera_id] = {"model": model, "width": width, "height": height, "params": params}
    return cameras


def read_images(path):
    images = []
    with open(path, "rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            _, qw, qx, qy, qz, tx, ty, tz, camera_id = struct.unpack("<idddddddi", f.read(64))
            name = bytearray()
            while (c := f.read(1)) != b"\0":
                name += c
            f.seek(24 * struct.unpack("<Q", f.read(8))[0], 1)
            images.append(
                {
                    "name": name.decode(),
                    "camera_id": camera_id,
                    "R": quat_to_matrix(np.array([[qw, qx, qy, qz]]))[0],
                    "t": np.array([tx, ty, tz]),
                }
            )
    return sorted(images, key=lambda image: image["name"])


def quat_to_matrix(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        axis=1,
    ).reshape(-1, 3, 3)


def read_image_any(path):
    # cv2.imread cannot open non-ASCII paths on Windows.
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)


class Lens:
    """Projection at a chosen resolution; depth is ray distance for fisheye, z for pinhole."""

    def __init__(self, camera, width, height, device):
        self.model = camera["model"]
        self.width, self.height = width, height
        self.device = device
        p = camera["params"]
        sx, sy = width / camera["width"], height / camera["height"]
        self.fx, self.fy = float(p[0] * sx), float(p[1] * sy)
        self.cx, self.cy = float(p[2] * sx), float(p[3] * sy)
        self.k = [float(v) for v in p[4:8]] if self.model == FISHEYE else [0.0] * 4
        self.max_angle = self._fisheye_max_angle() if self.model == FISHEYE else None

    def _fisheye_max_angle(self):
        # The distortion polynomial can turn back at wide angles, which folds points from behind
        # the camera onto the image center. Same limit as OpenCVFisheyeCameraModel in Cameras.cuh.
        angles = torch.linspace(1e-3, np.pi, 4096, dtype=torch.float64)
        turned = self._distort(angles)[1] <= 0
        turning = angles[turned.nonzero()[0, 0]].item() if turned.any() else np.pi
        radius = np.hypot(max(self.width - self.cx, self.cx), max(self.height - self.cy, self.cy))
        return min(turning, np.pi, max(radius / self.fx, radius / self.fy))

    def in_field_of_view(self, p):
        if self.model == PINHOLE:
            return p[:, 2] > MIN_RANGE_M
        return torch.atan2(torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2), p[:, 2]) < self.max_angle

    def valid_pixels(self):
        """Pixels whose ray exists; the corners of a fisheye image lie beyond its widest angle."""
        v, u = torch.meshgrid(
            torch.arange(self.height, device=self.device, dtype=torch.float32) + 0.5,
            torch.arange(self.width, device=self.device, dtype=torch.float32) + 0.5,
            indexing="ij",
        )
        if self.model == PINHOLE:
            return torch.ones_like(u, dtype=torch.bool)
        limit = self._distort(torch.tensor(self.max_angle, device=self.device, dtype=torch.float32))[0]
        return torch.hypot((u - self.cx) / self.fx, (v - self.cy) / self.fy) < limit

    def _distort(self, theta):
        k1, k2, k3, k4 = self.k
        t2 = theta * theta
        value = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
        slope = 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))
        return value, slope

    def project(self, p):
        """Returns uv, depth and the 3x3 Jacobian of (u, v, depth) with respect to p."""
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        jac = torch.zeros((p.shape[0], 3, 3), device=p.device, dtype=p.dtype)
        if self.model == PINHOLE:
            iz = 1 / z
            uv = torch.stack([self.fx * x * iz + self.cx, self.fy * y * iz + self.cy], dim=1)
            jac[:, 0, 0] = self.fx * iz
            jac[:, 0, 2] = -self.fx * x * iz * iz
            jac[:, 1, 1] = self.fy * iz
            jac[:, 1, 2] = -self.fy * y * iz * iz
            jac[:, 2, 2] = 1
            return uv, z, jac
        rho = torch.sqrt(x * x + y * y).clamp_min(1e-9)
        r2 = x * x + y * y + z * z
        r = torch.sqrt(r2)
        theta = torch.atan2(rho, z)
        value, slope = self._distort(theta)
        s = value / rho
        uv = torch.stack([self.fx * s * x + self.cx, self.fy * s * y + self.cy], dim=1)
        # ds/dq = (slope dtheta/dq - s drho/dq) / rho, with q in (x, y, z).
        dtheta = torch.stack([x * z / (rho * r2), y * z / (rho * r2), -rho / r2], dim=1)
        drho = torch.stack([x / rho, y / rho, torch.zeros_like(x)], dim=1)
        ds = (slope[:, None] * dtheta - s[:, None] * drho) / rho[:, None]
        jac[:, 0] = self.fx * x[:, None] * ds
        jac[:, 1] = self.fy * y[:, None] * ds
        jac[:, 0, 0] += self.fx * s
        jac[:, 1, 1] += self.fy * s
        jac[:, 2] = p / r[:, None]
        return uv, r, jac

    def pixel_rays(self):
        """Camera-space rays scaled so that ray * depth is the point."""
        v, u = torch.meshgrid(
            torch.arange(self.height, device=self.device, dtype=torch.float32) + 0.5,
            torch.arange(self.width, device=self.device, dtype=torch.float32) + 0.5,
            indexing="ij",
        )
        mx, my = (u - self.cx) / self.fx, (v - self.cy) / self.fy
        if self.model == PINHOLE:
            return torch.stack([mx, my, torch.ones_like(mx)], dim=-1)
        target = torch.sqrt(mx * mx + my * my)
        theta = target.clone()
        for _ in range(10):
            value, slope = self._distort(theta)
            theta = theta - (value - target) / slope
        scale = torch.sin(theta) / target.clamp_min(1e-9)
        return torch.stack([mx * scale, my * scale, torch.cos(theta)], dim=-1)


def render_depths(splats, lens, R, t, chunk=FRAGMENT_CHUNK, kinds=("center", "tilted"), levels=()):
    """Composites splats front to back and returns the per-method depth maps and alpha.

    For each transmittance in levels, also returns "<kind>_t<level>": the depth where it is first reached.
    """
    device = lens.device
    R = torch.as_tensor(R, device=device, dtype=torch.float32)
    t = torch.as_tensor(t, device=device, dtype=torch.float32)
    p = splats["means"] @ R.T + t
    uv, depth, jac = lens.project(p)
    keep = (depth > MIN_RANGE_M) & lens.in_field_of_view(p)
    margin = MAX_HALF_WIDTH_PX
    keep &= (uv[:, 0] > -margin) & (uv[:, 0] < lens.width + margin)
    keep &= (uv[:, 1] > -margin) & (uv[:, 1] < lens.height + margin)
    index = torch.nonzero(keep).squeeze(1)
    uv, depth, jac = uv[index], depth[index], jac[index]

    # Covariance of (u, v, depth): J R_cam diag(s^2) R_cam^T J^T, and the affine conditional
    # mean of depth given (u, v) gives the tilted depth of RaDe-GS.
    axes = jac @ (R @ splats["rotations"][index]) * splats["scales"][index][:, None, :]
    cov = axes @ axes.transpose(1, 2)
    a = cov[:, 0, 0] + 1e-4
    b = cov[:, 0, 1]
    c = cov[:, 1, 1] + 1e-4
    det = (a * c - b * b).clamp_min(1e-10)
    inv_a, inv_b, inv_c = c / det, -b / det, a / det
    tilt_u = cov[:, 2, 0] * inv_a + cov[:, 2, 1] * inv_b
    tilt_v = cov[:, 2, 0] * inv_b + cov[:, 2, 1] * inv_c
    sigma_depth = torch.sqrt(cov[:, 2, 2].clamp_min(0))
    opacity = splats["opacity"][index]

    half_w = (3 * torch.sqrt(a)).ceil().clamp_max(MAX_HALF_WIDTH_PX).long()
    half_h = (3 * torch.sqrt(c)).ceil().clamp_max(MAX_HALF_WIDTH_PX).long()
    u_min = (torch.floor(uv[:, 0]).long() - half_w).clamp_min(0)
    u_max = (torch.floor(uv[:, 0]).long() + half_w).clamp_max(lens.width - 1)
    v_min = torch.floor(uv[:, 1]).long() - half_h
    v_max = torch.floor(uv[:, 1]).long() + half_h

    result = {"fragments": 0}
    maps = {f"{kind}_{mode}": torch.full((lens.height, lens.width), float("nan"), device=device)
            for kind in kinds for mode in ("expected", "median")}
    maps["alpha"] = torch.zeros((lens.height, lens.width), device=device)
    for kind in kinds:
        for level in levels:
            maps[f"{kind}_t{level}"] = torch.full((lens.height, lens.width), float("nan"), device=device)

    # Bands of rows keep the fragments of one sort small enough for the GPU.
    for band_start in range(0, lens.height, BAND_ROWS):
        band_stop = min(band_start + BAND_ROWS, lens.height)
        band = torch.nonzero((u_max >= u_min) & (v_max >= band_start) & (v_min < band_stop)).squeeze(1)
        row_min = v_min[band].clamp_min(band_start)
        columns = u_max[band] - u_min[band] + 1
        counts = columns * (v_max[band].clamp_max(band_stop - 1) - row_min + 1)
        pixels_out, center_out, tilted_out, log_t_out = [], [], [], []
        start = 0
        while start < band.numel():
            running = torch.cumsum(counts[start:], dim=0)
            stop = start + max(1, int(torch.searchsorted(running, chunk).item()))
            local = torch.repeat_interleave(torch.arange(start, stop, device=device), counts[start:stop])
            ends = torch.cumsum(counts[start:stop], dim=0)
            offset = torch.arange(local.numel(), device=device) - torch.repeat_interleave(
                ends - counts[start:stop], counts[start:stop]
            )
            rep = band[local]
            px = u_min[rep] + offset % columns[local]
            py = row_min[local] + offset // columns[local]
            du = px.float() + 0.5 - uv[rep, 0]
            dv = py.float() + 0.5 - uv[rep, 1]
            mahalanobis = inv_a[rep] * du * du + 2 * inv_b[rep] * du * dv + inv_c[rep] * dv * dv
            alpha = opacity[rep] * torch.exp(-0.5 * mahalanobis)
            solid = alpha > MIN_FRAGMENT_ALPHA
            rep, du, dv, px, py, alpha = rep[solid], du[solid], dv[solid], px[solid], py[solid], alpha[solid]
            offset_depth = tilt_u[rep] * du + tilt_v[rep] * dv
            limit = 3 * sigma_depth[rep]
            pixels_out.append((py - band_start) * lens.width + px)
            center_out.append(depth[rep])
            tilted_out.append(depth[rep] + torch.maximum(torch.minimum(offset_depth, limit), -limit))
            log_t_out.append(torch.log1p(-alpha.clamp_max(0.9999)))
            start = stop
        if not pixels_out:
            continue
        pixels = torch.cat(pixels_out)
        log_t = torch.cat(log_t_out)
        result["fragments"] += int(pixels.numel())
        band_pixels = (band_stop - band_start) * lens.width
        for kind in kinds:
            values = torch.cat(center_out if kind == "center" else tilted_out)
            expected, median, alpha_map = composite(pixels, values, log_t, band_pixels)
            maps[f"{kind}_expected"][band_start:band_stop] = expected.view(-1, lens.width)
            maps[f"{kind}_median"][band_start:band_stop] = median.view(-1, lens.width)
            maps["alpha"][band_start:band_stop] = alpha_map.view(-1, lens.width)
            for level, depth_map in zip(levels, transmittance_depths(pixels, values, log_t, band_pixels, levels)):
                maps[f"{kind}_t{level}"][band_start:band_stop] = depth_map.view(-1, lens.width)
    result.update(maps)
    return result


def composite(pixels, depth, log_t, pixel_count):
    device = depth.device
    expected = torch.full((pixel_count,), float("nan"), device=device)
    median = torch.full((pixel_count,), float("nan"), device=device)
    alpha = torch.zeros(pixel_count, device=device)
    if pixels.numel() == 0:
        return expected, median, alpha
    order = torch.argsort(depth)
    order = order[torch.argsort(pixels[order], stable=True)]
    pixels, depth, log_t = pixels[order], depth[order], log_t[order]
    is_first = torch.ones_like(pixels, dtype=torch.bool)
    is_first[1:] = pixels[1:] != pixels[:-1]
    group = torch.cumsum(is_first, dim=0) - 1
    # Double precision: tens of millions of terms would drift in a float32 running sum.
    running = torch.cumsum(log_t.double(), dim=0)
    first_index = torch.nonzero(is_first).squeeze(1)
    before_group = running[first_index] - log_t[first_index].double()
    accumulated = running - before_group[group]
    transmit = torch.exp(accumulated)
    weight = torch.exp(accumulated - log_t.double()) - transmit
    first_pixels = pixels[first_index]
    weight_sum = torch.zeros(first_index.numel(), device=device, dtype=torch.float64)
    depth_sum = torch.zeros_like(weight_sum)
    weight_sum.scatter_add_(0, group, weight)
    depth_sum.scatter_add_(0, group, weight * depth.double())
    expected[first_pixels] = (depth_sum / weight_sum.clamp_min(1e-12)).float()
    alpha[first_pixels] = weight_sum.float()
    below = torch.nonzero(transmit < 0.5).squeeze(1)
    if below.numel() > 0:
        groups, first_below = torch.unique_consecutive(group[below], return_inverse=True)
        chosen = torch.full((groups.numel(),), below.numel(), device=device, dtype=torch.long)
        chosen.scatter_reduce_(0, first_below, torch.arange(below.numel(), device=device), "amin")
        median[first_pixels[groups]] = depth[below[chosen]]
    return expected, median, alpha


def transmittance_depths(pixels, depth, log_t, pixel_count, levels):
    """Per pixel and level, the depth of the first fragment that takes transmittance below the level."""
    device = depth.device
    order = torch.argsort(depth)
    order = order[torch.argsort(pixels[order], stable=True)]
    pixels, depth, log_t = pixels[order], depth[order], log_t[order]
    is_first = torch.ones_like(pixels, dtype=torch.bool)
    is_first[1:] = pixels[1:] != pixels[:-1]
    group = torch.cumsum(is_first, dim=0) - 1
    running = torch.cumsum(log_t.double(), dim=0)
    first_index = torch.nonzero(is_first).squeeze(1)
    transmit = torch.exp(running - (running[first_index] - log_t[first_index].double())[group])
    maps = []
    for level in levels:
        result = torch.full((pixel_count,), float("nan"), device=device)
        below = torch.nonzero(transmit < level).squeeze(1)
        if below.numel() > 0:
            groups, inverse = torch.unique_consecutive(group[below], return_inverse=True)
            chosen = torch.full((groups.numel(),), below.numel(), device=device, dtype=torch.long)
            chosen.scatter_reduce_(0, inverse, torch.arange(below.numel(), device=device), "amin")
            result[pixels[first_index][groups]] = depth[below[chosen]]
        maps.append(result)
    return maps


def voxel_dedupe(points, voxel, rng):
    """One random point per voxel, so the result does not depend on the sampling density."""
    keys = np.floor(points / voxel).astype(np.int64)
    keys -= keys.min(axis=0)
    scale = keys.max(axis=0) + 1
    key = keys[:, 0] + scale[0] * (keys[:, 1] + scale[1] * keys[:, 2])
    order = rng.permutation(key.size)
    _, first = np.unique(key[order], return_index=True)
    return points[order[first]]


def voxel_means(points, voxel):
    keys = np.floor(points / voxel).astype(np.int64)
    keys -= keys.min(axis=0)
    scale = keys.max(axis=0) + 1
    key = keys[:, 0] + scale[0] * (keys[:, 1] + scale[1] * keys[:, 2])
    _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    return np.stack([np.bincount(inverse, points[:, i]) / counts for i in range(3)], axis=1)


def nearest(tree, points, chunk=2_000_000):
    out = np.empty(points.shape[0], dtype=np.float32)
    for start in range(0, points.shape[0], chunk):
        out[start : start + chunk] = tree.query(points[start : start + chunk], workers=-1)[0]
    return out


def f_scores(pred_to_gt, gt_to_pred):
    report = {"accuracy_median_mm": float(np.median(pred_to_gt) * 1000),
              "completeness_median_mm": float(np.median(gt_to_pred) * 1000)}
    for tau in TAUS_M:
        precision = float((pred_to_gt < tau).mean())
        recall = float((gt_to_pred < tau).mean())
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        name = f"{int(round(tau * 100))}cm"
        report[f"precision_{name}"] = precision
        report[f"recall_{name}"] = recall
        report[f"f1_{name}"] = f1
    return report


def surface_report(pred, gt, gt_tree, voxel, rng):
    """gt_tree is built on gt. Both protocols use one tree for each direction."""
    pred_tree = cKDTree(pred, balanced_tree=False, compact_nodes=False)
    raw = f_scores(nearest(gt_tree, pred), nearest(pred_tree, gt))
    pred_uniform = voxel_dedupe(pred, voxel, rng)
    gt_uniform = voxel_dedupe(gt, voxel, rng)
    uniform_pred_tree = cKDTree(pred_uniform, balanced_tree=False, compact_nodes=False)
    uniform = f_scores(nearest(gt_tree, pred_uniform), nearest(uniform_pred_tree, gt_uniform))
    return {"virtual_scan": raw, "uniform": uniform, "points": int(pred.shape[0]),
            "uniform_points": int(pred_uniform.shape[0])}


def legacy_report(ply, las):
    """The per-splat plane metric: planes on 1.5 cm LAS voxel means within 6 cm."""
    centers = voxel_means(las, 0.015)
    tree = cKDTree(centers, balanced_tree=False, compact_nodes=False)
    scales = np.sort(ply["scales"], axis=1)
    all_weight = ply["opacity"] * scales[:, 1] * scales[:, 2]
    opaque = ply["opacity"] > 0.5
    means = ply["means"][opaque]
    weight = all_weight[opaque]
    rotations = quat_to_matrix(ply["quats"][opaque])
    thin_axis = np.argmin(ply["scales"][opaque], axis=1)
    normals = rotations[np.arange(means.shape[0]), :, thin_axis]

    distance, neighbor = tree.query(means, k=32, distance_upper_bound=0.06, workers=-1)
    valid = np.isfinite(distance)
    fit = valid.sum(axis=1) >= 5
    neighbor = np.where(valid, neighbor, 0)[fit]
    mask = valid[fit][..., None].astype(np.float64)
    pts = centers[neighbor]
    centroid = (pts * mask).sum(axis=1) / mask.sum(axis=1)
    d = (pts - centroid[:, None]) * mask
    _, vectors = np.linalg.eigh(np.einsum("nki,nkj->nij", d, d))
    plane_normal = vectors[:, :, 0]
    plane_distance = np.abs(((means[fit] - centroid) * plane_normal).sum(axis=1))
    angle = np.degrees(np.arccos(np.clip(np.abs((normals[fit] * plane_normal).sum(axis=1)), 0, 1)))
    w = weight[fit]

    def weighted(values, q):
        order = np.argsort(values)
        cumulative = np.cumsum(w[order])
        return float(values[order][np.searchsorted(cumulative, q * cumulative[-1])])

    return {
        "plane_mm_median_w": weighted(plane_distance, 0.5) * 1000,
        "plane_mm_p90_w": weighted(plane_distance, 0.9) * 1000,
        "las_normal_deg_median_w": weighted(angle, 0.5),
        "far_area_share": float(weight[~fit].sum() / weight.sum()),
        "opaque_splats": int(means.shape[0]),
        "splats": int(ply["means"].shape[0]),
    }


def scan_views(dataset, las, ply, every, device, rng, voxel, max_points_per_view=400_000):
    root = Path(dataset)
    cameras = read_cameras(root / "sparse/0/cameras.bin")
    images = read_images(root / "sparse/0/images.bin")[::every]
    splats = {
        "means": torch.as_tensor(ply["means"], device=device),
        "opacity": torch.as_tensor(ply["opacity"], device=device),
        "scales": torch.as_tensor(ply["scales"], device=device),
        "rotations": torch.as_tensor(quat_to_matrix(ply["quats"]), device=device),
    }
    errors = {method: [] for method in DEPTH_METHODS}
    points = {method: [] for method in DEPTH_METHODS}
    scan_points = []
    prior_alpha = []
    lenses = {}
    fragments = 0
    for i, image in enumerate(images):
        depth_path = root / "depths" / f"{image['name']}.png"
        mask_path = root / "masks" / f"{image['name']}.png"
        prior_u16 = read_image_any(depth_path)
        height, width = prior_u16.shape
        key = (image["camera_id"], width, height)
        if key not in lenses:
            lens = Lens(cameras[image["camera_id"]], width, height, device)
            lenses[key] = (lens, lens.pixel_rays())
        lens, rays = lenses[key]
        static = cv2.resize(read_image_any(mask_path), (width, height), interpolation=cv2.INTER_NEAREST) > 127
        static = torch.as_tensor(static, device=device)
        prior = torch.as_tensor(prior_u16.astype(np.float32) / 1000.0, device=device)
        has_prior = (prior > 0) & static

        rendered = render_depths(splats, lens, image["R"], image["t"])
        fragments += rendered["fragments"]
        alpha = rendered["alpha"]
        solid = alpha > 0.5
        scored = has_prior & solid
        prior_alpha.append(float(solid[has_prior].float().mean()))
        R = torch.as_tensor(image["R"], device=device, dtype=torch.float32)
        t = torch.as_tensor(image["t"], device=device, dtype=torch.float32)

        def to_world(mask, depth):
            return ((rays[mask] * depth[:, None] - t) @ R).double().cpu().numpy()

        for method in DEPTH_METHODS:
            depth = rendered[method]
            errors[method].append((depth[scored] - prior[scored]).cpu().numpy())
            visible = solid & static & torch.isfinite(depth)
            cloud = to_world(visible, depth[visible])
            if cloud.shape[0] > max_points_per_view:
                cloud = cloud[rng.choice(cloud.shape[0], max_points_per_view, replace=False)]
            points[method].append(cloud)
        cloud = to_world(has_prior, prior[has_prior])
        if cloud.shape[0] > max_points_per_view:
            cloud = cloud[rng.choice(cloud.shape[0], max_points_per_view, replace=False)]
        scan_points.append(cloud)
        if i % 10 == 0:
            print(f"  view {i + 1}/{len(images)}", flush=True)

    report = {"views": len(images), "fragments_per_view": fragments // max(len(images), 1),
              "prior_pixels_with_alpha_over_half": float(np.mean(prior_alpha)), "depth": {}}
    for method in DEPTH_METHODS:
        e = np.concatenate(errors[method])
        e = e[np.isfinite(e)]
        report["depth"][method] = {
            "err_median_mm": float(np.median(np.abs(e)) * 1000),
            "err_p90_mm": float(np.percentile(np.abs(e), 90) * 1000),
            "bias_mm": float(np.median(e) * 1000),
            "over_5cm": float((np.abs(e) > 0.05).mean()),
            "pixels": int(e.size),
        }

    gt = voxel_dedupe(las, voxel, rng)
    print(f"  LAS voxels: {gt.shape[0]}; building tree", flush=True)
    gt_tree = cKDTree(gt, balanced_tree=False, compact_nodes=False)
    scan = np.concatenate(scan_points)
    # The LiDAR's own scan from these cameras: LAS regions no camera sees do not count against recall.
    scan_tree = cKDTree(scan, balanced_tree=False, compact_nodes=False)
    report["surface"] = {}
    for method in ("center_median", "tilted_median"):
        pred = np.concatenate(points[method])
        report["surface"][method] = surface_report(pred, scan, scan_tree, voxel, rng)
        report["surface"][method]["distance_to_full_las_median_mm"] = float(np.median(nearest(gt_tree, pred)) * 1000)
        print(f"  surface {method}: done", flush=True)
    return report


def selftest():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)

    camera = {"model": FISHEYE, "width": 2544, "height": 2584,
              "params": np.array([966.0, 966.3, 1271.3, 1304.9, -0.029, -0.0057, 0.0021, -0.00066])}
    lens = Lens(camera, 636, 646, device)
    p = torch.as_tensor(rng.normal(size=(2000, 3)) + [0.0, 0.0, 2.0], device=device, dtype=torch.float64)
    p = p[p[:, 2] > 0.2]
    uv, depth, jac = lens.project(p)
    eps = 1e-6
    for axis in range(3):
        step = torch.zeros(3, device=device, dtype=torch.float64)
        step[axis] = eps
        uv2, depth2, _ = lens.project(p + step)
        uv1, depth1, _ = lens.project(p - step)
        numeric = torch.cat([(uv2 - uv1) / (2 * eps), ((depth2 - depth1) / (2 * eps))[:, None]], dim=1)
        assert (numeric - jac[:, :, axis]).abs().max() < 1e-3, "Jacobian disagrees with finite differences"

    rays = lens.pixel_rays()
    row, col = 300, 200
    ray = rays[row, col].double()[None]
    uv, _, _ = lens.project(ray)
    assert abs(uv[0, 0].item() - (col + 0.5)) < 1e-2 and abs(uv[0, 1].item() - (row + 0.5)) < 1e-2, "ray roundtrip"
    assert abs(rays[row, col].norm().item() - 1) < 1e-5, "fisheye rays must be unit length"

    # A splat straight behind a fisheye camera must not fold back onto the image.
    behind = {
        "means": torch.tensor([[0.01, 0.0, -2.0]], device=device),
        "opacity": torch.tensor([0.95], device=device),
        "scales": torch.tensor([[0.2, 0.2, 0.2]], device=device),
        "rotations": torch.eye(3, device=device)[None],
    }
    assert render_depths(behind, lens, np.eye(3), np.zeros(3))["fragments"] == 0, "splat behind the camera was drawn"

    # A tilted plane of large opaque discs. Tilted depth should recover it, center depth cannot.
    pinhole = Lens({"model": PINHOLE, "width": 640, "height": 640, "params": np.array([400.0, 400.0, 320.0, 320.0])},
                   640, 640, device)
    for current, name in ((pinhole, "pinhole"), (lens, "fisheye")):
        grid = np.stack(np.meshgrid(np.arange(-1, 1, 0.1), np.arange(-1, 1, 0.1)), -1).reshape(-1, 2)
        slope = 0.7
        means = np.stack([grid[:, 0], grid[:, 1], 2 + slope * grid[:, 0]], axis=1).astype(np.float32)
        ex = np.array([1, 0, slope]) / np.hypot(1, slope)
        n = np.cross(ex, [0, 1, 0])
        rotation = np.stack([ex, [0, 1, 0], n], axis=1)
        splats = {
            "means": torch.as_tensor(means, device=device),
            "opacity": torch.full((means.shape[0],), 0.95, device=device),
            "scales": torch.as_tensor(np.tile([0.1, 0.1, 0.001], (means.shape[0], 1)), device=device, dtype=torch.float32),
            "rotations": torch.as_tensor(np.tile(rotation, (means.shape[0], 1, 1)), device=device, dtype=torch.float32),
        }
        out = render_depths(splats, current, np.eye(3), np.zeros(3))
        solid = out["alpha"] > 0.5
        rays_here = current.pixel_rays()
        # Exact range along each ray to the plane z = 2 + slope x.
        truth = 2 / (rays_here[..., 2] - slope * rays_here[..., 0])
        central = solid & (rays_here[..., 2] > 0.75)
        center_err = (out["center_median"] - truth)[central].abs().median().item()
        tilted_err = (out["tilted_median"] - truth)[central].abs().median().item()
        print(f"  {name}: center median error {center_err * 1000:.1f} mm, tilted {tilted_err * 1000:.1f} mm")
        assert tilted_err < center_err / 2, f"{name}: tilted depth should beat center depth on a tilted plane"
        assert tilted_err < 0.02, f"{name}: tilted depth should sit on the plane"

    # Expected depth at one opaque layer in front of another stays near the front one.
    pixels = torch.zeros(2, dtype=torch.long, device=device)
    expected, median, alpha = composite(pixels, torch.tensor([3.0, 1.0], device=device),
                                        torch.log1p(-torch.tensor([0.9, 0.9], device=device)), 1)
    assert abs(median[0].item() - 1) < 1e-6, "median must pick the front layer"
    assert abs(alpha[0].item() - 0.99) < 1e-5, "alpha must be 1 - product of transmittance"
    assert abs(expected[0].item() - (0.9 * 1 + 0.09 * 3) / 0.99) < 1e-5, "expected depth weights front to back"

    # Surface metric: a perfect scan scores 1, noise lowers it, and oversampling one region
    # changes the raw protocol but not the uniform one.
    gt = rng.uniform(0, 3, size=(600_000, 3)) * [1, 1, 0]
    tree = cKDTree(gt)
    same = surface_report(gt.copy(), gt, tree, 0.01, rng)
    assert same["uniform"]["f1_1cm"] > 0.99, "identical clouds must score 1"
    noisy = gt + rng.normal(0, 0.02, size=gt.shape) * [0, 0, 1]
    noisy_report = surface_report(noisy, gt, tree, 0.01, rng)
    assert noisy_report["uniform"]["f1_1cm"] < same["uniform"]["f1_1cm"] - 0.2, "noise must lower the score"

    # 10% of the area sits 5.5 cm off the surface; sampling that part 30x denser must not count against it.
    # Mid-voxel on purpose: a layer straddling a voxel boundary counts twice, so thick surfaces still weigh more.
    shifted = gt + (gt[:, 0] < 0.3)[:, None] * np.array([0, 0, 0.055])
    bad = shifted[gt[:, 0] < 0.3]
    dense_cloud = np.concatenate([shifted] + [bad + rng.normal(0, 1e-4, bad.shape) for _ in range(30)])
    sparse = surface_report(shifted, gt, tree, 0.01, rng)
    dense = surface_report(dense_cloud, gt, tree, 0.01, rng)
    raw_change = abs(dense["virtual_scan"]["precision_1cm"] - sparse["virtual_scan"]["precision_1cm"])
    uniform_change = abs(dense["uniform"]["precision_1cm"] - sparse["uniform"]["precision_1cm"])
    print(f"  oversampling: raw precision moved {raw_change:.4f}, uniform {uniform_change:.4f}")
    assert raw_change > 0.3, "the raw protocol should depend on sampling density here"
    assert uniform_change < 0.02, "the uniform protocol must not depend on sampling density"
    print("selftest passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ply")
    parser.add_argument("--dataset")
    parser.add_argument("--las", help="xyz as a float64 .npy of shape [N, 3]")
    parser.add_argument("--every", type=int, default=8, help="use every Nth image sorted by name")
    parser.add_argument("--voxel", type=float, default=0.01, help="uniform protocol voxel size in meters")
    parser.add_argument("--out")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    if not (args.ply and args.dataset and args.las):
        parser.error("--ply, --dataset and --las are required")

    begin = time.time()
    rng = np.random.default_rng(0)
    device = "cuda"
    las = np.load(args.las)
    ply = read_ply(args.ply)
    print(f"{ply['means'].shape[0]} splats, {las.shape[0]} LAS points", flush=True)
    report = {"ply": args.ply, "legacy": legacy_report(ply, las)}
    print(json.dumps(report["legacy"], indent=2), flush=True)
    with torch.no_grad():
        report["scan"] = scan_views(args.dataset, las, ply, args.every, device, rng, args.voxel)
    report["seconds"] = time.time() - begin
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
