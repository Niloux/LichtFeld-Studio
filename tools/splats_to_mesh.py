#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Extract a vertex-colored mesh from a trained splat PLY, offline.

Renders transmittance median depth from the training cameras of a GeosunPre delivery, fuses it
into a sparse TSDF, extracts the surface with naive surface nets and colors it from the photos.
Every 8th image (LichtFeld's held-out views) is left out of the fusion so --eval can score the
mesh on views it never saw. The fisheye projection is shared with tools/splat_vs_las.py.

The application's mesh subcommand does the same natively; this tool stays as its reference and to
score meshes, with --mesh. Needs numpy, scipy, opencv and torch with CUDA. Example:

    python tools/splats_to_mesh.py --ply splat.ply --dataset work/scene --out mesh_dir \\
        --las scene.las.xyz.npy --eval
    python tools/splats_to_mesh.py --mesh mesh.ply --dataset work/scene
    python tools/splats_to_mesh.py --selftest
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import splat_vs_las as sv

KEY_OFFSET = 1 << 20
KEY_BITS = 21
KEY_MASK = (1 << KEY_BITS) - 1
AXIS_STEP = (1, 1 << KEY_BITS, 1 << (2 * KEY_BITS))
TILE_PX = 32
MAX_TSDF_BYTES = 18 * 1024**3
HELD_OUT_EVERY = 8


def encode(ijk):
    return (ijk[..., 0] + KEY_OFFSET) | ((ijk[..., 1] + KEY_OFFSET) << KEY_BITS) | ((ijk[..., 2] + KEY_OFFSET) << (2 * KEY_BITS))


def decode(keys):
    return torch.stack(
        [(keys & KEY_MASK) - KEY_OFFSET, ((keys >> KEY_BITS) & KEY_MASK) - KEY_OFFSET,
         ((keys >> (2 * KEY_BITS)) & KEY_MASK) - KEY_OFFSET], dim=-1)


def remove_depth_edges(depth, relative=0.03, absolute=0.02):
    """Drops pixels on depth steps, whose values mix the surfaces on both sides."""
    valid = depth > 0
    fill = depth.clone()
    fill[~valid] = float("inf")
    low = -F.max_pool2d(-fill[None, None], 3, 1, 1)[0, 0]
    fill[~valid] = float("-inf")
    high = F.max_pool2d(fill[None, None], 3, 1, 1)[0, 0]
    return torch.where(valid & (high - low < relative * depth + absolute), depth, torch.zeros_like(depth))


class SparseTsdf:
    def __init__(self, voxel, block, trunc_voxels, device, full_weight_range=None):
        self.voxel, self.block = voxel, block
        self.full_weight_range = full_weight_range
        self.trunc = voxel * trunc_voxels
        self.device = device
        self.block_size = voxel * block
        self.radius = float(np.sqrt(3) / 2 * self.block_size)
        self.keys = torch.empty(0, device=device, dtype=torch.int64)
        axis = torch.arange(block, device=device)
        self.local = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(-1, 3)

    def touch(self, depth, rays, R, t):
        """Allocates the blocks around this view's surface."""
        valid = depth > 0
        valid[1::2] = False
        valid[:, 1::2] = False
        keys = [self.keys]
        for offset in (-self.trunc, 0.0, self.trunc):
            world = (rays[valid] * (depth[valid] + offset)[:, None] - t) @ R
            keys.append(torch.unique(encode(torch.floor(world / self.block_size).long())))
        self.keys = torch.unique(torch.cat(keys))

    def allocate(self):
        bytes_needed = self.keys.numel() * self.local.shape[0] * 10
        if bytes_needed > MAX_TSDF_BYTES:
            raise RuntimeError(f"{self.keys.numel()} blocks need {bytes_needed / 1024**3:.1f} GB; use a larger --voxel")
        count = self.keys.numel() * self.local.shape[0]
        self.sum = torch.zeros(count, device=self.device)
        self.weight = torch.zeros(count, device=self.device)
        self.views = torch.zeros(count, device=self.device, dtype=torch.int16)
        self.carve = torch.zeros(self.keys.numel(), device=self.device)
        self.block_ijk = decode(self.keys)
        self.block_centers = ((self.block_ijk + 0.5) * self.block_size).float()

    def integrate(self, lens, depth, R, t, chunk_blocks=20_000):
        """Blocks wholly in front of the surface only count as free space; blocks wholly behind it are skipped."""
        device = self.device
        p = self.block_centers @ R.T + t
        uv, r, _ = lens.project(p)
        px, py = torch.floor(uv[:, 0]).long(), torch.floor(uv[:, 1]).long()
        visible = lens.in_field_of_view(p) & (r > sv.MIN_RANGE_M) & (px >= 0) & (px < lens.width) & (py >= 0) & (py < lens.height)
        index = torch.nonzero(visible).squeeze(1)
        if index.numel() == 0:
            return
        low, high = self._tile_bounds(depth)
        tx, ty = px[index] // TILE_PX, py[index] // TILE_PX
        block_low, block_high = low[ty, tx], high[ty, tx]
        range_c = r[index]
        seen = torch.isfinite(block_high)
        spans_tiles = max(lens.fx, lens.fy) * self.radius / range_c > TILE_PX
        behind = ~spans_tiles & (range_c - self.radius > block_high + self.trunc)
        front = ~spans_tiles & seen & (range_c + self.radius < block_low - self.trunc)
        self.carve[index[front]] += self._view_weight(range_c[front])
        near = index[(seen | spans_tiles) & ~behind & ~front]
        size = self.local.shape[0]
        for start in range(0, near.numel(), chunk_blocks):
            blocks = near[start : start + chunk_blocks]
            ijk = self.block_ijk[blocks][:, None] * self.block + self.local[None]
            ids = (blocks[:, None] * size + torch.arange(size, device=device)[None]).reshape(-1)
            p = ijk.reshape(-1, 3).float() * self.voxel @ R.T + t
            uv, r, _ = lens.project(p)
            px, py = torch.floor(uv[:, 0]).long(), torch.floor(uv[:, 1]).long()
            ok = lens.in_field_of_view(p) & (r > sv.MIN_RANGE_M) & (px >= 0) & (px < lens.width) & (py >= 0) & (py < lens.height)
            ids, r, px, py = ids[ok], r[ok], px[ok], py[ok]
            measured = depth[py, px]
            sdf = measured - r
            keep = (measured > 0) & (sdf > -self.trunc)
            weight = self._view_weight(r[keep])
            self.sum[ids[keep]] += weight * (sdf[keep] / self.trunc).clamp_max(1)
            self.weight[ids[keep]] += weight
            self.views[ids[keep]] += 1

    def _view_weight(self, r):
        # A pixel's footprint and depth noise grow with range, so close views should win where they exist.
        if self.full_weight_range is None:
            return torch.ones_like(r)
        return (self.full_weight_range / r).clamp_max(1) ** 2

    def _tile_bounds(self, depth):
        valid = depth > 0
        fill = depth.clone()
        fill[~valid] = float("inf")
        low = -F.max_pool2d(-fill[None, None], TILE_PX, TILE_PX, ceil_mode=True)
        fill[~valid] = float("-inf")
        high = F.max_pool2d(fill[None, None], TILE_PX, TILE_PX, ceil_mode=True)
        low = -F.max_pool2d(-low, 3, 1, 1)[0, 0]
        high = F.max_pool2d(high, 3, 1, 1)[0, 0]
        high[high == float("-inf")] = float("nan")
        return low, high

    def voxels(self, min_weight):
        """Returns sorted voxel keys and TSDF values inside the truncation band."""
        size = self.local.shape[0]
        carve = self.carve.repeat_interleave(size)
        total = self.weight + carve
        tsdf = (self.sum + carve) / total.clamp_min(1e-6)
        keep = torch.nonzero((self.views >= min_weight) & (tsdf.abs() < 1)).squeeze(1)
        ijk = self.block_ijk[keep // size] * self.block + self.local[keep % size]
        keys = encode(ijk)
        order = torch.argsort(keys)
        return keys[order], tsdf[keep][order]


def surface_nets(keys, tsdf, voxel, max_step=1.5):
    """Naive surface nets on sorted sparse voxels. Returns vertices and outward-facing triangles."""
    device = keys.device
    positions, cell_keys, faces_axis = [], [], []
    for axis in range(3):
        other = keys + AXIS_STEP[axis]
        found = torch.searchsorted(keys, other).clamp_max(keys.numel() - 1)
        hit = keys[found] == other
        a_index = torch.nonzero(hit).squeeze(1)
        b_index = found[hit]
        a, b = tsdf[a_index], tsdf[b_index]
        # A jump from free space to the far end of the band is where two observations meet, not a surface.
        # Grazing views steepen real crossings well past one voxel's share of the band, so this stays permissive.
        crossing = ((a < 0) != (b < 0)) & ((a - b).abs() < max_step)
        a_index, a, b = a_index[crossing], a[crossing], b[crossing]
        base_keys = keys[a_index]
        point = decode(base_keys).float()
        point[:, axis] += a / (a - b)
        j, k = (axis + 1) % 3, (axis + 2) % 3
        cells = torch.stack([base_keys, base_keys - AXIS_STEP[j], base_keys - AXIS_STEP[j] - AXIS_STEP[k],
                             base_keys - AXIS_STEP[k]], dim=1)
        flip = a >= 0
        cells[flip] = cells[flip].flip(1)
        positions.append(point * voxel)
        cell_keys.append(cells)
    positions = torch.cat(positions)
    cells = torch.cat(cell_keys)
    if cells.numel() == 0:
        return torch.empty((0, 3), device=device), torch.empty((0, 3), device=device, dtype=torch.long)
    unique, inverse = torch.unique(cells.reshape(-1), return_inverse=True)
    inverse = inverse.reshape(-1, 4)
    vertices = torch.zeros((unique.numel(), 3), device=device, dtype=torch.float64)
    counts = torch.zeros(unique.numel(), device=device, dtype=torch.float64)
    repeated = positions.double().repeat_interleave(4, dim=0)
    vertices.index_add_(0, inverse.reshape(-1), repeated)
    counts.index_add_(0, inverse.reshape(-1), torch.ones(repeated.shape[0], device=device, dtype=torch.float64))
    vertices = vertices / counts[:, None]
    faces = torch.cat([inverse[:, [0, 1, 2]], inverse[:, [0, 2, 3]]])
    return vertices.float(), faces


def remove_small_components(vertices, faces, min_faces):
    f = faces.cpu().numpy()
    v = vertices.shape[0]
    graph = coo_matrix((np.ones(f.shape[0] * 2, dtype=bool), (np.r_[f[:, 0], f[:, 0]], np.r_[f[:, 1], f[:, 2]])), shape=(v, v))
    _, labels = connected_components(graph, directed=False)
    face_labels = labels[f[:, 0]]
    sizes = np.bincount(face_labels, minlength=labels.max() + 1)
    keep_face = sizes[face_labels] >= min_faces
    used, remap = np.unique(f[keep_face], return_inverse=True)
    return vertices[torch.as_tensor(used, device=vertices.device)], \
        torch.as_tensor(remap.reshape(-1, 3), device=vertices.device), int((sizes >= min_faces).sum()), int((sizes > 0).sum())


def vertex_normals(vertices, faces):
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    n = torch.cross(b - a, c - a, dim=1)
    out = torch.zeros_like(vertices)
    for i in range(3):
        out.index_add_(0, faces[:, i], n)
    return F.normalize(out, dim=1), n.norm(dim=1) * 0.5


def write_ply(path, vertices, normals, colors, faces):
    count = vertices.shape[0]
    record = np.dtype([("p", "<f4", 3), ("n", "<f4", 3), ("c", "u1", 3)])
    data = np.empty(count, dtype=record)
    data["p"], data["n"], data["c"] = vertices.cpu().numpy(), normals.cpu().numpy(), colors.cpu().numpy()
    face_record = np.dtype([("n", "u1"), ("v", "<i4", 3)])
    face_data = np.empty(faces.shape[0], dtype=face_record)
    face_data["n"] = 3
    face_data["v"] = faces.cpu().numpy()
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {count}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property float nx\nproperty float ny\nproperty float nz\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              f"element face {faces.shape[0]}\nproperty list uchar int vertex_indices\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        data.tofile(f)
        face_data.tofile(f)


def load_view(root, image, cameras, lenses, device):
    prior = sv.read_image_any(root / "depths" / f"{image['name']}.png")
    height, width = prior.shape
    key = (image["camera_id"], width, height)
    if key not in lenses:
        lens = sv.Lens(cameras[image["camera_id"]], width, height, device)
        lenses[key] = (lens, lens.pixel_rays())
    lens, rays = lenses[key]
    mask = cv2.resize(sv.read_image_any(root / "masks" / f"{image['name']}.png"), (width, height), interpolation=cv2.INTER_NEAREST) > 127
    R = torch.as_tensor(image["R"], device=device, dtype=torch.float32)
    t = torch.as_tensor(image["t"], device=device, dtype=torch.float32)
    return lens, rays, torch.as_tensor(mask, device=device) & lens.valid_pixels(), R, t, prior


def load_splats(path, device):
    ply = sv.read_ply(path)
    return {"means": torch.as_tensor(ply["means"], device=device), "opacity": torch.as_tensor(ply["opacity"], device=device),
            "scales": torch.as_tensor(ply["scales"], device=device),
            "rotations": torch.as_tensor(sv.quat_to_matrix(ply["quats"]), device=device)}


def render_cache(args, images, cameras, root, cache, device):
    """Median depth per view as uint16 mm PNGs, so the fusion can be re-tuned without rendering again."""
    cache.mkdir(parents=True, exist_ok=True)
    splats, lenses = None, {}
    for i, image in enumerate(images):
        path = cache / f"{image['index']:05d}.png"
        if path.exists():
            continue
        if splats is None:
            splats = load_splats(args.ply, device)
        lens, _, mask, R, t, _ = load_view(root, image, cameras, lenses, device)
        out = sv.render_depths(splats, lens, image["R"], image["t"], kinds=(args.depth_kind.split("_")[0],))
        depth = torch.nan_to_num(out[args.depth_kind], nan=0.0)
        depth[~mask] = 0
        millimeters = (depth * 1000).clamp(0, 65535).round().cpu().numpy().astype(np.uint16)
        alpha = (out["alpha"] * 255).clamp(0, 255).round().byte().cpu().numpy()
        ok, encoded = cv2.imencode(".png", alpha)
        assert ok
        encoded.tofile(str(cache / f"{image['index']:05d}_alpha.png"))
        # Written last: its presence marks the view as complete.
        ok, encoded = cv2.imencode(".png", millimeters)
        assert ok
        encoded.tofile(str(path))
        if i % 25 == 0:
            print(f"  rendered {i + 1}/{len(images)}", flush=True)


def read_cached_depth(cache, image, device, min_alpha):
    millimeters = sv.read_image_any(cache / f"{image['index']:05d}.png").astype(np.float32)
    alpha = sv.read_image_any(cache / f"{image['index']:05d}_alpha.png").astype(np.float32) / 255.0
    millimeters[alpha <= min_alpha] = 0
    return torch.as_tensor(millimeters / 1000.0, device=device)


def view_report(held_out, cameras, root, vertices, faces, lenses, device):
    """Coverage needs no LAS: a pixel without LiDAR depth is where the mesh has to fill a LAS gap."""
    hit_all, total_all, hit_gap, total_gap = 0, 0, 0, 0
    errors = []
    for image in held_out:
        lens, _, mask, R, t, prior = load_view(root, image, cameras, lenses, device)
        prior = torch.as_tensor(prior.astype(np.float32) / 1000.0, device=device)
        depth, _ = rasterize(vertices, faces, lens, R, t)
        hit = torch.isfinite(depth) & mask
        gap = mask & (prior <= 0)
        hit_all += int(hit.sum())
        total_all += int(mask.sum())
        hit_gap += int((hit & gap).sum())
        total_gap += int(gap.sum())
        both = hit & (prior > 0)
        errors.append((depth - prior)[both].abs().cpu().numpy())
    e = np.concatenate(errors)
    return {"coverage_of_view_pixels": hit_all / max(total_all, 1),
            "coverage_of_lidar_gaps": hit_gap / max(total_gap, 1),
            "lidar_gap_share_of_pixels": total_gap / max(total_all, 1),
            "depth_err_median_mm": float(np.median(e) * 1000), "depth_err_p90_mm": float(np.percentile(e, 90) * 1000)}


def floor_report(vertices, normals, band=0.06):
    """Real floors are flat, so any relief on the lowest large horizontal surface is an error, with no LAS needed."""
    points = vertices.cpu().numpy().astype(np.float64)
    n = normals.cpu().numpy().astype(np.float64)
    # The scan's z axis need not be vertical, so a floor can span many z bins: fit a plane, not a z band.
    candidates = points[np.abs(n[:, 2]) > 0.9]
    lowest_half = candidates[candidates[:, 2] < np.median(candidates[:, 2])]
    rng = np.random.default_rng(0)
    subset = lowest_half[rng.choice(lowest_half.shape[0], min(200_000, lowest_half.shape[0]), replace=False)]
    best, plane_normal, center = -1, np.array([0.0, 0.0, 1.0]), subset.mean(axis=0)
    for _ in range(500):
        a, b, c = subset[rng.choice(subset.shape[0], 3, replace=False)]
        trial = np.cross(b - a, c - a)
        length = np.linalg.norm(trial)
        if length < 1e-9 or abs(trial[2]) / length < 0.9:
            continue
        trial /= length
        inliers = int((np.abs((subset - a) @ trial) < 0.03).sum())
        if inliers > best:
            best, plane_normal, center = inliers, trial, a
    for _ in range(3):
        keep = np.abs((points - center) @ plane_normal) < band
        center = points[keep].mean(axis=0)
        plane_normal = np.linalg.svd(points[keep][:: max(1, keep.sum() // 500_000)] - center, full_matrices=False)[2][2]
    floor = np.abs((points - center) @ plane_normal) < band
    pts = points[floor]
    residual = np.abs((pts - center) @ plane_normal)
    angle = np.degrees(np.arccos(np.clip(np.abs(n[floor] @ plane_normal), 0, 1)))
    return {"floor_z_m": float(center[2]), "floor_area_share": float(floor.mean()),
            "floor_tilt_deg": float(np.degrees(np.arccos(min(abs(plane_normal[2]), 1.0)))),
            "floor_residual_mm_p50": float(np.median(residual) * 1000),
            "floor_residual_mm_p90": float(np.percentile(residual, 90) * 1000),
            "floor_normal_deg_p50": float(np.median(angle)), "floor_normal_deg_p90": float(np.percentile(angle, 90)),
            "floor_patch": roughness_report(vertices[torch.as_tensor(floor, device=vertices.device)])}


def roughness_report(vertices, radius=0.1, samples=30_000, min_neighbors=20, k=256):
    """Plane residual in small patches. Most of a room is flat, so the low percentiles measure bumpiness."""
    points = vertices.cpu().numpy().astype(np.float64)
    rng = np.random.default_rng(0)
    query = points[rng.choice(points.shape[0], min(samples, points.shape[0]), replace=False)]
    distance, neighbor = cKDTree(points, balanced_tree=False, compact_nodes=False).query(
        query, k=k, distance_upper_bound=radius, workers=-1)
    valid = np.isfinite(distance)
    keep = valid.sum(axis=1) >= min_neighbors
    mask = valid[keep][..., None].astype(np.float64)
    pts = np.where(valid[keep][..., None], points[np.where(valid[keep], neighbor[keep], 0)], 0.0)
    count = mask.sum(axis=1)
    centered = (pts - ((pts * mask).sum(axis=1) / count)[:, None]) * mask
    eigenvalues = np.linalg.eigvalsh(np.einsum("nki,nkj->nij", centered, centered))
    rms = np.sqrt(np.maximum(eigenvalues[:, 0], 0) / count[:, 0]) * 1000
    return {"plane_rms_mm_p10": float(np.percentile(rms, 10)), "plane_rms_mm_p25": float(np.percentile(rms, 25)),
            "plane_rms_mm_p50": float(np.percentile(rms, 50)), "patches": int(rms.size)}


def color_vertices(vertices, images, cameras, root, cache, lenses, device, min_alpha, tolerance=0.05):
    """Averages the photos where a vertex is the surface the view actually sees."""
    total = torch.zeros((vertices.shape[0], 3), device=device)
    count = torch.zeros(vertices.shape[0], device=device)
    for image in images:
        lens, _, _, R, t, _ = load_view(root, image, cameras, lenses, device)
        depth = read_cached_depth(cache, image, device, min_alpha)
        photo = sv.read_image_any(root / "images" / image["name"])
        photo = cv2.resize(photo, (lens.width, lens.height), interpolation=cv2.INTER_AREA)
        photo = torch.as_tensor(photo[..., ::-1].copy(), device=device, dtype=torch.float32)
        for start in range(0, vertices.shape[0], 5_000_000):
            part = vertices[start : start + 5_000_000]
            p = part @ R.T + t
            uv, r, _ = lens.project(p)
            px, py = torch.floor(uv[:, 0]).long(), torch.floor(uv[:, 1]).long()
            ok = lens.in_field_of_view(p) & (px >= 0) & (px < lens.width) & (py >= 0) & (py < lens.height)
            ids = torch.nonzero(ok).squeeze(1)
            px, py = px[ids], py[ids]
            seen = (depth[py, px] > 0) & ((depth[py, px] - r[ids]).abs() < tolerance)
            ids, px, py = ids[seen] + start, px[seen], py[seen]
            total[ids] += photo[py, px]
            count[ids] += 1
    colors = total / count.clamp_min(1)[:, None]
    colors[count == 0] = 128
    return colors.round().clamp(0, 255).byte(), float((count > 0).float().mean())


def rasterize(vertices, faces, lens, R, t, attributes=None, chunk=20_000_000, max_box_px=128):
    """Z-buffers the triangles through the lens. Returns depth (NaN where empty) and interpolated attributes."""
    device = vertices.device
    p = vertices @ R.T + t
    uv, r, _ = lens.project(p)
    usable = lens.in_field_of_view(p) & (r > sv.MIN_RANGE_M)
    keep = usable[faces].all(dim=1)
    corners_u, corners_v = uv[faces, 0], uv[faces, 1]
    u0 = torch.floor(corners_u.min(dim=1).values - 0.5).long().clamp_min(0)
    u1 = torch.ceil(corners_u.max(dim=1).values - 0.5).long().clamp_max(lens.width - 1)
    v0 = torch.floor(corners_v.min(dim=1).values - 0.5).long().clamp_min(0)
    v1 = torch.ceil(corners_v.max(dim=1).values - 0.5).long().clamp_max(lens.height - 1)
    # Beyond this a triangle is stretched across the fisheye, where straight edges no longer hold.
    keep &= (u1 >= u0) & (v1 >= v0) & (u1 - u0 <= max_box_px) & (v1 - v0 <= max_box_px)
    index = torch.nonzero(keep).squeeze(1)
    columns = (u1 - u0 + 1)[index]
    counts = columns * (v1 - v0 + 1)[index]
    size = lens.width * lens.height
    zbuffer = torch.full((size,), float("inf"), device=device)
    if attributes is not None:
        image = torch.zeros((size, attributes.shape[1]), device=device)
    start = 0
    while start < index.numel():
        running = torch.cumsum(counts[start:], dim=0)
        stop = start + max(1, int(torch.searchsorted(running, chunk).item()))
        local = torch.repeat_interleave(torch.arange(start, stop, device=device), counts[start:stop])
        ends = torch.cumsum(counts[start:stop], dim=0)
        offset = torch.arange(local.numel(), device=device) - torch.repeat_interleave(ends - counts[start:stop], counts[start:stop])
        face = index[local]
        px = u0[face] + offset % columns[local]
        py = v0[face] + offset // columns[local]
        x, y = px.float() + 0.5, py.float() + 0.5
        ax, ay = corners_u[face, 0], corners_v[face, 0]
        bx, by = corners_u[face, 1], corners_v[face, 1]
        cx, cy = corners_u[face, 2], corners_v[face, 2]
        area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        w1 = ((x - ax) * (cy - ay) - (y - ay) * (cx - ax)) / area
        w2 = ((bx - ax) * (y - ay) - (by - ay) * (x - ax)) / area
        w0 = 1 - w1 - w2
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0) & (area != 0)
        face, px, py, w0, w1, w2 = face[inside], px[inside], py[inside], w0[inside], w1[inside], w2[inside]
        tri = faces[face]
        depth = w0 * r[tri[:, 0]] + w1 * r[tri[:, 1]] + w2 * r[tri[:, 2]]
        pixels = py * lens.width + px
        zbuffer.scatter_reduce_(0, pixels, depth, "amin")
        if attributes is not None:
            won = depth == zbuffer[pixels]
            image[pixels[won]] = (w0[won, None] * attributes[tri[won, 0]] + w1[won, None] * attributes[tri[won, 1]]
                                  + w2[won, None] * attributes[tri[won, 2]])
        start = stop
    zbuffer[torch.isinf(zbuffer)] = float("nan")
    shape = (lens.height, lens.width)
    return zbuffer.view(shape), image.view(*shape, -1) if attributes is not None else None


def write_preview(path, vertices, faces, colors, normals, lens, rays, R, t, photo):
    """Photo, colored mesh and shaded mesh from one camera."""
    attributes = torch.cat([colors.float(), normals], dim=1)
    depth, image = rasterize(vertices, faces, lens, R, t, attributes)
    empty = torch.isnan(depth)
    world_rays = rays @ R
    shade = (F.normalize(image[..., 3:], dim=-1) * world_rays).sum(-1).abs()
    shade = (40 + 215 * shade).clamp(0, 255)
    shade[empty] = 0
    color = image[..., :3].clamp(0, 255)
    color[empty] = 0
    color_image = color.byte().cpu().numpy()[..., ::-1]
    shade_image = np.repeat(shade.byte().cpu().numpy()[..., None], 3, axis=2)
    panel = np.hstack([cv2.resize(photo, (lens.width, lens.height), interpolation=cv2.INTER_AREA), color_image, shade_image])
    cv2.imencode(".jpg", cv2.resize(panel, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))[1].tofile(str(path))


def sample_surface(vertices, faces, area, spacing, rng_seed=0):
    generator = torch.Generator(device=vertices.device).manual_seed(rng_seed)
    count = int(area.sum().item() / (spacing * spacing))
    # torch.multinomial stops at 2^24 faces.
    cumulative = torch.cumsum(area.double(), dim=0)
    picks = torch.rand(min(count, 1 << 26), device=vertices.device, dtype=torch.float64, generator=generator) * cumulative[-1]
    chosen = torch.searchsorted(cumulative, picks).clamp_max(faces.shape[0] - 1)
    tri = faces[chosen]
    r = torch.rand((chosen.numel(), 2), device=vertices.device, generator=generator)
    s = torch.sqrt(r[:, 0])
    return ((1 - s)[:, None] * vertices[tri[:, 0]] + (s * (1 - r[:, 1]))[:, None] * vertices[tri[:, 1]]
            + (s * r[:, 1])[:, None] * vertices[tri[:, 2]]).double().cpu().numpy()


def evaluate(args, held_out, cameras, root, vertices, faces, area, device):
    """Precision of each cloud against the full LAS, recall of the held-out views' LiDAR depth maps."""
    rng = np.random.default_rng(0)
    las = np.load(args.las)
    print("  building LAS tree", flush=True)
    las_voxels = sv.voxel_dedupe(las, 0.01, rng)
    del las
    las_tree = cKDTree(las_voxels, balanced_tree=False, compact_nodes=False)
    splats = load_splats(args.ply, device)
    lenses = {}
    scan, baseline, mesh_scan = [], [], []
    per_view = 400_000

    def sampled_world(mask, depth, rays, R, t):
        cloud = ((rays[mask] * depth[mask][:, None] - t) @ R).cpu().numpy()
        return cloud[rng.choice(cloud.shape[0], min(per_view, cloud.shape[0]), replace=False)]

    splat_errors, mesh_errors, common_splat, common_mesh = [], [], [], []
    covered, total_pixels = 0, 0
    for image in held_out:
        lens, rays, mask, R, t, prior = load_view(root, image, cameras, lenses, device)
        prior = torch.as_tensor(prior.astype(np.float32) / 1000.0, device=device)
        has = (prior > 0) & mask
        scan.append(sampled_world(has, prior, rays, R, t))
        out = sv.render_depths(splats, lens, image["R"], image["t"], kinds=("center",))
        depth = out["center_median"]
        visible = (out["alpha"] > args.min_alpha) & mask & torch.isfinite(depth)
        baseline.append(sampled_world(visible, depth, rays, R, t))
        mesh_depth, _ = rasterize(vertices, faces, lens, R, t)
        mesh_scan.append(sampled_world(torch.isfinite(mesh_depth) & mask, mesh_depth, rays, R, t))
        mesh_hit = torch.isfinite(mesh_depth) & has
        total_pixels += int(has.sum())
        covered += int(mesh_hit.sum())
        splat_errors.append((depth - prior)[visible & has].abs().cpu().numpy())
        mesh_errors.append((mesh_depth - prior)[mesh_hit].abs().cpu().numpy())
        both = visible & mesh_hit
        common_splat.append((depth - prior)[both].abs().cpu().numpy())
        common_mesh.append((mesh_depth - prior)[both].abs().cpu().numpy())
    del splats

    def error_stats(parts):
        e = np.concatenate(parts)
        return {"err_median_mm": float(np.median(e) * 1000), "err_p90_mm": float(np.percentile(e, 90) * 1000),
                "over_5cm": float((e > 0.05).mean())}

    report = {"depth": {"mesh_coverage_of_lidar_pixels": covered / max(total_pixels, 1),
                        "splat": error_stats(splat_errors), "mesh": error_stats(mesh_errors),
                        "splat_on_common_pixels": error_stats(common_splat), "mesh_on_common_pixels": error_stats(common_mesh)}}
    print(f"  {report['depth']}", flush=True)
    scan = sv.voxel_dedupe(np.concatenate(scan), 0.01, rng)
    # splat_render and mesh_from_views are the same protocol: pixels back-projected from the held-out views.
    # mesh_surface samples the whole mesh once, so it has no multi-view thickness and also covers unseen parts.
    clouds = {"splat_render": sv.voxel_dedupe(np.concatenate(baseline), 0.01, rng),
              "mesh_from_views": sv.voxel_dedupe(np.concatenate(mesh_scan), 0.01, rng),
              "mesh_surface": sv.voxel_dedupe(sample_surface(vertices, faces, area, 0.005).astype(np.float32), 0.01, rng)}
    del baseline, mesh_scan
    for name, cloud in clouds.items():
        tree = cKDTree(cloud, balanced_tree=False, compact_nodes=False)
        report[name] = sv.f_scores(sv.nearest(las_tree, cloud), sv.nearest(tree, scan))
        report[name]["points"] = int(cloud.shape[0])
        print(f"  {name}: {report[name]}", flush=True)
    return report


def build_mesh(args):
    device = "cuda"
    begin = time.time()
    root, out = Path(args.dataset), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cameras = sv.read_cameras(root / "sparse/0/cameras.bin")
    images = sv.read_images(root / "sparse/0/images.bin")
    for index, image in enumerate(images):
        image["index"] = index
    held_out = images[::HELD_OUT_EVERY]
    training = [image for image in images if image["index"] % HELD_OUT_EVERY != 0][:: args.every]
    print(f"{len(training)} fusion views, {len(held_out)} held out", flush=True)
    cache = (Path(args.depth_cache) if args.depth_cache else out / "depth_cache") / args.depth_kind
    with torch.no_grad():
        render_cache(args, training, cameras, root, cache, device)
        report = {"seconds_render": time.time() - begin, "voxel_m": args.voxel, "views": len(training)}

        step = time.time()
        tsdf = SparseTsdf(args.voxel, args.block, args.trunc_voxels, device, args.full_weight_range)
        lenses = {}
        depths = {}
        for image in training:
            lens, rays, _, R, t, _ = load_view(root, image, cameras, lenses, device)
            depth = read_cached_depth(cache, image, device, args.min_alpha)
            if args.drop_depth_edges:
                depth = remove_depth_edges(depth)
            depths[image["index"]] = depth.cpu()
            tsdf.touch(depth.clone(), rays, R, t)
        tsdf.allocate()
        print(f"  {tsdf.keys.numel()} blocks", flush=True)
        for i, image in enumerate(training):
            lens, _, _, R, t, _ = load_view(root, image, cameras, lenses, device)
            tsdf.integrate(lens, depths[image["index"]].to(device), R, t)
            if i % 25 == 0:
                print(f"  fused {i + 1}/{len(training)}", flush=True)
        del depths
        report.update(blocks=int(tsdf.keys.numel()), seconds_fusion=time.time() - step)

        step = time.time()
        keys, values = tsdf.voxels(args.min_weight)
        report["band_voxels"] = int(keys.numel())
        del tsdf
        vertices, faces = surface_nets(keys, values, args.voxel)
        del keys, values
        report.update(raw_vertices=int(vertices.shape[0]), raw_faces=int(faces.shape[0]))
        vertices, faces, kept, total = remove_small_components(vertices, faces, args.min_component_faces)
        normals, area = vertex_normals(vertices, faces)
        report.update(vertices=int(vertices.shape[0]), faces=int(faces.shape[0]), components_kept=kept,
                      components_total=total, area_m2=float(area.sum()), seconds_extract=time.time() - step)
        print(f"  {report['vertices']} vertices, {report['faces']} faces", flush=True)

        step = time.time()
        colors, colored = color_vertices(vertices, training[:: args.color_every], cameras, root, cache, lenses, device, args.min_alpha)
        report.update(colored_vertex_share=colored, seconds_color=time.time() - step)
        write_ply(out / "mesh.ply", vertices, normals, colors, faces)
        report["held_out"] = view_report(held_out, cameras, root, vertices, faces, lenses, device)
        report["roughness"] = roughness_report(vertices)
        report["floor"] = floor_report(vertices, normals)
        print(f"  {report['held_out']}\n  {report['roughness']}\n  {report['floor']}", flush=True)

        for image in held_out[:: max(1, len(held_out) // 4)]:
            lens, rays, _, R, t, _ = load_view(root, image, cameras, lenses, device)
            photo = sv.read_image_any(root / "images" / image["name"])
            write_preview(out / f"preview_{image['index']:05d}.jpg", vertices, faces, colors, normals, lens, rays, R, t, photo)

        if args.eval:
            report["eval"] = evaluate(args, held_out, cameras, root, vertices, faces, area, device)
    report["seconds_total"] = time.time() - begin
    (out / "mesh_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def read_mesh(path):
    with open(path, "rb") as f:
        counts, fields, current = {}, [], None
        while True:
            parts = f.readline().decode("ascii").split()
            if parts == ["end_header"]:
                break
            if parts[0] == "element":
                current = parts[1]
                counts[current] = int(parts[2])
            elif parts[0] == "property" and current == "vertex":
                fields.append((parts[2], {"float": "<f4", "uchar": "u1"}[parts[1]]))
        data = np.fromfile(f, dtype=np.dtype(fields), count=counts["vertex"])
        faces = np.fromfile(f, dtype=np.dtype([("n", "u1"), ("v", "<i4", 3)]), count=counts["face"])
    return np.stack([data["x"], data["y"], data["z"]], axis=1), faces["v"].astype(np.int64)


def score_mesh(args):
    """Reports on a mesh made elsewhere, such as by the mesh subcommand, on the held-out views."""
    device = "cuda"
    root = Path(args.dataset)
    cameras = sv.read_cameras(root / "sparse/0/cameras.bin")
    images = sv.read_images(root / "sparse/0/images.bin")
    for index, image in enumerate(images):
        image["index"] = index
    held_out = images[::HELD_OUT_EVERY]
    v, f = read_mesh(args.mesh)
    vertices = torch.as_tensor(v, device=device)
    faces = torch.as_tensor(f, device=device)
    normals, area = vertex_normals(vertices, faces)
    report = {"vertices": int(v.shape[0]), "faces": int(f.shape[0]), "area_m2": float(area.sum()),
              "held_out": view_report(held_out, cameras, root, vertices, faces, {}, device),
              "roughness": roughness_report(vertices), "floor": floor_report(vertices, normals)}
    if args.eval:
        report["eval"] = evaluate(args, held_out, cameras, root, vertices, faces, area, device)
    print(json.dumps(report, indent=2))
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "mesh_score.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def selftest():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    center_radius, voxel = 1.0, 0.02
    lens = sv.Lens({"model": sv.FISHEYE, "width": 800, "height": 800,
                    "params": np.array([255.0, 255.0, 400.0, 400.0, -0.029, -0.0057, 0.0021, -0.00066])}, 800, 800, device)
    rays = lens.pixel_rays()
    valid = lens.valid_pixels()
    assert not valid[0, 0] and valid[400, 400], "fisheye corners lie beyond the widest angle"
    tsdf = SparseTsdf(voxel, 8, 4, device)
    views = []
    rng = np.random.default_rng(0)
    for _ in range(24):
        eye = rng.normal(size=3)
        eye = eye / np.linalg.norm(eye) * 3.0
        forward = -eye / np.linalg.norm(eye)
        right = np.cross(forward, [0.3, 0.2, 1.0])
        right /= np.linalg.norm(right)
        R = np.stack([right, np.cross(forward, right), forward])
        R_t = torch.as_tensor(R, device=device, dtype=torch.float32)
        t_t = torch.as_tensor(-R @ eye, device=device, dtype=torch.float32)
        # Exact range along each ray to the sphere at the origin.
        d = rays @ R_t
        o = torch.as_tensor(eye, device=device, dtype=torch.float32)
        b = d @ o
        disc = b * b - (o @ o - center_radius**2)
        depth = torch.where(disc > 0, -b - torch.sqrt(disc.clamp_min(0)), torch.zeros_like(b))
        depth = torch.nan_to_num(depth, nan=0.0).clamp_min(0) * valid
        views.append((depth, R_t, t_t))
        tsdf.touch(depth.clone(), rays, R_t, t_t)
    tsdf.allocate()
    for depth, R_t, t_t in views:
        tsdf.integrate(lens, depth, R_t, t_t)
    keys, values = tsdf.voxels(2)
    vertices, faces = surface_nets(keys, values, voxel)
    normals, area = vertex_normals(vertices, faces)
    radius = vertices.norm(dim=1)
    print(f"  sphere: {vertices.shape[0]} vertices, radius err median {(radius - 1).abs().median().item() * 1000:.2f} mm, "
          f"area {area.sum().item():.3f} (exact {4 * np.pi:.3f}), outward {(normals * vertices).sum(1).gt(0).float().mean().item():.4f}")
    assert (radius - 1).abs().median() < voxel / 4, "surface must sit on the sphere"
    assert abs(area.sum().item() / (4 * np.pi) - 1) < 0.1, "area must match the sphere"
    assert (normals * vertices).sum(1).gt(0).float().mean() > 0.999, "normals must point toward the cameras"

    # A floater seen by one view must be carved away by the others.
    tsdf = SparseTsdf(voxel, 8, 4, device)
    floater = []
    for i, (depth, R_t, t_t) in enumerate(views):
        d = depth.clone()
        if i == 0:
            d[390:410, 390:410] = 1.0
        floater.append(d)
        tsdf.touch(d.clone(), rays, R_t, t_t)
    tsdf.allocate()
    for d, (_, R_t, t_t) in zip(floater, views):
        tsdf.integrate(lens, d, R_t, t_t)
    keys, values = tsdf.voxels(2)
    vertices, _ = surface_nets(keys, values, voxel)
    assert (vertices.norm(dim=1) - 1).abs().max() < 0.1, "a one-view floater must not survive the fusion"

    # The rasterizer sees the sphere where the exact ray intersection does, at the same range.
    tsdf = SparseTsdf(voxel, 8, 4, device)
    for depth, R_t, t_t in views:
        tsdf.touch(depth.clone(), rays, R_t, t_t)
    tsdf.allocate()
    for depth, R_t, t_t in views:
        tsdf.integrate(lens, depth, R_t, t_t)
    keys, values = tsdf.voxels(2)
    vertices, faces = surface_nets(keys, values, voxel)
    depth, R_t, t_t = views[3]
    drawn, shaded = rasterize(vertices, faces, lens, R_t, t_t, vertices)
    both = torch.isfinite(drawn) & (depth > 0)
    assert both.sum() > 0.9 * (depth > 0).sum(), "rasterizer must cover the sphere"
    assert (drawn - depth)[both].abs().median() < voxel / 2, "rasterized depth must match the sphere"
    world = (rays[both] * drawn[both][:, None] - t_t) @ R_t
    assert (world - shaded[both]).norm(dim=1).median() < voxel, "interpolated attributes must match the pixel"

    normals, area = vertex_normals(vertices, faces)
    samples = sample_surface(vertices, faces, area, 0.01)
    assert abs(samples.shape[0] / (area.sum().item() / 1e-4) - 1) < 0.01, "sample count follows the area"
    assert np.abs(np.linalg.norm(samples, axis=1) - 1).max() < 0.03, "samples must lie on the surface"

    # Range weighting: where a close and a far view disagree, the surface follows the close one, and a voxel
    # seen only from far away still counts as observed.
    pinhole = sv.Lens({"model": sv.PINHOLE, "width": 200, "height": 200, "params": np.array([200.0, 200.0, 100.0, 100.0])}, 200, 200, device)
    pinhole_rays = pinhole.pixel_rays()
    results = {}
    for setting in (None, 1.0):
        tsdf = SparseTsdf(0.02, 8, 4, device, setting)
        poses = []
        for camera_z, surface_z in ((-1.0, 0.0), (-5.0, 0.04)):
            R_t = torch.eye(3, device=device)
            t_t = torch.tensor([0.0, 0.0, -camera_z], device=device)
            depth = (surface_z - camera_z) / pinhole_rays[..., 2]
            poses.append((depth, R_t, t_t))
            tsdf.touch(depth.clone(), pinhole_rays, R_t, t_t)
        tsdf.allocate()
        for depth, R_t, t_t in poses:
            tsdf.integrate(pinhole, depth, R_t, t_t)
        keys, values = tsdf.voxels(1)
        v, _ = surface_nets(keys, values, 0.02)
        central = v[(v[:, 0].abs() < 0.1) & (v[:, 1].abs() < 0.1)]
        results[setting] = central[:, 2].median().item()
    print(f"  close view at z=0, far view at z=0.04: equal weights {results[None]:.4f}, range weighted {results[1.0]:.4f}")
    assert 0.015 < results[None] < 0.025, "equal weights average the two surfaces"
    assert results[1.0] < 0.005, "range weighting must follow the close view"
    far_only = SparseTsdf(0.02, 8, 4, device, 1.0)
    depth, R_t, t_t = poses[1]
    far_only.touch(depth.clone(), pinhole_rays, R_t, t_t)
    far_only.allocate()
    far_only.integrate(pinhole, depth, R_t, t_t)
    assert far_only.voxels(1)[0].numel() > 0, "a voxel seen only from far must still count as observed"

    # Roughness: a bumpy plane scores its bump height, a flat one about zero.
    grid = torch.stack(torch.meshgrid(torch.arange(0, 2, 0.02), torch.arange(0, 2, 0.02), indexing="ij"), dim=-1).reshape(-1, 2)
    flat = torch.cat([grid, torch.zeros(grid.shape[0], 1)], dim=1)
    bumpy = flat.clone()
    bumpy[:, 2] = 0.005 * torch.sin(grid[:, 0] * 2 * np.pi / 0.1)
    flat_rms = roughness_report(flat)["plane_rms_mm_p50"]
    bumpy_rms = roughness_report(bumpy)["plane_rms_mm_p50"]
    print(f"  roughness: flat {flat_rms:.3f} mm, 5 mm bumps {bumpy_rms:.3f} mm")
    assert flat_rms < 0.01, "a plane must have no roughness"
    assert 2.5 < bumpy_rms < 4.5, "a 5 mm sine has about 3.5 mm rms"

    # Floor: a bumpy floor spanning 20 cm of z under a ceiling reports the bumps, not the tilt and not the ceiling.
    grid = torch.stack(torch.meshgrid(torch.arange(0, 4, 0.02), torch.arange(0, 4, 0.02), indexing="ij"), dim=-1).reshape(-1, 2)
    bump = 0.005 * torch.sin(grid[:, 1] * 2 * np.pi / 0.2)
    floor_points = torch.cat([grid, (0.05 * grid[:, 0] + bump)[:, None] - 1.0], dim=1)
    ceiling = torch.cat([grid, torch.full((grid.shape[0], 1), 3.0)], dim=1)
    points = torch.cat([floor_points, ceiling]).to(device)
    up = torch.zeros_like(points)
    up[:, 2] = 1
    result = floor_report(points, up)
    print(f"  floor: tilt {result['floor_tilt_deg']:.2f} deg, residual p50 {result['floor_residual_mm_p50']:.2f} mm, share {result['floor_area_share']:.2f}")
    assert abs(result["floor_area_share"] - 0.5) < 0.01, "must find the whole tilted floor, not a slice or the ceiling"
    assert abs(result["floor_tilt_deg"] - np.degrees(np.arctan(0.05))) < 0.3, "must measure the tilt"
    assert 2.5 < result["floor_residual_mm_p50"] < 4.5, "a 5 mm sine has a median |residual| of about 3.5 mm"

    # Small pieces go, large pieces stay.
    v = torch.rand((7, 3), device=device)
    f = torch.tensor([[0, 1, 2], [0, 2, 3], [0, 3, 1], [4, 5, 6]], device=device)
    v2, f2, kept, total = remove_small_components(v, f, 2)
    assert v2.shape[0] == 4 and f2.shape[0] == 3 and kept == 1 and total == 2, "component filter"
    print("selftest passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ply")
    parser.add_argument("--mesh", help="score this mesh instead of building one; needs --dataset")
    parser.add_argument("--dataset")
    parser.add_argument("--out")
    parser.add_argument("--las", help="xyz as a float64 .npy of shape [N, 3]; needed by --eval")
    parser.add_argument("--depth-cache", help="reuse the depth maps of an earlier run with the same model and views")
    parser.add_argument("--depth-kind", default="center_median", choices=sv.DEPTH_METHODS)
    parser.add_argument("--drop-depth-edges", action="store_true", help="do not fuse pixels on depth steps; costs thin objects")
    parser.add_argument("--full-weight-range", type=float, default=1.0, help="views beyond this range in meters count less, by 1/r^2")
    parser.add_argument("--voxel", type=float, default=0.02, help="voxel size in meters")
    parser.add_argument("--block", type=int, default=8, help="voxels per block side")
    parser.add_argument("--trunc-voxels", type=float, default=4, help="truncation band in voxels")
    parser.add_argument("--every", type=int, default=2, help="use every Nth training image for the fusion")
    parser.add_argument("--color-every", type=int, default=2, help="use every Nth fusion view for the colors")
    parser.add_argument("--min-alpha", type=float, default=0.5, help="drop pixels rendered with less alpha")
    parser.add_argument("--min-weight", type=float, default=3, help="views a voxel needs to count as observed")
    parser.add_argument("--min-component-faces", type=int, default=100, help="at 2 cm a robot's arm is only a few hundred faces")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        with torch.no_grad():
            selftest()
        return
    if args.mesh:
        if not args.dataset:
            parser.error("--mesh needs --dataset")
        if args.eval and not (args.las and args.ply):
            parser.error("--eval needs --las and --ply")
        with torch.no_grad():
            score_mesh(args)
        return
    if not (args.ply and args.dataset and args.out):
        parser.error("--ply, --dataset and --out are required")
    if args.eval and not args.las:
        parser.error("--eval needs --las")
    build_mesh(args)


if __name__ == "__main__":
    main()
