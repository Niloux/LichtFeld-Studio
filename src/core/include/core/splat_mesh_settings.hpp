/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

namespace lfs::core {

    // Distances are in scene units. The defaults are tuned on metric indoor scans.
    struct SplatMeshSettings {
        float voxel_size = .02f;
        int block_voxels = 8;
        float trunc_voxels = 4.f;
        float full_weight_range = 1.f; // views beyond this range count less, by 1/r^2; 0 = equal weights
        int view_stride = 2;           // fuse every Nth view
        int color_stride = 2;          // color from every Nth fused view
        int min_views = 3;             // views a voxel needs to count as observed
        int min_component_faces = 100;
        int max_width = 1600;          // render and photo width limit
        bool invert_mask = false;
        float mask_threshold = .5f;
    };

} // namespace lfs::core
