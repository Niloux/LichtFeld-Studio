/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "Common.h"
#include <cstddef>
#include <cstdint>

namespace gsplat_lfs {

    // Sparse TSDF voxels keep 21 bits per axis, block and voxel coordinates alike.
    inline constexpr int kTsdfKeyOffset = 1 << 20;
    inline constexpr int kTsdfKeyBits = 21;
    inline constexpr uint32_t kTsdfTilePixels = 32;
    inline constexpr uint64_t kTsdfNoKey = ~uint64_t{0};

    // One camera, by value: the kernels build its model once per thread.
    struct TsdfView {
        CameraModelType model = CameraModelType::PINHOLE;
        ShutterType shutter = ShutterType::GLOBAL;
        uint32_t width = 0;
        uint32_t height = 0;
        float fx = 0.f;
        float fy = 0.f;
        float cx = 0.f;
        float cy = 0.f;
        float radial[6] = {};
        float tangential[2] = {};
        float thin_prism[4] = {};
        bool has_distortion = false;
        float viewmat[16] = {};   // reference world-to-camera, row-major; depth is measured from it
        float readout0[16] = {};  // world-to-camera of the first readout line
        float readout1[16] = {};  // and of the last; used when shutter is not GLOBAL
    };

    struct TsdfGrid {
        float voxel = 0.f;
        uint32_t block = 0;      // voxels per block side
        float trunc = 0.f;       // scene units
        float full_weight_range = 0.f; // views beyond this range count less, by 1/r^2; 0 = equal weights
    };

    // Block keys around the surface of a depth map, sampled at every 2nd pixel at
    // depth and depth +- trunc. keys holds 3 * ceil(W / 2) * ceil(H / 2); kTsdfNoKey fills the rest.
    void tsdf_touch_keys(const float* depth, const float* rays, const TsdfView& view, const TsdfGrid& grid,
                         uint64_t* keys, cudaStream_t stream);

    // Sorts keys in place and removes duplicates and kTsdfNoKey. Returns how many remain.
    // scratch holds count keys, counter one on the device.
    [[nodiscard]] size_t tsdf_sort_unique(uint64_t* keys, uint64_t* scratch, size_t count, uint64_t* counter, cudaStream_t stream);

    [[nodiscard]] inline __host__ __device__ uint32_t tsdf_tile_count(const uint32_t pixels) {
        return (pixels + kTsdfTilePixels - 1) / kTsdfTilePixels;
    }

    // Per tile the nearest and farthest depth of the tile and its neighbours; +inf and -inf where
    // there is none. raw_* and low, high hold tile_count(W) * tile_count(H) floats.
    void tsdf_tile_bounds(const float* depth, uint32_t width, uint32_t height,
                          float* raw_low, float* raw_high, float* low, float* high, cudaStream_t stream);

    // Blocks wholly in front of the surface only count as free space, blocks wholly behind it are
    // skipped, and the rest go to near. near holds block_count indices. Returns how many are near.
    [[nodiscard]] uint32_t tsdf_classify_blocks(const uint64_t* block_keys, uint32_t block_count,
                                                const TsdfView& view, const TsdfGrid& grid,
                                                const float* low, const float* high,
                                                float* carve, uint32_t* near_blocks, uint32_t* near_count,
                                                cudaStream_t stream);

    void tsdf_integrate(const uint64_t* block_keys, const uint32_t* near_blocks, uint32_t near_count,
                        const TsdfView& view, const TsdfGrid& grid, const float* depth,
                        float* sum, float* weight, uint16_t* views, cudaStream_t stream);

    // Voxels seen by at least min_views views with |tsdf| < 1. With null keys it only counts them.
    [[nodiscard]] uint64_t tsdf_collect_voxels(const uint64_t* block_keys, uint32_t block_count, const TsdfGrid& grid,
                                               const float* sum, const float* weight, const uint16_t* views,
                                               const float* carve, uint32_t min_views,
                                               uint64_t* keys, float* tsdf, uint64_t* counter, cudaStream_t stream);

    // Adds the photo [3, H, W] to the vertices this view sees as its surface.
    void mesh_color_accumulate(const float* vertices, uint32_t vertex_count, const TsdfView& view,
                               const float* depth, const float* photo, float tolerance,
                               float* color_sum, uint32_t* color_count, cudaStream_t stream);

} // namespace gsplat_lfs
