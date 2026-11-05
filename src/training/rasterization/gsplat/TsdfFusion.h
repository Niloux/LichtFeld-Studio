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
        float max_range = 0.f;   // depth beyond this touches no blocks; 0 = no limit
    };

    // Block keys around the surface of a depth map, sampled at every 2nd pixel at
    // depth and depth +- trunc, up to grid.max_range. keys holds 3 * ceil(W / 2) * ceil(H / 2); kTsdfNoKey fills the rest.
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

    // Voxels of blocks [first_block, last_block) with a face neighbor of the opposite sign, all
    // that surface extraction reads. A voxel counts when at least min_views views saw it and
    // |tsdf| < 1. Neighbors may lie in any block of block_keys, which is sorted. With null keys
    // it only counts them.
    [[nodiscard]] uint64_t tsdf_collect_voxels(const uint64_t* block_keys, uint32_t block_count, uint32_t first_block,
                                               uint32_t last_block, const TsdfGrid& grid,
                                               const float* sum, const float* weight, const uint16_t* views,
                                               const float* carve, uint32_t min_views,
                                               uint64_t* keys, float* tsdf, uint64_t* counter, cudaStream_t stream);

    // Adds the photo [3, H, W] to the vertices this view sees as its surface, weighted by 1 / distance.
    // No weight by angle: on a flat floor seen at a low angle, the noise in the normals makes such a
    // weight vary from vertex to vertex, and the colors mottle.
    void mesh_color_accumulate(const float* vertices, uint32_t vertex_count, const TsdfView& view, const float* depth, const float* photo,
                               float tolerance, float* color_sum, float* color_weight, cudaStream_t stream);

    // Blurs the photo [3, H, W] over pixels of positive depth, for mesh_texture_accumulate.
    // scratch holds a photo and weights two channels of H * W.
    void mesh_photo_low_pass(const float* photo, const float* depth, uint32_t width, uint32_t height, int radius,
                             float* scratch, float* weights, float* low, cudaStream_t stream);

    // How well this view sees each face: 0 unless at least half of 7 points on it lie on the depth map's surface,
    // else that share squared times cos(angle)^6 / distance^2. Footprint alone would use a single cosine, but a surface
    // is never where its photo saw it: raised lettering is not in the mesh, and an oblique view lays it across the wall
    // by that height times tan(angle), so a close oblique view slants what a farther square-on view keeps upright.
    // Faces are triangles of vertices [3 * V], facing outward by their winding.
    // tolerances [face_count] say how far from the depth map a point may lie and still count as on it.
    void mesh_face_view_score(const float* vertices, const int32_t* faces, uint32_t face_count, const TsdfView& view, const float* depth,
                              const float* tolerances, float* score, cudaStream_t stream);

    // Adds this view to the texels at world positions texels [3 * count]; a texel needs all four pixels it
    // samples to lie on its surface. low_sum / low_weight is the mean of the blurred photo over views, each
    // counting by 1 / distance, so levels agree across views. high keeps photo - low_photo of the view that
    // labels [count] chose for the texel; where that view has no clean sample, of the closest view that does.
    void mesh_texture_accumulate(const float* texels, uint32_t count, const TsdfView& view, int32_t view_label, const int32_t* labels,
                                 const float* depth, const float* photo, const float* low_photo, const float* tolerances, float* low_sum,
                                 float* low_weight, float* high, float* best_weight, cudaStream_t stream);

} // namespace gsplat_lfs
