/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "TsdfFusion.h"
#include "Cameras.cuh"
#include "core/cuda_error.hpp"
#include "core/tensor/internal/cub_workspace.hpp"
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_select.cuh>

namespace gsplat_lfs {
    namespace {
        constexpr float kMinRange = 0.05f;
        constexpr uint32_t kThreads = 256;
        constexpr uint32_t kVerticesPerThread = 64;
        // Rolling shutter iterates from the poses of the frame's two ends; a point just outside
        // the frame at either end can still land inside it.
        constexpr float kReadoutMargin = 0.1f;
        constexpr uint32_t kNoBlock = ~uint32_t{0};
        // Keep a view that sees a vertex at all from dropping to nothing, and depth noise on a
        // very close view from taking over.
        constexpr float kColorMinDistance = 0.5f;

        // Calls body once with this view's camera model.
        template <class Body>
        __device__ __forceinline__ void with_model(const TsdfView& v, Body&& body) {
            if (v.model == CameraModelType::FISHEYE) {
                OpenCVFisheyeCameraModel<>::Parameters p = {};
                p.resolution = {v.width, v.height};
                p.shutter_type = v.shutter;
                p.principal_point = {v.cx, v.cy};
                p.focal_length = {v.fx, v.fy};
                p.radial_coeffs = {v.radial[0], v.radial[1], v.radial[2], v.radial[3]};
                const OpenCVFisheyeCameraModel<> model(p);
                body(model);
            } else if (v.model == CameraModelType::THIN_PRISM_FISHEYE) {
                ThinPrismFisheyeCameraModel<>::Parameters p = {};
                p.resolution = {v.width, v.height};
                p.shutter_type = v.shutter;
                p.principal_point = {v.cx, v.cy};
                p.focal_length = {v.fx, v.fy};
                p.radial_coeffs = {v.radial[0], v.radial[1], v.radial[2], v.radial[3]};
                p.thin_prism_coeffs = {v.thin_prism[0], v.thin_prism[1], v.thin_prism[2], v.thin_prism[3]};
                const ThinPrismFisheyeCameraModel<> model(p);
                body(model);
            } else if (v.model == CameraModelType::EQUIRECTANGULAR) {
                EquirectangularCameraModel::Parameters p = {};
                p.resolution = {v.width, v.height};
                p.shutter_type = ShutterType::GLOBAL;
                const EquirectangularCameraModel model(p);
                body(model);
            } else if (v.has_distortion) {
                OpenCVPinholeCameraModel<>::Parameters p = {};
                p.resolution = {v.width, v.height};
                p.shutter_type = v.shutter;
                p.principal_point = {v.cx, v.cy};
                p.focal_length = {v.fx, v.fy};
                p.radial_coeffs = {v.radial[0], v.radial[1], v.radial[2], v.radial[3], v.radial[4], v.radial[5]};
                p.tangential_coeffs = {v.tangential[0], v.tangential[1]};
                p.thin_prism_coeffs = {v.thin_prism[0], v.thin_prism[1], v.thin_prism[2], v.thin_prism[3]};
                const OpenCVPinholeCameraModel<> model(p);
                body(model);
            } else {
                PerfectPinholeCameraModel::Parameters p = {};
                p.resolution = {v.width, v.height};
                p.shutter_type = v.shutter;
                p.principal_point = {v.cx, v.cy};
                p.focal_length = {v.fx, v.fy};
                const PerfectPinholeCameraModel model(p);
                body(model);
            }
        }

        struct Pixel {
            uint32_t x = 0;
            uint32_t y = 0;
            float u = 0.f; // where in the image, with pixel x spanning [x, x + 1)
            float v = 0.f;
            float range = 0.f;
            bool valid = false;
        };

        // The range is what the depth channel holds: camera z for pinhole, else distance.
        template <class Model>
        __device__ __forceinline__ Pixel project(const Model& model, const TsdfView& v, const RollingShutterParameters& readout,
                                                 const glm::fvec3& world) {
            const float* m = v.viewmat;
            const glm::fvec3 p(m[0] * world.x + m[1] * world.y + m[2] * world.z + m[3],
                               m[4] * world.x + m[5] * world.y + m[6] * world.z + m[7],
                               m[8] * world.x + m[9] * world.y + m[10] * world.z + m[11]);
            Pixel out;
            out.range = v.model == CameraModelType::PINHOLE ? p.z : glm::length(p);
            if (!(out.range > kMinRange))
                return out;
            const auto image = model.template world_point_to_image_point_shutter_pose<>(world, readout, kReadoutMargin);
            if (!image.valid_flag)
                return out;
            const float u = floorf(image.imagePoint.x), w = floorf(image.imagePoint.y);
            if (u < 0.f || w < 0.f || u >= static_cast<float>(v.width) || w >= static_cast<float>(v.height))
                return out;
            out.u = image.imagePoint.x;
            out.v = image.imagePoint.y;
            out.x = static_cast<uint32_t>(u);
            out.y = static_cast<uint32_t>(w);
            out.valid = true;
            return out;
        }

        __device__ __forceinline__ RollingShutterParameters readout_of(const TsdfView& v) {
            return v.shutter == ShutterType::GLOBAL ? RollingShutterParameters(v.viewmat, nullptr)
                                                    : RollingShutterParameters(v.readout0, v.readout1);
        }

        __device__ __forceinline__ uint64_t encode_key(const int x, const int y, const int z) {
            return static_cast<uint64_t>(x + kTsdfKeyOffset) |
                   (static_cast<uint64_t>(y + kTsdfKeyOffset) << kTsdfKeyBits) |
                   (static_cast<uint64_t>(z + kTsdfKeyOffset) << (2 * kTsdfKeyBits));
        }

        __device__ __forceinline__ glm::ivec3 decode_key(const uint64_t key) {
            const uint64_t mask = (uint64_t{1} << kTsdfKeyBits) - 1;
            return {static_cast<int>(key & mask) - kTsdfKeyOffset,
                    static_cast<int>((key >> kTsdfKeyBits) & mask) - kTsdfKeyOffset,
                    static_cast<int>((key >> (2 * kTsdfKeyBits)) & mask) - kTsdfKeyOffset};
        }

        // A pixel's footprint and depth noise grow with range, so close views win where they exist.
        __device__ __forceinline__ float view_weight(const TsdfGrid& grid, const float range) {
            if (grid.full_weight_range <= 0.f)
                return 1.f;
            const float ratio = fminf(grid.full_weight_range / range, 1.f);
            return ratio * ratio;
        }

        __global__ void touch_kernel(const float* depth, const float* rays, const TsdfView view, const TsdfGrid grid, uint64_t* keys) {
            const uint32_t half_w = (view.width + 1) / 2, half_h = (view.height + 1) / 2;
            const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= half_w * half_h)
                return;
            const uint32_t pixel = (i / half_w) * 2 * view.width + (i % half_w) * 2;
            const float d = depth[pixel];
            const float* m = view.viewmat;
            const float block_size = grid.voxel * static_cast<float>(grid.block);
            for (int k = 0; k < 3; ++k) {
                uint64_t key = kTsdfNoKey;
                if (d > 0.f && (grid.max_range <= 0.f || d <= grid.max_range)) {
                    const float along = d + static_cast<float>(k - 1) * grid.trunc;
                    const float px = rays[3 * pixel] * along - m[3];
                    const float py = rays[3 * pixel + 1] * along - m[7];
                    const float pz = rays[3 * pixel + 2] * along - m[11];
                    // World = R^T (p - t).
                    const float wx = m[0] * px + m[4] * py + m[8] * pz;
                    const float wy = m[1] * px + m[5] * py + m[9] * pz;
                    const float wz = m[2] * px + m[6] * py + m[10] * pz;
                    if (isfinite(wx) && isfinite(wy) && isfinite(wz)) {
                        key = encode_key(static_cast<int>(floorf(wx / block_size)), static_cast<int>(floorf(wy / block_size)),
                                         static_cast<int>(floorf(wz / block_size)));
                    }
                }
                keys[3 * static_cast<uint64_t>(i) + k] = key;
            }
        }

        __global__ void raw_bounds_kernel(const float* depth, const uint32_t width, const uint32_t height, float* low, float* high) {
            const uint32_t tiles_x = tsdf_tile_count(width);
            const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= tiles_x * tsdf_tile_count(height))
                return;
            const uint32_t x0 = (i % tiles_x) * kTsdfTilePixels, y0 = (i / tiles_x) * kTsdfTilePixels;
            float lo = INFINITY, hi = -INFINITY;
            for (uint32_t y = y0; y < min(y0 + kTsdfTilePixels, height); ++y) {
                for (uint32_t x = x0; x < min(x0 + kTsdfTilePixels, width); ++x) {
                    const float d = depth[y * width + x];
                    if (d > 0.f) {
                        lo = fminf(lo, d);
                        hi = fmaxf(hi, d);
                    }
                }
            }
            low[i] = lo;
            high[i] = hi;
        }

        __global__ void dilate_bounds_kernel(const float* raw_low, const float* raw_high, const uint32_t tiles_x, const uint32_t tiles_y, float* low, float* high) {
            const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= tiles_x * tiles_y)
                return;
            const int tx = static_cast<int>(i % tiles_x), ty = static_cast<int>(i / tiles_x);
            float lo = INFINITY, hi = -INFINITY;
            for (int y = max(ty - 1, 0); y <= min(ty + 1, static_cast<int>(tiles_y) - 1); ++y) {
                for (int x = max(tx - 1, 0); x <= min(tx + 1, static_cast<int>(tiles_x) - 1); ++x) {
                    lo = fminf(lo, raw_low[y * tiles_x + x]);
                    hi = fmaxf(hi, raw_high[y * tiles_x + x]);
                }
            }
            low[i] = lo;
            high[i] = hi;
        }

        __global__ void classify_kernel(const uint64_t* block_keys, const uint32_t block_count, const TsdfView view, const TsdfGrid grid,
                                        const float* low, const float* high, float* carve, uint32_t* near_blocks, uint32_t* near_count) {
            const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
            if (b >= block_count)
                return;
            const float block_size = grid.voxel * static_cast<float>(grid.block);
            const float radius = 0.8660254f * block_size;
            const glm::ivec3 ijk = decode_key(block_keys[b]);
            const glm::fvec3 center((static_cast<float>(ijk.x) + 0.5f) * block_size, (static_cast<float>(ijk.y) + 0.5f) * block_size,
                                    (static_cast<float>(ijk.z) + 0.5f) * block_size);
            const RollingShutterParameters readout = readout_of(view);
            Pixel pixel;
            with_model(view, [&](const auto& model) { pixel = project(model, view, readout, center); });
            if (!pixel.valid)
                return;
            const uint32_t tile = (pixel.y / kTsdfTilePixels) * tsdf_tile_count(view.width) + pixel.x / kTsdfTilePixels;
            const float block_low = low[tile], block_high = high[tile];
            const bool seen = block_high >= block_low;
            const float focal = view.model == CameraModelType::EQUIRECTANGULAR
                                    ? static_cast<float>(view.width) / 6.2831853f
                                    : fmaxf(view.fx, view.fy);
            const bool spans_tiles = focal * radius / pixel.range > static_cast<float>(kTsdfTilePixels);
            const bool behind = !spans_tiles && seen && pixel.range - radius > block_high + grid.trunc;
            const bool front = !spans_tiles && seen && pixel.range + radius < block_low - grid.trunc;
            if (front)
                carve[b] += view_weight(grid, pixel.range);
            if ((seen || spans_tiles) && !behind && !front)
                near_blocks[atomicAdd(near_count, 1u)] = b;
        }

        __global__ void integrate_kernel(const uint64_t* block_keys, const uint32_t* near_blocks, const uint32_t near_count,
                                         const TsdfView view, const TsdfGrid grid, const float* depth,
                                         float* sum, float* weight, uint16_t* views) {
            const uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;
            if (n >= near_count)
                return;
            const uint32_t b = near_blocks[n];
            const glm::ivec3 ijk = decode_key(block_keys[b]);
            const uint32_t side = grid.block;
            const uint64_t first = static_cast<uint64_t>(b) * side * side * side;
            const RollingShutterParameters readout = readout_of(view);
            with_model(view, [&](const auto& model) {
                for (uint32_t i = 0; i < side * side * side; ++i) {
                    // Voxel keys run x fastest, as tsdf_collect_voxels decodes them.
                    const int lx = static_cast<int>(i % side), ly = static_cast<int>((i / side) % side), lz = static_cast<int>(i / (side * side));
                    const glm::fvec3 world(static_cast<float>(ijk.x * static_cast<int>(side) + lx) * grid.voxel,
                                           static_cast<float>(ijk.y * static_cast<int>(side) + ly) * grid.voxel,
                                           static_cast<float>(ijk.z * static_cast<int>(side) + lz) * grid.voxel);
                    const Pixel pixel = project(model, view, readout, world);
                    if (!pixel.valid)
                        continue;
                    const float measured = depth[pixel.y * view.width + pixel.x];
                    const float sdf = measured - pixel.range;
                    if (!(measured > 0.f) || !(sdf > -grid.trunc))
                        continue;
                    const float w = view_weight(grid, pixel.range);
                    sum[first + i] += w * fminf(sdf / grid.trunc, 1.f);
                    weight[first + i] += w;
                    views[first + i] += 1;
                }
            });
        }

        __device__ __forceinline__ uint32_t find_block(const uint64_t* block_keys, const uint32_t block_count, const uint64_t key) {
            uint32_t low = 0, high = block_count;
            while (low < high) {
                const uint32_t middle = low + (high - low) / 2;
                if (block_keys[middle] < key)
                    low = middle + 1;
                else
                    high = middle;
            }
            return low < block_count && block_keys[low] == key ? low : kNoBlock;
        }

        __global__ void collect_kernel(const uint64_t* block_keys, const uint32_t block_count, const uint32_t first_block, const uint32_t last_block,
                                       const TsdfGrid grid, const float* sum, const float* weight, const uint16_t* views, const float* carve,
                                       const uint32_t min_views, uint64_t* keys, float* tsdf, uint64_t* counter) {
            const uint32_t b = first_block + blockIdx.x * blockDim.x + threadIdx.x;
            if (b >= last_block)
                return;
            const glm::ivec3 ijk = decode_key(block_keys[b]);
            const uint32_t side = grid.block;
            const uint32_t voxels = side * side * side;
            const auto read = [&](const uint32_t block, const uint32_t i, float& value) {
                const uint64_t at = static_cast<uint64_t>(block) * voxels + i;
                if (views[at] < min_views)
                    return false;
                value = (sum[at] + carve[block]) / fmaxf(weight[at] + carve[block], 1e-6f);
                return fabsf(value) < 1.f;
            };
            // Found on first use: most blocks' voxels never look across a face.
            uint32_t faces[6] = {kNoBlock, kNoBlock, kNoBlock, kNoBlock, kNoBlock, kNoBlock};
            bool looked[6] = {};
            const uint32_t stride[3] = {1, side, side * side};
            const auto opposite = [&](const uint32_t i, const uint32_t local, const int axis, const bool up, const float value) {
                uint32_t block = b, neighbor;
                if (up ? local + 1 < side : local > 0) {
                    neighbor = up ? i + stride[axis] : i - stride[axis];
                } else {
                    const int face = 2 * axis + (up ? 1 : 0);
                    if (!looked[face]) {
                        looked[face] = true;
                        glm::ivec3 next = ijk;
                        next[axis] += up ? 1 : -1;
                        faces[face] = find_block(block_keys, block_count, encode_key(next.x, next.y, next.z));
                    }
                    block = faces[face];
                    if (block == kNoBlock)
                        return false;
                    neighbor = up ? i - (side - 1) * stride[axis] : i + (side - 1) * stride[axis];
                }
                float other = 0.f;
                return read(block, neighbor, other) && (other < 0.f) != (value < 0.f);
            };
            for (uint32_t i = 0; i < voxels; ++i) {
                float value = 0.f;
                if (!read(b, i, value))
                    continue;
                const uint32_t lx = i % side, ly = (i / side) % side, lz = i / (side * side);
                if (!opposite(i, lx, 0, true, value) && !opposite(i, lx, 0, false, value) && !opposite(i, ly, 1, true, value) &&
                    !opposite(i, ly, 1, false, value) && !opposite(i, lz, 2, true, value) && !opposite(i, lz, 2, false, value))
                    continue;
                const uint64_t slot = atomicAdd(reinterpret_cast<unsigned long long*>(counter), 1ull);
                if (keys != nullptr) {
                    const int side_i = static_cast<int>(side);
                    keys[slot] = encode_key(ijk.x * side_i + static_cast<int>(lx), ijk.y * side_i + static_cast<int>(ly), ijk.z * side_i + static_cast<int>(lz));
                    tsdf[slot] = value;
                }
            }
        }

        __global__ void color_kernel(const float* vertices, const uint32_t vertex_count, const TsdfView view, const float* depth,
                                     const float* photo, const float tolerance, float* color_sum, float* color_weight) {
            const uint32_t first = (blockIdx.x * blockDim.x + threadIdx.x) * kVerticesPerThread;
            if (first >= vertex_count)
                return;
            const uint32_t pixels = view.width * view.height;
            const RollingShutterParameters readout = readout_of(view);
            const float* m = view.viewmat;
            // Center = -R^T t. The middle of the readout is close enough for a weight.
            const glm::fvec3 center(-(m[0] * m[3] + m[4] * m[7] + m[8] * m[11]), -(m[1] * m[3] + m[5] * m[7] + m[9] * m[11]),
                                    -(m[2] * m[3] + m[6] * m[7] + m[10] * m[11]));
            with_model(view, [&](const auto& model) {
                for (uint32_t v = first; v < min(first + kVerticesPerThread, vertex_count); ++v) {
                    const glm::fvec3 point(vertices[3 * v], vertices[3 * v + 1], vertices[3 * v + 2]);
                    const Pixel pixel = project(model, view, readout, point);
                    if (!pixel.valid)
                        continue;
                    const uint32_t at = pixel.y * view.width + pixel.x;
                    const float measured = depth[at];
                    if (!(measured > 0.f) || !(fabsf(measured - pixel.range) < tolerance))
                        continue;
                    const float w = 1.f / fmaxf(glm::length(center - point), kColorMinDistance);
                    for (uint32_t c = 0; c < 3; ++c)
                        color_sum[3 * v + c] += w * photo[c * pixels + at];
                    color_weight[v] += w;
                }
            });
        }

        // Averages valid pixels of each channel along one axis, weighted by a triangle of the given radius.
        __global__ void blur_pass_kernel(const float* source, const float* source_weight, const uint32_t width, const uint32_t height,
                                         const int radius, const bool along_x, float* target, float* target_weight) {
            const uint32_t pixels = width * height;
            const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= pixels)
                return;
            const int x = static_cast<int>(i % width), y = static_cast<int>(i / width);
            float sum[3] = {0.f, 0.f, 0.f}, total = 0.f;
            for (int k = -radius; k <= radius; ++k) {
                const int sx = along_x ? x + k : x, sy = along_x ? y : y + k;
                if (sx < 0 || sy < 0 || sx >= static_cast<int>(width) || sy >= static_cast<int>(height))
                    continue;
                const uint32_t at = static_cast<uint32_t>(sy) * width + static_cast<uint32_t>(sx);
                const float w = static_cast<float>(radius + 1 - abs(k)) * source_weight[at];
                total += w;
                for (uint32_t c = 0; c < 3; ++c)
                    sum[c] += w * source[c * pixels + at];
            }
            for (uint32_t c = 0; c < 3; ++c)
                target[c * pixels + i] = total > 0.f ? sum[c] / total : 0.f;
            target_weight[i] = total > 0.f ? total / static_cast<float>((radius + 1) * (radius + 1)) : 0.f;
        }

        __global__ void valid_pixels_kernel(const float* depth, const uint32_t pixels, float* weight) {
            const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < pixels)
                weight[i] = depth[i] > 0.f ? 1.f : 0.f;
        }

        // The bilinear sample of an image [3, H, W] around pixel at.
        __device__ __forceinline__ void sample_bilinear(const float* image, const uint32_t pixels, const uint32_t at, const uint32_t width, const float fx, const float fy, float* value) {
            for (uint32_t c = 0; c < 3; ++c) {
                const float* p = image + c * pixels + at;
                value[c] = (1.f - fy) * ((1.f - fx) * p[0] + fx * p[1]) + fy * ((1.f - fx) * p[width] + fx * p[width + 1]);
            }
        }

        __global__ void face_score_kernel(const float* vertices, const int32_t* faces, const uint32_t face_count, const TsdfView view, const float* depth,
                                          const float* tolerances, float* score) {
            const uint32_t first = (blockIdx.x * blockDim.x + threadIdx.x) * kVerticesPerThread;
            if (first >= face_count)
                return;
            const RollingShutterParameters readout = readout_of(view);
            const float* m = view.viewmat;
            const glm::fvec3 center(-(m[0] * m[3] + m[4] * m[7] + m[8] * m[11]), -(m[1] * m[3] + m[5] * m[7] + m[9] * m[11]),
                                    -(m[2] * m[3] + m[6] * m[7] + m[10] * m[11]));
            with_model(view, [&](const auto& model) {
                for (uint32_t f = first; f < min(first + kVerticesPerThread, face_count); ++f) {
                    const auto vertex = [&](const int c) {
                        const int32_t v = faces[3 * static_cast<uint64_t>(f) + c];
                        return glm::fvec3(vertices[3 * v], vertices[3 * v + 1], vertices[3 * v + 2]);
                    };
                    const glm::fvec3 a = vertex(0), b = vertex(1), c = vertex(2);
                    const glm::fvec3 cross = glm::cross(b - a, c - a);
                    const glm::fvec3 middle = (a + b + c) / 3.f;
                    const glm::fvec3 to_camera = center - middle;
                    const float distance = glm::length(to_camera), area = glm::length(cross);
                    const float cosine = area > 0.f && distance > 0.f ? glm::dot(cross, to_camera) / (area * distance) : 0.f;
                    float result = 0.f;
                    if (cosine > 0.f) {
                        const glm::fvec3 points[7] = {a, b, c, .5f * (a + b), .5f * (b + c), .5f * (c + a), middle};
                        int seen = 0;
                        for (const auto& point : points) {
                            const Pixel pixel = project(model, view, readout, point);
                            if (!pixel.valid)
                                continue;
                            const float measured = depth[pixel.y * view.width + pixel.x];
                            seen += measured > 0.f && fabsf(measured - pixel.range) < tolerances[f];
                        }
                        if (2 * seen >= 7) {
                            const float share = static_cast<float>(seen) / 7.f;
                            const float d = fmaxf(distance, kColorMinDistance);
                            const float square = cosine * cosine;
                            result = share * share * square * square * square / (d * d);
                        }
                    }
                    score[f] = result;
                }
            });
        }

        __global__ void texture_kernel(const float* texels, const uint32_t count, const TsdfView view, const int32_t view_label, const int32_t* labels, const float* depth, const float* photo,
                                       const float* low_photo, const float* tolerances,
                                       float* low_sum, float* low_weight, float* high, float* best_weight) {
            const uint32_t first = (blockIdx.x * blockDim.x + threadIdx.x) * kVerticesPerThread;
            if (first >= count)
                return;
            const uint32_t width = view.width, pixels = view.width * view.height;
            const RollingShutterParameters readout = readout_of(view);
            const float* m = view.viewmat;
            const glm::fvec3 center(-(m[0] * m[3] + m[4] * m[7] + m[8] * m[11]), -(m[1] * m[3] + m[5] * m[7] + m[9] * m[11]),
                                    -(m[2] * m[3] + m[6] * m[7] + m[10] * m[11]));
            with_model(view, [&](const auto& model) {
                for (uint32_t t = first; t < min(first + kVerticesPerThread, count); ++t) {
                    const glm::fvec3 point(texels[3 * t], texels[3 * t + 1], texels[3 * t + 2]);
                    const Pixel pixel = project(model, view, readout, point);
                    if (!pixel.valid)
                        continue;
                    const float sx = pixel.u - .5f, sy = pixel.v - .5f;
                    const float fx0 = floorf(sx), fy0 = floorf(sy);
                    if (fx0 < 0.f || fy0 < 0.f || fx0 + 1.f >= static_cast<float>(view.width) || fy0 + 1.f >= static_cast<float>(view.height))
                        continue;
                    const uint32_t at = static_cast<uint32_t>(fy0) * width + static_cast<uint32_t>(fx0);
                    // A texel next to an occluding edge or a mask has no clean color in this view.
                    const float corners[4] = {depth[at], depth[at + 1], depth[at + width], depth[at + width + 1]};
                    bool clean = true;
                    for (const float measured : corners)
                        clean = clean && measured > 0.f && fabsf(measured - pixel.range) < tolerances[t];
                    if (!clean)
                        continue;
                    const float w = 1.f / fmaxf(glm::length(center - point), kColorMinDistance);
                    float color[3], low[3];
                    sample_bilinear(photo, pixels, at, width, sx - fx0, sy - fy0, color);
                    sample_bilinear(low_photo, pixels, at, width, sx - fx0, sy - fy0, low);
                    for (uint32_t c = 0; c < 3; ++c)
                        low_sum[3 * static_cast<uint64_t>(t) + c] += w * low[c];
                    low_weight[t] += w;
                    // The chosen view keeps its whole color, and another view only its detail, over the levels of all.
                    const bool chosen = labels[t] == view_label;
                    const float rank = chosen ? INFINITY : w;
                    if (rank > best_weight[t]) {
                        best_weight[t] = rank;
                        for (uint32_t c = 0; c < 3; ++c)
                            high[3 * static_cast<uint64_t>(t) + c] = chosen ? color[c] : color[c] - low[c];
                    }
                }
            });
        }

        [[nodiscard]] uint32_t launch_blocks(const uint64_t count) {
            return static_cast<uint32_t>((count + kThreads - 1) / kThreads);
        }
    } // namespace

    void tsdf_touch_keys(const float* depth, const float* rays, const TsdfView& view, const TsdfGrid& grid, uint64_t* keys, cudaStream_t stream) {
        const uint64_t samples = static_cast<uint64_t>((view.width + 1) / 2) * ((view.height + 1) / 2);
        touch_kernel<<<launch_blocks(samples), kThreads, 0, stream>>>(depth, rays, view, grid, keys);
        LFS_CUDA_LAUNCH_CHECK(stream, "tsdf.touch");
    }

    size_t tsdf_sort_unique(uint64_t* keys, uint64_t* scratch, const size_t count, uint64_t* counter, cudaStream_t stream) {
        if (count == 0)
            return 0;
        lfs::core::tensor_ops::run_cub_operation("tsdf.sort_keys", stream, [&](void* workspace, size_t& bytes) {
            return cub::DeviceRadixSort::SortKeys(workspace, bytes, keys, scratch, count, 0, 64, stream);
        });
        lfs::core::tensor_ops::run_cub_operation("tsdf.unique_keys", stream, [&](void* workspace, size_t& bytes) {
            return cub::DeviceSelect::Unique(workspace, bytes, scratch, keys, counter, static_cast<int64_t>(count), stream);
        });
        uint64_t unique = 0;
        LFS_CUDA_CHECK_MSG(cudaMemcpyAsync(&unique, counter, sizeof(uint64_t), cudaMemcpyDeviceToHost, stream), "tsdf.unique count");
        LFS_CUDA_CHECK_MSG(cudaStreamSynchronize(stream), "tsdf.unique sync");
        uint64_t last = 0;
        if (unique > 0) {
            LFS_CUDA_CHECK_MSG(cudaMemcpyAsync(&last, keys + unique - 1, sizeof(uint64_t), cudaMemcpyDeviceToHost, stream), "tsdf.unique last");
            LFS_CUDA_CHECK_MSG(cudaStreamSynchronize(stream), "tsdf.unique last sync");
        }
        return unique > 0 && last == kTsdfNoKey ? unique - 1 : unique;
    }

    void tsdf_tile_bounds(const float* depth, const uint32_t width, const uint32_t height,
                          float* raw_low, float* raw_high, float* low, float* high, cudaStream_t stream) {
        const uint32_t tiles_x = tsdf_tile_count(width), tiles_y = tsdf_tile_count(height);
        raw_bounds_kernel<<<launch_blocks(uint64_t{tiles_x} * tiles_y), kThreads, 0, stream>>>(depth, width, height, raw_low, raw_high);
        LFS_CUDA_LAUNCH_CHECK(stream, "tsdf.raw_bounds");
        dilate_bounds_kernel<<<launch_blocks(uint64_t{tiles_x} * tiles_y), kThreads, 0, stream>>>(raw_low, raw_high, tiles_x, tiles_y, low, high);
        LFS_CUDA_LAUNCH_CHECK(stream, "tsdf.dilate_bounds");
    }

    uint32_t tsdf_classify_blocks(const uint64_t* block_keys, const uint32_t block_count, const TsdfView& view, const TsdfGrid& grid,
                                  const float* low, const float* high, float* carve, uint32_t* near_blocks, uint32_t* near_count,
                                  cudaStream_t stream) {
        LFS_CUDA_CHECK_MSG(cudaMemsetAsync(near_count, 0, sizeof(uint32_t), stream), "tsdf.classify clear");
        classify_kernel<<<launch_blocks(block_count), kThreads, 0, stream>>>(block_keys, block_count, view, grid, low, high, carve, near_blocks, near_count);
        LFS_CUDA_LAUNCH_CHECK(stream, "tsdf.classify");
        uint32_t count = 0;
        LFS_CUDA_CHECK_MSG(cudaMemcpyAsync(&count, near_count, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), "tsdf.classify count");
        LFS_CUDA_CHECK_MSG(cudaStreamSynchronize(stream), "tsdf.classify sync");
        return count;
    }

    void tsdf_integrate(const uint64_t* block_keys, const uint32_t* near_blocks, const uint32_t near_count,
                        const TsdfView& view, const TsdfGrid& grid, const float* depth,
                        float* sum, float* weight, uint16_t* views, cudaStream_t stream) {
        if (near_count == 0)
            return;
        integrate_kernel<<<launch_blocks(near_count), kThreads, 0, stream>>>(block_keys, near_blocks, near_count, view, grid, depth, sum, weight, views);
        LFS_CUDA_LAUNCH_CHECK(stream, "tsdf.integrate");
    }

    uint64_t tsdf_collect_voxels(const uint64_t* block_keys, const uint32_t block_count, const uint32_t first_block,
                                 const uint32_t last_block, const TsdfGrid& grid,
                                 const float* sum, const float* weight, const uint16_t* views, const float* carve,
                                 const uint32_t min_views, uint64_t* keys, float* tsdf, uint64_t* counter, cudaStream_t stream) {
        LFS_CUDA_CHECK_MSG(cudaMemsetAsync(counter, 0, sizeof(uint64_t), stream), "tsdf.collect clear");
        collect_kernel<<<launch_blocks(last_block - first_block), kThreads, 0, stream>>>(block_keys, block_count, first_block, last_block, grid, sum, weight, views, carve,
                                                                                         min_views, keys, tsdf, counter);
        LFS_CUDA_LAUNCH_CHECK(stream, "tsdf.collect");
        uint64_t count = 0;
        LFS_CUDA_CHECK_MSG(cudaMemcpyAsync(&count, counter, sizeof(uint64_t), cudaMemcpyDeviceToHost, stream), "tsdf.collect count");
        LFS_CUDA_CHECK_MSG(cudaStreamSynchronize(stream), "tsdf.collect sync");
        return count;
    }

    void mesh_color_accumulate(const float* vertices, const uint32_t vertex_count, const TsdfView& view, const float* depth, const float* photo,
                               const float tolerance, float* color_sum, float* color_weight, cudaStream_t stream) {
        const uint64_t threads = (static_cast<uint64_t>(vertex_count) + kVerticesPerThread - 1) / kVerticesPerThread;
        color_kernel<<<launch_blocks(threads), kThreads, 0, stream>>>(vertices, vertex_count, view, depth, photo, tolerance, color_sum, color_weight);
        LFS_CUDA_LAUNCH_CHECK(stream, "mesh.color");
    }

    namespace {
        __global__ void pack_mask_kernel(const float* mask, const uint32_t pixels, uint32_t* bits) {
            const uint32_t word = blockIdx.x * blockDim.x + threadIdx.x;
            if (uint64_t{word} * 32 >= pixels)
                return;
            uint32_t value = 0;
            for (uint32_t b = 0; b < 32; ++b) {
                const uint64_t p = uint64_t{word} * 32 + b;
                if (p < pixels && mask[p] != 0.f)
                    value |= 1u << b;
            }
            bits[word] = value;
        }

        __global__ void apply_mask_kernel(float* depth, const uint32_t pixels, const uint32_t* bits) {
            const uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
            if (p >= pixels)
                return;
            depth[p] *= (bits[p >> 5] >> (p & 31)) & 1u ? 1.f : 0.f;
        }
    } // namespace

    void mesh_pack_mask(const float* mask, const uint32_t pixels, uint32_t* bits, cudaStream_t stream) {
        const uint64_t words = (uint64_t{pixels} + 31) / 32;
        pack_mask_kernel<<<launch_blocks(words), kThreads, 0, stream>>>(mask, pixels, bits);
        LFS_CUDA_LAUNCH_CHECK(stream, "mesh.pack_mask");
    }

    void mesh_apply_mask(float* depth, const uint32_t pixels, const uint32_t* bits, cudaStream_t stream) {
        apply_mask_kernel<<<launch_blocks(pixels), kThreads, 0, stream>>>(depth, pixels, bits);
        LFS_CUDA_LAUNCH_CHECK(stream, "mesh.apply_mask");
    }

    void mesh_photo_low_pass(const float* photo, const float* depth, const uint32_t width, const uint32_t height, const int radius,
                             float* scratch, float* weights, float* low, cudaStream_t stream) {
        const uint64_t pixels = uint64_t{width} * height;
        valid_pixels_kernel<<<launch_blocks(pixels), kThreads, 0, stream>>>(depth, static_cast<uint32_t>(pixels), weights);
        LFS_CUDA_LAUNCH_CHECK(stream, "mesh.valid_pixels");
        blur_pass_kernel<<<launch_blocks(pixels), kThreads, 0, stream>>>(photo, weights, width, height, radius, true, scratch, weights + pixels);
        LFS_CUDA_LAUNCH_CHECK(stream, "mesh.blur_x");
        blur_pass_kernel<<<launch_blocks(pixels), kThreads, 0, stream>>>(scratch, weights + pixels, width, height, radius, false, low, weights);
        LFS_CUDA_LAUNCH_CHECK(stream, "mesh.blur_y");
    }

    void mesh_face_view_score(const float* vertices, const int32_t* faces, const uint32_t face_count, const TsdfView& view, const float* depth,
                              const float* tolerances, float* score, cudaStream_t stream) {
        const uint64_t threads = (static_cast<uint64_t>(face_count) + kVerticesPerThread - 1) / kVerticesPerThread;
        face_score_kernel<<<launch_blocks(threads), kThreads, 0, stream>>>(vertices, faces, face_count, view, depth, tolerances, score);
        LFS_CUDA_LAUNCH_CHECK(stream, "mesh.face_score");
    }

    void mesh_texture_accumulate(const float* texels, const uint32_t count, const TsdfView& view, const int32_t view_label, const int32_t* labels, const float* depth, const float* photo,
                                 const float* low_photo, const float* tolerances, float* low_sum, float* low_weight,
                                 float* high, float* best_weight, cudaStream_t stream) {
        const uint64_t threads = (static_cast<uint64_t>(count) + kVerticesPerThread - 1) / kVerticesPerThread;
        texture_kernel<<<launch_blocks(threads), kThreads, 0, stream>>>(texels, count, view, view_label, labels, depth, photo, low_photo, tolerances, low_sum, low_weight, high, best_weight);
        LFS_CUDA_LAUNCH_CHECK(stream, "mesh.texture");
    }

} // namespace gsplat_lfs
