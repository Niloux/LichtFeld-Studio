/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <cooperative_groups.h>
#include <cstdlib>
#include <cuda_runtime.h>

#include "Cameras.cuh"
#include "Common.h"
#include "FromWorldRay.cuh"
#include "Rasterization.h"
#include "Utils.cuh"

namespace gsplat_lfs {

    namespace cg = cooperative_groups;

    ////////////////////////////////////////////////////////////////
    // Forward Kernel
    ////////////////////////////////////////////////////////////////

    // kMipFilter is a template parameter so the unfiltered kernel compiles exactly as before.
    template <uint32_t CDIM, typename scalar_t, bool kPerfectPinhole, bool kBatched, bool kMipFilter>
    __global__ void rasterize_to_pixels_from_world_3dgs_fwd_kernel(
        const uint32_t C,
        const uint32_t N,
        const uint32_t n_isects,
        const bool packed,
        const vec3* __restrict__ means,           // [N, 3]
        const vec4* __restrict__ quats,           // [N, 4]
        const vec3* __restrict__ scales,          // [N, 3]
        const scalar_t* __restrict__ colors,      // [C, N, CDIM] or [nnz, CDIM]
        const scalar_t* __restrict__ opacities,   // [C, N] or [nnz]
        const scalar_t* __restrict__ backgrounds, // [C, CDIM] - solid color (mutually exclusive with bg_images)
        const scalar_t* __restrict__ bg_images,   // [C, CDIM, H, W] - per-pixel background (mutually exclusive with backgrounds)
        const bool* __restrict__ masks,           // [C, tile_height, tile_width]
        const scalar_t* __restrict__ compensations,
        const scalar_t* __restrict__ dilations,
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        const uint32_t tile_width,
        const uint32_t tile_height,
        // camera model
        const scalar_t* __restrict__ viewmats0, // [C, 4, 4]
        const scalar_t* __restrict__ viewmats1, // [C, 4, 4] optional for rolling shutter
        const scalar_t* __restrict__ Ks,        // [C, 3, 3]
        const CameraModelType camera_model_type,
        // uncented transform
        const UnscentedTransformParameters ut_params,
        const ShutterType rs_type,
        const scalar_t* __restrict__ radial_coeffs,     // [C, 6] or [C, 4] optional
        const scalar_t* __restrict__ tangential_coeffs, // [C, 2] optional
        const scalar_t* __restrict__ thin_prism_coeffs, // [C, 2] optional
        // intersections
        const int32_t* __restrict__ tile_offsets, // [C, tile_height, tile_width]
        const int32_t* __restrict__ flatten_ids,  // [n_isects]
        scalar_t* __restrict__ render_colors,     // [C, CDIM, image_height, image_width]
        scalar_t* __restrict__ render_alphas,     // [C, image_height, image_width, 1]
        int32_t* __restrict__ last_ids,           // [C, image_height, image_width]
        const TileRange tiles,
        float* __restrict__ median_depth,
        int32_t* __restrict__ median_ids) {
        // each thread draws one pixel, but also timeshares caching gaussians in a
        // shared tile

        auto block = cg::this_thread_block();
        int32_t cid = block.group_index().x;
        int32_t tile_id =
            block.group_index().y * tile_width + block.group_index().z;
        if constexpr (kBatched) {
            if (tile_id < tiles.begin || tile_id >= tiles.end)
                return;
        }
        uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
        uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

        tile_offsets += cid * tile_height * tile_width;
        render_colors += cid * image_height * image_width * CDIM;
        render_alphas += cid * image_height * image_width;
        last_ids += cid * image_height * image_width;
        float* const median_out =
            median_depth != nullptr ? median_depth + cid * image_height * image_width : nullptr;
        int32_t* const median_ids_out =
            median_ids != nullptr ? median_ids + cid * image_height * image_width : nullptr;
        if (backgrounds != nullptr) {
            backgrounds += cid * CDIM;
        }
        if (bg_images != nullptr) {
            bg_images += cid * 3u * image_height * image_width;
        }
        if (masks != nullptr) {
            masks += cid * tile_height * tile_width;
        }

        float px = (float)j + 0.5f;
        float py = (float)i + 0.5f;
        int32_t pix_id = i * image_width + j;

        const WorldRay ray = from_world_pixel_ray<kPerfectPinhole>(
            camera_model_type, rs_type, image_width, image_height, px, py,
            viewmats0, viewmats1, Ks, cid,
            radial_coeffs, tangential_coeffs, thin_prism_coeffs);
        const vec3 ray_d = ray.ray_dir;
        const vec3 ray_o = ray.ray_org;
        const float depth_scale =
            (camera_model_type == CameraModelType::PINHOLE) ? ray.cam_ray_z : 1.0f;
        constexpr bool kHasRayDepth = (CDIM == 1 || CDIM == 4 || CDIM == 8);
        constexpr uint32_t kDepthChannel = (CDIM == 1) ? 0u : 3u;
        constexpr bool kHasDepthSq = (CDIM == 8);
        constexpr uint32_t kDepthSqChannel = 7u;

        // return if out of bounds
        // keep not rasterizing threads around for reading data
        bool inside = (i < image_height && j < image_width);
        bool done = (!inside) || (!ray.valid_flag);

        // when the mask is provided, render the background color and return
        // if this tile is labeled as False
        if (masks != nullptr && inside && !masks[tile_id]) {
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                float bg_val = 0.0f;
                if (bg_images != nullptr && k < 3u) {
                    // bg_images is [CDIM, H, W] for this camera
                    bg_val = bg_images[k * image_height * image_width + pix_id];
                } else if (backgrounds != nullptr && k < 3u) {
                    bg_val = backgrounds[k];
                }
                render_colors[chw_pix(k, pix_id, image_height, image_width)] = bg_val;
            }
            if (median_out != nullptr)
                median_out[pix_id] = 0.0f;
            if (median_ids_out != nullptr)
                median_ids_out[pix_id] = -1;
            return;
        }

        // have all threads in tile process the same gaussians in batches
        // first collect gaussians between range.x and range.y in batches
        // which gaussians to look through in this tile
        int32_t range_start = tile_offsets[tile_id];
        int32_t range_end = tile_offsets[tile_id + 1];
        const uint32_t block_size = block.size();
        const uint32_t stage_n =
            (CDIM == 3 && block_size == 256u) ? 128u : block_size;
        uint32_t num_batches =
            (range_end - range_start + stage_n - 1) / stage_n;

        extern __shared__ int s[];
        vec4* xyz_opacity_batch = reinterpret_cast<vec4*>(s); // [stage_n]
        mat3* iscl_rot_batch =
            reinterpret_cast<mat3*>(&xyz_opacity_batch[stage_n]); // [stage_n]
        float* rgbs_batch =
            reinterpret_cast<float*>(&iscl_rot_batch[stage_n]); // [stage_n * CDIM]

        // current visibility left to render
        // transmittance is gonna be used in the backward pass which requires a high
        // numerical precision so we use double for it. However double make bwd 1.5x
        // slower so we stick with float for now.
        float T = 1.0f;
        // index of most recent gaussian to write to this thread's pixel
        uint32_t cur_idx = 0;
        float median = 0.0f;
        int32_t median_id = -1;

        // collect and process batches of gaussians
        // each thread loads one gaussian at a time before rasterizing its
        // designated pixel
        uint32_t tr = block.thread_rank();

        float pix_out[CDIM] = {0.f};
        for (uint32_t b = 0; b < num_batches; ++b) {
            // resync all threads before beginning next batch
            // end early if entire tile is done
            if (__syncthreads_count(done) >= block_size) {
                break;
            }

            // each thread fetch 1 gaussian from front to back
            // index of gaussian to load
            uint32_t batch_start = range_start + stage_n * b;
            uint32_t idx = batch_start + tr;
            if (tr < stage_n && idx < range_end) {
                int32_t g = flatten_ids[idx]; // flatten index in [C * N] or [nnz]
                const float3 xyz_f = load_float3(reinterpret_cast<const float*>(&means[g]));
                float opac = activated_opacity(opacities[g]);
                if constexpr (kMipFilter)
                    opac *= compensations[g];
                xyz_opacity_batch[tr] = {xyz_f.x, xyz_f.y, xyz_f.z, opac};

                const vec4 quat = quats[g];
                vec3 scale = activated_scale(scales[g]);
                mat3 R = quat_to_rotmat(quat);
                mat3 S = mat3(
                    1.0f / scale[0],
                    0.f,
                    0.f,
                    0.f,
                    1.0f / scale[1],
                    0.f,
                    0.f,
                    0.f,
                    1.0f / scale[2]);
                iscl_rot_batch[tr] = S * glm::transpose(R);
                if constexpr (kMipFilter)
                    iscl_rot_batch[tr] = inverse_scale_matrix(dilated_scale(scale, dilations[g])) * glm::transpose(R);
                if constexpr (CDIM == 3) {
                    const float3 c = load_float3(colors + g * 3);
                    rgbs_batch[tr * 3u + 0] = c.x;
                    rgbs_batch[tr * 3u + 1] = c.y;
                    rgbs_batch[tr * 3u + 2] = c.z;
                } else if constexpr (CDIM == 4) {
                    const float4 c = load_float4(colors + g * 4);
                    rgbs_batch[tr * 4u + 0] = c.x;
                    rgbs_batch[tr * 4u + 1] = c.y;
                    rgbs_batch[tr * 4u + 2] = c.z;
                    rgbs_batch[tr * 4u + 3] = c.w;
                } else {
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        rgbs_batch[tr * CDIM + k] = colors[g * CDIM + k];
                    }
                }
            }

            // wait for other threads to collect the gaussians in batch
            block.sync();

            // process gaussians in the current batch for this pixel
            uint32_t batch_size = min(stage_n, range_end - batch_start);
            for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
                const float4 xyz_opac = load_float4(reinterpret_cast<const float*>(&xyz_opacity_batch[t]));
                const float opac = xyz_opac.w;
                const vec3 xyz = {xyz_opac.x, xyz_opac.y, xyz_opac.z};
                const mat3 iscl_rot = iscl_rot_batch[t];

                const vec3 gro = iscl_rot * (ray_o - xyz);
                vec3 grd_n;
                vec3 gcrod;
                float power;
                float depth = 0.0f;
                if constexpr (kHasRayDepth) {
                    float inv_grd_len = 0.0f;
                    float t_hit = 0.0f;
                    ray_gaussian_eval(gro, iscl_rot * ray_d, depth_scale,
                                      grd_n, gcrod, inv_grd_len, t_hit, depth, power);
                } else {
                    grd_n = safe_normalize(iscl_rot * ray_d);
                    gcrod = glm::cross(grd_n, gro);
                    const float grayDist = glm::dot(gcrod, gcrod);
                    power = -0.5f * grayDist;
                }

                float alpha = min(0.999f, opac * __expf(power));
                if (alpha < 1.f / 255.f) {
                    continue;
                }

                const float next_T = T * (1.0f - alpha);
                if (next_T <= 1e-4f) { // this pixel is done: exclusive
                    done = true;
                    break;
                }

                const float splat_depth = kHasRayDepth ? depth : 0.0f;
                if ((median_out != nullptr || median_ids_out != nullptr) && T >= 0.5f && next_T < 0.5f) {
                    median = splat_depth;
                    median_id = static_cast<int32_t>(batch_start + t);
                }

                const float vis = alpha * T;
                if constexpr (CDIM == 3) {
                    const float3 c = load_float3(&rgbs_batch[t * 3u]);
                    pix_out[0] += c.x * vis;
                    pix_out[1] += c.y * vis;
                    pix_out[2] += c.z * vis;
                } else if constexpr (CDIM == 4) {
                    const float4 c = load_float4(&rgbs_batch[t * 4u]);
                    pix_out[0] += c.x * vis;
                    pix_out[1] += c.y * vis;
                    pix_out[2] += c.z * vis;
                    pix_out[3] += depth * vis;
                } else {
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        const float c_val = (kHasRayDepth && k == kDepthChannel)
                                                ? depth
                                                : ((kHasDepthSq && k == kDepthSqChannel)
                                                       ? depth * depth
                                                       : rgbs_batch[t * CDIM + k]);
                        pix_out[k] += c_val * vis;
                    }
                }
                cur_idx = batch_start + t;

                T = next_T;
            }
        }

        if (inside) {
            // Here T is the transmittance AFTER the last gaussian in this pixel.
            // We (should) store double precision as T would be used in backward
            // pass and it can be very small and causing large diff in gradients
            // with float32. However, double precision makes the backward pass 1.5x
            // slower so we stick with float for now.
            render_alphas[pix_id] = 1.0f - T;
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                float bg_val = 0.0f;
                if (bg_images != nullptr && k < 3u) {
                    // bg_images is [CDIM, H, W] for this camera
                    bg_val = bg_images[k * image_height * image_width + pix_id];
                } else if (backgrounds != nullptr && k < 3u) {
                    bg_val = backgrounds[k];
                }
                render_colors[chw_pix(k, pix_id, image_height, image_width)] = pix_out[k] + T * bg_val;
            }
            // index in bin of last gaussian in this pixel
            last_ids[pix_id] = static_cast<int32_t>(cur_idx);
            if (median_out != nullptr)
                median_out[pix_id] = median;
            if (median_ids_out != nullptr)
                median_ids_out[pix_id] = median_id;
        }
    }

    // Micro-tile forward. A block of 64 threads takes an 8x8 sub-tile of a tile and
    // walks the tile's list front to back. Each round tests 64 entries against the
    // sub-tile and compacts the survivors into shared memory, loading them on the
    // way, so pixels only evaluate splats whose ellipse can reach them. The test reads
    // the pair's sub-tile mask from its key when the intersector wrote one, else it
    // runs the per-splat backward's cull directly; both give the same answer, and
    // last_ids still index the tile's list, so the backward replays the same splats.
    constexpr uint32_t kMicroSub = kCullSubTile;
    constexpr uint32_t kMicroThreads = kMicroSub * kMicroSub;
    constexpr uint32_t kMicroStage = kMicroThreads + kMicroThreads / 2;

    bool use_micro_forward() {
        static const bool enabled = [] {
            const char* value = std::getenv("LFS_GUT_MICRO_FWD");
            return value == nullptr || value[0] != '0';
        }();
        return enabled;
    }

    template <uint32_t CDIM>
    constexpr uint32_t fwd_splat_color_count() {
        if constexpr (CDIM == 1)
            return 0u;
        if constexpr (CDIM == 4)
            return 3u;
        if constexpr (CDIM == 8)
            return 6u;
        return CDIM;
    }

    template <uint32_t CDIM>
    __device__ __forceinline__ constexpr uint32_t fwd_splat_channel(const uint32_t s) {
        if constexpr (CDIM == 8)
            return s < 3u ? s : s + 1u;
        return s;
    }

    template <uint32_t CDIM>
    __device__ __forceinline__ constexpr uint32_t fwd_channel_to_slot(const uint32_t k) {
        if constexpr (CDIM == 8)
            return k < 3u ? k : k - 1u;
        return k;
    }

    template <uint32_t CDIM, bool kPerfectPinhole, bool kBatched, bool kMipFilter, bool kCachedRays = false,
              bool kGlobalShutter = false>
    __global__ void __launch_bounds__(kMicroThreads) rasterize_to_pixels_from_world_3dgs_fwd_micro_kernel(
        const uint32_t C,
        const vec3* __restrict__ means,
        const vec4* __restrict__ quats,
        const vec3* __restrict__ scales,
        const float* __restrict__ colors,
        const float* __restrict__ opacities,
        const float* __restrict__ backgrounds,
        const float* __restrict__ bg_images,
        const bool* __restrict__ masks,
        const float* __restrict__ compensations,
        const float* __restrict__ dilations,
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        const uint32_t tile_width,
        const uint32_t tile_height,
        const float* __restrict__ viewmats0,
        const float* __restrict__ viewmats1,
        const float* __restrict__ Ks,
        const CameraModelType camera_model_type,
        const ShutterType rs_type,
        const float* __restrict__ radial_coeffs,
        const float* __restrict__ tangential_coeffs,
        const float* __restrict__ thin_prism_coeffs,
        const int32_t* __restrict__ tile_offsets,
        const int32_t* __restrict__ flatten_ids,
        float* __restrict__ render_colors,
        float* __restrict__ render_alphas,
        int32_t* __restrict__ last_ids,
        const TileRange tiles,
        float* __restrict__ median_depth,
        const ScreenFootprint footprint,
        int32_t* __restrict__ median_ids) {
        const uint32_t subs = tile_size / kMicroSub;
        uint32_t cid = 0, tile_id = 0, sub_bit = 0, sub_y = 0, sub_x = 0;
        if (footprint.tile_order != nullptr) {
            cid = blockIdx.y;
            const uint32_t tile_rank = blockIdx.x / (subs * subs);
            sub_bit = blockIdx.x % (subs * subs);
            tile_id = static_cast<uint32_t>(footprint.tile_order[tile_rank]);
            sub_y = (tile_id / tile_width) * subs + (sub_bit / subs);
            sub_x = (tile_id % tile_width) * subs + (sub_bit % subs);
        } else {
            cid = blockIdx.x;
            sub_y = blockIdx.y;
            sub_x = blockIdx.z;
            tile_id = (sub_y / subs) * tile_width + sub_x / subs;
            sub_bit = (sub_y % subs) * subs + sub_x % subs;
        }
        if constexpr (kBatched) {
            if (tile_id < tiles.begin || tile_id >= tiles.end)
                return;
        }
        const uint32_t tr = threadIdx.x;
        const uint32_t lane = tr & 31u;
        const uint32_t warp = tr >> 5;
        const uint32_t i = sub_y * kMicroSub + tr / kMicroSub;
        const uint32_t j = sub_x * kMicroSub + tr % kMicroSub;

        tile_offsets += cid * tile_height * tile_width;
        render_colors += cid * image_height * image_width * CDIM;
        render_alphas += cid * image_height * image_width;
        last_ids += cid * image_height * image_width;
        float* const median_out =
            median_depth != nullptr ? median_depth + cid * image_height * image_width : nullptr;
        int32_t* const median_ids_out =
            median_ids != nullptr ? median_ids + cid * image_height * image_width : nullptr;
        if (backgrounds != nullptr) {
            backgrounds += cid * CDIM;
        }
        if (bg_images != nullptr) {
            bg_images += cid * 3u * image_height * image_width;
        }
        if (masks != nullptr) {
            masks += cid * tile_height * tile_width;
        }

        const int32_t pix_id = i * image_width + j;
        const bool inside = i < image_height && j < image_width;
        const bool sub_culled =
            (masks != nullptr && !masks[tile_id]) ||
            (footprint.tile_sub_masks != nullptr && !((footprint.tile_sub_masks[tile_id] >> sub_bit) & 1u));
        if (sub_culled) {
            if (!inside)
                return;
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                float bg_val = 0.0f;
                if (bg_images != nullptr && k < 3u) {
                    bg_val = bg_images[k * image_height * image_width + pix_id];
                } else if (backgrounds != nullptr && k < 3u) {
                    bg_val = backgrounds[k];
                }
                render_colors[chw_pix(k, pix_id, image_height, image_width)] = bg_val;
            }
            render_alphas[pix_id] = 0.0f;
            last_ids[pix_id] = 0;
            if (median_out != nullptr)
                median_out[pix_id] = 0.0f;
            if (median_ids_out != nullptr)
                median_ids_out[pix_id] = -1;
            return;
        }

        vec3 ray_d{0.f};
        vec3 ray_o{0.f};
        float depth_scale = 1.0f;
        bool valid_ray = false;
        if constexpr (kCachedRays) {
            __shared__ float3 s_ray_o[kMicroSub];
            __shared__ float s_R_inv[kMicroSub][9];
            const bool row_shutter = (rs_type != ShutterType::ROLLING_LEFT_TO_RIGHT &&
                                      rs_type != ShutterType::ROLLING_RIGHT_TO_LEFT);
            if constexpr (kGlobalShutter) {
                if (tr == 0) {
                    const RollingShutterParameters rs_params(viewmats0 + cid * 16, nullptr);
                    const ShutterPose pose = interpolate_shutter_pose(0.f, rs_params);
                    const mat3 R_inv = glm::mat3_cast(glm::inverse(pose.q));
                    const vec3 org = -(R_inv * pose.t);
                    s_ray_o[0] = make_float3(org.x, org.y, org.z);
#pragma unroll
                    for (int c = 0; c < 3; ++c) {
#pragma unroll
                        for (int r = 0; r < 3; ++r)
                            s_R_inv[0][c * 3 + r] = R_inv[c][r];
                    }
                }
            } else if (tr < kMicroSub) {
                const uint32_t coord = row_shutter ? (sub_y * kMicroSub + tr) : (sub_x * kMicroSub + tr);
                const uint32_t extent = row_shutter ? image_height : image_width;
                float t_shutter = 0.f;
                if (viewmats1 != nullptr && rs_type != ShutterType::GLOBAL && extent > 1u) {
                    const float clamped = static_cast<float>(min(coord, extent - 1u));
                    const float denom = static_cast<float>(extent - 1u);
                    t_shutter = (rs_type == ShutterType::ROLLING_BOTTOM_TO_TOP ||
                                 rs_type == ShutterType::ROLLING_RIGHT_TO_LEFT)
                                    ? (denom - clamped) / denom
                                    : clamped / denom;
                }
                const RollingShutterParameters rs_params(
                    viewmats0 + cid * 16,
                    viewmats1 == nullptr ? nullptr : viewmats1 + cid * 16);
                const ShutterPose pose = interpolate_shutter_pose(t_shutter, rs_params);
                const mat3 R_inv = glm::mat3_cast(glm::inverse(pose.q));
                const vec3 org = -(R_inv * pose.t);
                s_ray_o[tr] = make_float3(org.x, org.y, org.z);
#pragma unroll
                for (int c = 0; c < 3; ++c) {
#pragma unroll
                    for (int r = 0; r < 3; ++r)
                        s_R_inv[tr][c * 3 + r] = R_inv[c][r];
                }
            }
            __syncthreads();
            const uint32_t u = kGlobalShutter ? 0u : (row_shutter ? (tr / kMicroSub) : (tr % kMicroSub));
            const float3 org3 = s_ray_o[u];
            ray_o = {org3.x, org3.y, org3.z};
            if (inside) {
                const float* const ur =
                    footprint.unit_rays + (static_cast<size_t>(cid) * image_height * image_width + pix_id) * 3u;
                const float cx = ur[0];
                const float cy = ur[1];
                const float cz = ur[2];
                valid_ray = isfinite(cx) && isfinite(cy) && isfinite(cz);
                if (valid_ray) {
                    const float* const Rm = s_R_inv[u];
                    ray_d = {
                        Rm[0] * cx + Rm[3] * cy + Rm[6] * cz,
                        Rm[1] * cx + Rm[4] * cy + Rm[7] * cz,
                        Rm[2] * cx + Rm[5] * cy + Rm[8] * cz};
                }
            }
        } else {
            const WorldRay ray = from_world_pixel_ray<kPerfectPinhole>(
                camera_model_type, rs_type, image_width, image_height,
                static_cast<float>(j) + 0.5f, static_cast<float>(i) + 0.5f,
                viewmats0, viewmats1, Ks, cid,
                radial_coeffs, tangential_coeffs, thin_prism_coeffs);
            ray_d = ray.ray_dir;
            ray_o = ray.ray_org;
            depth_scale = (camera_model_type == CameraModelType::PINHOLE) ? ray.cam_ray_z : 1.0f;
            valid_ray = ray.valid_flag;
        }
        constexpr bool kHasRayDepth = (CDIM == 1 || CDIM == 4 || CDIM == 8);
        constexpr uint32_t kDepthChannel = (CDIM == 1) ? 0u : 3u;
        constexpr bool kHasDepthSq = (CDIM == 8);
        constexpr uint32_t kDepthSqChannel = 7u;
        constexpr uint32_t kSplatColors = fwd_splat_color_count<CDIM>();
        constexpr uint32_t kSmemColors = kSplatColors > 0u ? kSplatColors : 1u;

        bool done = !inside || !valid_ray;

        __shared__ vec4 s_xyz_opacity[kMicroStage];
        __shared__ mat3 s_iscl_rot[kMicroStage];
        __shared__ float s_rgbs[kMicroStage * kSmemColors];
        __shared__ int32_t s_bin[kMicroStage]; // the survivor's position in the tile's list
        __shared__ uint8_t s_hit[kMicroStage]; // whether the survivor blended any pixel
        __shared__ uint32_t s_warp_count[kMicroThreads / 32];
        auto* const prune = reinterpret_cast<unsigned long long*>(footprint.prune_tile_keys);
        uint32_t* const prune32 = footprint.prune_tile_keys32;

        const int32_t range_end = tile_offsets[tile_id + 1];
        int32_t cursor = tile_offsets[tile_id];
        const float sub_x0 = static_cast<float>(sub_x * kMicroSub);
        const float sub_y0 = static_cast<float>(sub_y * kMicroSub);

        float T = 1.0f;
        uint32_t cur_idx = 0;
        float median = 0.0f;
        int32_t median_id = -1;
        float pix_out[CDIM] = {0.f};

        // The trip counts below depend only on shared values, so the loops are
        // block-uniform and every barrier is reached by all threads.
        while (cursor < range_end) {
            // Also keeps the previous batch alive until every pixel has used it.
            if (__syncthreads_count(done) >= kMicroThreads)
                break;

            uint32_t count = 0;
            while (count + kMicroThreads <= kMicroStage && cursor < range_end) {
                const int32_t gi = cursor + static_cast<int32_t>(tr);
                int32_t g = 0;
                bool keep = false;
                if (gi < range_end) {
                    if (footprint.tile_keys32 != nullptr) {
                        keep = (footprint.tile_keys32[gi] >> sub_bit) & 1u;
                        if (keep)
                            g = flatten_ids[gi];
                    } else if (footprint.tile_keys != nullptr) {
                        keep = (static_cast<uint64_t>(footprint.tile_keys[gi]) >> sub_bit) & 1u;
                        if (keep)
                            g = flatten_ids[gi];
                    } else {
                        g = flatten_ids[gi];
                        keep = footprint_overlaps_square(footprint.means2d, footprint.conics,
                                                         footprint.radii, g, sub_x0, sub_y0, kMicroSub);
                    }
                }
                const uint32_t ballot = __ballot_sync(0xffffffffu, keep);
                if (lane == 0)
                    s_warp_count[warp] = __popc(ballot);
                __syncthreads();
                if (keep) {
                    const uint32_t slot = count + (warp != 0 ? s_warp_count[0] : 0u) +
                                          __popc(ballot & ((1u << lane) - 1u));
                    s_bin[slot] = gi;
                    s_hit[slot] = 0;
                    const float3 xyz_f = load_float3(reinterpret_cast<const float*>(&means[g]));
                    float opacity = activated_opacity(opacities[g]);
                    vec3 scale = activated_scale(scales[g]);
                    if constexpr (kMipFilter) {
                        opacity *= compensations[g];
                        scale = dilated_scale(scale, dilations[g]);
                    }
                    const mat3 iscl_rot = inverse_scale_matrix(scale) * glm::transpose(quat_to_rotmat(quats[g]));
                    s_iscl_rot[slot] = iscl_rot;
                    if constexpr (kGlobalShutter) {
                        const vec3 gro = iscl_rot * (ray_o - vec3{xyz_f.x, xyz_f.y, xyz_f.z});
                        s_xyz_opacity[slot] = {gro.x, gro.y, gro.z, opacity};
                    } else {
                        s_xyz_opacity[slot] = {xyz_f.x, xyz_f.y, xyz_f.z, opacity};
                    }
#pragma unroll
                    for (uint32_t s = 0; s < kSplatColors; ++s) {
                        s_rgbs[slot * kSmemColors + s] = colors[g * CDIM + fwd_splat_channel<CDIM>(s)];
                    }
                }
                count += s_warp_count[0] + s_warp_count[1];
                cursor += static_cast<int32_t>(kMicroThreads);
                // The counts are reread above; the next round must not overwrite them early.
                __syncthreads();
            }

            for (uint32_t t = 0; t < count && !done; ++t) {
                const float4 xyz_opac = load_float4(reinterpret_cast<const float*>(&s_xyz_opacity[t]));
                const mat3 iscl_rot = s_iscl_rot[t];

                const vec3 gro = kGlobalShutter
                                     ? vec3{xyz_opac.x, xyz_opac.y, xyz_opac.z}
                                     : iscl_rot * (ray_o - vec3{xyz_opac.x, xyz_opac.y, xyz_opac.z});
                vec3 grd_n;
                vec3 gcrod;
                float inv_grd_len = 0.0f;
                float grayDist = 0.0f;
                ray_gaussian_power(gro, iscl_rot * ray_d, grd_n, gcrod, inv_grd_len, grayDist);
                if (!(grayDist <= 11.09f)) {
                    continue;
                }

                const float alpha = min(0.999f, xyz_opac.w * __expf(-0.5f * grayDist));
                if (alpha < 1.f / 255.f) {
                    continue;
                }
                const float next_T = T * (1.0f - alpha);
                if (next_T <= 1e-4f) { // this pixel is done: exclusive
                    done = true;
                    break;
                }
                float depth = 0.0f;
                if constexpr (kHasRayDepth) {
                    const float t_hit = inv_grd_len > 0.0f ? (-glm::dot(grd_n, gro) * inv_grd_len) : 0.0f;
                    depth = t_hit > 0.0f ? t_hit * depth_scale : 0.0f;
                }
                const float splat_depth = kHasRayDepth ? depth : 0.0f;
                if ((median_out != nullptr || median_ids_out != nullptr) && T >= 0.5f && next_T < 0.5f) {
                    median = splat_depth;
                    median_id = s_bin[t];
                }
                const float vis = alpha * T;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    float c_val;
                    if constexpr (kHasRayDepth) {
                        if (k == kDepthChannel)
                            c_val = depth;
                        else if constexpr (kHasDepthSq) {
                            if (k == kDepthSqChannel)
                                c_val = depth * depth;
                            else
                                c_val = s_rgbs[t * kSmemColors + fwd_channel_to_slot<CDIM>(k)];
                        } else {
                            c_val = s_rgbs[t * kSmemColors + fwd_channel_to_slot<CDIM>(k)];
                        }
                    } else {
                        c_val = s_rgbs[t * kSmemColors + k];
                    }
                    pix_out[k] += c_val * vis;
                }
                cur_idx = static_cast<uint32_t>(s_bin[t]);
                s_hit[t] = 1;
                T = next_T;
            }

            // A survivor that blended no pixel of the sub-tile adds nothing to its
            // gradient either, so it loses the sub-tile's bit and the backward skips it.
            // Clearing it cannot change any render: no pixel here blends that pair.
            if (prune32 != nullptr || prune != nullptr) {
                __syncthreads();
                for (uint32_t t = tr; t < count; t += kMicroThreads) {
                    if (!s_hit[t]) {
                        if (prune32 != nullptr) {
                            atomicAnd(&prune32[s_bin[t]], ~(1u << sub_bit));
                        } else {
                            atomicAnd(&prune[s_bin[t]], ~(1ull << sub_bit));
                        }
                    }
                }
            }
        }

        if (inside) {
            render_alphas[pix_id] = 1.0f - T;
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                float bg_val = 0.0f;
                if (bg_images != nullptr && k < 3u) {
                    bg_val = bg_images[k * image_height * image_width + pix_id];
                } else if (backgrounds != nullptr && k < 3u) {
                    bg_val = backgrounds[k];
                }
                render_colors[chw_pix(k, pix_id, image_height, image_width)] = pix_out[k] + T * bg_val;
            }
            last_ids[pix_id] = static_cast<int32_t>(cur_idx);
            if (median_out != nullptr)
                median_out[pix_id] = median;
            if (median_ids_out != nullptr)
                median_ids_out[pix_id] = median_id;
        }
    }

    ////////////////////////////////////////////////////////////////
    // Launch Function
    ////////////////////////////////////////////////////////////////

    template <uint32_t CDIM>
    void launch_rasterize_to_pixels_from_world_3dgs_fwd_kernel(
        const float* means,
        const float* quats,
        const float* scales,
        const float* colors,
        const float* opacities,
        const float* backgrounds,
        const float* bg_images,
        const bool* masks,
        const float* compensations,
        const float* dilations,
        uint32_t C,
        uint32_t N,
        uint32_t n_isects,
        uint32_t image_width,
        uint32_t image_height,
        uint32_t tile_size,
        const float* viewmats0,
        const float* viewmats1,
        const float* Ks,
        CameraModelType camera_model,
        const UnscentedTransformParameters& ut_params,
        ShutterType rs_type,
        const float* radial_coeffs,
        const float* tangential_coeffs,
        const float* thin_prism_coeffs,
        const int32_t* tile_offsets,
        const int32_t* flatten_ids,
        float* renders,
        float* alphas,
        int32_t* last_ids,
        cudaStream_t stream, TileRange tiles,
        float* median_depth,
        const ScreenFootprint& footprint,
        int32_t* median_ids) {
        const bool packed = false; // Only support non-packed for now
        const uint32_t tile_width = (image_width + tile_size - 1) / tile_size;
        const uint32_t tile_height = (image_height + tile_size - 1) / tile_size;

        // Each block covers a tile on the image. In total there are
        // C * tile_height * tile_width blocks.
        dim3 threads = {tile_size, tile_size, 1};
        dim3 grid = {C, tile_height, tile_width};

        const bool global_shutter = is_global_shutter_launch(rs_type, viewmats1);
        const bool perfect_pinhole = is_perfect_pinhole_launch(
            camera_model, radial_coeffs, tangential_coeffs, thin_prism_coeffs);

        const uint32_t n_stage =
            (CDIM == 3 && tile_size == 16) ? 128u : (tile_size * tile_size);
        int64_t shmem_size =
            n_stage *
            (sizeof(vec4) + sizeof(mat3) + sizeof(float) * CDIM);

        if (n_isects == 0 && tiles.end == UINT32_MAX &&
            backgrounds == nullptr && bg_images == nullptr) {
            // Skip kernel launch if no intersections
            // Still need to clear output buffers
            LFS_CUDA_CHECK_MSG(
                cudaMemsetAsync(renders, 0,
                                C * image_height * image_width * CDIM * sizeof(float), stream),
                "gsplat empty-forward render clear");
            LFS_CUDA_CHECK_MSG(
                cudaMemsetAsync(alphas, 0,
                                C * image_height * image_width * sizeof(float), stream),
                "gsplat empty-forward alpha clear");
            LFS_CUDA_CHECK_MSG(
                cudaMemsetAsync(last_ids, 0,
                                C * image_height * image_width * sizeof(int32_t), stream),
                "gsplat empty-forward last-id clear");
            if (median_depth != nullptr) {
                LFS_CUDA_CHECK_MSG(
                    cudaMemsetAsync(median_depth, 0,
                                    C * image_height * image_width * sizeof(float), stream),
                    "gsplat empty-forward median depth clear");
            }
            if (median_ids != nullptr) {
                LFS_CUDA_CHECK_MSG(
                    cudaMemsetAsync(median_ids, 0xff,
                                    C * image_height * image_width * sizeof(int32_t), stream),
                    "gsplat empty-forward median ids clear");
            }
            return;
        }

        auto launch = [&](auto kernel) {
            set_kernel_max_dynamic_smem(
                kernel, static_cast<int>(shmem_size), "gsplat forward");
            kernel<<<grid, threads, shmem_size, stream>>>(
                C,
                N,
                n_isects,
                packed,
                reinterpret_cast<const vec3*>(means),
                reinterpret_cast<const vec4*>(quats),
                reinterpret_cast<const vec3*>(scales),
                colors,
                opacities,
                backgrounds,
                bg_images,
                masks,
                compensations,
                dilations,
                image_width,
                image_height,
                tile_size,
                tile_width,
                tile_height,
                viewmats0,
                viewmats1,
                Ks,
                camera_model,
                ut_params,
                rs_type,
                radial_coeffs,
                tangential_coeffs,
                thin_prism_coeffs,
                tile_offsets,
                flatten_ids,
                renders,
                alphas,
                last_ids, tiles, median_depth, median_ids);
            LFS_CUDA_LAUNCH_CHECK(stream, "gsplat.rasterize_to_pixels_fwd");
        };

        auto dispatch = [&]<bool kBatched, bool kMipFilter>() {
            if constexpr (CDIM <= 8) {
                if (use_micro_forward() && footprint.valid() && tile_size % kMicroSub == 0) {
                    const uint32_t subs = tile_size / kMicroSub;
                    const dim3 micro_grid =
                        (footprint.tile_order != nullptr && C == 1 && !kBatched)
                            ? dim3{tile_height * tile_width * subs * subs, 1u, 1u}
                            : dim3{C, tile_height * subs, tile_width * subs};
                    auto launch_micro = [&](auto kernel) {
                        cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                             cudaSharedmemCarveoutMaxShared);
                        kernel<<<micro_grid, kMicroThreads, 0, stream>>>(
                            C,
                            reinterpret_cast<const vec3*>(means),
                            reinterpret_cast<const vec4*>(quats),
                            reinterpret_cast<const vec3*>(scales),
                            colors, opacities, backgrounds, bg_images, masks, compensations, dilations,
                            image_width, image_height, tile_size, tile_width, tile_height,
                            viewmats0, viewmats1, Ks, camera_model, rs_type,
                            radial_coeffs, tangential_coeffs, thin_prism_coeffs,
                            tile_offsets, flatten_ids, renders, alphas, last_ids,
                            tiles, median_depth, footprint, median_ids);
                        LFS_CUDA_LAUNCH_CHECK(stream, "gsplat.rasterize_to_pixels_fwd_micro");
                    };
                    if (global_shutter && perfect_pinhole)
                        launch_micro(rasterize_to_pixels_from_world_3dgs_fwd_micro_kernel<CDIM, true, kBatched, kMipFilter, false, true>);
                    else if (!perfect_pinhole && footprint.unit_rays != nullptr &&
                             camera_model != CameraModelType::PINHOLE && camera_model != CameraModelType::ORTHO) {
                        if (global_shutter)
                            launch_micro(rasterize_to_pixels_from_world_3dgs_fwd_micro_kernel<CDIM, false, kBatched, kMipFilter, true, true>);
                        else
                            launch_micro(rasterize_to_pixels_from_world_3dgs_fwd_micro_kernel<CDIM, false, kBatched, kMipFilter, true, false>);
                    } else
                        launch_micro(rasterize_to_pixels_from_world_3dgs_fwd_micro_kernel<CDIM, false, kBatched, kMipFilter, false, false>);
                    return;
                }
            }
            if constexpr (CDIM == 3) {
                if (global_shutter && perfect_pinhole) {
                    launch(rasterize_to_pixels_from_world_3dgs_fwd_kernel<CDIM, float, true, kBatched, kMipFilter>);
                    return;
                }
            }
            launch(rasterize_to_pixels_from_world_3dgs_fwd_kernel<CDIM, float, false, kBatched, kMipFilter>);
        };
        auto dispatch_batched = [&]<bool kMipFilter>() {
            if (tiles.end == UINT32_MAX)
                dispatch.template operator()<false, kMipFilter>();
            else
                dispatch.template operator()<true, kMipFilter>();
        };
        if (dilations != nullptr)
            dispatch_batched.template operator()<true>();
        else
            dispatch_batched.template operator()<false>();
    }

    ////////////////////////////////////////////////////////////////
    // Explicit Instantiations
    ////////////////////////////////////////////////////////////////

#define __INS__(CDIM)                                                          \
    template void launch_rasterize_to_pixels_from_world_3dgs_fwd_kernel<CDIM>( \
        const float* means,                                                    \
        const float* quats,                                                    \
        const float* scales,                                                   \
        const float* colors,                                                   \
        const float* opacities,                                                \
        const float* backgrounds,                                              \
        const float* bg_images,                                                \
        const bool* masks,                                                     \
        const float* compensations,                                            \
        const float* dilations,                                                \
        uint32_t C,                                                            \
        uint32_t N,                                                            \
        uint32_t n_isects,                                                     \
        uint32_t image_width,                                                  \
        uint32_t image_height,                                                 \
        uint32_t tile_size,                                                    \
        const float* viewmats0,                                                \
        const float* viewmats1,                                                \
        const float* Ks,                                                       \
        CameraModelType camera_model,                                          \
        const UnscentedTransformParameters& ut_params,                         \
        ShutterType rs_type,                                                   \
        const float* radial_coeffs,                                            \
        const float* tangential_coeffs,                                        \
        const float* thin_prism_coeffs,                                        \
        const int32_t* tile_offsets,                                           \
        const int32_t* flatten_ids,                                            \
        float* renders,                                                        \
        float* alphas,                                                         \
        int32_t* last_ids,                                                     \
        cudaStream_t stream, TileRange tiles,                                  \
        float* median_depth,                                                   \
        const ScreenFootprint& footprint,                                      \
        int32_t* median_ids);

    __INS__(1)
    __INS__(2)
    __INS__(3)
    __INS__(4)
    __INS__(5)
    __INS__(8)
    __INS__(9)
    __INS__(16)
    __INS__(17)
    __INS__(32)
    __INS__(33)
    __INS__(64)
    __INS__(65)
    __INS__(128)
    __INS__(129)
    __INS__(256)
    __INS__(257)
    __INS__(512)
    __INS__(513)
#undef __INS__

} // namespace gsplat_lfs
