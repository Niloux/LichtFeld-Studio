/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <cooperative_groups.h>
#include <cstdlib>
#include <cuda_runtime.h>

#include "Cameras.cuh"
#include "Common.h"
#include "FreeSpace.h"
#include "FromWorldRay.cuh"
#include "Rasterization.h"
#include "Utils.cuh"

namespace gsplat_lfs {

    namespace cg = cooperative_groups;

    ////////////////////////////////////////////////////////////////
    // Forward Kernel
    ////////////////////////////////////////////////////////////////

    template <uint32_t CDIM, typename scalar_t, bool kPerfectPinhole, bool kBatched>
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
        const FreeSpaceRender free_space) {
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
        float* const T_front_out =
            free_space.T_front != nullptr && free_space.gt_depth != nullptr
                ? free_space.T_front + cid * image_height * image_width
                : nullptr;
        if (backgrounds != nullptr) {
            backgrounds += cid * CDIM;
        }
        if (bg_images != nullptr) {
            bg_images += cid * CDIM * image_height * image_width;
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
            if (T_front_out != nullptr)
                T_front_out[pix_id] = 1.0f;
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
        // transmittance of the splats in front of the scan surface
        float T_front = 1.0f;
        const float free_limit = T_front_out != nullptr && inside
                                     ? free_space.limit(cid, j, i, image_width, image_height)
                                     : 0.0f;

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
                const float opac = activated_opacity(opacities[g]);
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
                const vec3 grd = safe_normalize(iscl_rot * ray_d);
                const vec3 gcrod = glm::cross(grd, gro);
                const float grayDist = glm::dot(gcrod, gcrod);
                const float power = -0.5f * grayDist;

                float alpha = min(0.999f, opac * __expf(power));
                if (alpha < 1.f / 255.f) {
                    continue;
                }

                const float next_T = T * (1.0f - alpha);
                if (next_T <= 1e-4f) { // this pixel is done: exclusive
                    done = true;
                    break;
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
                    pix_out[3] += c.w * vis;
                } else {
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        pix_out[k] += rgbs_batch[t * CDIM + k] * vis;
                    }
                }
                cur_idx = batch_start + t;
                if (free_limit > 0.0f && rgbs_batch[t * CDIM + free_space.depth_channel] < free_limit) {
                    T_front *= 1.0f - alpha;
                }

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
            if (T_front_out != nullptr)
                T_front_out[pix_id] = T_front;
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
    constexpr uint32_t kMicroStage = 2 * kMicroThreads;

    bool use_micro_forward() {
        static const bool enabled = [] {
            const char* value = std::getenv("LFS_GUT_MICRO_FWD");
            return value == nullptr || value[0] != '0';
        }();
        return enabled;
    }

    template <uint32_t CDIM, bool kPerfectPinhole, bool kBatched>
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
        const FreeSpaceRender free_space,
        const ScreenFootprint footprint) {
        const uint32_t cid = blockIdx.x;
        const uint32_t subs = tile_size / kMicroSub;
        const uint32_t sub_y = blockIdx.y;
        const uint32_t sub_x = blockIdx.z;
        const uint32_t tile_id = (sub_y / subs) * tile_width + sub_x / subs;
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
        float* const T_front_out =
            free_space.T_front != nullptr && free_space.gt_depth != nullptr
                ? free_space.T_front + cid * image_height * image_width
                : nullptr;
        if (backgrounds != nullptr) {
            backgrounds += cid * CDIM;
        }
        if (bg_images != nullptr) {
            bg_images += cid * CDIM * image_height * image_width;
        }
        if (masks != nullptr) {
            masks += cid * tile_height * tile_width;
        }

        const int32_t pix_id = i * image_width + j;
        const WorldRay ray = from_world_pixel_ray<kPerfectPinhole>(
            camera_model_type, rs_type, image_width, image_height,
            static_cast<float>(j) + 0.5f, static_cast<float>(i) + 0.5f,
            viewmats0, viewmats1, Ks, cid,
            radial_coeffs, tangential_coeffs, thin_prism_coeffs);
        const vec3 ray_d = ray.ray_dir;
        const vec3 ray_o = ray.ray_org;

        const bool inside = i < image_height && j < image_width;
        bool done = !inside || !ray.valid_flag;

        // A masked-out tile renders its background; the mask is uniform per block.
        if (masks != nullptr && !masks[tile_id]) {
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
            if (T_front_out != nullptr)
                T_front_out[pix_id] = 1.0f;
            return;
        }

        __shared__ vec4 s_xyz_opacity[kMicroStage];
        __shared__ mat3 s_iscl_rot[kMicroStage];
        __shared__ float s_rgbs[kMicroStage * CDIM];
        __shared__ int32_t s_bin[kMicroStage]; // the survivor's position in the tile's list
        __shared__ uint8_t s_hit[kMicroStage]; // whether the survivor blended any pixel
        __shared__ uint32_t s_warp_count[kMicroThreads / 32];
        auto* const prune = reinterpret_cast<unsigned long long*>(footprint.prune_tile_keys);

        const int32_t range_end = tile_offsets[tile_id + 1];
        int32_t cursor = tile_offsets[tile_id];
        const float sub_x0 = static_cast<float>(sub_x * kMicroSub);
        const float sub_y0 = static_cast<float>(sub_y * kMicroSub);
        const uint32_t sub_bit = (sub_y % subs) * subs + sub_x % subs;

        float T = 1.0f;
        uint32_t cur_idx = 0;
        float T_front = 1.0f;
        const float free_limit = T_front_out != nullptr && inside
                                     ? free_space.limit(cid, j, i, image_width, image_height)
                                     : 0.0f;
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
                    if (footprint.tile_keys != nullptr) {
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
                    s_xyz_opacity[slot] = {xyz_f.x, xyz_f.y, xyz_f.z, activated_opacity(opacities[g])};
                    const vec3 scale = activated_scale(scales[g]);
                    const mat3 R = quat_to_rotmat(quats[g]);
                    const mat3 S = mat3(
                        1.0f / scale[0], 0.f, 0.f,
                        0.f, 1.0f / scale[1], 0.f,
                        0.f, 0.f, 1.0f / scale[2]);
                    s_iscl_rot[slot] = S * glm::transpose(R);
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        s_rgbs[slot * CDIM + k] = colors[g * CDIM + k];
                    }
                }
                count += s_warp_count[0] + s_warp_count[1];
                cursor += static_cast<int32_t>(kMicroThreads);
                // The counts are reread above; the next round must not overwrite them early.
                __syncthreads();
            }

            for (uint32_t t = 0; t < count && !done; ++t) {
                const float4 xyz_opac = load_float4(reinterpret_cast<const float*>(&s_xyz_opacity[t]));
                const vec3 xyz = {xyz_opac.x, xyz_opac.y, xyz_opac.z};
                const mat3 iscl_rot = s_iscl_rot[t];

                const vec3 gro = iscl_rot * (ray_o - xyz);
                const vec3 grd = safe_normalize(iscl_rot * ray_d);
                const vec3 gcrod = glm::cross(grd, gro);
                const float power = -0.5f * glm::dot(gcrod, gcrod);

                const float alpha = min(0.999f, xyz_opac.w * __expf(power));
                if (alpha < 1.f / 255.f) {
                    continue;
                }
                const float next_T = T * (1.0f - alpha);
                if (next_T <= 1e-4f) { // this pixel is done: exclusive
                    done = true;
                    break;
                }
                const float vis = alpha * T;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    pix_out[k] += s_rgbs[t * CDIM + k] * vis;
                }
                cur_idx = static_cast<uint32_t>(s_bin[t]);
                s_hit[t] = 1;
                if (free_limit > 0.0f && s_rgbs[t * CDIM + free_space.depth_channel] < free_limit) {
                    T_front *= 1.0f - alpha;
                }
                T = next_T;
            }

            // A survivor that blended no pixel of the sub-tile adds nothing to its
            // gradient either, so it loses the sub-tile's bit and the backward skips it.
            // Clearing it cannot change any render: no pixel here blends that pair.
            if (prune != nullptr) {
                __syncthreads();
                for (uint32_t t = tr; t < count; t += kMicroThreads) {
                    if (!s_hit[t])
                        atomicAnd(&prune[s_bin[t]], ~(1ull << sub_bit));
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
            if (T_front_out != nullptr)
                T_front_out[pix_id] = T_front;
        }
    }

    namespace {
        __global__ void fill_ones_kernel(float* out, const uint32_t n) {
            const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n)
                out[i] = 1.0f;
        }
    } // namespace

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
        const FreeSpaceRender& free_space_in,
        const ScreenFootprint& footprint) {
        const bool packed = false; // Only support non-packed for now
        // The free-space test reads the splat depth from a render channel.
        const FreeSpaceRender free_space =
            free_space_in.depth_channel < CDIM ? free_space_in : FreeSpaceRender{};
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

        if (n_isects == 0 && tiles.end == UINT32_MAX) {
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
            if (free_space.T_front != nullptr) {
                fill_ones_kernel<<<(C * image_height * image_width + 255) / 256, 256, 0, stream>>>(
                    free_space.T_front, C * image_height * image_width);
                LFS_CUDA_LAUNCH_CHECK(stream, "gsplat empty-forward T_front fill");
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
                last_ids, tiles, free_space);
            LFS_CUDA_LAUNCH_CHECK(stream, "gsplat.rasterize_to_pixels_fwd");
        };

        auto dispatch = [&]<bool kBatched>() {
            if constexpr (CDIM <= 8) {
                if (use_micro_forward() && footprint.valid() && tile_size % kMicroSub == 0) {
                    const uint32_t subs = tile_size / kMicroSub;
                    const dim3 micro_grid = {C, tile_height * subs, tile_width * subs};
                    auto launch_micro = [&](auto kernel) {
                        kernel<<<micro_grid, kMicroThreads, 0, stream>>>(
                            C,
                            reinterpret_cast<const vec3*>(means),
                            reinterpret_cast<const vec4*>(quats),
                            reinterpret_cast<const vec3*>(scales),
                            colors, opacities, backgrounds, bg_images, masks,
                            image_width, image_height, tile_size, tile_width, tile_height,
                            viewmats0, viewmats1, Ks, camera_model, rs_type,
                            radial_coeffs, tangential_coeffs, thin_prism_coeffs,
                            tile_offsets, flatten_ids, renders, alphas, last_ids,
                            tiles, free_space, footprint);
                        LFS_CUDA_LAUNCH_CHECK(stream, "gsplat.rasterize_to_pixels_fwd_micro");
                    };
                    if (global_shutter && perfect_pinhole)
                        launch_micro(rasterize_to_pixels_from_world_3dgs_fwd_micro_kernel<CDIM, true, kBatched>);
                    else
                        launch_micro(rasterize_to_pixels_from_world_3dgs_fwd_micro_kernel<CDIM, false, kBatched>);
                    return;
                }
            }
            if constexpr (CDIM == 3) {
                if (global_shutter && perfect_pinhole) {
                    launch(rasterize_to_pixels_from_world_3dgs_fwd_kernel<CDIM, float, true, kBatched>);
                    return;
                }
            }
            launch(rasterize_to_pixels_from_world_3dgs_fwd_kernel<CDIM, float, false, kBatched>);
        };
        if (tiles.end == UINT32_MAX)
            dispatch.template operator()<false>();
        else
            dispatch.template operator()<true>();
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
        const FreeSpaceRender& free_space,                                     \
        const ScreenFootprint& footprint);

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
