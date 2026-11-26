/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <cooperative_groups.h>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "Cameras.cuh"
#include "Common.h"
#include "FromWorldRay.cuh"
#include "Rasterization.h"
#include "Utils.cuh"
#include "core/cuda/sigmoid.cuh"

// Use standard CUDA atomic add
#ifndef gpuAtomicAdd
#define gpuAtomicAdd atomicAdd
#endif

namespace gsplat_lfs {

    namespace cg = cooperative_groups;

    // 16x16 tile, 128 threads, two horizontally adjacent pixels per thread.
    constexpr uint32_t kBwdDualThreads = 128;
    constexpr uint32_t kBwdDualBatch = 256;

    __device__ __forceinline__ void record_blend_peak(float* out, int32_t g, float peak) {
        if (out && peak > 0.0f)
            atomicMax(reinterpret_cast<int*>(out + g), __float_as_int(peak));
    }

    __device__ __forceinline__ float load_edge_weight(const void* map, const size_t idx, const bool fp16) {
        return fp16
                   ? __half2float(static_cast<const __half*>(map)[idx])
                   : static_cast<const float*>(map)[idx];
    }

    __device__ __forceinline__ const void* offset_edge_weight_map(
        const void* map, const size_t elem_offset, const bool fp16) {
        if (!map)
            return nullptr;
        return fp16
                   ? static_cast<const void*>(static_cast<const __half*>(map) + elem_offset)
                   : static_cast<const void*>(static_cast<const float*>(map) + elem_offset);
    }

    struct PixelBwd {
        bool active;
        int32_t pix_id;
        int32_t bin_final;
        vec3 ray_o;
        vec3 ray_d;
        float T;
        float T_final;
        float buffer[3];
        float v_render_c[3];
        float v_render_a;
        float bg_accum;
        float pixel_error;
    };

    template <bool kPerfectPinhole>
    __device__ __forceinline__ void init_pixel_bwd(
        PixelBwd& pix,
        const uint32_t i,
        const uint32_t j,
        const uint32_t image_width,
        const uint32_t image_height,
        const CameraModelType camera_model_type,
        const ShutterType rs_type,
        const float* viewmats0,
        const float* viewmats1,
        const float* Ks,
        const uint32_t cid,
        const float* radial_coeffs,
        const float* tangential_coeffs,
        const float* thin_prism_coeffs,
        const float* render_alphas,
        const int32_t* last_ids,
        const float* v_render_colors,
        const float* v_render_alphas,
        const float* backgrounds,
        const float* bg_images,
        const bool have_bg,
        const bool do_dens,
        const float* densification_error_map) {
        const float px = (float)j + 0.5f;
        const float py = (float)i + 0.5f;
        pix.pix_id = min(i * image_width + j, image_width * image_height - 1);
        const WorldRay ray = from_world_pixel_ray<kPerfectPinhole>(
            camera_model_type, rs_type, image_width, image_height, px, py,
            viewmats0, viewmats1, Ks, cid,
            radial_coeffs, tangential_coeffs, thin_prism_coeffs);
        pix.ray_d = ray.ray_dir;
        pix.ray_o = ray.ray_org;
        pix.active = (i < image_height && j < image_width) && ray.valid_flag;
        pix.T_final = 1.0f - render_alphas[pix.pix_id];
        pix.T = pix.T_final;
#pragma unroll
        for (uint32_t k = 0; k < 3; ++k) {
            pix.buffer[k] = 0.f;
            pix.v_render_c[k] = v_render_colors[chw_pix(k, pix.pix_id, image_height, image_width)];
        }
        pix.bin_final = pix.active ? last_ids[pix.pix_id] : 0;
        pix.v_render_a = v_render_alphas[pix.pix_id];
        pix.bg_accum = 0.f;
        if (have_bg) {
#pragma unroll
            for (uint32_t k = 0; k < 3; ++k) {
                float bg_val;
                if (bg_images != nullptr) {
                    bg_val = bg_images[k * image_height * image_width + pix.pix_id];
                } else {
                    bg_val = backgrounds[k];
                }
                pix.bg_accum += bg_val * pix.v_render_c[k];
            }
        }
        pix.pixel_error = do_dens ? densification_error_map[pix.pix_id] : 0.f;
    }

    template <bool kEdgeScoring>
    __device__ __forceinline__ bool contribute_pixel(
        PixelBwd& pix,
        const int32_t isect_idx,
        const float opac,
        const mat3 Mt,
        const vec3 gro,
        const vec3 o_minus_mu,
        const vec3 scale_act,
        const vec4 quat,
        const float rgb0,
        const float rgb1,
        const float rgb2,
        const bool have_bg,
        const bool do_dens,
        float v_rgb[3],
        vec3& v_mean,
        vec3& v_scale,
        vec4& v_quat,
        float& v_opacity,
        float& dens_w,
        float& dens_e,
        const void* edge_weight_map,
        const bool edge_weight_fp16,
        float& edge_score,
        float& blend_peak,
        mat3& R,
        bool& have_R) {
        bool valid = pix.active;
        if (isect_idx > pix.bin_final) {
            valid = false;
        }
        if (!valid) {
            return false;
        }
        const vec3 grd = Mt * pix.ray_d;
        const vec3 grd_n = safe_normalize(grd);
        const vec3 gcrod = glm::cross(grd_n, gro);
        const float grayDist = glm::dot(gcrod, gcrod);
        const float power = -0.5f * grayDist;
        const float vis = __expf(power);
        const float alpha = min(0.999f, opac * vis);
        if (power > 0.f || alpha < 1.f / 255.f) {
            return false;
        }

        const float ra = 1.0f / (1.0f - alpha);
        pix.T *= ra;
        const float fac = alpha * pix.T;
        blend_peak = fmaxf(blend_peak, fac);
        if constexpr (kEdgeScoring) {
            const float edge_weight = load_edge_weight(edge_weight_map, pix.pix_id, edge_weight_fp16);
            const float edge_contribution = fac * edge_weight;
            if (edge_weight > 0.0f && isfinite(edge_contribution)) {
                edge_score += edge_contribution;
            }
        }
        v_rgb[0] += fac * pix.v_render_c[0];
        v_rgb[1] += fac * pix.v_render_c[1];
        v_rgb[2] += fac * pix.v_render_c[2];
        if (do_dens) {
            dens_w += fac;
            dens_e += fac * pix.pixel_error;
        }
        float v_alpha = (rgb0 * pix.T - pix.buffer[0] * ra) * pix.v_render_c[0] +
                        (rgb1 * pix.T - pix.buffer[1] * ra) * pix.v_render_c[1] +
                        (rgb2 * pix.T - pix.buffer[2] * ra) * pix.v_render_c[2];
        v_alpha += pix.T_final * ra * pix.v_render_a;
        if (have_bg) {
            v_alpha += -pix.T_final * ra * pix.bg_accum;
        }

        if (opac * vis <= 0.999f) {
            const float v_vis = opac * v_alpha;
            const float v_gradDist = -0.5f * vis * v_vis;
            const vec3 v_gcrod = 2.0f * v_gradDist * gcrod;
            const vec3 v_grd_n = -glm::cross(v_gcrod, gro);
            const vec3 v_gro = glm::cross(v_gcrod, grd_n);
            const vec3 v_grd = safe_normalize_bw(grd, v_grd_n);
            const mat3 v_Mt = glm::outerProduct(v_grd, pix.ray_d) +
                              glm::outerProduct(v_gro, o_minus_mu);
            const vec3 v_o_minus_mu = glm::transpose(Mt) * v_gro;
            v_mean += -v_o_minus_mu;
            if (!have_R) {
                R = quat_to_rotmat(quat);
                have_R = true;
            }
            quat_scale_to_preci_half_vjp(
                quat, scale_act, R, glm::transpose(v_Mt), v_quat, v_scale);
            v_opacity += vis * v_alpha;
        }

        pix.buffer[0] += rgb0 * fac;
        pix.buffer[1] += rgb1 * fac;
        pix.buffer[2] += rgb2 * fac;
        return true;
    }

    template <bool kSharedOrigin, bool kEdgeScoring>
    __device__ __forceinline__ void contribute_pixel_pair(
        PixelBwd& pix0,
        PixelBwd& pix1,
        const int32_t isect_idx,
        const vec4 xyz_opac,
        const mat3 Mt,
        const vec3 scale_act,
        const vec4 quat,
        const float rgb0,
        const float rgb1,
        const float rgb2,
        const bool have_bg,
        const bool do_dens,
        float v_rgb[3],
        vec3& v_mean,
        vec3& v_scale,
        vec4& v_quat,
        float& v_opacity,
        float& dens_w,
        float& dens_e,
        const void* edge_weight_map,
        const bool edge_weight_fp16,
        float& edge_score,
        float& blend_peak,
        bool& hit0,
        bool& hit1) {
        const float opac = xyz_opac[3];
        const vec3 xyz = {xyz_opac[0], xyz_opac[1], xyz_opac[2]};
        vec3 o_minus_mu0, o_minus_mu1, gro0, gro1;
        if constexpr (kSharedOrigin) {
            const vec3 ray_o = pix0.active ? pix0.ray_o : pix1.ray_o;
            o_minus_mu0 = ray_o - xyz;
            gro0 = Mt * o_minus_mu0;
            o_minus_mu1 = o_minus_mu0;
            gro1 = gro0;
        } else {
            o_minus_mu0 = pix0.ray_o - xyz;
            gro0 = Mt * o_minus_mu0;
            o_minus_mu1 = pix1.ray_o - xyz;
            gro1 = Mt * o_minus_mu1;
        }
        mat3 R;
        bool have_R = false;
        hit0 = contribute_pixel<kEdgeScoring>(
            pix0, isect_idx, opac, Mt, gro0, o_minus_mu0, scale_act, quat,
            rgb0, rgb1, rgb2, have_bg, do_dens, v_rgb, v_mean, v_scale, v_quat,
            v_opacity, dens_w, dens_e, edge_weight_map, edge_weight_fp16, edge_score, blend_peak, R, have_R);
        hit1 = contribute_pixel<kEdgeScoring>(
            pix1, isect_idx, opac, Mt, gro1, o_minus_mu1, scale_act, quat,
            rgb0, rgb1, rgb2, have_bg, do_dens, v_rgb, v_mean, v_scale, v_quat,
            v_opacity, dens_w, dens_e, edge_weight_map, edge_weight_fp16, edge_score, blend_peak, R, have_R);
    }

    __device__ __forceinline__ float reduce_field(float v, const unsigned n_contrib, const int src_lane) {
        if (n_contrib == 1u) {
            return __shfl_sync(0xffffffffu, v, src_lane);
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            v += __shfl_xor_sync(0xffffffffu, v, offset);
        }
        return v;
    }

    // d s / d log(raw scale) for the scale s the kernel used: s itself, or s0^2 / s for s = sqrt(s0^2 + sigma2)
    // under the mip filter, whose sigma2 is a constant.
    __device__ __forceinline__ vec3 log_scale_chain(const vec3 scale_act, const vec3* scales, const float* dilations,
                                                    const int32_t g) {
        if (dilations == nullptr)
            return scale_act;
        const vec3 s0 = activated_scale(scales[g]);
        return s0 * s0 / scale_act;
    }

    __device__ __forceinline__ void flush_splat_grads(
        const int32_t g,
        const uint32_t N,
        const bool do_dens,
        const float v_rgb[3],
        const vec3 v_mean,
        const vec3 v_scale,
        const vec4 v_quat,
        const float v_opacity,
        const float dens_w,
        const float dens_e,
        const vec3 scale_act,
        const float opacity_raw,
        vec3* v_means,
        vec4* v_quats,
        vec3* v_scales,
        float* v_colors,
        float* v_opacities,
        float* densification_weight,
        float* densification_error) {
        float* v_rgb_ptr = v_colors + 3 * g;
        gpuAtomicAdd(v_rgb_ptr, v_rgb[0]);
        gpuAtomicAdd(v_rgb_ptr + 1, v_rgb[1]);
        gpuAtomicAdd(v_rgb_ptr + 2, v_rgb[2]);
        float* v_mean_ptr = (float*)(v_means) + 3 * g;
        gpuAtomicAdd(v_mean_ptr, v_mean.x);
        gpuAtomicAdd(v_mean_ptr + 1, v_mean.y);
        gpuAtomicAdd(v_mean_ptr + 2, v_mean.z);
        float* v_scale_ptr = (float*)(v_scales) + 3 * g;
        gpuAtomicAdd(v_scale_ptr, v_scale.x * scale_act.x);
        gpuAtomicAdd(v_scale_ptr + 1, v_scale.y * scale_act.y);
        gpuAtomicAdd(v_scale_ptr + 2, v_scale.z * scale_act.z);
        float* v_quat_ptr = (float*)(v_quats) + 4 * g;
        gpuAtomicAdd(v_quat_ptr, v_quat.x);
        gpuAtomicAdd(v_quat_ptr + 1, v_quat.y);
        gpuAtomicAdd(v_quat_ptr + 2, v_quat.z);
        gpuAtomicAdd(v_quat_ptr + 3, v_quat.w);
        gpuAtomicAdd(v_opacities + g, v_opacity * lfs::core::sigmoid_derivative_from_logit(opacity_raw));
        if (do_dens) {
            if (densification_weight != nullptr)
                gpuAtomicAdd(densification_weight + g, dens_w);
            gpuAtomicAdd(densification_error + g, dens_e);
        }
    }

    template <typename scalar_t, bool kSharedOrigin, bool kPerfectPinhole, bool kEdgeScoring, bool kBatched>
    __global__ void rasterize_to_pixels_from_world_3dgs_bwd_dual_kernel(
        const uint32_t C,
        const uint32_t N,
        const uint32_t n_isects,
        const bool packed,
        const vec3* __restrict__ means,
        const vec4* __restrict__ quats,
        const vec3* __restrict__ scales,
        const scalar_t* __restrict__ colors,
        const scalar_t* __restrict__ opacities,
        const scalar_t* __restrict__ backgrounds,
        const scalar_t* __restrict__ bg_images,
        const bool* __restrict__ masks,
        const scalar_t* __restrict__ compensations,
        const scalar_t* __restrict__ dilations,
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        const uint32_t tile_width,
        const uint32_t tile_height,
        const scalar_t* __restrict__ viewmats0,
        const scalar_t* __restrict__ viewmats1,
        const scalar_t* __restrict__ Ks,
        const CameraModelType camera_model_type,
        const UnscentedTransformParameters ut_params,
        const ShutterType rs_type,
        const scalar_t* __restrict__ radial_coeffs,
        const scalar_t* __restrict__ tangential_coeffs,
        const scalar_t* __restrict__ thin_prism_coeffs,
        const int32_t* __restrict__ tile_offsets,
        const int32_t* __restrict__ flatten_ids,
        const scalar_t* __restrict__ render_alphas,
        const int32_t* __restrict__ last_ids,
        const scalar_t* __restrict__ v_render_colors,
        const scalar_t* __restrict__ v_render_alphas,
        vec3* __restrict__ v_means,
        vec4* __restrict__ v_quats,
        vec3* __restrict__ v_scales,
        scalar_t* __restrict__ v_colors,
        scalar_t* __restrict__ v_opacities,
        float* __restrict__ densification_weight,
        float* __restrict__ densification_error,
        const scalar_t* __restrict__ densification_error_map,
        const void* __restrict__ edge_weight_map,
        float* __restrict__ edge_score_out, const TileRange tiles,
        const ColorVarianceRender, float* __restrict__ blend_peak_out,
        const RenderGradPlanes, const bool edge_weight_fp16) {
        auto block = cg::this_thread_block();
        const uint32_t cid = block.group_index().x;
        const uint32_t tile_id =
            block.group_index().y * tile_width + block.group_index().z;
        if constexpr (kBatched) {
            if (tile_id < tiles.begin || tile_id >= tiles.end)
                return;
        }
        const uint32_t tx = block.thread_index().x;
        const uint32_t ty = block.thread_index().y;
        const uint32_t i = block.group_index().y * tile_size + ty;
        const uint32_t j0 = block.group_index().z * tile_size + tx * 2u;
        const uint32_t j1 = j0 + 1u;

        tile_offsets += cid * tile_height * tile_width;
        render_alphas += cid * image_height * image_width;
        last_ids += cid * image_height * image_width;
        v_render_colors += cid * image_height * image_width * 3u;
        v_render_alphas += cid * image_height * image_width;
        if (backgrounds != nullptr) {
            backgrounds += cid * 3u;
        }
        if (bg_images != nullptr) {
            bg_images += cid * 3u * image_height * image_width;
        }
        if (masks != nullptr) {
            masks += cid * tile_height * tile_width;
        }
        if constexpr (kEdgeScoring) {
            edge_weight_map = offset_edge_weight_map(
                edge_weight_map, static_cast<size_t>(cid) * image_height * image_width, edge_weight_fp16);
            edge_score_out += cid * N;
        }
        if (masks != nullptr && !masks[tile_id]) {
            return;
        }

        const bool have_bg = bg_images != nullptr || backgrounds != nullptr;
        const bool do_dens = densification_error != nullptr && densification_error_map != nullptr;

        PixelBwd pix0, pix1;
        init_pixel_bwd<kPerfectPinhole>(
            pix0, i, j0, image_width, image_height,
            camera_model_type, rs_type, viewmats0, viewmats1, Ks, cid,
            radial_coeffs, tangential_coeffs, thin_prism_coeffs,
            render_alphas, last_ids, v_render_colors, v_render_alphas,
            backgrounds, bg_images, have_bg, do_dens, densification_error_map);
        init_pixel_bwd<kPerfectPinhole>(
            pix1, i, j1, image_width, image_height,
            camera_model_type, rs_type, viewmats0, viewmats1, Ks, cid,
            radial_coeffs, tangential_coeffs, thin_prism_coeffs,
            render_alphas, last_ids, v_render_colors, v_render_alphas,
            backgrounds, bg_images, have_bg, do_dens, densification_error_map);

        const int32_t range_start = tile_offsets[tile_id];
        const int32_t range_end = tile_offsets[tile_id + 1];
        const uint32_t num_batches =
            (range_end - range_start + kBwdDualBatch - 1) / kBwdDualBatch;

        extern __shared__ int s[];
        int32_t* id_batch = (int32_t*)s;
        vec4* xyz_opacity_batch = reinterpret_cast<vec4*>(&id_batch[kBwdDualBatch]);
        vec3* scale_batch = reinterpret_cast<vec3*>(&xyz_opacity_batch[kBwdDualBatch]);
        vec4* quat_batch = reinterpret_cast<vec4*>(&scale_batch[kBwdDualBatch]);
        mat3* mt_batch = reinterpret_cast<mat3*>(&quat_batch[kBwdDualBatch]);
        float* rgbs_batch = reinterpret_cast<float*>(&mt_batch[kBwdDualBatch]);
        float* edge_score_batch = nullptr;
        if constexpr (kEdgeScoring) {
            edge_score_batch = reinterpret_cast<float*>(&rgbs_batch[kBwdDualBatch * 3u]);
        }

        const uint32_t tr = block.thread_rank();
        cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
        const int32_t warp_bin_final =
            cg::reduce(warp, max(pix0.bin_final, pix1.bin_final), cg::greater<int>());

        auto stage_splat = [&](const int32_t idx, const uint32_t slot) {
            if (idx >= range_start) {
                const int32_t g = flatten_ids[idx];
                id_batch[slot] = g;
                const vec3 xyz = means[g];
                float opac = activated_opacity(opacities[g]);
                if (compensations != nullptr)
                    opac *= compensations[g];
                xyz_opacity_batch[slot] = {xyz.x, xyz.y, xyz.z, opac};
                const vec3 scale_act = activated_scale(scales[g]);
                const vec4 quat = quats[g];
                scale_batch[slot] = scale_act;
                quat_batch[slot] = quat;
                const mat3 R = quat_to_rotmat(quat);
                const mat3 S = mat3(
                    1.0f / scale_act[0], 0.f, 0.f,
                    0.f, 1.0f / scale_act[1], 0.f,
                    0.f, 0.f, 1.0f / scale_act[2]);
                mt_batch[slot] = glm::transpose(R * S);
                if (dilations != nullptr) { // mip filter; kept apart so the unfiltered path is computed as before
                    scale_batch[slot] = dilated_scale(scale_act, dilations[g]);
                    mt_batch[slot] = glm::transpose(R * inverse_scale_matrix(scale_batch[slot]));
                }
                rgbs_batch[slot * 3u + 0] = colors[g * 3u + 0];
                rgbs_batch[slot * 3u + 1] = colors[g * 3u + 1];
                rgbs_batch[slot * 3u + 2] = colors[g * 3u + 2];
            }
        };

        for (uint32_t b = 0; b < num_batches; ++b) {
            block.sync();
            if constexpr (kEdgeScoring) {
                for (uint32_t slot = tr; slot < kBwdDualBatch; slot += block.size()) {
                    edge_score_batch[slot] = 0.f;
                }
                block.sync();
            }
            const int32_t batch_end = range_end - 1 - static_cast<int32_t>(kBwdDualBatch * b);
            const int32_t batch_size = min(static_cast<int32_t>(kBwdDualBatch), batch_end + 1 - range_start);
            stage_splat(batch_end - static_cast<int32_t>(tr), tr);
            stage_splat(batch_end - static_cast<int32_t>(tr + kBwdDualThreads), tr + kBwdDualThreads);
            block.sync();

            for (uint32_t t = max(0, batch_end - warp_bin_final); t < static_cast<uint32_t>(batch_size); ++t) {
                const int32_t isect_idx = batch_end - static_cast<int32_t>(t);
                const vec4 xyz_opac = xyz_opacity_batch[t];
                const mat3 Mt = mt_batch[t];
                const vec3 scale_act = scale_batch[t];
                const vec4 quat = quat_batch[t];
                const float rgb0 = rgbs_batch[t * 3u + 0];
                const float rgb1 = rgbs_batch[t * 3u + 1];
                const float rgb2 = rgbs_batch[t * 3u + 2];

                float v_rgb_local[3] = {0.f, 0.f, 0.f};
                vec3 v_mean_local = {0.f, 0.f, 0.f};
                vec3 v_scale_local = {0.f, 0.f, 0.f};
                vec4 v_quat_local = {0.f, 0.f, 0.f, 0.f};
                float v_opacity_local = 0.f;
                float densification_weight_local = 0.f;
                float densification_error_weighted_local = 0.f;
                float edge_score_local = 0.f;
                float blend_peak = 0.f;

                bool hit0 = false;
                bool hit1 = false;
                contribute_pixel_pair<kSharedOrigin, kEdgeScoring>(
                    pix0, pix1, isect_idx, xyz_opac, Mt, scale_act, quat, rgb0, rgb1, rgb2,
                    have_bg, do_dens, v_rgb_local, v_mean_local, v_scale_local, v_quat_local,
                    v_opacity_local, densification_weight_local, densification_error_weighted_local,
                    edge_weight_map, edge_weight_fp16, edge_score_local, blend_peak, hit0, hit1);

                const unsigned valid_mask = __ballot_sync(0xffffffffu, hit0 || hit1);
                if (valid_mask == 0u) {
                    continue;
                }
                const unsigned n_contrib = __popc(valid_mask);
                const int src_lane = __ffs(valid_mask) - 1;
                v_rgb_local[0] = reduce_field(v_rgb_local[0], n_contrib, src_lane);
                v_rgb_local[1] = reduce_field(v_rgb_local[1], n_contrib, src_lane);
                v_rgb_local[2] = reduce_field(v_rgb_local[2], n_contrib, src_lane);
                v_mean_local.x = reduce_field(v_mean_local.x, n_contrib, src_lane);
                v_mean_local.y = reduce_field(v_mean_local.y, n_contrib, src_lane);
                v_mean_local.z = reduce_field(v_mean_local.z, n_contrib, src_lane);
                v_scale_local.x = reduce_field(v_scale_local.x, n_contrib, src_lane);
                v_scale_local.y = reduce_field(v_scale_local.y, n_contrib, src_lane);
                v_scale_local.z = reduce_field(v_scale_local.z, n_contrib, src_lane);
                v_quat_local.x = reduce_field(v_quat_local.x, n_contrib, src_lane);
                v_quat_local.y = reduce_field(v_quat_local.y, n_contrib, src_lane);
                v_quat_local.z = reduce_field(v_quat_local.z, n_contrib, src_lane);
                v_quat_local.w = reduce_field(v_quat_local.w, n_contrib, src_lane);
                v_opacity_local = reduce_field(v_opacity_local, n_contrib, src_lane);
                densification_weight_local = reduce_field(densification_weight_local, n_contrib, src_lane);
                densification_error_weighted_local =
                    reduce_field(densification_error_weighted_local, n_contrib, src_lane);
                if constexpr (kEdgeScoring) {
                    edge_score_local = reduce_field(edge_score_local, n_contrib, src_lane);
                }
                if (blend_peak_out)
                    warpMax(blend_peak, warp);
                if (warp.thread_rank() == 0) {
                    record_blend_peak(blend_peak_out, id_batch[t], blend_peak);
                    // The compensation scales the opacity gradient only and stays detached from the
                    // covariance: feeding this determinant ratio into the scale and rotation gradients
                    // creates long splats.
                    if (compensations != nullptr)
                        v_opacity_local *= compensations[id_batch[t]];
                    flush_splat_grads(
                        id_batch[t], N, do_dens, v_rgb_local, v_mean_local, v_scale_local,
                        v_quat_local, v_opacity_local, densification_weight_local,
                        densification_error_weighted_local,
                        log_scale_chain(scale_act, scales, dilations, id_batch[t]), opacities[id_batch[t]],
                        v_means, v_quats, v_scales, v_colors, v_opacities, densification_weight,
                        densification_error);
                    if constexpr (kEdgeScoring) {
                        gpuAtomicAdd(&edge_score_batch[t], edge_score_local);
                    }
                }
            }
            if constexpr (kEdgeScoring) {
                block.sync();
                for (uint32_t t = tr; t < static_cast<uint32_t>(batch_size); t += block.size()) {
                    gpuAtomicAdd(&edge_score_out[id_batch[t]], edge_score_batch[t]);
                }
                block.sync();
            }
        }
    }

    template <uint32_t CDIM>
    constexpr uint32_t bwd_splat_color_count() {
        if constexpr (CDIM == 1)
            return 0u;
        if constexpr (CDIM == 4)
            return 3u;
        if constexpr (CDIM == 8)
            return 6u;
        return CDIM;
    }

    template <uint32_t CDIM>
    __device__ __forceinline__ constexpr uint32_t bwd_splat_channel(const uint32_t s) {
        if constexpr (CDIM == 8)
            return s < 3u ? s : s + 1u;
        return s;
    }

    template <uint32_t CDIM>
    __device__ __forceinline__ float load_grad_plane(
        const RenderGradPlanes& gp,
        const float* __restrict__ v_render_colors,
        const uint32_t cid,
        const uint32_t k,
        const uint32_t pix,
        const uint32_t H,
        const uint32_t W) {
        if (!gp.enabled()) {
            return v_render_colors != nullptr
                       ? v_render_colors[chw_pix(k, pix, H, W)]
                       : 0.0f;
        }
        const size_t pixels = static_cast<size_t>(H) * W;
        if constexpr (CDIM == 1u) {
            return gp.v_depth ? gp.v_depth[cid * pixels + pix] : 0.0f;
        }
        if (k < 3u) {
            return gp.v_rgb ? gp.v_rgb[(cid * 3u + k) * pixels + pix] : 0.0f;
        }
        if constexpr (CDIM == 4u) {
            return gp.v_depth ? gp.v_depth[cid * pixels + pix] : 0.0f;
        }
        if constexpr (CDIM == 8u) {
            if (k == 3u)
                return gp.v_depth ? gp.v_depth[cid * pixels + pix] : 0.0f;
            if (k < 7u)
                return gp.v_normal ? gp.v_normal[(cid * 3u + (k - 4u)) * pixels + pix] : 0.0f;
            return gp.v_depth_sq ? gp.v_depth_sq[cid * pixels + pix] : 0.0f;
        }
        return 0.0f;
    }

    template <uint32_t CDIM, typename scalar_t, bool kPerfectPinhole, bool kEdgeScoring, bool kBatched>
    __global__ void rasterize_to_pixels_from_world_3dgs_bwd_kernel(
        const uint32_t C,
        const uint32_t N,
        const uint32_t n_isects,
        const bool packed,
        // fwd inputs
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
        // fwd outputs
        const scalar_t* __restrict__ render_alphas, // [C, image_height, image_width, 1]
        const int32_t* __restrict__ last_ids,       // [C, image_height, image_width]
        // grad outputs
        const scalar_t* __restrict__ v_render_colors, // [C, CDIM, image_height, image_width]
        const scalar_t* __restrict__ v_render_alphas, // [C, image_height, image_width, 1]
        // grad inputs
        vec3* __restrict__ v_means,                           // [N, 3]
        vec4* __restrict__ v_quats,                           // [N, 4]
        vec3* __restrict__ v_scales,                          // [N, 3]
        scalar_t* __restrict__ v_colors,                      // [C, N, CDIM] or [nnz, CDIM]
        scalar_t* __restrict__ v_opacities,                   // [C, N] or [nnz]
        float* __restrict__ densification_weight,             // [N] or nullptr
        float* __restrict__ densification_error,              // [N] or nullptr
        const scalar_t* __restrict__ densification_error_map, // [H, W] or nullptr
        const void* __restrict__ edge_weight_map,
        float* __restrict__ edge_score_out, const TileRange tiles,
        const ColorVarianceRender color_variance, float* __restrict__ blend_peak_out,
        const RenderGradPlanes grad_planes, const bool edge_weight_fp16) {
        auto block = cg::this_thread_block();
        uint32_t cid = block.group_index().x;
        uint32_t tile_id =
            block.group_index().y * tile_width + block.group_index().z;
        if constexpr (kBatched) {
            if (tile_id < tiles.begin || tile_id >= tiles.end)
                return;
        }
        uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
        uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

        tile_offsets += cid * tile_height * tile_width;
        render_alphas += cid * image_height * image_width;
        last_ids += cid * image_height * image_width;
        if (v_render_colors != nullptr) {
            v_render_colors += cid * image_height * image_width * CDIM;
        }
        v_render_alphas += cid * image_height * image_width;
        if (backgrounds != nullptr) {
            backgrounds += cid * CDIM;
        }
        if (bg_images != nullptr) {
            bg_images += cid * 3u * image_height * image_width;
        }
        if (masks != nullptr) {
            masks += cid * tile_height * tile_width;
        }
        if constexpr (kEdgeScoring) {
            edge_weight_map = offset_edge_weight_map(
                edge_weight_map, static_cast<size_t>(cid) * image_height * image_width, edge_weight_fp16);
            edge_score_out += cid * N;
        }

        // when the mask is provided, do nothing and return if
        // this tile is labeled as False
        if (masks != nullptr && !masks[tile_id]) {
            return;
        }

        const float px = (float)j + 0.5f;
        const float py = (float)i + 0.5f;
        // clamp this value to the last pixel
        const int32_t pix_id =
            min(i * image_width + j, image_width * image_height - 1);

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
        constexpr uint32_t kUsed = CDIM;

        // keep not rasterizing threads around for reading data
        const bool active = (i < image_height && j < image_width) && ray.valid_flag;

        // have all threads in tile process the same gaussians in batches
        // first collect gaussians between range.x and range.y in batches
        // which gaussians to look through in this tile
        int32_t range_start = tile_offsets[tile_id];
        int32_t range_end = tile_offsets[tile_id + 1];
        const uint32_t block_size = block.size();
        const uint32_t num_batches =
            (range_end - range_start + block_size - 1) / block_size;

        extern __shared__ int s[];
        int32_t* id_batch = (int32_t*)s; // [block_size]
        vec4* xyz_opacity_batch =
            reinterpret_cast<vec4*>(&id_batch[block_size]); // [block_size]
        vec3* scale_batch =
            reinterpret_cast<vec3*>(&xyz_opacity_batch[block_size]); // [block_size]
        vec4* quat_batch =
            reinterpret_cast<vec4*>(&scale_batch[block_size]); // [block_size]
        mat3* mt_batch =
            reinterpret_cast<mat3*>(&quat_batch[block_size]); // [block_size]
        float* rgbs_batch =
            reinterpret_cast<float*>(&mt_batch[block_size]); // [block_size * CDIM]
        float* edge_score_batch = nullptr;
        if constexpr (kEdgeScoring) {
            edge_score_batch = reinterpret_cast<float*>(&rgbs_batch[block_size * CDIM]);
        }

        // this is the T AFTER the last gaussian in this pixel
        float T_final = 1.0f - render_alphas[pix_id];
        float T = T_final;
        // the contribution from gaussians behind the current one
        float buffer[kUsed] = {0.f};
        // index of last gaussian to contribute to this pixel
        const int32_t bin_final = active ? last_ids[pix_id] : 0;

        // df/d_out for this pixel
        float v_render_c[kUsed];
#pragma unroll
        for (uint32_t k = 0; k < kUsed; ++k) {
            v_render_c[k] = load_grad_plane<CDIM>(
                grad_planes, v_render_colors, cid, k, pix_id, image_height, image_width);
        }
        const float v_render_a = v_render_alphas[pix_id];

        const bool have_bg = bg_images != nullptr || backgrounds != nullptr;
        float bg_accum = 0.f;
        if (have_bg) {
#pragma unroll
            for (uint32_t k = 0; k < kUsed; ++k) {
                float bg_val;
                if (bg_images != nullptr && k < 3u) {
                    bg_val = bg_images[k * image_height * image_width + pix_id];
                } else if (backgrounds != nullptr && k < 3u) {
                    bg_val = backgrounds[k];
                } else {
                    bg_val = 0.0f;
                }
                bg_accum += bg_val * v_render_c[k];
            }
        }
        const bool do_dens = densification_error != nullptr && densification_error_map != nullptr;
        const float pixel_error = do_dens ? densification_error_map[pix_id] : 0.f;

        const bool do_cv = (CDIM >= 3) && color_variance.enabled();
        float4 cv_target = make_float4(0.f, 0.f, 0.f, 0.f);
        float cv_buf = 0.f;
        if constexpr (CDIM >= 3) {
            if (do_cv && active) {
                const float cov = 1.0f - T_final;
                if (cov >= 0.5f / 255.f) {
                    const uint32_t pixels = image_height * image_width;
                    float pw_scale = 0.f;
                    if (color_variance.pixel_weight == nullptr) {
                        pw_scale = color_variance.weight / static_cast<float>(pixels);
                    } else {
                        const float pw_sum = *color_variance.pixel_weight_sum;
                        if (pw_sum > 0.f) {
                            pw_scale = color_variance.weight *
                                       color_variance.pixel_weight[static_cast<size_t>(cid) * pixels + pix_id] / pw_sum;
                        }
                    }
                    if (pw_scale > 0.f) {
                        const float* rc = color_variance.render_colors + static_cast<size_t>(cid) * CDIM * pixels;
                        float bg_c[3] = {0.f, 0.f, 0.f};
                        if (have_bg) {
#pragma unroll
                            for (uint32_t c = 0; c < 3u; ++c) {
                                bg_c[c] = bg_images != nullptr ? bg_images[c * pixels + pix_id] : backgrounds[c];
                            }
                        }
                        cv_target = make_float4(
                            (rc[chw_pix(0, pix_id, image_height, image_width)] - T_final * bg_c[0]) / cov,
                            (rc[chw_pix(1, pix_id, image_height, image_width)] - T_final * bg_c[1]) / cov,
                            (rc[chw_pix(2, pix_id, image_height, image_width)] - T_final * bg_c[2]) / cov,
                            pw_scale);
                    }
                }
            }
        }

        // collect and process batches of gaussians
        // each thread loads one gaussian at a time before rasterizing
        const uint32_t tr = block.thread_rank();
        cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
        const int32_t warp_bin_final =
            cg::reduce(warp, bin_final, cg::greater<int>());
        for (uint32_t b = 0; b < num_batches; ++b) {
            // resync all threads before writing next batch of shared mem
            block.sync();
            if constexpr (kEdgeScoring) {
                for (uint32_t slot = tr; slot < block_size; slot += block.size()) {
                    edge_score_batch[slot] = 0.f;
                }
                block.sync();
            }

            // each thread fetch 1 gaussian from back to front
            // 0 index will be furthest back in batch
            // index of gaussian to load
            // batch end is the index of the last gaussian in the batch
            // These values can be negative so must be int32 instead of uint32
            const int32_t batch_end = range_end - 1 - block_size * b;
            const int32_t batch_size = min(block_size, batch_end + 1 - range_start);
            const int32_t idx = batch_end - tr;
            if (idx >= range_start) {
                int32_t g = flatten_ids[idx]; // flatten index in [C * N] or [nnz]
                id_batch[tr] = g;
                const vec3 xyz = means[g];
                float opac = activated_opacity(opacities[g]);
                if (compensations != nullptr)
                    opac *= compensations[g];
                xyz_opacity_batch[tr] = {xyz.x, xyz.y, xyz.z, opac};
                const vec3 scale_act = activated_scale(scales[g]);
                const vec4 quat = quats[g];
                scale_batch[tr] = scale_act;
                quat_batch[tr] = quat;
                const mat3 R = quat_to_rotmat(quat);
                const mat3 S = mat3(
                    1.0f / scale_act[0], 0.f, 0.f,
                    0.f, 1.0f / scale_act[1], 0.f,
                    0.f, 0.f, 1.0f / scale_act[2]);
                mt_batch[tr] = glm::transpose(R * S);
                if (dilations != nullptr) { // mip filter, as above
                    scale_batch[tr] = dilated_scale(scale_act, dilations[g]);
                    mt_batch[tr] = glm::transpose(R * inverse_scale_matrix(scale_batch[tr]));
                }
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    rgbs_batch[tr * CDIM + k] = colors[g * CDIM + k];
                }
            }
            // wait for other threads to collect the gaussians in batch
            block.sync();
            // process gaussians in the current batch for this pixel
            // 0 index is the furthest back gaussian in the batch
            for (uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size;
                 ++t) {
                bool valid = active;
                if (batch_end - static_cast<int32_t>(t) > bin_final) {
                    valid = false;
                }
                float alpha = 0.f;
                float opac = 0.f;
                float vis = 0.f;
                float inv_grd_len = 0.f;
                float t_hit = 0.f;
                float depth = 0.f;
                vec3 xyz, o_minus_mu, gro, grd, grd_n, gcrod;
                mat3 Mt;
                if (valid) {
                    const vec4 xyz_opac = xyz_opacity_batch[t];
                    opac = xyz_opac[3];
                    xyz = {xyz_opac[0], xyz_opac[1], xyz_opac[2]};
                    Mt = mt_batch[t];
                    o_minus_mu = ray_o - xyz;
                    gro = Mt * o_minus_mu;
                    grd = Mt * ray_d;
                    float power;
                    if constexpr (kHasRayDepth) {
                        ray_gaussian_eval(gro, grd, depth_scale,
                                          grd_n, gcrod, inv_grd_len, t_hit, depth, power);
                    } else {
                        grd_n = safe_normalize(grd);
                        gcrod = glm::cross(grd_n, gro);
                        const float grayDist = glm::dot(gcrod, gcrod);
                        power = -0.5f * grayDist;
                    }
                    vis = __expf(power);
                    alpha = min(0.999f, opac * vis);
                    if (power > 0.f || alpha < 1.f / 255.f) {
                        valid = false;
                    }
                }

                const unsigned valid_mask = __ballot_sync(0xffffffffu, valid);
                if (valid_mask == 0u) {
                    if constexpr (!kEdgeScoring) {
                        continue;
                    }
                }
                float v_rgb_local[CDIM] = {0.f};
                vec3 v_mean_local = {0.f, 0.f, 0.f};
                vec3 v_scale_local = {0.f, 0.f, 0.f};
                vec4 v_quat_local = {0.f, 0.f, 0.f, 0.f};
                float v_opacity_local = 0.f;
                float densification_weight_local = 0.f;
                float densification_error_weighted_local = 0.f;
                float edge_score_local = 0.f;
                float blend_peak = 0.f;
                if (valid_mask != 0u) {
                    const float ra = 1.0f / (1.0f - alpha);
                    if (valid) {
                        T *= ra;
                        const float fac = alpha * T;
                        blend_peak = fac;
                        float v_depth = 0.f;
                        float v_alpha = 0.f;
#pragma unroll
                        for (uint32_t k = 0; k < kUsed; ++k) {
                            const float v_c = v_render_c[k];
                            float c_val;
                            if constexpr (kHasRayDepth) {
                                if (k == kDepthChannel) {
                                    c_val = depth;
                                    v_depth += fac * v_c;
                                } else if constexpr (kHasDepthSq) {
                                    if (k == kDepthSqChannel) {
                                        c_val = depth * depth;
                                        v_depth += fac * (2.0f * depth) * v_c;
                                    } else {
                                        c_val = rgbs_batch[t * CDIM + k];
                                        v_rgb_local[k] = fac * v_c;
                                    }
                                } else {
                                    c_val = rgbs_batch[t * CDIM + k];
                                    v_rgb_local[k] = fac * v_c;
                                }
                            } else {
                                c_val = rgbs_batch[t * CDIM + k];
                                v_rgb_local[k] = fac * v_c;
                            }
                            v_alpha += (c_val * T - buffer[k] * ra) * v_c;
                            buffer[k] += c_val * fac;
                        }
                        if (do_dens) {
                            densification_weight_local = fac;
                            densification_error_weighted_local = fac * pixel_error;
                        }
                        if constexpr (kEdgeScoring) {
                            const float edge_weight = load_edge_weight(edge_weight_map, pix_id, edge_weight_fp16);
                            const float edge_contribution = fac * edge_weight;
                            if (edge_weight > 0.0f && isfinite(edge_contribution)) {
                                edge_score_local = edge_contribution;
                            }
                        }
                        v_alpha += T_final * ra * v_render_a;
                        if (have_bg) {
                            v_alpha += -T_final * ra * bg_accum;
                        }
                        if constexpr (CDIM >= 3) {
                            if (cv_target.w > 0.f) {
                                const float r0 = rgbs_batch[t * CDIM + 0] - cv_target.x;
                                const float r1 = rgbs_batch[t * CDIM + 1] - cv_target.y;
                                const float r2 = rgbs_batch[t * CDIM + 2] - cv_target.z;
                                const float u = cv_target.w * (r0 * r0 + r1 * r1 + r2 * r2);
                                const float two_w = 2.0f * cv_target.w * fac;
                                v_rgb_local[0] += two_w * r0;
                                v_rgb_local[1] += two_w * r1;
                                v_rgb_local[2] += two_w * r2;
                                v_alpha += u * T - cv_buf * ra;
                                cv_buf += u * fac;
                            }
                        }

                        const bool alpha_grad = (opac * vis <= 0.999f);
                        const bool depth_grad =
                            kHasRayDepth && (v_depth != 0.f && t_hit > 0.f && inv_grd_len > 0.f);
                        if (alpha_grad || depth_grad) {
                            vec3 v_gro{0.f};
                            vec3 v_grd{0.f};
                            if (alpha_grad) {
                                const float v_vis = opac * v_alpha;
                                const float v_gradDist = -0.5f * vis * v_vis;
                                const vec3 v_gcrod = 2.0f * v_gradDist * gcrod;
                                const vec3 v_grd_n = -glm::cross(v_gcrod, gro);
                                v_gro = glm::cross(v_gcrod, grd_n);
                                v_grd = safe_normalize_bw(grd, v_grd_n);
                                v_opacity_local = vis * v_alpha;
                            }
                            if constexpr (kHasRayDepth) {
                                if (depth_grad) {
                                    ray_gaussian_depth_vjp(gro, grd_n, inv_grd_len, t_hit,
                                                           depth_scale, v_depth, v_gro, v_grd);
                                }
                            }
                            const mat3 v_Mt = glm::outerProduct(v_grd, ray_d) +
                                              glm::outerProduct(v_gro, o_minus_mu);
                            const vec3 v_o_minus_mu = glm::transpose(Mt) * v_gro;
                            v_mean_local += -v_o_minus_mu;
                            const vec3 scale_act = scale_batch[t];
                            const vec4 quat = quat_batch[t];
                            const mat3 R = quat_to_rotmat(quat);
                            quat_scale_to_preci_half_vjp(
                                quat, scale_act, R, glm::transpose(v_Mt), v_quat_local, v_scale_local);
                        }
                    }
                    warpSum<CDIM>(v_rgb_local, warp);
                    warpSum(v_mean_local, warp);
                    warpSum(v_scale_local, warp);
                    warpSum(v_quat_local, warp);
                    warpSum(v_opacity_local, warp);
                    warpSum(densification_weight_local, warp);
                    warpSum(densification_error_weighted_local, warp);
                    if (blend_peak_out)
                        warpMax(blend_peak, warp);
                    if (warp.thread_rank() == 0) {
                        const int32_t g = id_batch[t];
                        record_blend_peak(blend_peak_out, g, blend_peak);
                        float* v_rgb_ptr = (float*)(v_colors) + CDIM * g;
#pragma unroll
                        for (uint32_t k = 0; k < CDIM; ++k) {
                            if constexpr (kHasRayDepth) {
                                if (k == kDepthChannel || (kHasDepthSq && k == kDepthSqChannel))
                                    continue;
                            }
                            gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
                        }
                        float* v_mean_ptr = (float*)(v_means) + 3 * g;
                        gpuAtomicAdd(v_mean_ptr, v_mean_local.x);
                        gpuAtomicAdd(v_mean_ptr + 1, v_mean_local.y);
                        gpuAtomicAdd(v_mean_ptr + 2, v_mean_local.z);

                        float* v_scale_ptr = (float*)(v_scales) + 3 * g;
                        const vec3 scale_act = log_scale_chain(scale_batch[t], scales, dilations, g);
                        gpuAtomicAdd(v_scale_ptr, v_scale_local.x * scale_act.x);
                        gpuAtomicAdd(v_scale_ptr + 1, v_scale_local.y * scale_act.y);
                        gpuAtomicAdd(v_scale_ptr + 2, v_scale_local.z * scale_act.z);

                        float* v_quat_ptr = (float*)(v_quats) + 4 * g;
                        gpuAtomicAdd(v_quat_ptr, v_quat_local.x);
                        gpuAtomicAdd(v_quat_ptr + 1, v_quat_local.y);
                        gpuAtomicAdd(v_quat_ptr + 2, v_quat_local.z);
                        gpuAtomicAdd(v_quat_ptr + 3, v_quat_local.w);

                        // Detached from the covariance, as above.
                        if (compensations != nullptr)
                            v_opacity_local *= compensations[g];
                        const float derivative = lfs::core::sigmoid_derivative_from_logit(opacities[g]);
                        gpuAtomicAdd(v_opacities + g, v_opacity_local * derivative);
                        if (do_dens) {
                            if (densification_weight != nullptr)
                                gpuAtomicAdd(densification_weight + g, densification_weight_local);
                            gpuAtomicAdd(densification_error + g, densification_error_weighted_local);
                        }
                    }
                }
                if constexpr (kEdgeScoring) {
                    warpSum(edge_score_local, warp);
                    if (warp.thread_rank() == 0) {
                        gpuAtomicAdd(&edge_score_batch[t], edge_score_local);
                    }
                }
            }
            if constexpr (kEdgeScoring) {
                block.sync();
                for (uint32_t t = tr; t < static_cast<uint32_t>(batch_size); t += block.size()) {
                    gpuAtomicAdd(edge_score_out + id_batch[t], edge_score_batch[t]);
                }
                block.sync();
            }
        }
        if constexpr (CDIM >= 3) {
            if (do_cv && color_variance.loss_out != nullptr) {
                warpSum(cv_buf, warp);
                if (warp.thread_rank() == 0 && cv_buf != 0.f) {
                    gpuAtomicAdd(color_variance.loss_out, cv_buf);
                }
            }
        }
    }

    // Per-splat backward. One warp takes an 8x8 sub-tile of a 16x16 tile. Each lane
    // owns one splat and walks the sub-tile's pixels on a diagonal: at step t lane l
    // handles pixel t - l, so every pixel meets the splats back to front, one lane
    // per step. Pixel state lives in shared memory and the splat's gradient in
    // registers, flushed once per sub-tile instead of reduced across the warp for
    // every splat. Splats whose ellipse misses the sub-tile are dropped first.
    constexpr uint32_t kSplatBwdSub = kCullSubTile;
    constexpr uint32_t kSplatBwdPixels = kSplatBwdSub * kSplatBwdSub;
    constexpr int kSplatBwdSurvivors = 96;

    template <uint32_t CDIM, bool kPerfectPinhole, bool kEdgeScoring, bool kBatched,
              bool kColorVariance = false, bool kCachedRays = false, bool kGlobalShutter = false>
    __global__ void __launch_bounds__(32, 20) rasterize_to_pixels_from_world_3dgs_bwd_splat_kernel(
        const uint32_t C,
        const uint32_t N,
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
        const float* __restrict__ render_alphas,
        const int32_t* __restrict__ last_ids,
        const float* __restrict__ v_render_colors,
        const float* __restrict__ v_render_alphas,
        vec3* __restrict__ v_means,
        vec4* __restrict__ v_quats,
        vec3* __restrict__ v_scales,
        float* __restrict__ v_colors,
        float* __restrict__ v_opacities,
        float* __restrict__ densification_weight,
        float* __restrict__ densification_error,
        const float* __restrict__ densification_error_map,
        const void* __restrict__ edge_weight_map,
        float* __restrict__ edge_score_out,
        const TileRange tiles,
        const ScreenFootprint footprint,
        const ColorVarianceRender color_variance,
        float* __restrict__ blend_peak_out,
        const RenderGradPlanes grad_planes,
        const bool edge_weight_fp16) {
        const uint32_t subs = tile_size / kSplatBwdSub;
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
        if (footprint.tile_sub_masks != nullptr && !((footprint.tile_sub_masks[tile_id] >> sub_bit) & 1u))
            return;

        const uint32_t lane = threadIdx.x;
        const uint32_t pixels = image_width * image_height;

        tile_offsets += cid * tile_height * tile_width;
        render_alphas += cid * pixels;
        last_ids += cid * pixels;
        if (v_render_colors != nullptr) {
            v_render_colors += cid * pixels * CDIM;
        }
        v_render_alphas += cid * pixels;
        if (backgrounds != nullptr) {
            backgrounds += cid * CDIM;
        }
        if (bg_images != nullptr) {
            bg_images += cid * 3u * pixels;
        }
        if (masks != nullptr) {
            masks += cid * tile_height * tile_width;
            if (!masks[tile_id])
                return;
        }
        if constexpr (kEdgeScoring) {
            edge_weight_map = offset_edge_weight_map(
                edge_weight_map, static_cast<size_t>(cid) * pixels, edge_weight_fp16);
            edge_score_out += cid * N;
        }
        const bool have_bg = bg_images != nullptr || backgrounds != nullptr;
        const bool do_dens = densification_error != nullptr && densification_error_map != nullptr;
        constexpr bool kUseCv = kColorVariance && (CDIM >= 3);
        constexpr uint32_t kUsed = CDIM;
        constexpr uint32_t kSplatColors = bwd_splat_color_count<CDIM>();
        constexpr uint32_t kRegColors = kSplatColors > 0u ? kSplatColors : 1u;
        constexpr bool kHasRayDepth = (CDIM == 1 || CDIM == 4 || CDIM == 8);
        constexpr uint32_t kDepthChannel = (CDIM == 1) ? 0u : 3u;
        constexpr bool kHasDepthSq = (CDIM == 8);
        constexpr uint32_t kDepthSqChannel = 7u;

        __shared__ float s_T[kSplatBwdPixels];
        // Each pixel uses the same color gradient for every splat. Store the
        // dot product with its accumulated color instead of all channels.
        __shared__ float s_buffer_dot[kSplatBwdPixels];
        __shared__ float s_v_c[kUsed][kSplatBwdPixels];
        __shared__ float s_tail[kSplatBwdPixels];
        __shared__ float s_error[kSplatBwdPixels];
        __shared__ float s_edge[kEdgeScoring ? kSplatBwdPixels : 1];
        __shared__ float4 s_cv_target[kUseCv ? kSplatBwdPixels : 1];
        __shared__ float s_cv_buf[kUseCv ? kSplatBwdPixels : 1];
        __shared__ float3 s_ray_o[kSplatBwdSub];
        __shared__ float s_ray_dx[kSplatBwdPixels];
        __shared__ float s_ray_dy[kSplatBwdPixels];
        __shared__ float s_ray_dz[kSplatBwdPixels];
        __shared__ float s_depth_scale[kCachedRays ? 1 : kSplatBwdPixels];
        __shared__ int32_t s_bin[kSplatBwdPixels]; // last contributing pair; -1 inactive
        union SplatBwdScratch {
            int32_t surv[kSplatBwdSurvivors];
            float R_inv[kSplatBwdSub][9];
        };
        __shared__ SplatBwdScratch s_scratch;

        float cv_pw_sum = 0.f;
        if constexpr (kUseCv) {
            if (color_variance.pixel_weight != nullptr) {
                if (lane == 0) {
                    cv_pw_sum = *color_variance.pixel_weight_sum;
                }
                cv_pw_sum = __shfl_sync(0xffffffffu, cv_pw_sum, 0);
            }
        }

        const int32_t range_start = tile_offsets[tile_id];
        const int32_t range_end = tile_offsets[tile_id + 1];
        if (range_start >= range_end)
            return;

        const bool row_shutter = (rs_type != ShutterType::ROLLING_LEFT_TO_RIGHT &&
                                  rs_type != ShutterType::ROLLING_RIGHT_TO_LEFT);
        if constexpr (kGlobalShutter) {
            if (lane == 0) {
                const RollingShutterParameters rs_params(viewmats0 + cid * 16, nullptr);
                const ShutterPose pose = interpolate_shutter_pose(0.f, rs_params);
                const mat3 R_inv = glm::mat3_cast(glm::inverse(pose.q));
                const vec3 org = -(R_inv * pose.t);
                s_ray_o[0] = make_float3(org.x, org.y, org.z);
                if constexpr (kCachedRays) {
#pragma unroll
                    for (int c = 0; c < 3; ++c) {
#pragma unroll
                        for (int r = 0; r < 3; ++r)
                            s_scratch.R_inv[0][c * 3 + r] = R_inv[c][r];
                    }
                }
            }
        } else if (lane < kSplatBwdSub) {
            const uint32_t coord = row_shutter ? (sub_y * kSplatBwdSub + lane) : (sub_x * kSplatBwdSub + lane);
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
            s_ray_o[lane] = make_float3(org.x, org.y, org.z);
            if constexpr (kCachedRays) {
#pragma unroll
                for (int c = 0; c < 3; ++c) {
#pragma unroll
                    for (int r = 0; r < 3; ++r)
                        s_scratch.R_inv[lane][c * 3 + r] = R_inv[c][r];
                }
            }
        }
        __syncwarp();

        int32_t bin_max = -1;
        for (uint32_t p = lane; p < kSplatBwdPixels; p += 32) {
            const uint32_t x = sub_x * kSplatBwdSub + p % kSplatBwdSub;
            const uint32_t y = sub_y * kSplatBwdSub + p / kSplatBwdSub;
            s_bin[p] = -1;
            if constexpr (kUseCv)
                s_cv_buf[p] = 0.f;
            if (x >= image_width || y >= image_height)
                continue;

            const uint32_t pix = y * image_width + x;
            const float render_alpha = render_alphas[pix];
            const int32_t last_id = last_ids[pix];
            if (render_alpha == 0.f || last_id < range_start)
                continue;

            const float v_a = v_render_alphas[pix];
            float v_color[kUsed];
            bool has_gradient = v_a != 0.f;
#pragma unroll
            for (uint32_t k = 0; k < kUsed; ++k) {
                v_color[k] = load_grad_plane<CDIM>(
                    grad_planes, v_render_colors, cid, k, pix, image_height, image_width);
                has_gradient |= v_color[k] != 0.f;
            }
            // These auxiliary paths can contribute even when the image loss has no gradient.
            if constexpr (!kUseCv && !kEdgeScoring) {
                if (!do_dens && !has_gradient && !blend_peak_out)
                    continue;
            }

            if constexpr (kCachedRays) {
                const float* const ur =
                    footprint.unit_rays + (static_cast<size_t>(cid) * pixels + pix) * 3u;
                const float cx = ur[0];
                const float cy = ur[1];
                const float cz = ur[2];
                if (!isfinite(cx) || !isfinite(cy) || !isfinite(cz))
                    continue;
                const uint32_t u = kGlobalShutter ? 0u : (row_shutter ? (p / kSplatBwdSub) : (p % kSplatBwdSub));
                const float* const Rm = s_scratch.R_inv[u];
                s_ray_dx[p] = Rm[0] * cx + Rm[3] * cy + Rm[6] * cz;
                s_ray_dy[p] = Rm[1] * cx + Rm[4] * cy + Rm[7] * cz;
                s_ray_dz[p] = Rm[2] * cx + Rm[5] * cy + Rm[8] * cz;
            } else {
                const WorldRay ray = from_world_pixel_ray<kPerfectPinhole>(
                    camera_model_type, rs_type, image_width, image_height,
                    static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f,
                    viewmats0, viewmats1, Ks, cid, radial_coeffs, tangential_coeffs, thin_prism_coeffs);
                if (!ray.valid_flag)
                    continue;
                s_ray_dx[p] = ray.ray_dir.x;
                s_ray_dy[p] = ray.ray_dir.y;
                s_ray_dz[p] = ray.ray_dir.z;
                s_depth_scale[p] = (camera_model_type == CameraModelType::PINHOLE) ? ray.cam_ray_z : 1.0f;
            }

            s_bin[p] = last_id;
            bin_max = max(bin_max, last_id);
            const float T_final = 1.0f - render_alpha;
            s_T[p] = T_final;
            s_buffer_dot[p] = 0.f;
            float bg = 0.f;
#pragma unroll
            for (uint32_t k = 0; k < kUsed; ++k) {
                const float v_c = v_color[k];
                s_v_c[k][p] = v_c;
                // Geometry channels have a zero background; only RGB contributes.
                if (have_bg && k < 3u) {
                    bg += (bg_images != nullptr ? bg_images[k * pixels + pix] : backgrounds[k]) * v_c;
                }
            }
            s_tail[p] = T_final * (v_a - (have_bg ? bg : 0.f));
            s_error[p] = do_dens ? densification_error_map[pix] : 0.f;
            if constexpr (kEdgeScoring) {
                s_edge[p] = load_edge_weight(edge_weight_map, pix, edge_weight_fp16);
            }
            if constexpr (kUseCv) {
                float4 cv_t = make_float4(0.f, 0.f, 0.f, 0.f);
                if (color_variance.enabled()) {
                    const float cov = 1.0f - T_final;
                    if (cov >= 0.5f / 255.f) {
                        float pw_scale = 0.f;
                        if (color_variance.pixel_weight == nullptr) {
                            pw_scale = color_variance.weight / static_cast<float>(pixels);
                        } else if (cv_pw_sum > 0.f) {
                            pw_scale = color_variance.weight *
                                       color_variance.pixel_weight[static_cast<size_t>(cid) * pixels + pix] / cv_pw_sum;
                        }
                        if (pw_scale > 0.f) {
                            const float* rc = color_variance.render_colors + static_cast<size_t>(cid) * CDIM * pixels;
                            float bg_c[3] = {0.f, 0.f, 0.f};
                            if (have_bg) {
#pragma unroll
                                for (uint32_t c = 0; c < 3u; ++c) {
                                    bg_c[c] = bg_images != nullptr ? bg_images[c * pixels + pix] : backgrounds[c];
                                }
                            }
                            cv_t = make_float4(
                                (rc[chw_pix(0, pix, image_height, image_width)] - T_final * bg_c[0]) / cov,
                                (rc[chw_pix(1, pix, image_height, image_width)] - T_final * bg_c[1]) / cov,
                                (rc[chw_pix(2, pix, image_height, image_width)] - T_final * bg_c[2]) / cov,
                                pw_scale);
                        }
                    }
                }
                s_cv_target[p] = cv_t;
            }
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            bin_max = max(bin_max, __shfl_xor_sync(0xffffffffu, bin_max, offset));
        }
        __syncwarp();
        if (bin_max < range_start)
            return;

        const float sub_x0 = static_cast<float>(sub_x * kSplatBwdSub);
        const float sub_y0 = static_cast<float>(sub_y * kSplatBwdSub);
        const bool cull = footprint.means2d != nullptr && footprint.conics != nullptr &&
                          footprint.radii != nullptr;

        // Nothing past the deepest contributing pair of the sub-tile matters.
        for (int32_t scan_end = min(range_end - 1, bin_max); scan_end >= range_start;) {
            // Compact back-to-front survivors; the loop bounds are warp-uniform.
            int count = 0;
            int32_t s = scan_end;
            while (s >= range_start && count <= kSplatBwdSurvivors - 32) {
                const int32_t gi = s - static_cast<int32_t>(lane);
                bool keep = false;
                if (gi >= range_start) {
                    if (footprint.tile_keys32 != nullptr) {
                        keep = (footprint.tile_keys32[gi] >> sub_bit) & 1u;
                    } else if (footprint.tile_keys != nullptr) {
                        keep = (static_cast<uint64_t>(footprint.tile_keys[gi]) >> sub_bit) & 1u;
                    } else {
                        keep = !cull || footprint_overlaps_square(
                                            footprint.means2d, footprint.conics, footprint.radii,
                                            flatten_ids[gi], sub_x0, sub_y0, kSplatBwdSub);
                    }
                }
                const uint32_t ballot = __ballot_sync(0xffffffffu, keep);
                if (keep) {
                    s_scratch.surv[count + __popc(ballot & ((1u << lane) - 1u))] = gi;
                }
                count += __popc(ballot);
                s -= 32;
            }
            scan_end = s;
            __syncwarp();

            for (int base = 0; base < count; base += 32) {
                const int batch = min(32, count - base);
                const bool owns = static_cast<int>(lane) < batch;
                // Lane 0 owns the back-most splat of the batch.
                const int32_t isect_idx = owns ? s_scratch.surv[base + lane] : 0;
                const int32_t g = owns ? flatten_ids[isect_idx] : 0;
                vec3 splat_pos_or_gro{0.f};
                float opac = 0.f;
                float max_grayDist = -1.f;
                mat3 Mt{1.f};
                float rgb[kRegColors];
#pragma unroll
                for (uint32_t sc = 0; sc < kRegColors; ++sc) {
                    rgb[sc] = 0.f;
                }
                if (owns) {
                    const vec3 xyz = means[g];
                    opac = activated_opacity(opacities[g]);
                    if (compensations != nullptr)
                        opac *= compensations[g];
                    if (opac >= (1.f / 255.f))
                        max_grayDist = 2.0f * __logf(255.0f * opac) + 1e-3f;
                    vec3 scale_init = activated_scale(scales[g]);
                    if (dilations != nullptr)
                        scale_init = dilated_scale(scale_init, dilations[g]);
                    const vec4 quat_init = quats[g];
                    const mat3 S = mat3(
                        1.0f / scale_init[0], 0.f, 0.f,
                        0.f, 1.0f / scale_init[1], 0.f,
                        0.f, 0.f, 1.0f / scale_init[2]);
                    Mt = glm::transpose(quat_to_rotmat(quat_init) * S);
                    if constexpr (kGlobalShutter) {
                        const float3 o3 = s_ray_o[0];
                        splat_pos_or_gro = Mt * (vec3{o3.x, o3.y, o3.z} - xyz);
                    } else {
                        splat_pos_or_gro = xyz;
                    }
#pragma unroll
                    for (uint32_t sc = 0; sc < kSplatColors; ++sc) {
                        rgb[sc] = colors[g * CDIM + bwd_splat_channel<CDIM>(sc)];
                    }
                }

                float v_rgb[kRegColors];
#pragma unroll
                for (uint32_t sc = 0; sc < kRegColors; ++sc) {
                    v_rgb[sc] = 0.f;
                }
                // The mean and quaternion/scale VJPs are linear in v_gro and v_Mt, so
                // the pixels' terms are summed here and pushed through them once.
                vec3 v_gro_sum{0.f};
                mat3 v_Mt_sum{0.f};
                float v_opacity = 0.f;
                float dens_w = 0.f;
                float dens_e = 0.f;
                float edge_score = 0.f;
                float blend_peak = 0.f;
                bool touched = false;

                for (int t = 0; t < batch + static_cast<int>(kSplatBwdPixels) - 1; ++t) {
                    const int p = t - static_cast<int>(lane);
                    if (owns && p >= 0 && p < static_cast<int>(kSplatBwdPixels) && isect_idx <= s_bin[p]) {
                        vec3 o_minus_mu{0.f};
                        vec3 gro;
                        if constexpr (kGlobalShutter) {
                            gro = splat_pos_or_gro;
                        } else {
                            const int u = static_cast<int>(row_shutter ? (static_cast<uint32_t>(p) >> 3)
                                                                       : (static_cast<uint32_t>(p) & 7u));
                            const float3 o3 = s_ray_o[u];
                            o_minus_mu = vec3{o3.x, o3.y, o3.z} - splat_pos_or_gro;
                            gro = Mt * o_minus_mu;
                        }
                        const vec3 ray_d{s_ray_dx[p], s_ray_dy[p], s_ray_dz[p]};
                        float depth_scale = 1.0f;
                        if constexpr (!kCachedRays) {
                            if (camera_model_type == CameraModelType::PINHOLE)
                                depth_scale = s_depth_scale[p];
                        }
                        const vec3 grd = Mt * ray_d;
                        vec3 grd_n;
                        vec3 gcrod;
                        float inv_grd_len = 0.f;
                        float grayDist = 0.f;
                        ray_gaussian_power(gro, grd, grd_n, gcrod, inv_grd_len, grayDist);
                        if (grayDist <= max_grayDist) {
                            const float vis = __expf(-0.5f * grayDist);
                            const float alpha = min(0.999f, opac * vis);
                            if (alpha >= 1.f / 255.f) {
                                touched = true;
                                float t_hit = 0.f;
                                float depth = 0.f;
                                if constexpr (kHasRayDepth) {
                                    t_hit = inv_grd_len > 0.f ? (-glm::dot(grd_n, gro) * inv_grd_len) : 0.f;
                                    depth = t_hit > 0.f ? t_hit * depth_scale : 0.f;
                                }
                                const float ra = 1.0f / (1.0f - alpha);
                                const float T = s_T[p] * ra;
                                const float fac = alpha * T;
                                if (blend_peak_out)
                                    blend_peak = fmaxf(blend_peak, fac);
                                float v_depth = 0.f;
                                float color_dot = 0.f;
#pragma unroll
                                for (uint32_t sc = 0; sc < kSplatColors; ++sc) {
                                    const uint32_t ch = bwd_splat_channel<CDIM>(sc);
                                    const float v_c = s_v_c[ch][p];
                                    v_rgb[sc] += fac * v_c;
                                    color_dot += rgb[sc] * v_c;
                                }
                                if constexpr (kHasRayDepth) {
                                    const float v_d = s_v_c[kDepthChannel][p];
                                    color_dot += depth * v_d;
                                    v_depth += fac * v_d;
                                    if constexpr (kHasDepthSq) {
                                        const float v_d2 = s_v_c[kDepthSqChannel][p];
                                        color_dot += (depth * depth) * v_d2;
                                        v_depth += fac * (2.0f * depth) * v_d2;
                                    }
                                }
                                const float buffer_dot = s_buffer_dot[p];
                                s_buffer_dot[p] = buffer_dot + color_dot * fac;
                                float v_alpha = color_dot * T - buffer_dot * ra + ra * s_tail[p];
                                if constexpr (kUseCv) {
                                    const float4 cv_t = s_cv_target[p];
                                    if (cv_t.w > 0.f) {
                                        const float r0 = rgb[0] - cv_t.x;
                                        const float r1 = rgb[1] - cv_t.y;
                                        const float r2 = rgb[2] - cv_t.z;
                                        const float u_cv = cv_t.w * (r0 * r0 + r1 * r1 + r2 * r2);
                                        const float two_w = 2.0f * cv_t.w * fac;
                                        v_rgb[0] += two_w * r0;
                                        v_rgb[1] += two_w * r1;
                                        v_rgb[2] += two_w * r2;
                                        const float cv_buf = s_cv_buf[p];
                                        v_alpha += u_cv * T - cv_buf * ra;
                                        s_cv_buf[p] = cv_buf + u_cv * fac;
                                    }
                                }
                                if (do_dens) {
                                    dens_w += fac;
                                    dens_e += fac * s_error[p];
                                }
                                if constexpr (kEdgeScoring) {
                                    const float ew = s_edge[p];
                                    const float edge_contribution = fac * ew;
                                    if (ew > 0.0f && isfinite(edge_contribution)) {
                                        edge_score += edge_contribution;
                                    }
                                }
                                const bool alpha_grad = (opac * vis <= 0.999f);
                                const bool depth_grad =
                                    kHasRayDepth && (v_depth != 0.f && t_hit > 0.f && inv_grd_len > 0.f);
                                if (alpha_grad || depth_grad) {
                                    vec3 v_gro{0.f};
                                    vec3 v_grd{0.f};
                                    if (alpha_grad) {
                                        const float v_vis = opac * v_alpha;
                                        const float v_gradDist = -0.5f * vis * v_vis;
                                        const vec3 v_gcrod = 2.0f * v_gradDist * gcrod;
                                        const vec3 v_grd_n = -glm::cross(v_gcrod, gro);
                                        v_gro = glm::cross(v_gcrod, grd_n);
                                        v_grd = safe_normalize_bw_precomp(grd_n, inv_grd_len, v_grd_n);
                                        v_opacity += vis * v_alpha;
                                    }
                                    if constexpr (kHasRayDepth) {
                                        if (depth_grad) {
                                            ray_gaussian_depth_vjp(gro, grd_n, inv_grd_len, t_hit,
                                                                   depth_scale, v_depth, v_gro, v_grd);
                                        }
                                    }
                                    if constexpr (kGlobalShutter) {
                                        v_Mt_sum += glm::outerProduct(v_grd, ray_d);
                                    } else {
                                        v_Mt_sum += glm::outerProduct(v_grd, ray_d) +
                                                    glm::outerProduct(v_gro, o_minus_mu);
                                    }
                                    v_gro_sum += v_gro;
                                }
                                s_T[p] = T;
                            }
                        }
                    }
                    __syncwarp();
                }

                if (touched) {
                    record_blend_peak(blend_peak_out, g, blend_peak);
                    const vec3 v_mean = -(glm::transpose(Mt) * v_gro_sum);
                    if constexpr (kGlobalShutter) {
                        const float3 o3 = s_ray_o[0];
                        const vec3 o_minus_mu = vec3{o3.x, o3.y, o3.z} - means[g];
                        v_Mt_sum += glm::outerProduct(v_gro_sum, o_minus_mu);
                    }
                    const vec4 quat = quats[g];
                    vec3 scale_act = activated_scale(scales[g]);
                    if (dilations != nullptr)
                        scale_act = dilated_scale(scale_act, dilations[g]);
                    vec3 v_scale{0.f};
                    vec4 v_quat{0.f};
                    quat_scale_to_preci_half_vjp(
                        quat, scale_act, quat_to_rotmat(quat), glm::transpose(v_Mt_sum), v_quat, v_scale);
                    float* v_rgb_ptr = v_colors + CDIM * g;
#pragma unroll
                    for (uint32_t sc = 0; sc < kSplatColors; ++sc) {
                        if (v_rgb[sc] != 0.f) {
                            gpuAtomicAdd(v_rgb_ptr + bwd_splat_channel<CDIM>(sc), v_rgb[sc]);
                        }
                    }
                    float* v_mean_ptr = reinterpret_cast<float*>(v_means) + 3 * g;
                    if (v_mean.x != 0.f)
                        gpuAtomicAdd(v_mean_ptr, v_mean.x);
                    if (v_mean.y != 0.f)
                        gpuAtomicAdd(v_mean_ptr + 1, v_mean.y);
                    if (v_mean.z != 0.f)
                        gpuAtomicAdd(v_mean_ptr + 2, v_mean.z);
                    float* v_scale_ptr = reinterpret_cast<float*>(v_scales) + 3 * g;
                    const vec3 scale_chain = log_scale_chain(scale_act, scales, dilations, g);
                    const vec3 v_log_scale = v_scale * scale_chain;
                    if (v_log_scale.x != 0.f)
                        gpuAtomicAdd(v_scale_ptr, v_log_scale.x);
                    if (v_log_scale.y != 0.f)
                        gpuAtomicAdd(v_scale_ptr + 1, v_log_scale.y);
                    if (v_log_scale.z != 0.f)
                        gpuAtomicAdd(v_scale_ptr + 2, v_log_scale.z);
                    float* v_quat_ptr = reinterpret_cast<float*>(v_quats) + 4 * g;
                    if (v_quat.x != 0.f)
                        gpuAtomicAdd(v_quat_ptr, v_quat.x);
                    if (v_quat.y != 0.f)
                        gpuAtomicAdd(v_quat_ptr + 1, v_quat.y);
                    if (v_quat.z != 0.f)
                        gpuAtomicAdd(v_quat_ptr + 2, v_quat.z);
                    if (v_quat.w != 0.f)
                        gpuAtomicAdd(v_quat_ptr + 3, v_quat.w);
                    if (compensations != nullptr)
                        v_opacity *= compensations[g];
                    const float v_raw_opac = v_opacity * lfs::core::sigmoid_derivative_from_logit(opacities[g]);
                    if (v_raw_opac != 0.f)
                        gpuAtomicAdd(v_opacities + g, v_raw_opac);
                    if (do_dens) {
                        if (densification_weight != nullptr && dens_w != 0.f)
                            gpuAtomicAdd(densification_weight + g, dens_w);
                        if (dens_e != 0.f)
                            gpuAtomicAdd(densification_error + g, dens_e);
                    }
                    if constexpr (kEdgeScoring) {
                        if (edge_score != 0.f)
                            gpuAtomicAdd(edge_score_out + g, edge_score);
                    }
                }
            }
            __syncwarp();
        }
        if constexpr (kUseCv) {
            if (color_variance.enabled() && color_variance.loss_out != nullptr) {
                __syncwarp();
                float local_loss = 0.f;
                for (uint32_t p = lane; p < kSplatBwdPixels; p += 32) {
                    local_loss += s_cv_buf[p];
                }
                for (int offset = 16; offset > 0; offset /= 2) {
                    local_loss += __shfl_xor_sync(0xffffffffu, local_loss, offset);
                }
                if (lane == 0 && local_loss != 0.f) {
                    gpuAtomicAdd(color_variance.loss_out, local_loss);
                }
            }
        }
    }

    // The per-splat backward is the default; LFS_GUT_SPLAT_BWD=0 falls back to the
    // per-pixel kernels.
    bool use_splat_backward() {
        static const bool enabled = [] {
            const char* value = std::getenv("LFS_GUT_SPLAT_BWD");
            return value == nullptr || value[0] != '0';
        }();
        return enabled;
    }

    ////////////////////////////////////////////////////////////////
    // Launch Function
    ////////////////////////////////////////////////////////////////

    template <uint32_t CDIM>
    void launch_rasterize_to_pixels_from_world_3dgs_bwd_kernel(
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
        const float* render_alphas,
        const int32_t* last_ids,
        const float* v_render_colors,
        const float* v_render_alphas,
        float* v_means,
        float* v_quats,
        float* v_scales,
        float* v_colors,
        float* v_opacities,
        float* densification_weight,
        float* densification_error,
        const float* densification_error_map,
        const void* edge_weight_map,
        float* edge_score_out,
        cudaStream_t stream, TileRange tiles,
        const ScreenFootprint& footprint,
        const ColorVarianceRender& color_variance,
        float* blend_peak_out,
        const RenderGradPlanes& grad_planes,
        const bool edge_weight_fp16) {
        const bool packed = false; // Only support non-packed for now
        const uint32_t tile_width = (image_width + tile_size - 1) / tile_size;
        const uint32_t tile_height = (image_height + tile_size - 1) / tile_size;

        if (n_isects == 0) {
            // Skip kernel launch if no intersections
            return;
        }

        const bool global_shutter = is_global_shutter_launch(rs_type, viewmats1);
        const bool perfect_pinhole = is_perfect_pinhole_launch(
            camera_model, radial_coeffs, tangential_coeffs, thin_prism_coeffs);

        auto launch_args = [&](auto kernel, dim3 grid, dim3 threads, int64_t shmem_size) {
            set_kernel_max_dynamic_smem(
                kernel, static_cast<int>(shmem_size), "gsplat backward");
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
                render_alphas,
                last_ids,
                v_render_colors,
                v_render_alphas,
                reinterpret_cast<vec3*>(v_means),
                reinterpret_cast<vec4*>(v_quats),
                reinterpret_cast<vec3*>(v_scales),
                v_colors,
                v_opacities,
                densification_weight,
                densification_error,
                densification_error_map,
                edge_weight_map,
                edge_score_out, tiles, color_variance, blend_peak_out, grad_planes, edge_weight_fp16);
            LFS_CUDA_LAUNCH_CHECK(stream, "gsplat.rasterize_to_pixels_bwd");
        };

        auto dispatch = [&]<bool kBatched>() {
            if constexpr (CDIM <= 8) {
                if (use_splat_backward() && footprint.valid() && tile_size % kSplatBwdSub == 0 &&
                    camera_model != CameraModelType::ORTHO) {
                    const uint32_t subs = tile_size / kSplatBwdSub;
                    const dim3 threads = {32, 1, 1};
                    const dim3 grid = (footprint.tile_order != nullptr && C == 1 && !kBatched)
                                          ? dim3{tile_height * tile_width * subs * subs, 1u, 1u}
                                          : dim3{C, tile_height * subs, tile_width * subs};
                    auto launch_splat = [&](auto kernel) {
                        cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                             cudaSharedmemCarveoutMaxShared);
                        kernel<<<grid, threads, 0, stream>>>(
                            C, N,
                            reinterpret_cast<const vec3*>(means),
                            reinterpret_cast<const vec4*>(quats),
                            reinterpret_cast<const vec3*>(scales),
                            colors, opacities, backgrounds, bg_images, masks, compensations, dilations,
                            image_width, image_height, tile_size, tile_width, tile_height,
                            viewmats0, viewmats1, Ks, camera_model, rs_type,
                            radial_coeffs, tangential_coeffs, thin_prism_coeffs,
                            tile_offsets, flatten_ids, render_alphas, last_ids,
                            v_render_colors, v_render_alphas,
                            reinterpret_cast<vec3*>(v_means),
                            reinterpret_cast<vec4*>(v_quats),
                            reinterpret_cast<vec3*>(v_scales),
                            v_colors, v_opacities, densification_weight, densification_error, densification_error_map,
                            edge_weight_map, edge_score_out, tiles, footprint, color_variance, blend_peak_out,
                            grad_planes, edge_weight_fp16);
                        LFS_CUDA_LAUNCH_CHECK(stream, "gsplat.rasterize_to_pixels_bwd_splat");
                    };
                    const bool edges = edge_weight_map != nullptr && edge_score_out != nullptr;
                    const bool cached_rays = !perfect_pinhole && footprint.unit_rays != nullptr &&
                                             camera_model != CameraModelType::PINHOLE;
                    auto dispatch_splat = [&]<bool kUseCv>() {
                        if (global_shutter && perfect_pinhole) {
                            if (edges)
                                launch_splat(rasterize_to_pixels_from_world_3dgs_bwd_splat_kernel<CDIM, true, true, kBatched, kUseCv, false, true>);
                            else
                                launch_splat(rasterize_to_pixels_from_world_3dgs_bwd_splat_kernel<CDIM, true, false, kBatched, kUseCv, false, true>);
                        } else if (cached_rays) {
                            if (global_shutter) {
                                if (edges)
                                    launch_splat(rasterize_to_pixels_from_world_3dgs_bwd_splat_kernel<CDIM, false, true, kBatched, kUseCv, true, true>);
                                else
                                    launch_splat(rasterize_to_pixels_from_world_3dgs_bwd_splat_kernel<CDIM, false, false, kBatched, kUseCv, true, true>);
                            } else {
                                if (edges)
                                    launch_splat(rasterize_to_pixels_from_world_3dgs_bwd_splat_kernel<CDIM, false, true, kBatched, kUseCv, true, false>);
                                else
                                    launch_splat(rasterize_to_pixels_from_world_3dgs_bwd_splat_kernel<CDIM, false, false, kBatched, kUseCv, true, false>);
                            }
                        } else {
                            if (edges)
                                launch_splat(rasterize_to_pixels_from_world_3dgs_bwd_splat_kernel<CDIM, false, true, kBatched, kUseCv, false, false>);
                            else
                                launch_splat(rasterize_to_pixels_from_world_3dgs_bwd_splat_kernel<CDIM, false, false, kBatched, kUseCv, false, false>);
                        }
                    };
                    if constexpr (CDIM >= 3) {
                        if (color_variance.enabled()) {
                            dispatch_splat.template operator()<true>();
                            return;
                        }
                    }
                    dispatch_splat.template operator()<false>();
                    return;
                }
            }
            if constexpr (CDIM == 3) {
                if (tile_size == 16 && !color_variance.enabled()) {
                    dim3 threads = {8, 16, 1};
                    dim3 grid = {C, tile_height, tile_width};
                    int64_t shmem_size =
                        int64_t(kBwdDualBatch) *
                        (sizeof(int32_t) + sizeof(vec4) + sizeof(vec3) + sizeof(vec4) +
                         sizeof(mat3) + sizeof(float) * CDIM);
                    const int64_t edge_shmem_size =
                        shmem_size + int64_t(kBwdDualBatch) * sizeof(float);
                    if (global_shutter) {
                        if (perfect_pinhole) {
                            if (edge_weight_map != nullptr && edge_score_out != nullptr) {
                                launch_args(
                                    rasterize_to_pixels_from_world_3dgs_bwd_dual_kernel<float, true, true, true, kBatched>,
                                    grid, threads, edge_shmem_size);
                            } else {
                                launch_args(
                                    rasterize_to_pixels_from_world_3dgs_bwd_dual_kernel<float, true, true, false, kBatched>,
                                    grid, threads, shmem_size);
                            }
                        } else {
                            if (edge_weight_map != nullptr && edge_score_out != nullptr) {
                                launch_args(
                                    rasterize_to_pixels_from_world_3dgs_bwd_dual_kernel<float, true, false, true, kBatched>,
                                    grid, threads, edge_shmem_size);
                            } else {
                                launch_args(
                                    rasterize_to_pixels_from_world_3dgs_bwd_dual_kernel<float, true, false, false, kBatched>,
                                    grid, threads, shmem_size);
                            }
                        }
                        return;
                    }
                    if (edge_weight_map != nullptr && edge_score_out != nullptr) {
                        launch_args(
                            rasterize_to_pixels_from_world_3dgs_bwd_dual_kernel<float, false, false, true, kBatched>,
                            grid, threads, edge_shmem_size);
                    } else {
                        launch_args(
                            rasterize_to_pixels_from_world_3dgs_bwd_dual_kernel<float, false, false, false, kBatched>,
                            grid, threads, shmem_size);
                    }
                    return;
                }
            }

            dim3 threads = {tile_size, tile_size, 1};
            dim3 grid = {C, tile_height, tile_width};
            int64_t shmem_size =
                tile_size * tile_size *
                (sizeof(int32_t) + sizeof(vec4) + sizeof(vec3) + sizeof(vec4) +
                 sizeof(mat3) + sizeof(float) * CDIM);
            if constexpr (CDIM == 3) {
                constexpr int64_t kOccCapBytes = 49152;
                if (shmem_size < kOccCapBytes) {
                    shmem_size = kOccCapBytes;
                }
            }
            const int64_t edge_shmem_size = shmem_size + tile_size * tile_size * sizeof(float);
            if constexpr (CDIM == 3) {
                if (global_shutter && perfect_pinhole) {
                    if (edge_weight_map != nullptr && edge_score_out != nullptr) {
                        launch_args(
                            rasterize_to_pixels_from_world_3dgs_bwd_kernel<CDIM, float, true, true, kBatched>,
                            grid, threads, edge_shmem_size);
                    } else {
                        launch_args(
                            rasterize_to_pixels_from_world_3dgs_bwd_kernel<CDIM, float, true, false, kBatched>,
                            grid, threads, shmem_size);
                    }
                    return;
                }
            }
            if (edge_weight_map != nullptr && edge_score_out != nullptr) {
                launch_args(
                    rasterize_to_pixels_from_world_3dgs_bwd_kernel<CDIM, float, false, true, kBatched>,
                    grid, threads, edge_shmem_size);
            } else {
                launch_args(
                    rasterize_to_pixels_from_world_3dgs_bwd_kernel<CDIM, float, false, false, kBatched>,
                    grid, threads, shmem_size);
            }
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
    template void launch_rasterize_to_pixels_from_world_3dgs_bwd_kernel<CDIM>( \
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
        const float* render_alphas,                                            \
        const int32_t* last_ids,                                               \
        const float* v_render_colors,                                          \
        const float* v_render_alphas,                                          \
        float* v_means,                                                        \
        float* v_quats,                                                        \
        float* v_scales,                                                       \
        float* v_colors,                                                       \
        float* v_opacities,                                                    \
        float* densification_weight,                                           \
        float* densification_error,                                            \
        const float* densification_error_map,                                  \
        const void* edge_weight_map,                                           \
        float* edge_score_out,                                                 \
        cudaStream_t stream, TileRange tiles,                                  \
        const ScreenFootprint& footprint,                                      \
        const ColorVarianceRender& color_variance,                            \
        float* blend_peak_out,                                                 \
        const RenderGradPlanes& grad_planes,                                   \
        bool edge_weight_fp16);

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
