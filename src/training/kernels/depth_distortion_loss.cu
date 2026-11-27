/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "depth_distortion_loss.hpp"
#include "core/cuda_error.hpp"

#include "kernel_stream.hpp"

namespace lfs::training::kernels {
    namespace {
        constexpr int kThreadsPerBlock = 256;

        __device__ __forceinline__ bool target_depth_is_edge(
            const float* __restrict__ target_depth,
            const int x,
            const int y,
            const int width,
            const int height,
            const float anchor_scale,
            const float anchor_shift) {
            if (!target_depth || !(anchor_scale > 0.0f)) {
                return false;
            }
            const size_t idx = static_cast<size_t>(y) * width + x;
            const float tc = target_depth[idx];
            if (!(tc >= kDepthDistortionMinTargetCode) || !isfinite(tc)) {
                return true;
            }
            const float dc = anchor_scale * tc + anchor_shift;
            if (!(dc >= kDepthDistortionMinExpectedDepth) || !isfinite(dc)) {
                return true;
            }
            for (int dy = -1; dy <= 1; ++dy) {
                const int ny = y + dy;
                if (ny < 0 || ny >= height) {
                    continue;
                }
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) {
                        continue;
                    }
                    const int nx = x + dx;
                    if (nx < 0 || nx >= width) {
                        continue;
                    }
                    const float tn = target_depth[static_cast<size_t>(ny) * width + nx];
                    if (!(tn >= kDepthDistortionMinTargetCode) || !isfinite(tn)) {
                        return true;
                    }
                    const float dn = anchor_scale * tn + anchor_shift;
                    if (!(dn >= kDepthDistortionMinExpectedDepth) || !isfinite(dn)) {
                        return true;
                    }
                    if (fabsf(dn - dc) > kDepthDistortionEdgeRelJump * fminf(dn, dc)) {
                        return true;
                    }
                }
            }
            return false;
        }

        __global__ void depth_distortion_kernel(
            const float* __restrict__ s1_buf,
            const float* __restrict__ s2_buf,
            const float* __restrict__ alpha_buf,
            const float* __restrict__ pixel_weight,
            const float* __restrict__ median_depth,
            const float* __restrict__ target_depth,
            float* __restrict__ grad_s1,
            float* __restrict__ grad_s2,
            float* __restrict__ grad_alpha,
            float* __restrict__ grad_median_depth,
            const int width,
            const int height,
            const float inv_pixels,
            const float dist_weight,
            const float mm_weight,
            const float anchor_scale,
            const float anchor_shift) {
            const size_t num_pixels = static_cast<size_t>(width) * height;
            const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (idx >= num_pixels) {
                return;
            }
            grad_s2[idx] = 0.0f;
            const float a = alpha_buf[idx];
            const float s1 = s1_buf[idx];
            const float s2 = s2_buf[idx];
            const float w = pixel_weight ? pixel_weight[idx] : 1.0f;
            if (!(a > kDepthDistortionMinAlpha) || !(w > 0.0f) || !(s1 > 0.0f) ||
                !isfinite(s1) || !isfinite(s2) || !isfinite(a)) {
                return;
            }
            const float e = s1 / a;
            if (!(e > kDepthDistortionMinExpectedDepth)) {
                return;
            }
            const int x = static_cast<int>(idx % width);
            const int y = static_cast<int>(idx / width);
            if (target_depth_is_edge(target_depth, x, y, width, height, anchor_scale, anchor_shift)) {
                return;
            }

            if (dist_weight > 0.0f) {
                float r2 = 0.0f;
                float dr2_ds1 = 0.0f;
                float dr2_ds2 = 0.0f;
                float dr2_da = 0.0f;
                if (e >= kDepthDistortionDepthFloor) {
                    const float inv_s1 = 1.0f / s1;
                    const float inv_s1_sq = inv_s1 * inv_s1;
                    const float raw = (s2 * a - s1 * s1) * inv_s1_sq;
                    if (raw > 1.0e-6f) {
                        r2 = raw;
                        dr2_ds2 = a * inv_s1_sq;
                        dr2_ds1 = -2.0f * s2 * a * inv_s1_sq * inv_s1;
                        dr2_da = s2 * inv_s1_sq;
                    }
                } else {
                    const float inv_a = 1.0f / a;
                    const float inv_floor_sq = 1.0f / (kDepthDistortionDepthFloor * kDepthDistortionDepthFloor);
                    const float var = s2 * inv_a - e * e;
                    const float raw = var * inv_floor_sq;
                    if (raw > 1.0e-6f) {
                        r2 = raw;
                        dr2_ds2 = inv_a * inv_floor_sq;
                        dr2_ds1 = -2.0f * e * inv_a * inv_floor_sq;
                        dr2_da = (2.0f * e * e - s2 * inv_a) * inv_a * inv_floor_sq;
                    }
                }
                if (r2 > 0.0f) {
                    const float rho = sqrtf(r2 + kDepthDistortionRelEpsilon * kDepthDistortionRelEpsilon);
                    const float g = (dist_weight * inv_pixels) * w *
                                    (0.5f / (rho * (1.0f + rho / kDepthDistortionRelSaturation)));
                    grad_s1[idx] += g * dr2_ds1;
                    grad_s2[idx] = g * dr2_ds2;
                    grad_alpha[idx] += g * dr2_da;
                }
            }

            if (median_depth && grad_median_depth && mm_weight > 0.0f) {
                const float d_med = median_depth[idx];
                if (d_med > 1.0e-4f && isfinite(d_med)) {
                    const float u = logf(e) - logf(d_med);
                    const float denom = sqrtf(
                        u * u + kDepthDistortionMeanMedianEpsilon * kDepthDistortionMeanMedianEpsilon);
                    const float g_mm = (mm_weight * inv_pixels) * w * (u / denom);
                    grad_s1[idx] += g_mm / s1;
                    grad_alpha[idx] += -g_mm / a;
                    grad_median_depth[idx] += -g_mm / d_med;
                }
            }
        }
    } // namespace

    void launch_depth_distortion_loss(
        const float* rendered_depth_accum,
        const float* rendered_depth_sq_accum,
        const float* rendered_alpha,
        float* grad_depth_accum,
        float* grad_depth_sq_accum,
        float* grad_alpha,
        const int width,
        const int height,
        const float weight,
        cudaStream_t stream,
        const float* pixel_weight,
        const float* median_depth,
        float* grad_median_depth,
        const float mean_median_weight,
        const float* target_depth,
        const float anchor_scale,
        const float anchor_shift) {
        stream = resolve_stream(stream);
        const size_t num_pixels = static_cast<size_t>(width) * height;
        if (num_pixels == 0) {
            return;
        }
        depth_distortion_kernel<<<(num_pixels + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
            rendered_depth_accum,
            rendered_depth_sq_accum,
            rendered_alpha,
            pixel_weight,
            median_depth,
            target_depth,
            grad_depth_accum,
            grad_depth_sq_accum,
            grad_alpha,
            grad_median_depth,
            width,
            height,
            1.0f / static_cast<float>(num_pixels),
            weight,
            mean_median_weight,
            anchor_scale,
            anchor_shift);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.depth_distortion");
    }

} // namespace lfs::training::kernels
