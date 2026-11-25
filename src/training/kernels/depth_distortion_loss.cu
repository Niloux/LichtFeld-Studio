/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "depth_distortion_loss.hpp"
#include "core/cuda_error.hpp"

#include "kernel_stream.hpp"

namespace lfs::training::kernels {
    namespace {
        constexpr int kThreadsPerBlock = 256;

        __global__ void depth_distortion_kernel(
            const float* __restrict__ s1,
            const float* __restrict__ s2,
            const float* __restrict__ alpha,
            const float* __restrict__ pixel_weight,
            float* __restrict__ grad_s1,
            float* __restrict__ grad_s2,
            float* __restrict__ grad_alpha,
            const size_t num_pixels,
            const float scale) {
            const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (idx >= num_pixels) {
                return;
            }
            grad_s2[idx] = 0.0f;
            const float a = alpha[idx];
            const float w = pixel_weight ? pixel_weight[idx] : 1.0f;
            if (!(a > kDepthDistortionMinAlpha) || !(w > 0.0f) || !isfinite(s1[idx]) || !isfinite(s2[idx])) {
                return;
            }
            const float inv = 1.0f / a;
            const float expected = s1[idx] * inv;
            const float variance = fmaxf(s2[idx] * inv - expected * expected, 0.0f);
            if (variance > kDepthDistortionMaxStdDev * kDepthDistortionMaxStdDev) {
                return;
            }
            const float g = scale * w * 0.5f / sqrtf(variance + kDepthDistortionEpsilon * kDepthDistortionEpsilon);
            grad_s1[idx] += g * -2.0f * expected * inv;
            grad_s2[idx] = g * inv;
            grad_alpha[idx] += g * inv * (2.0f * expected * expected - s2[idx] * inv);
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
        const float* pixel_weight) {
        stream = resolve_stream(stream);
        const size_t num_pixels = static_cast<size_t>(width) * height;
        depth_distortion_kernel<<<(num_pixels + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
            rendered_depth_accum, rendered_depth_sq_accum, rendered_alpha, pixel_weight,
            grad_depth_accum, grad_depth_sq_accum, grad_alpha, num_pixels, weight / static_cast<float>(num_pixels));
        LFS_CUDA_LAUNCH_CHECK(stream, "training.depth_distortion");
    }

} // namespace lfs::training::kernels
