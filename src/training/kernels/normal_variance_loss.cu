/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "normal_variance_loss.hpp"
#include "core/cuda_error.hpp"

#include "kernel_stream.hpp"

namespace lfs::training::kernels {
    namespace {
        constexpr int kThreadsPerBlock = 256;

        __global__ void normal_variance_kernel(
            const float* __restrict__ rendered_normal,
            const float* __restrict__ rendered_alpha,
            const float* __restrict__ pixel_weight,
            float* __restrict__ grad_normal,
            float* __restrict__ grad_alpha,
            const size_t num_pixels,
            const float scale) {
            const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (idx >= num_pixels) {
                return;
            }
            const float a = rendered_alpha[idx];
            const float w = pixel_weight ? pixel_weight[idx] : 1.0f;
            if (!(a > kNormalVarianceMinAlpha) || !(w > 0.0f) || !isfinite(a) || !isfinite(w)) {
                return;
            }
            const float nx = rendered_normal[idx];
            const float ny = rendered_normal[num_pixels + idx];
            const float nz = rendered_normal[2 * num_pixels + idx];
            if (!isfinite(nx) || !isfinite(ny) || !isfinite(nz)) {
                return;
            }
            const float norm_sq = nx * nx + ny * ny + nz * nz;
            const float inv_a = 1.0f / a;
            const float inv_a2 = inv_a * inv_a;
            const float mean_norm_sq = norm_sq * inv_a2;
            if (mean_norm_sq >= 1.0f || mean_norm_sq < kNormalVarianceMinNorm * kNormalVarianceMinNorm) {
                return;
            }
            const float coeff = 2.0f * scale * w * inv_a2;
            grad_normal[idx] -= coeff * nx;
            grad_normal[num_pixels + idx] -= coeff * ny;
            grad_normal[2 * num_pixels + idx] -= coeff * nz;
            grad_alpha[idx] += coeff * norm_sq * inv_a;
        }
    } // namespace

    void launch_normal_variance_loss(
        const float* rendered_normal,
        const float* rendered_alpha,
        float* grad_normal,
        float* grad_alpha,
        const int width,
        const int height,
        const float weight,
        cudaStream_t stream,
        const float* pixel_weight) {
        if (width <= 0 || height <= 0 || !(weight > 0.0f)) {
            return;
        }
        stream = resolve_stream(stream);
        const size_t num_pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
        normal_variance_kernel<<<(num_pixels + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
            rendered_normal, rendered_alpha, pixel_weight, grad_normal, grad_alpha,
            num_pixels, weight / static_cast<float>(num_pixels));
        LFS_CUDA_LAUNCH_CHECK(stream, "training.normal_variance");
    }

} // namespace lfs::training::kernels
