/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"
#include "kernel_stream.hpp"
#include "lfs/core/warp_reduce.cuh"
#include "lfs/kernels/gut_regularization.cuh"
#include "lfs/training/effective_rank.cuh"

namespace lfs::training::kernels {
    namespace {
        // Mirrors the erank and dc_reg terms of FastGS's fused backward (kernels_backward.cuh).
        __global__ void gut_param_regularization_kernel(
            const float* __restrict__ log_scales, float* __restrict__ scales_grad,
            const float* __restrict__ sh0, float* __restrict__ sh0_grad, const int sh0_stride,
            const int n, const float erank_weight, const float dc_weight,
            float* __restrict__ erank_loss_out, float* __restrict__ dc_loss_out) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            float erank_loss = 0.0f;
            float dc_loss = 0.0f;
            if (i < n) {
                if (erank_weight > 0.0f) {
                    const float* s = log_scales + 3 * i;
                    const auto penalty = lfs::training::effective_rank_penalty(s[0], s[1], s[2]);
                    const float coef = erank_weight / static_cast<float>(n);
                    erank_loss = coef * penalty.value;
                    for (int axis = 0; axis < 3; ++axis)
                        scales_grad[3 * i + axis] += coef * penalty.gradient[axis];
                }
                if (dc_weight > 0.0f) {
                    constexpr float kSh0 = 0.28209479177387814f;
                    const float limit = 0.5f / kSh0;
                    // The gradient stays per splat; the reported loss is the mean over splats.
                    const float reported_loss_weight = dc_weight / (3.0f * static_cast<float>(n));
                    const float* coeff = sh0 + i * sh0_stride;
                    for (int c = 0; c < 3; ++c) {
                        float delta = 0.0f;
                        if (coeff[c] > limit)
                            delta = coeff[c] - limit;
                        else if (coeff[c] < -limit)
                            delta = coeff[c] + limit;
                        sh0_grad[i * sh0_stride + c] += (2.0f * dc_weight / 3.0f) * delta;
                        dc_loss += reported_loss_weight * delta * delta;
                    }
                }
            }
            if (erank_loss_out != nullptr) {
                const float block_sum = lfs::core::warp_ops::block_reduce_sum(erank_loss);
                if (threadIdx.x == 0 && block_sum != 0.0f)
                    atomicAdd(erank_loss_out, block_sum);
            }
            if (dc_loss_out != nullptr) {
                const float block_sum = lfs::core::warp_ops::block_reduce_sum(dc_loss);
                if (threadIdx.x == 0 && block_sum != 0.0f)
                    atomicAdd(dc_loss_out, block_sum);
            }
        }
    } // namespace

    void launch_gut_param_regularization(
        const float* log_scales, float* scales_grad, const float* sh0, float* sh0_grad,
        const int sh0_stride, const int n, const float erank_weight, const float dc_weight,
        float* erank_loss_out, float* dc_loss_out, cudaStream_t stream) {
        stream = resolve_stream(stream);
        if (n <= 0 || (erank_weight <= 0.0f && dc_weight <= 0.0f))
            return;
        constexpr int kBlock = 256;
        gut_param_regularization_kernel<<<(n + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
            log_scales, scales_grad, sh0, sh0_grad, sh0_stride, n,
            erank_weight, dc_weight,
            erank_weight > 0.0f ? erank_loss_out : nullptr,
            dc_weight > 0.0f ? dc_loss_out : nullptr);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.regularization.gut_params");
    }

} // namespace lfs::training::kernels
