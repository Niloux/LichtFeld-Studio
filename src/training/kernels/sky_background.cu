/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/cuda_error.hpp"
#include "sky_background.hpp"
#include "mask_preprocess.hpp"
#include "lfs/core/warp_reduce.cuh"
#include <algorithm>
#include <cmath>
#include <type_traits>
namespace lfs::training::kernels {
    namespace {
        template <typename T>
        __device__ float normalized_sample(const T* samples, int i) {
            if constexpr (std::is_same_v<T, uint8_t>)
                return samples[i] / 255.f;
            return samples[i];
        }

        template <typename TargetT, typename MaskT>
        __global__ void masked_l1_kernel(const float* image, const TargetT* target,
                                         const MaskT* mask, const float* roi, float* grad,
                                         float* partial, int pixels) {
            float loss = 0.f, weight_sum = 0.f;
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < pixels;
                 i += blockDim.x * gridDim.x) {
                const float m = normalized_sample(mask, i);
                const float weight = m >= kMaskSegmentMin && m <= kMaskKeepMin
                                         ? (roi ? roi[i] : 1.f)
                                         : 0.f;
                weight_sum += 3.f * weight;
                for (int c = 0; c < 3; ++c) {
                    const int j = c * pixels + i;
                    float g = 0.f;
                    if (weight > 0.f) {
                        const float diff = image[j] - normalized_sample(target, j);
                        loss += weight * fabsf(diff);
                        g = weight * ((diff > 0.f) - (diff < 0.f));
                    }
                    grad[j] = g;
                }
            }
            loss = core::warp_ops::block_reduce_sum(loss);
            if (threadIdx.x == 0)
                partial[blockIdx.x] = loss;
            __syncthreads();
            weight_sum = core::warp_ops::block_reduce_sum(weight_sum);
            if (threadIdx.x == 0)
                partial[gridDim.x + blockIdx.x] = weight_sum;
        }

        __global__ void reduce_color_loss_kernel(const float* partial, float* stats, int blocks) {
            float loss = 0.f, weight = 0.f;
            for (int i = threadIdx.x; i < blocks; i += blockDim.x) {
                loss += partial[i];
                weight += partial[blocks + i];
            }
            loss = core::warp_ops::block_reduce_sum(loss);
            if (threadIdx.x == 0)
                stats[0] = loss;
            __syncthreads();
            weight = core::warp_ops::block_reduce_sum(weight);
            if (threadIdx.x == 0) {
                stats[1] = weight > 0.f ? 1.f / weight : 0.f;
                stats[0] *= stats[1];
            }
        }

        __global__ void normalize_color_gradient_kernel(float* grad, const float* stats, int n) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n)
                grad[i] *= stats[1];
        }

        template <typename MaskT>
        __global__ void foreground_gradient_kernel(float* gradient, const MaskT* mask, int pixels) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < pixels && normalized_sample(mask, i) <= kMaskKeepMin) {
                for (int c = 0; c < 3; ++c)
                    gradient[c * pixels + i] = 0.f;
            }
        }

        __global__ void adam_kernel(float* sh0, float* grad, float* m, float* v,
                                    int n, float lr, float inv_b1, float inv_b2) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n)
                return;
            const float g = grad[i];
            m[i] = .9f * m[i] + .1f * g;
            v[i] = .999f * v[i] + .001f * g * g;
            const float value = sh0[i] - lr * (m[i] * inv_b1) / (sqrtf(v[i] * inv_b2) + 1e-15f);
            constexpr float c0 = .28209479177387814f;
            sh0[i] = fminf((1.f - .5f) / c0, fmaxf((1.f / 255.f - .5f) / c0, value));
            grad[i] = 0.f;
        }
    } // namespace
    void sky_masked_l1(const float* image, const void* target, bool target_u8,
                       const void* mask, bool mask_u8, const float* roi, float* grad,
                       float* stats, float* partial, int pixels, cudaStream_t stream) {
        const int blocks = std::min((pixels + 255) / 256, 1024);
        if (target_u8 && mask_u8)
            masked_l1_kernel<<<blocks, 256, 0, stream>>>(image, static_cast<const uint8_t*>(target), static_cast<const uint8_t*>(mask), roi, grad, partial, pixels);
        else if (target_u8)
            masked_l1_kernel<<<blocks, 256, 0, stream>>>(image, static_cast<const uint8_t*>(target), static_cast<const float*>(mask), roi, grad, partial, pixels);
        else if (mask_u8)
            masked_l1_kernel<<<blocks, 256, 0, stream>>>(image, static_cast<const float*>(target), static_cast<const uint8_t*>(mask), roi, grad, partial, pixels);
        else
            masked_l1_kernel<<<blocks, 256, 0, stream>>>(image, static_cast<const float*>(target), static_cast<const float*>(mask), roi, grad, partial, pixels);
        reduce_color_loss_kernel<<<1, 256, 0, stream>>>(partial, stats, blocks);
        normalize_color_gradient_kernel<<<(3 * pixels + 255) / 256, 256, 0, stream>>>(grad, stats, 3 * pixels);
        LFS_CUDA_CHECK(cudaGetLastError());
    }

    void sky_mask_foreground_gradient(float* gradient, const void* mask, bool mask_u8,
                                      int pixels, cudaStream_t stream) {
        if (mask_u8)
            foreground_gradient_kernel<<<(pixels + 255) / 256, 256, 0, stream>>>(gradient, static_cast<const uint8_t*>(mask), pixels);
        else
            foreground_gradient_kernel<<<(pixels + 255) / 256, 256, 0, stream>>>(gradient, static_cast<const float*>(mask), pixels);
        LFS_CUDA_CHECK(cudaGetLastError());
    }
    void sky_adam(float* sh0, float* gradient, float* first, float* second, int count,
                  std::uint64_t step, float lr, cudaStream_t stream) {
        const int n = count * 3;
        adam_kernel<<<(n + 255) / 256, 256, 0, stream>>>(sh0, gradient, first, second, n, lr,
                                                         1.f / static_cast<float>(1. - std::pow(.9, static_cast<double>(step))),
                                                         1.f / static_cast<float>(1. - std::pow(.999, static_cast<double>(step))));
        LFS_CUDA_CHECK(cudaGetLastError());
    }
} // namespace lfs::training::kernels
