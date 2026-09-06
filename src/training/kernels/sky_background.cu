/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/cuda_error.hpp"
#include "sky_background.hpp"
#include <cmath>
namespace lfs::training::kernels {
    namespace {
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
    void sky_adam(float* sh0, float* gradient, float* first, float* second, int count,
                  std::uint64_t step, float lr, cudaStream_t stream) {
        const int n = count * 3;
        adam_kernel<<<(n + 255) / 256, 256, 0, stream>>>(sh0, gradient, first, second, n, lr,
                                                         1.f / static_cast<float>(1. - std::pow(.9, static_cast<double>(step))),
                                                         1.f / static_cast<float>(1. - std::pow(.999, static_cast<double>(step))));
        LFS_CUDA_CHECK(cudaGetLastError());
    }
} // namespace lfs::training::kernels
