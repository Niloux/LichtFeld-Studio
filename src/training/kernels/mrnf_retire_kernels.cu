/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"
#include "kernel_stream.hpp"
#include "mrnf_kernels.hpp"
#include <cuda_runtime.h>

namespace lfs::training::mrnf_strategy {

    // A row's reach: clamp(peak / max(sigmoid(raw opacity), 1e-6), 0, 1).
    __device__ __forceinline__ float row_reach(const float peak, const float raw_opacity) {
        const float opacity = fmaxf(1.0f / (1.0f + expf(-raw_opacity)), 1e-6f);
        // Explicit at and above the opacity: the fast-math division need not give exactly 1 there.
        return peak >= opacity ? 1.0f : fminf(fmaxf(peak / opacity, 0.0f), 1.0f);
    }

    __global__ void blend_peak_update_kernel(float* __restrict__ step_peak, float* __restrict__ peak,
                                             float* __restrict__ renders, const float peak_decay, const size_t N,
                                             float* __restrict__ rendered_count) {
        const size_t idx = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
        if (idx >= N)
            return;
        const float step = step_peak[idx];
        if (step != 0.0f)
            step_peak[idx] = 0.0f;
        if (step > 0.0f) {
            renders[idx] += 1.0f;
            if (rendered_count)
                rendered_count[idx] += 1.0f;
        }
        peak[idx] = fmaxf(peak[idx] * peak_decay, step);
    }
    __global__ void blend_peak_update_vec4_kernel(float* __restrict__ step_peak, float* __restrict__ peak,
                                                  float* __restrict__ renders, const float peak_decay, const size_t N,
                                                  float* __restrict__ rendered_count) {
        const size_t idx = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
        const size_t idx4 = idx * 4;
        if (idx4 + 3 < N) {
            const float4 s = reinterpret_cast<const float4*>(step_peak)[idx];
            if (s.x != 0.0f || s.y != 0.0f || s.z != 0.0f || s.w != 0.0f) {
                reinterpret_cast<float4*>(step_peak)[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                float4 r = reinterpret_cast<const float4*>(renders)[idx];
                if (s.x > 0.0f) r.x += 1.0f;
                if (s.y > 0.0f) r.y += 1.0f;
                if (s.z > 0.0f) r.z += 1.0f;
                if (s.w > 0.0f) r.w += 1.0f;
                reinterpret_cast<float4*>(renders)[idx] = r;
                if (rendered_count) {
                    float4 rc = reinterpret_cast<const float4*>(rendered_count)[idx];
                    if (s.x > 0.0f) rc.x += 1.0f;
                    if (s.y > 0.0f) rc.y += 1.0f;
                    if (s.z > 0.0f) rc.z += 1.0f;
                    if (s.w > 0.0f) rc.w += 1.0f;
                    reinterpret_cast<float4*>(rendered_count)[idx] = rc;
                }
            }
            float4 p = reinterpret_cast<const float4*>(peak)[idx];
            p.x = fmaxf(p.x * peak_decay, s.x);
            p.y = fmaxf(p.y * peak_decay, s.y);
            p.z = fmaxf(p.z * peak_decay, s.z);
            p.w = fmaxf(p.w * peak_decay, s.w);
            reinterpret_cast<float4*>(peak)[idx] = p;
            return;
        }
        for (size_t i = idx4; i < N; ++i) {
            const float step = step_peak[i];
            if (step != 0.0f)
                step_peak[i] = 0.0f;
            if (step > 0.0f) {
                renders[i] += 1.0f;
                if (rendered_count)
                    rendered_count[i] += 1.0f;
            }
            peak[i] = fmaxf(peak[i] * peak_decay, step);
        }
    }
    void launch_blend_peak_update(float* step_peak, float* peak, float* renders, const float peak_decay,
                                  const size_t N, void* stream, float* rendered_count) {
        if (N == 0)
            return;
        constexpr int threads = 256;
        cudaStream_t s = resolve_stream(stream);
        const bool aligned16 =
            ((reinterpret_cast<uintptr_t>(step_peak) | reinterpret_cast<uintptr_t>(peak) |
              reinterpret_cast<uintptr_t>(renders) | reinterpret_cast<uintptr_t>(rendered_count)) &
             15u) == 0;
        if (aligned16 && N >= 4) {
            const size_t n4 = (N + 3) / 4;
            const int blocks = static_cast<int>((n4 + threads - 1) / threads);
            blend_peak_update_vec4_kernel<<<blocks, threads, 0, s>>>(
                step_peak, peak, renders, peak_decay, N, rendered_count);
        } else {
            const int blocks = static_cast<int>((N + threads - 1) / threads);
            blend_peak_update_kernel<<<blocks, threads, 0, s>>>(
                step_peak, peak, renders, peak_decay, N, rendered_count);
        }
        LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.blend_peak_update");
    }
    __global__ void mrnf_covered_factor_kernel(
        const float* __restrict__ peak,
        const float* __restrict__ renders,
        const float* __restrict__ raw_opacities,
        const float* __restrict__ log_scales,
        float grace,
        float keep_reach,
        float log_size_floor,
        size_t N,
        float* __restrict__ factor) {

        const size_t idx = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
        if (idx >= N)
            return;
        // Within the grace: unknown, treated as visible.
        const float r = renders[idx] >= grace ? row_reach(peak[idx], raw_opacities[idx]) : 1.0f;
        const float largest = fmaxf(fmaxf(log_scales[idx * 3 + 0], log_scales[idx * 3 + 1]), log_scales[idx * 3 + 2]);
        factor[idx] = r < keep_reach && largest >= log_size_floor ? r : 1.0f;
    }
    void launch_mrnf_covered_factor(
        const float* peak,
        const float* renders,
        const float* raw_opacities,
        const float* log_scales,
        float grace,
        float keep_reach,
        float log_size_floor,
        size_t N,
        float* factor,
        void* stream) {

        if (N == 0)
            return;
        constexpr int threads = 256;
        const int blocks = static_cast<int>((N + threads - 1) / threads);
        cudaStream_t s = resolve_stream(stream);
        mrnf_covered_factor_kernel<<<blocks, threads, 0, s>>>(peak, renders, raw_opacities, log_scales, grace,
                                                              keep_reach, log_size_floor, N, factor);
        LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.covered_factor");
    }

    __global__ void mrnf_retire_key_kernel(const float* __restrict__ peak, const float* __restrict__ raw_opacities,
                                           const float dead_peak, const size_t N, float* __restrict__ key) {
        const size_t idx = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
        if (idx >= N)
            return;
        const float p = peak[idx];
        // Dead rows land in [-1, -1 + dead_peak), below every reach in [0, 1].
        key[idx] = p < dead_peak ? p - 1.0f : row_reach(p, raw_opacities[idx]);
    }
    void launch_mrnf_retire_key(const float* peak, const float* raw_opacities, const float dead_peak, const size_t N,
                                float* key, void* stream) {
        if (N == 0)
            return;
        constexpr int threads = 256;
        const int blocks = static_cast<int>((N + threads - 1) / threads);
        cudaStream_t s = resolve_stream(stream);
        mrnf_retire_key_kernel<<<blocks, threads, 0, s>>>(peak, raw_opacities, dead_peak, N, key);
        LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.retire_key");
    }

} // namespace lfs::training::mrnf_strategy
