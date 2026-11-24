/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "FromWorldRay.cuh"
#include "GeometryFeatures.h"
#include "Utils.cuh"
#include "core/cuda_error.hpp"

namespace gsplat_lfs {
    namespace {
        __global__ void expected_depth_kernel(const float* depth, const float* alpha, float* gd, float* ga, uint32_t pixels) {
            const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= pixels)
                return;
            const float a = fmaxf(alpha[i], 1.e-10f);
            const float g = gd[i];
            gd[i] = g / a;
            if (alpha[i] >= 1.e-10f)
                ga[i] -= g * depth[i] / (a * a);
        }
        __global__ void rays_kernel(float* rays, uint32_t w, uint32_t h, const float* K,
                                    CameraModelType model, const float* radial,
                                    const float* tangential, const float* thin_prism) {
            const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= w * h)
                return;
            const float identity[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
            const auto ray = from_world_pixel_ray_generic(model, ShutterType::GLOBAL, w, h,
                                                          float(i % w) + 0.5f, float(i / w) + 0.5f, identity, nullptr, K, 0, radial, tangential, thin_prism);
            const float scale = model == CameraModelType::PINHOLE ? 1.f / ray.ray_dir.z : 1.f;
            for (int c = 0; c < 3; ++c)
                rays[3 * i + c] = ray.valid_flag ? ray.ray_dir[c] * scale : nanf("");
        }
        __device__ mat3 camera_rotation(const float* v) {
            return mat3(v[0], v[4], v[8], v[1], v[5], v[9], v[2], v[6], v[10]);
        }
        __device__ int min_axis(const float* s) {
            return s[0] <= s[1] && s[0] <= s[2] ? 0 : s[1] <= s[2] ? 1
                                                                   : 2;
        }
        template <bool Backward>
        __global__ void features_kernel(const float* means, const float* quats, const float* scales,
                                        const float* views, float* features,
                                        const float* gf, float* gm, float* gq,
                                        uint32_t N, uint32_t C, uint32_t channels, CameraModelType /*model*/,
                                        const int32_t* radii) {
            if (channels != 8)
                return;
            const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= N * C)
                return;
            float gf0 = 0.f, gf1 = 0.f, gf2 = 0.f;
            if constexpr (!Backward) {
                if (radii != nullptr && (radii[2 * i] <= 0 || radii[2 * i + 1] <= 0))
                    return;
            } else {
                gf0 = gf[i * channels + 4];
                gf1 = gf[i * channels + 5];
                gf2 = gf[i * channels + 6];
                if (gf0 == 0.f && gf1 == 0.f && gf2 == 0.f)
                    return;
            }
            const uint32_t g = i % N;
            const float* v = views + (i / N) * 16;
            const mat3 V = camera_rotation(v);
            const vec3 p = V * vec3(means[3 * g], means[3 * g + 1], means[3 * g + 2]) + vec3(v[3], v[7], v[11]);
            (void)gm;
            const vec4 q(quats[4 * g], quats[4 * g + 1], quats[4 * g + 2], quats[4 * g + 3]);
            const int axis = min_axis(scales + 3 * g);
            const vec3 normal = V * quat_to_rotmat(q)[axis];
            const float sign = dot(normal, p) > 0 ? -1.f : 1.f;
            if constexpr (!Backward) {
                for (int c = 0; c < 3; ++c)
                    features[i * channels + 4 + c] = sign * normal[c];
            } else {
                const vec3 gn = sign * (transpose(V) * vec3(gf0, gf1, gf2));
                mat3 gR(0.f);
                gR[axis] = gn;
                vec4 grad(0.f);
                quat_to_rotmat_vjp(q, gR, grad);
                for (int c = 0; c < 4; ++c)
                    atomicAdd(gq + 4 * g + c, grad[c]);
            }
        }
        __global__ void flatten_kernel(const float* s, float* gs, uint32_t N, float weight, const float* blend_peak) {
            const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= N || (blend_peak && blend_peak[i] <= 0.0f))
                return;
            const int a = min_axis(s + 3 * i);
            gs[3 * i + a] += weight * expf(s[3 * i + a]) / N;
        }
    } // namespace
    void geometry_features_fwd(const float* m, const float* q, const float* s, const float* v,
                               float* f, uint32_t N, uint32_t C, uint32_t channels,
                               CameraModelType model, cudaStream_t stream,
                               const int32_t* radii) {
        if (!N || !C)
            return;
        features_kernel<false><<<(N * C + 255) / 256, 256, 0, stream>>>(m, q, s, v, f, nullptr, nullptr, nullptr, N, C, channels, model, radii);
        LFS_CUDA_LAUNCH_CHECK(stream, "gut.geometry.forward");
    }
    void geometry_features_bwd(const float* m, const float* q, const float* s, const float* v,
                               const float* gf, float* gm, float* gq,
                               uint32_t N, uint32_t C, uint32_t channels, CameraModelType model, cudaStream_t stream) {
        if (!N || !C)
            return;
        features_kernel<true><<<(N * C + 255) / 256, 256, 0, stream>>>(m, q, s, v, nullptr, gf, gm, gq, N, C, channels, model, nullptr);
        LFS_CUDA_LAUNCH_CHECK(stream, "gut.geometry.backward");
    }
    void flatten_scale_grad(const float* s, float* gs, uint32_t N, float weight, cudaStream_t stream,
                            const float* blend_peak) {
        if (!N || weight <= 0.f)
            return;
        flatten_kernel<<<(N + 255) / 256, 256, 0, stream>>>(s, gs, N, weight, blend_peak);
        LFS_CUDA_LAUNCH_CHECK(stream, "gut.geometry.flatten");
    }
    void geometry_camera_rays(float* rays, uint32_t w, uint32_t h, const float* K,
                              CameraModelType model, const float* radial, const float* tangential,
                              const float* thin_prism, cudaStream_t stream) {
        if (!w || !h)
            return;
        rays_kernel<<<(w * h + 255) / 256, 256, 0, stream>>>(rays, w, h, K, model, radial, tangential, thin_prism);
        LFS_CUDA_LAUNCH_CHECK(stream, "gut.geometry.rays");
    }
    void expected_depth_bwd(const float* depth, const float* alpha, float* gd, float* ga, uint32_t pixels, cudaStream_t stream) {
        if (!pixels)
            return;
        expected_depth_kernel<<<(pixels + 255) / 256, 256, 0, stream>>>(depth, alpha, gd, ga, pixels);
        LFS_CUDA_LAUNCH_CHECK(stream, "gut.geometry.expected_depth");
    }
} // namespace gsplat_lfs
