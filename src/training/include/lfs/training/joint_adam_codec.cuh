/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

/**
 * Device-side joint (u, log_s) Adam codec.
 * Mirrors host math in joint_adam_codec.hpp.
 */

#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::training::joint_adam {

    constexpr int kBlockSizeDevice = 256;
    constexpr float kEpsDevice = 1e-15f;

    // Stateless 64-bit SplitMix64-based hash returning uniform float in [0, 1).
    __device__ __forceinline__ float sr_uniform01(const uint64_t key, const uint32_t seed) {
        uint64_t x = key ^ (static_cast<uint64_t>(seed) * 0x9e3779b97f4a7c15ull);
        x ^= x >> 30;
        x *= 0xbf58476d1ce4e5b9ull;
        x ^= x >> 27;
        x *= 0x94d049bb133111ebull;
        x ^= x >> 31;
        return static_cast<float>(static_cast<uint32_t>(x >> 40)) * 5.9604644775390625e-8f;
    }

    template <int BITS>
    struct DeviceCodec {
        static_assert(BITS == 8 || BITS == 16, "joint Adam codec supports 8 or 16 bits");
        static constexpr int kBits = BITS;
        static constexpr float kQMax = static_cast<float>((1 << BITS) - 1);
        static constexpr float kInvQMax = 1.0f / kQMax;
        static constexpr int kBytesPerCell = (BITS == 16) ? 4 : 2;

        // Guarded fast transcode: __logf/__expf for the bulk of the range;
        // 5-term Horner polynomials near zero (< 1e-6 error, 0<->0 bit-exact)
        // to avoid log1pf/expm1f register pressure inside fused backward kernels.
        __device__ static __forceinline__ float forward_sqrt_g2(const float sqrt_g2) {
            const float x = fmaxf(sqrt_g2, 0.0f) * (1.0f / kEpsDevice);
            if (x > 0.125f)
                return __logf(1.0f + x);
            return x * (1.0f - x * (0.5f - x * (0.333333343f - x * (0.25f - 0.2f * x))));
        }
        __device__ static __forceinline__ float inverse_sqrt_g2(const float log_s) {
            float e1;
            if (log_s > 0.118f) {
                e1 = __expf(log_s) - 1.0f;
            } else {
                e1 = log_s * (1.0f + log_s * (0.5f + log_s * (0.166666672f + log_s * (0.041666668f + 0.008333334f * log_s))));
            }
            return kEpsDevice * e1;
        }

        __device__ static __forceinline__ float2 g1g2_to_us(const float g1, const float g2) {
            const float sqrt_g2 = sqrtf(fmaxf(g2, 0.0f));
            return make_float2(g1 / (sqrt_g2 + kEpsDevice), forward_sqrt_g2(sqrt_g2));
        }

        __device__ static inline float2 decode_us(const uint8_t* __restrict__ packed,
                                                  const int64_t idx,
                                                  const float4 mm) {
            float u_q, s_q;
            if constexpr (BITS == 16) {
                const auto* p = reinterpret_cast<const uint16_t*>(packed);
                u_q = static_cast<float>(p[idx * 2 + 0]);
                s_q = static_cast<float>(p[idx * 2 + 1]);
            } else {
                u_q = static_cast<float>(packed[idx * 2 + 0]);
                s_q = static_cast<float>(packed[idx * 2 + 1]);
            }
            return make_float2(mm.x + (mm.y - mm.x) * (u_q * kInvQMax),
                               mm.z + (mm.w - mm.z) * (s_q * kInvQMax));
        }

        __device__ static inline float2 decode_g1g2(const uint8_t* __restrict__ packed,
                                                    const int64_t idx,
                                                    const float4 mm) {
            const float2 prim = decode_us(packed, idx, mm);
            const float sqrt_g2 = inverse_sqrt_g2(prim.y);
            const float g2 = sqrt_g2 * sqrt_g2;
            // Match the host decoder: zero variance must not turn quantized
            // u=0 residue into momentum that Adam amplifies by 1/eps.
            const float g1 = (g2 == 0.0f) ? 0.0f : prim.x * (sqrt_g2 + kEpsDevice);
            return make_float2(g1, g2);
        }

        // Block-uniform ranges: caller hoists inv_u_range / inv_s_range once
        // per encode pass (2 fdiv → FMA per cell).
        __device__ static inline void encode_us(uint8_t* __restrict__ packed,
                                                const int64_t idx,
                                                const float u_val,
                                                const float log_s_val,
                                                const float umin,
                                                const float smin,
                                                const float inv_u_range,
                                                const float inv_s_range) {
            const float u_qf = fminf(fmaxf(roundf(kQMax * (u_val - umin) * inv_u_range), 0.0f), kQMax);
            const float s_qf = fminf(fmaxf(roundf(kQMax * (log_s_val - smin) * inv_s_range), 0.0f), kQMax);
            if constexpr (BITS == 16) {
                auto* p = reinterpret_cast<uint16_t*>(packed);
                p[idx * 2 + 0] = static_cast<uint16_t>(u_qf);
                p[idx * 2 + 1] = static_cast<uint16_t>(s_qf);
            } else {
                packed[idx * 2 + 0] = static_cast<uint8_t>(u_qf);
                packed[idx * 2 + 1] = static_cast<uint8_t>(s_qf);
            }
        }

        __device__ static inline void encode_us(uint8_t* __restrict__ packed,
                                                const int64_t idx,
                                                const float u_val,
                                                const float log_s_val,
                                                const float4 mm) {
            const float inv_u = 1.0f / fmaxf(mm.y - mm.x, kEpsDevice);
            const float inv_s = 1.0f / fmaxf(mm.w - mm.z, kEpsDevice);
            encode_us(packed, idx, u_val, log_s_val, mm.x, mm.z, inv_u, inv_s);
        }

        // Stochastic rounding variant of encode_us using stateless hash PRNG.
        // Preserves exact 0 and kQMax endpoints; sr_seed == 0 falls back to round-to-nearest.
        __device__ static inline void encode_us_sr(uint8_t* __restrict__ packed,
                                                   const int64_t idx,
                                                   const float u_val,
                                                   const float log_s_val,
                                                   const float umin,
                                                   const float smin,
                                                   const float inv_u_range,
                                                   const float inv_s_range,
                                                   const uint32_t sr_seed) {
            if (sr_seed == 0u) {
                encode_us(packed, idx, u_val, log_s_val, umin, smin, inv_u_range, inv_s_range);
                return;
            }
            const float u_scaled = fminf(fmaxf(kQMax * (u_val - umin) * inv_u_range, 0.0f), kQMax);
            const float s_scaled = fminf(fmaxf(kQMax * (log_s_val - smin) * inv_s_range, 0.0f), kQMax);
            const float ru = sr_uniform01(static_cast<uint64_t>(idx) << 1, sr_seed);
            const float rs = sr_uniform01((static_cast<uint64_t>(idx) << 1) | 1ull, sr_seed);
            const float u_qf = fminf(floorf(u_scaled + ru), kQMax);
            const float s_qf = fminf(floorf(s_scaled + rs), kQMax);
            if constexpr (BITS == 16) {
                auto* p = reinterpret_cast<uint16_t*>(packed);
                p[idx * 2 + 0] = static_cast<uint16_t>(u_qf);
                p[idx * 2 + 1] = static_cast<uint16_t>(s_qf);
            } else {
                packed[idx * 2 + 0] = static_cast<uint8_t>(u_qf);
                packed[idx * 2 + 1] = static_cast<uint8_t>(s_qf);
            }
        }
    };

} // namespace lfs::training::joint_adam
