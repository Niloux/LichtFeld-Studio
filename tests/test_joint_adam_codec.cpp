/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * joint (u, log_s) Adam codec unit tests.
 *
 * Host-side encode/decode, zero fixed point, bounds reduction, and
 * optimizer-step trajectory error vs fp32 reference (joint <= legacy).
 */

#include "adam_api.h"
#include "core/tensor.hpp"
#include "lfs/training/joint_adam_codec.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace lfs::training::joint_adam;

namespace {

    // Legacy uint8 m + sqrt(v) codec (matches adam_kernels.cuh) for SNR comparison.
    constexpr uint8_t kLegacyZeroPoint = 128;

    float legacy_dequant_m(const uint8_t q, const float scale) {
        return scale == 0.0f ? 0.0f : (static_cast<int>(q) - static_cast<int>(kLegacyZeroPoint)) * scale;
    }
    uint8_t legacy_quantize_m(const float value, const float scale) {
        if (scale == 0.0f)
            return kLegacyZeroPoint;
        const int q = static_cast<int>(std::round(value / scale)) + static_cast<int>(kLegacyZeroPoint);
        return static_cast<uint8_t>(std::min(255, std::max(0, q)));
    }
    float legacy_dequant_sqrt_v(const uint8_t q, const float scale) {
        return scale == 0.0f ? 0.0f : static_cast<float>(q) * scale;
    }
    uint8_t legacy_quantize_sqrt_v(const float v, const float scale) {
        if (scale == 0.0f)
            return 0;
        const float s = std::sqrt(std::max(v, 0.0f));
        int q = static_cast<int>(std::round(s / scale));
        if (v > 0.0f && q == 0)
            q = 1;
        return static_cast<uint8_t>(std::min(255, std::max(0, q)));
    }

    struct Moments {
        std::vector<float> m;
        std::vector<float> v;
    };

    void adam_step_fp32(Moments& st, const std::vector<float>& grad,
                        float beta1, float beta2, float lr, float eps, int step) {
        const float bc1 = 1.0f - std::pow(beta1, static_cast<float>(step));
        const float bc2 = 1.0f - std::pow(beta2, static_cast<float>(step));
        const float step_size = lr / bc1;
        const float bc2_sqrt_rcp = 1.0f / std::sqrt(bc2);
        for (size_t i = 0; i < st.m.size(); ++i) {
            st.m[i] = beta1 * st.m[i] + (1.0f - beta1) * grad[i];
            st.v[i] = beta2 * st.v[i] + (1.0f - beta2) * grad[i] * grad[i];
            // param update not stored — we track moment fidelity only
            (void)step_size;
            (void)bc2_sqrt_rcp;
            (void)eps;
        }
    }

    // One encode/decode roundtrip of moments with joint codec (block = all cells).
    template <int BITS>
    Moments joint_roundtrip(const Moments& in) {
        using C = Codec<BITS>;
        const size_t n = in.m.size();
        float bounds[4];
        C::reduce_bounds(in.m.data(), in.v.data(), n, bounds);
        std::vector<uint8_t> packed(n * C::kBytesPerCell, 0);
        for (size_t i = 0; i < n; ++i) {
            C::encode_g1g2(packed.data(), i, in.m[i], in.v[i],
                           bounds[0], bounds[1], bounds[2], bounds[3]);
        }
        Moments out;
        out.m.resize(n);
        out.v.resize(n);
        for (size_t i = 0; i < n; ++i) {
            C::decode_g1g2(packed.data(), i, bounds[0], bounds[1], bounds[2], bounds[3],
                           out.m[i], out.v[i]);
        }
        return out;
    }

    Moments legacy_roundtrip(const Moments& in) {
        const size_t n = in.m.size();
        float max_abs_m = 0.0f, max_v = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            max_abs_m = std::max(max_abs_m, std::abs(in.m[i]));
            max_v = std::max(max_v, std::max(in.v[i], 0.0f));
        }
        const float m_scale = max_abs_m > 0.0f ? max_abs_m / 127.0f : 0.0f;
        const float v_scale = max_v > 0.0f ? std::sqrt(max_v) / 255.0f : 0.0f;
        std::vector<uint8_t> mq(n), vq(n);
        for (size_t i = 0; i < n; ++i) {
            mq[i] = legacy_quantize_m(in.m[i], m_scale);
            vq[i] = legacy_quantize_sqrt_v(in.v[i], v_scale);
        }
        Moments out;
        out.m.resize(n);
        out.v.resize(n);
        for (size_t i = 0; i < n; ++i) {
            out.m[i] = legacy_dequant_m(mq[i], m_scale);
            const float s = legacy_dequant_sqrt_v(vq[i], v_scale);
            out.v[i] = s * s;
        }
        return out;
    }

    double mse(const std::vector<float>& a, const std::vector<float>& b) {
        double s = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
            s += d * d;
        }
        return s / static_cast<double>(a.size());
    }

} // namespace

TEST(JointAdamCodecTest, ZeroFixedPointDecodesToZero) {
    constexpr size_t n = 16;
    std::vector<uint8_t> packed16(n * Codec16::kBytesPerCell, 0);
    std::vector<uint8_t> packed8(n * Codec8::kBytesPerCell, 0);
    for (size_t i = 0; i < n; ++i) {
        float g1 = 1.0f, g2 = 1.0f;
        Codec16::decode_g1g2(packed16.data(), i, 0, 0, 0, 0, g1, g2);
        EXPECT_FLOAT_EQ(g1, 0.0f);
        EXPECT_FLOAT_EQ(g2, 0.0f);
        Codec8::decode_g1g2(packed8.data(), i, 0, 0, 0, 0, g1, g2);
        EXPECT_FLOAT_EQ(g1, 0.0f);
        EXPECT_FLOAT_EQ(g2, 0.0f);
    }
}

namespace {
    template <int Bits>
    void check_mixed_block_zero_moments() {
        using C = Codec<Bits>;
        std::vector<uint8_t> packed(C::kBytesPerCell);
        // This is also the pre-fix encoding: no new sentinel or codebook.
        C::encode_us(packed.data(), 0, 0.0f, 0.0f, -1.0f, 1.0f, 0.0f, 25.0f);
        float u, s, m, v;
        C::decode_us(packed.data(), 0, -1.0f, 1.0f, 0.0f, 25.0f, u, s);
        ASSERT_NE(u, 0.0f); // Zero lies between the two central u codes.
        ASSERT_EQ(s, 0.0f);
        C::decode_g1g2(packed.data(), 0, -1.0f, 1.0f, 0.0f, 25.0f, m, v);
        EXPECT_EQ(m, 0.0f);
        EXPECT_EQ(v, 0.0f);

        // A zero s code with positive smin still carries real variance/history.
        C::encode_us(packed.data(), 0, 0.5f, 1.0f, -1.0f, 1.0f, 1.0f, 25.0f);
        C::decode_g1g2(packed.data(), 0, -1.0f, 1.0f, 1.0f, 25.0f, m, v);
        EXPECT_GT(m, 0.0f);
        EXPECT_GT(v, 0.0f);
        EXPECT_LT(v, 1e-28f); // No positive variance floor.

        // Near-zero u with nonzero variance must retain the usual reconstruction.
        C::decode_us(packed.data(), 0, -1.0f, 1.0f, 1.0f, 25.0f, u, s);
        EXPECT_FLOAT_EQ(m, u * (C::inverse_sqrt_g2(s) + kEps));
        C::us_to_g1g2(1e-12f, 1.0f, m, v);
        EXPECT_GT(m, 0.0f);
        EXPECT_GT(v, 0.0f);
    }
} // namespace

TEST(JointAdamCodecTest, MixedSignBlockPreservesZero16Bit) {
    check_mixed_block_zero_moments<16>();
}

TEST(JointAdamCodecTest, MixedSignBlockPreservesZero8Bit) {
    check_mixed_block_zero_moments<8>();
}

TEST(JointAdamCodecTest, EndpointExactRoundtrip) {
    // Single cell: encode endpoints of a range, decode must hit lo/hi exactly for u,log_s.
    float g1s[2] = {-0.5f, 1.25f};
    float g2s[2] = {1e-10f, 4.0f};
    float bounds[4];
    Codec16::reduce_bounds(g1s, g2s, 2, bounds);

    std::vector<uint8_t> packed(2 * Codec16::kBytesPerCell, 0);
    Codec16::encode_g1g2(packed.data(), 0, g1s[0], g2s[0], bounds[0], bounds[1], bounds[2], bounds[3]);
    Codec16::encode_g1g2(packed.data(), 1, g1s[1], g2s[1], bounds[0], bounds[1], bounds[2], bounds[3]);

    float u0, s0, u1, s1;
    Codec16::decode_us(packed.data(), 0, bounds[0], bounds[1], bounds[2], bounds[3], u0, s0);
    Codec16::decode_us(packed.data(), 1, bounds[0], bounds[1], bounds[2], bounds[3], u1, s1);

    // Endpoint-exact: min/max of decoded (u,log_s) must recover bounds.
    EXPECT_NEAR(std::min(u0, u1), bounds[0], 1e-6f);
    EXPECT_NEAR(std::max(u0, u1), bounds[1], 1e-6f);
    EXPECT_NEAR(std::min(s0, s1), bounds[2], 1e-6f);
    EXPECT_NEAR(std::max(s0, s1), bounds[3], 1e-6f);
}

TEST(JointAdamCodecTest, RoundtripErrorBounded) {
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 0.1f);
    std::uniform_real_distribution<float> uv(1e-12f, 1e-2f);

    Moments ref;
    ref.m.resize(64);
    ref.v.resize(64);
    for (size_t i = 0; i < 64; ++i) {
        ref.m[i] = nd(rng);
        ref.v[i] = uv(rng);
    }

    const auto j16 = joint_roundtrip<16>(ref);
    const auto j8 = joint_roundtrip<8>(ref);
    const double mse_m16 = mse(ref.m, j16.m);
    const double mse_m8 = mse(ref.m, j8.m);
    // 16-bit should be tight; 8-bit coarser but finite.
    EXPECT_LT(mse_m16, 1e-6);
    EXPECT_LT(mse_m8, 1e-2);
    EXPECT_LT(mse_m16, mse_m8);
}

TEST(JointAdamCodecTest, BoundsReductionMatchesManual) {
    const float g1[] = {0.1f, -0.2f, 0.0f};
    const float g2[] = {1e-8f, 1e-4f, 0.0f};
    float bounds[4];
    Codec16::reduce_bounds(g1, g2, 3, bounds);

    float umin = 1e30f, umax = -1e30f, smin = 1e30f, smax = -1e30f;
    for (int i = 0; i < 3; ++i) {
        float u, s;
        Codec16::g1g2_to_us(g1[i], g2[i], u, s);
        umin = std::min(umin, u);
        umax = std::max(umax, u);
        smin = std::min(smin, s);
        smax = std::max(smax, s);
    }
    EXPECT_FLOAT_EQ(bounds[0], umin);
    EXPECT_FLOAT_EQ(bounds[1], umax);
    EXPECT_FLOAT_EQ(bounds[2], smin);
    EXPECT_FLOAT_EQ(bounds[3], smax);
}

TEST(JointAdamCodecTest, TrajectoryErrorNotWorseThanLegacy) {
    // Synthetic Adam moment trajectory: joint 16-bit roundtrip each step vs legacy.
    // Compare MSE of moments vs fp32 reference after K steps; joint must be <= legacy * 1.05.
    constexpr int kSteps = 50;
    constexpr size_t kN = 32;
    constexpr float beta1 = 0.9f, beta2 = 0.999f, lr = 1e-3f, eps = 1e-15f;

    std::mt19937 rng(7);
    std::normal_distribution<float> gd(0.0f, 0.05f);

    Moments ref, joint_st, legacy_st;
    ref.m.assign(kN, 0.0f);
    ref.v.assign(kN, 0.0f);
    joint_st = ref;
    legacy_st = ref;

    double joint_err = 0.0, legacy_err = 0.0;
    for (int step = 1; step <= kSteps; ++step) {
        std::vector<float> grad(kN);
        for (size_t i = 0; i < kN; ++i)
            grad[i] = gd(rng);

        adam_step_fp32(ref, grad, beta1, beta2, lr, eps, step);
        adam_step_fp32(joint_st, grad, beta1, beta2, lr, eps, step);
        adam_step_fp32(legacy_st, grad, beta1, beta2, lr, eps, step);

        // Requantize each path after the step (codec noise).
        joint_st = joint_roundtrip<16>(joint_st);
        legacy_st = legacy_roundtrip(legacy_st);

        joint_err += mse(ref.m, joint_st.m) + mse(ref.v, joint_st.v);
        legacy_err += mse(ref.m, legacy_st.m) + mse(ref.v, legacy_st.v);
    }

    // Joint (u,log_s) is designed for better small-v SNR; require not worse than legacy.
    EXPECT_LE(joint_err, legacy_err * 1.05 + 1e-12)
        << "joint_err=" << joint_err << " legacy_err=" << legacy_err;
}

TEST(JointAdamCodecTest, OptimBpsConstantsMatchFootprint) {
    EXPECT_EQ(kOptimBpsLegacySh3, 172u);
    EXPECT_EQ(kOptimBpsJointSh3, 152u); // LFS swizzled SH pad: 48 cells × 2 B
    EXPECT_LT(kOptimBpsJointSh3, kOptimBpsLegacySh3);
}

TEST(JointAdamCodecTest, StochasticRoundingTracksDecayAndPreservesEndpoints) {
    // 8-bit log_s under repeated zero-gradient decay (beta2 = 0.999) stalls under
    // round-to-nearest once per-step change < 0.5 bin, whereas SR remains unbiased.
    constexpr int kCells = 256;
    constexpr int kSteps = 500;
    constexpr float kBeta1 = 0.9f;
    constexpr float kBeta2 = 0.999f;

    // Pin the min and max anchor cells so the block bounds stay realistic while
    // interior decaying cells have per-step deltas smaller than half an 8-bit bin.
    std::vector<float> ref_m(kCells, 0.0f);
    std::vector<float> ref_v(kCells, 0.0f);
    for (int i = 0; i < kCells; ++i) {
        const float s = 1e-4f + 1e-2f * static_cast<float>(i) / static_cast<float>(kCells - 1);
        ref_m[i] = (i % 2 == 0 ? 0.02f : -0.02f) * (1.0f + 0.01f * static_cast<float>(i));
        ref_v[i] = s * s;
    }
    auto run_decay_step = [&](std::vector<uint8_t>& packed, float bounds[4],
                              std::vector<float>& ref_m_local, std::vector<float>& ref_v_local,
                              const uint32_t seed) {
        std::vector<float> u(kCells), s(kCells);
        float umin = 1e30f, umax = -1e30f, smin = 1e30f, smax = -1e30f;
        for (int i = 0; i < kCells; ++i) {
            float m = 0.0f, v = 0.0f;
            Codec8::decode_g1g2(packed.data(), i, bounds[0], bounds[1], bounds[2], bounds[3], m, v);
            if (i != 0 && i != kCells - 1) {
                m *= kBeta1;
                v *= kBeta2;
            }
            Codec8::g1g2_to_us(m, v, u[i], s[i]);
            umin = std::min(umin, u[i]);
            umax = std::max(umax, u[i]);
            smin = std::min(smin, s[i]);
            smax = std::max(smax, s[i]);
        }
        bounds[0] = umin;
        bounds[1] = umax;
        bounds[2] = smin;
        bounds[3] = smax;
        for (int i = 0; i < kCells; ++i) {
            Codec8::encode_us_sr(packed.data(), i, u[i], s[i], umin, umax, smin, smax, seed);
            if (i != 0 && i != kCells - 1) {
                ref_m_local[i] *= kBeta1;
                ref_v_local[i] *= kBeta2;
            }
        }
    };

    auto init_packed = [&](std::vector<uint8_t>& packed, float bounds[4]) {
        Codec8::reduce_bounds(ref_m.data(), ref_v.data(), kCells, bounds);
        packed.assign(kCells * Codec8::kBytesPerCell, 0);
        for (int i = 0; i < kCells; ++i) {
            Codec8::encode_g1g2(packed.data(), i, ref_m[i], ref_v[i], bounds[0], bounds[1], bounds[2], bounds[3]);
        }
    };

    std::vector<uint8_t> packed_sr, packed_rtn;
    float bounds_sr[4], bounds_rtn[4];
    init_packed(packed_sr, bounds_sr);
    init_packed(packed_rtn, bounds_rtn);

    std::vector<float> ref_m_sr = ref_m, ref_v_sr = ref_v;
    std::vector<float> ref_m_rtn = ref_m, ref_v_rtn = ref_v;
    for (int step = 1; step <= kSteps; ++step) {
        run_decay_step(packed_sr, bounds_sr, ref_m_sr, ref_v_sr, static_cast<uint32_t>(step));
        run_decay_step(packed_rtn, bounds_rtn, ref_m_rtn, ref_v_rtn, 0u);
    }

    double sum_ref = 0.0, sum_sr = 0.0, sum_rtn = 0.0;
    for (int i = 1; i < kCells - 1; ++i) {
        float m, v;
        Codec8::decode_g1g2(packed_sr.data(), i, bounds_sr[0], bounds_sr[1], bounds_sr[2], bounds_sr[3], m, v);
        sum_ref += ref_v_sr[i];
        sum_sr += v;
        Codec8::decode_g1g2(packed_rtn.data(), i, bounds_rtn[0], bounds_rtn[1], bounds_rtn[2], bounds_rtn[3], m, v);
        sum_rtn += v;
    }
    const double rel_err_sr = std::abs(sum_sr - sum_ref) / sum_ref;
    const double rel_err_rtn = std::abs(sum_rtn - sum_ref) / sum_ref;
    EXPECT_LT(rel_err_sr, 0.08) << "sum_sr=" << sum_sr << " sum_ref=" << sum_ref;
    EXPECT_LT(rel_err_sr, rel_err_rtn) << "sr_err=" << rel_err_sr << " rtn_err=" << rel_err_rtn;

    // Exact endpoint clamping: lo and hi map to 0 and kQMax regardless of seed.
    std::vector<uint8_t> endpoint_packed(2 * Codec8::kBytesPerCell, 0);
    Codec8::encode_us_sr(endpoint_packed.data(), 0, -2.0f, 1.0f, -2.0f, 3.0f, 1.0f, 5.0f, 12345u);
    Codec8::encode_us_sr(endpoint_packed.data(), 1, 3.0f, 5.0f, -2.0f, 3.0f, 1.0f, 5.0f, 12345u);
    EXPECT_EQ(endpoint_packed[0], 0u);
    EXPECT_EQ(endpoint_packed[1], 0u);
    EXPECT_EQ(endpoint_packed[2], 255u);
    EXPECT_EQ(endpoint_packed[3], 255u);
}

TEST(JointAdamCodecTest, PerBlockLazyStepAndFlushMatchDenseJointAdam) {
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::Tensor;

    constexpr int kN = 512; // Two 256-splat blocks: block 0 active, block 1 intermittently inactive.
    constexpr int kAttr = 3;
    constexpr int kBlocks = 2;
    constexpr int kTotalSteps = 12;
    constexpr float kLr = 1e-3f;
    constexpr float kBeta1 = 0.9f;
    constexpr float kBeta2 = 0.999f;
    constexpr float kEps = 1e-15f;

    std::vector<float> init_param(kN * kAttr);
    for (int i = 0; i < kN * kAttr; ++i) {
        init_param[i] = 0.1f * std::sin(0.07f * static_cast<float>(i + 1));
    }

    auto make_buf = [&](const std::vector<float>& vals) {
        return Tensor::from_vector(vals, {static_cast<size_t>(kN), static_cast<size_t>(kAttr)}, Device::CUDA);
    };

    Tensor p_dense = make_buf(init_param);
    Tensor m_dense = Tensor::zeros({static_cast<size_t>(kN), static_cast<size_t>(kAttr * 4)}, Device::CUDA, DataType::UInt8);
    Tensor b_dense = Tensor::zeros({static_cast<size_t>(kBlocks), size_t{4}}, Device::CUDA);

    Tensor p_lazy = make_buf(init_param);
    Tensor m_lazy = Tensor::zeros({static_cast<size_t>(kN), static_cast<size_t>(kAttr * 4)}, Device::CUDA, DataType::UInt8);
    Tensor b_lazy = Tensor::zeros({static_cast<size_t>(kBlocks), size_t{4}}, Device::CUDA);
    Tensor lazy_step = Tensor::zeros({static_cast<size_t>(kBlocks)}, Device::CUDA, DataType::Int32);

    std::vector<float> host_grad(kN * kAttr);
    std::vector<float> host_bp(kN);

    for (int step = 1; step <= kTotalSteps; ++step) {
        const bool block1_active = (step <= 5) || (step == 11);
        for (int prim = 0; prim < kN; ++prim) {
            const bool active = (prim < 256) || block1_active;
            host_bp[prim] = active ? 0.25f : 0.0f;
            for (int a = 0; a < kAttr; ++a) {
                host_grad[prim * kAttr + a] =
                    active ? (0.01f * std::cos(0.11f * static_cast<float>(prim + step * 7 + a))) : 0.0f;
            }
        }
        Tensor g_dense = make_buf(host_grad);
        Tensor g_lazy = make_buf(host_grad);
        Tensor bp = Tensor::from_vector(host_bp, {static_cast<size_t>(kN)}, Device::CUDA);

        const float bc1_rcp = 1.0f / (1.0f - std::pow(kBeta1, step));
        const float bc2_sqrt_rcp = 1.0f / std::sqrt(1.0f - std::pow(kBeta2, step));

        fast_lfs::optimizer::JointContiguousBatchEntry e_dense{};
        e_dense.param = p_dense.ptr<float>();
        e_dense.packed = m_dense.ptr<uint8_t>();
        e_dense.bounds = b_dense.ptr<float>();
        e_dense.grad = g_dense.ptr<float>();
        e_dense.n_prims = kN;
        e_dense.n_attr = kAttr;
        e_dense.lr = kLr;
        e_dense.bias_correction1_rcp = bc1_rcp;
        e_dense.bias_correction2_sqrt_rcp = bc2_sqrt_rcp;
        e_dense.zero_grad = 1;
        e_dense.step_count = step;

        fast_lfs::optimizer::JointContiguousBatchEntry e_lazy = e_dense;
        e_lazy.param = p_lazy.ptr<float>();
        e_lazy.packed = m_lazy.ptr<uint8_t>();
        e_lazy.bounds = b_lazy.ptr<float>();
        e_lazy.grad = g_lazy.ptr<float>();
        e_lazy.lazy_step = lazy_step.ptr<int32_t>();
        e_lazy.allow_lazy = 1;

        fast_lfs::optimizer::GutRegularizationArgs gut_reg{};
        gut_reg.blend_peak = bp.ptr<float>();

        fast_lfs::optimizer::adam_step_joint_contiguous_batched(
            &e_dense, 1, nullptr, 0, 0.0f, nullptr, 0, 1.0f, kBeta1, kBeta2, kEps, nullptr,
            nullptr, 0, 0.0f, nullptr, 0, nullptr, 0, 0.0f, 0.0f, gut_reg, false);
        fast_lfs::optimizer::adam_step_joint_contiguous_batched(
            &e_lazy, 1, nullptr, 0, 0.0f, nullptr, 0, 1.0f, kBeta1, kBeta2, kEps, nullptr,
            nullptr, 0, 0.0f, nullptr, 0, nullptr, 0, 0.0f, 0.0f, gut_reg, false);
    }

    fast_lfs::optimizer::JointContiguousBatchEntry e_flush{};
    e_flush.param = p_lazy.ptr<float>();
    e_flush.packed = m_lazy.ptr<uint8_t>();
    e_flush.bounds = b_lazy.ptr<float>();
    e_flush.n_prims = kN;
    e_flush.n_attr = kAttr;
    e_flush.lr = kLr;
    e_flush.bias_correction1_rcp = 1.0f / (1.0f - std::pow(kBeta1, kTotalSteps));
    e_flush.bias_correction2_sqrt_rcp = 1.0f / std::sqrt(1.0f - std::pow(kBeta2, kTotalSteps));
    e_flush.lazy_step = lazy_step.ptr<int32_t>();
    e_flush.step_count = kTotalSteps;
    fast_lfs::optimizer::adam_step_joint_contiguous_batched(
        &e_flush, 1, nullptr, 0, 0.0f, nullptr, 0, 1.0f, kBeta1, kBeta2, kEps, nullptr,
        nullptr, 0, 0.0f, nullptr, 0, nullptr, 0, 0.0f, 0.0f, {}, /*flush_only=*/true);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto out_dense = p_dense.cpu().to_vector();
    const auto out_lazy = p_lazy.cpu().to_vector();
    const auto steps_cpu = lazy_step.cpu().to_vector_int();
    ASSERT_EQ(steps_cpu.size(), 2u);
    EXPECT_EQ(steps_cpu[0], kTotalSteps);
    EXPECT_EQ(steps_cpu[1], kTotalSteps);

    float max_abs_diff = 0.0f;
    for (size_t i = 0; i < out_dense.size(); ++i) {
        ASSERT_TRUE(std::isfinite(out_lazy[i])) << "i=" << i;
        max_abs_diff = std::max(max_abs_diff, std::abs(out_dense[i] - out_lazy[i]));
    }
    EXPECT_LT(max_abs_diff, 5e-4f) << "max_abs_diff=" << max_abs_diff;
}
