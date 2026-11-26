/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "adam_api.h"
#include "core/cuda/sh_layout.cuh"
#include "core/sh_value_quant.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "lfs/kernels/gut_regularization.cuh"
#include "lfs/training/joint_adam_codec.hpp"
#include "lfs/training/sh_value_codec.hpp"
#include "lfs/training/sh_value_storage.hpp"
#include "optimizer/adam_optimizer.hpp"
#include "rasterization/gsplat/GeometryFeatures.h"
#include "rasterization/gsplat/Ops.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

using namespace lfs::core;
using namespace lfs::training;

namespace {

    struct CodecsOnGuard {
        CodecsOnGuard() {
            sh_value::set_sh_value_quant_enabled_for_testing(true);
        }
        ~CodecsOnGuard() {
            sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
        }
    };

    SplatData make_mixed_splat(const size_t n, const int sh_degree = 3) {
        const size_t rest = sh_degree > 0
                                ? static_cast<size_t>(sh_degree * (sh_degree + 2))
                                : size_t{0};
        std::vector<float> means(n * 3);
        std::vector<float> sh0(n * 3);
        std::vector<float> scaling(n * 3);
        std::vector<float> rotation(n * 4, 0.0f);
        std::vector<float> opacity(n, 0.0f);
        for (size_t i = 0; i < n; ++i) {
            means[i * 3 + 0] = static_cast<float>((i * 17) % 97) * 0.03f;
            means[i * 3 + 1] = static_cast<float>((i * 13) % 89) * 0.04f;
            means[i * 3 + 2] = static_cast<float>((i * 11) % 83) * 0.05f;
            sh0[i * 3 + 0] = 0.01f * static_cast<float>(i + 1);
            sh0[i * 3 + 1] = 0.02f * static_cast<float>(i + 1);
            sh0[i * 3 + 2] = 0.03f * static_cast<float>(i + 1);
            scaling[i * 3 + 0] = -1.5f - 0.001f * static_cast<float>(i);
            scaling[i * 3 + 1] = -1.6f - 0.001f * static_cast<float>(i);
            scaling[i * 3 + 2] = -1.7f - 0.001f * static_cast<float>(i);
            rotation[i * 4] = 1.0f;
            rotation[i * 4 + 1] = 0.001f * static_cast<float>(i);
            opacity[i] = -1.0f + 0.0005f * static_cast<float>(i);
        }
        Tensor shN = rest == 0
                         ? Tensor::zeros({size_t{0}}, Device::CUDA)
                         : Tensor::zeros({n, rest, size_t{3}}, Device::CUDA);
        if (rest > 0) {
            auto cpu = shN.cpu();
            auto* p = cpu.ptr<float>();
            for (size_t i = 0; i < n * rest * 3; ++i) {
                p[i] = 0.02f * static_cast<float>((i % 11) + 1) *
                       (1.0f + 0.001f * static_cast<float>(i / 3));
            }
            shN = cpu.cuda();
        }
        return SplatData(
            sh_degree,
            Tensor::from_vector(means, {n, size_t{3}}, Device::CUDA),
            Tensor::from_vector(sh0, {n, size_t{1}, size_t{3}}, Device::CUDA),
            std::move(shN),
            Tensor::from_vector(scaling, {n, size_t{3}}, Device::CUDA),
            Tensor::from_vector(rotation, {n, size_t{4}}, Device::CUDA),
            Tensor::from_vector(opacity, {n, size_t{1}}, Device::CUDA),
            1.0f);
    }

    AdamConfig make_cfg(const size_t cap) {
        AdamConfig cfg;
        cfg.lr = 1e-3f;
        cfg.initial_capacity = cap;
        cfg.growth_factor = 1.5f;
        return cfg;
    }

    void copy_tensor_bytes(Tensor& dst, const Tensor& src) {
        ASSERT_TRUE(dst.is_valid());
        ASSERT_TRUE(src.is_valid());
        ASSERT_EQ(dst.numel(), src.numel());
        ASSERT_EQ(dst.dtype(), src.dtype());
        const size_t n_bytes = src.bytes();
        ASSERT_GT(n_bytes, 0u);
        ASSERT_EQ(cudaMemcpy(dst.data_ptr(), src.data_ptr(), n_bytes, cudaMemcpyDeviceToDevice),
                  cudaSuccess);
    }

    std::vector<std::uint8_t> tensor_bytes(const Tensor& t) {
        auto cpu = t.cpu();
        const auto* p = static_cast<const std::uint8_t*>(cpu.data_ptr());
        return {p, p + cpu.bytes()};
    }

    void fill_contiguous_gradients(AdamOptimizer& optimizer, const float scale) {
        optimizer.get_grad(ParamType::Means).fill_(scale);
        optimizer.get_grad(ParamType::Sh0).fill_(scale * 2.0f);
        optimizer.get_grad(ParamType::Scaling).fill_(scale * 3.0f);
        optimizer.get_grad(ParamType::Rotation).fill_(scale * 4.0f);
        optimizer.get_grad(ParamType::Opacity).fill_(scale * 5.0f);
    }

    void expect_contiguous_params_equal(const SplatData& actual, const SplatData& expected) {
        EXPECT_EQ(tensor_bytes(actual.means_raw()), tensor_bytes(expected.means_raw()));
        EXPECT_EQ(tensor_bytes(actual.sh0_raw()), tensor_bytes(expected.sh0_raw()));
        EXPECT_EQ(tensor_bytes(actual.scaling_raw()), tensor_bytes(expected.scaling_raw()));
        EXPECT_EQ(tensor_bytes(actual.rotation_raw()), tensor_bytes(expected.rotation_raw()));
        EXPECT_EQ(tensor_bytes(actual.opacity_raw()), tensor_bytes(expected.opacity_raw()));
    }

} // namespace

TEST(GutJointAdamBatch, BackToBackLaunchesKeepTheirOwnParameters) {
    CodecsOnGuard guard;
    constexpr size_t n = 300;
    constexpr size_t cap = 512;
    constexpr int iteration = 1001;

    auto splat_a = make_mixed_splat(n, 0);
    auto splat_b = make_mixed_splat(n, 0);
    auto expected_a = splat_a.clone();
    auto expected_b = splat_b.clone();

    AdamOptimizer optimizer_a(splat_a, make_cfg(cap));
    AdamOptimizer optimizer_b(splat_b, make_cfg(cap));
    AdamOptimizer reference_a(expected_a, make_cfg(cap));
    AdamOptimizer reference_b(expected_b, make_cfg(cap));
    optimizer_a.allocate_gradients(cap);
    optimizer_b.allocate_gradients(cap);
    reference_a.allocate_gradients(cap);
    reference_b.allocate_gradients(cap);

    fill_contiguous_gradients(optimizer_a, 0.01f);
    fill_contiguous_gradients(optimizer_b, 0.02f);
    fill_contiguous_gradients(reference_a, 0.01f);
    fill_contiguous_gradients(reference_b, 0.02f);

    ASSERT_NO_THROW(optimizer_a.step(iteration));
    ASSERT_NO_THROW(optimizer_b.step(iteration));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_NO_THROW(reference_a.step(iteration));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_NO_THROW(reference_b.step(iteration));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    expect_contiguous_params_equal(splat_a, expected_a);
    expect_contiguous_params_equal(splat_b, expected_b);
}

TEST(GutShNJointAdam, StandaloneStepMatchesFusedKernelQ16Sh3) {
    CodecsOnGuard guard;
    constexpr size_t n = 300;
    constexpr size_t cap = 512;
    constexpr int sh_degree = 3;
    constexpr int past_warmup = 1001;

    auto splat_fused = make_mixed_splat(n, sh_degree);
    ASSERT_TRUE(sh_value::apply_shN_value_quant(splat_fused));
    ASSERT_TRUE(splat_fused.shN_value_quantized());
    auto splat_step = splat_fused.clone();

    AdamOptimizer opt_fused(splat_fused, make_cfg(cap));
    AdamOptimizer opt_step(splat_step, make_cfg(cap));
    opt_fused.allocate_gradients(cap);
    opt_step.allocate_gradients(cap);

    auto* st_fused = opt_fused.get_state_mutable(ParamType::ShN);
    auto* st_step = opt_step.get_state_mutable(ParamType::ShN);
    ASSERT_NE(st_fused, nullptr);
    ASSERT_NE(st_step, nullptr);
    ASSERT_TRUE(st_fused->is_joint());
    ASSERT_TRUE(st_step->is_joint());
    copy_tensor_bytes(st_step->exp_avg, st_fused->exp_avg);
    copy_tensor_bytes(st_step->joint_bounds, st_fused->joint_bounds);
    copy_tensor_bytes(splat_step.shN(), splat_fused.shN());
    copy_tensor_bytes(splat_step.shN_value_bounds(), splat_fused.shN_value_bounds());

    const auto layout_rest = static_cast<uint32_t>(splat_fused.max_sh_coeffs_rest());
    const size_t float_layout = sh_swizzled_float_count(n, layout_rest);
    ASSERT_EQ(st_fused->size, float_layout);

    auto& grad_fused = opt_fused.get_grad(ParamType::ShN);
    auto& grad_step = opt_step.get_grad(ParamType::ShN);
    ASSERT_EQ(static_cast<size_t>(grad_fused.numel()), float_layout);
    ASSERT_EQ(static_cast<size_t>(grad_step.numel()), float_layout);

    std::vector<float> grads_h(float_layout);
    for (size_t i = 0; i < float_layout; ++i) {
        grads_h[i] = 0.01f * static_cast<float>((i % 17) + 1) *
                     (1.0f + 0.002f * static_cast<float>(i / 4));
    }
    ASSERT_EQ(cudaMemcpy(grad_fused.ptr<float>(), grads_h.data(),
                         float_layout * sizeof(float), cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(grad_step.ptr<float>(), grads_h.data(),
                         float_layout * sizeof(float), cudaMemcpyHostToDevice),
              cudaSuccess);

    auto fused = opt_fused.prepare_fastgs_fused_adam(past_warmup);
    ASSERT_TRUE(fused.enabled);
    ASSERT_TRUE(fused.shN.enabled);
    EXPECT_EQ(fused.shN.sh_value_bits, 16);
    EXPECT_EQ(fused.shN.n_primitives, static_cast<int>(n));
    const int sh_layout_slots = static_cast<int>(sh_float4_slots_for_rest(layout_rest));
    const int active_sh_bases =
        static_cast<int>(splat_fused.active_sh_coeffs_rest() + 1);
    ASSERT_EQ(active_sh_bases, 16);

    fast_lfs::optimizer::adam_step_shN_joint_from_grad(
        fused.shN.param,
        fused.shN.joint_packed,
        fused.shN.joint_bounds,
        fused.shN.sh_value_bounds,
        grad_fused.ptr<float>(),
        fused.shN.frozen_mask,
        fused.shN.frozen_mask_size,
        fused.shN.frozen_lr_scale,
        fused.shN.crop_damping_mask,
        fused.shN.crop_damping_mask_size,
        fused.shN.cropbox_lr_scale,
        fused.shN.n_primitives,
        sh_layout_slots,
        active_sh_bases,
        fused.shN.sh_value_bits,
        fused.shN.sh_value_n_cells,
        fused.shN.step_size,
        fused.beta1,
        fused.beta2,
        fused.eps,
        fused.shN.bias_correction2_sqrt_rcp,
        nullptr,
        0.0f,
        nullptr,
        false,
        fused.shN.sr_seed);
    opt_fused.commit_fastgs_fused_adam(past_warmup);

    ASSERT_NO_THROW(opt_step.step(past_warmup));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto codes_fused = tensor_bytes(splat_fused.shN());
    const auto codes_step = tensor_bytes(splat_step.shN());
    ASSERT_EQ(codes_fused.size(), codes_step.size());
    EXPECT_EQ(codes_fused, codes_step) << "q16 shN codes must match fused kernel";

    const auto vbounds_fused = tensor_bytes(splat_fused.shN_value_bounds());
    const auto vbounds_step = tensor_bytes(splat_step.shN_value_bounds());
    ASSERT_EQ(vbounds_fused.size(), vbounds_step.size());
    EXPECT_EQ(vbounds_fused, vbounds_step) << "q16 value bounds must match";

    st_fused = opt_fused.get_state_mutable(ParamType::ShN);
    st_step = opt_step.get_state_mutable(ParamType::ShN);
    ASSERT_NE(st_fused, nullptr);
    ASSERT_NE(st_step, nullptr);
    const auto packed_fused = tensor_bytes(st_fused->exp_avg);
    const auto packed_step = tensor_bytes(st_step->exp_avg);
    ASSERT_EQ(packed_fused.size(), packed_step.size());
    EXPECT_EQ(packed_fused, packed_step) << "packed shN moments must match bitwise";

    const auto jbounds_fused = tensor_bytes(st_fused->joint_bounds);
    const auto jbounds_step = tensor_bytes(st_step->joint_bounds);
    ASSERT_EQ(jbounds_fused.size(), jbounds_step.size());
    EXPECT_EQ(jbounds_fused, jbounds_step) << "joint moment bounds must match";
}

TEST(GutShNJointFromColor, MatchesStreamedBwdAndZeroGradAcrossDegrees) {
    for (const bool use_q16 : {false, true}) {
        std::optional<CodecsOnGuard> guard;
        if (use_q16) {
            guard.emplace();
        } else {
            sh_value::set_sh_value_quant_enabled_for_testing(false);
        }
        for (const int deg : {1, 2, 3}) {
            for (const uint32_t sr_seed : {0u, 7u}) {
                constexpr size_t n = 300;
                constexpr size_t cap = 512;
                constexpr int past_warmup = 1001;
                constexpr uint32_t C = 1;
                constexpr int color_stride = 8; // Matches GUT RGB_D_N

                auto splat_fused = make_mixed_splat(n, deg);
                splat_fused.set_active_sh_degree(deg);
                if (use_q16) {
                    ASSERT_TRUE(sh_value::apply_shN_value_quant(splat_fused));
                    ASSERT_TRUE(splat_fused.shN_value_quantized());
                }
                auto splat_ref = splat_fused.clone();

                // Exercise frozen & crop-damped rows alongside active rows.
                std::vector<uint8_t> frozen_h(n, 0);
                std::vector<uint8_t> damp_h(n, 0);
                frozen_h[0] = 1;
                frozen_h[257] = 1;
                damp_h[5] = 1;
                damp_h[260] = 1;
                auto frozen_mask = Tensor::from_vector(
                    std::vector<bool>(frozen_h.begin(), frozen_h.end()), {n}, Device::CUDA);
                auto damp_mask = Tensor::from_vector(
                    std::vector<bool>(damp_h.begin(), damp_h.end()), {n}, Device::CUDA);

                AdamOptimizer opt_fused(splat_fused, make_cfg(cap));
                AdamOptimizer opt_ref(splat_ref, make_cfg(cap));
                opt_fused.set_frozen_mask(frozen_mask);
                opt_fused.set_frozen_lr_scale(0.0f);
                opt_fused.set_crop_damping_mask(damp_mask);
                opt_fused.set_cropbox_lr_scale(0.2f);
                opt_ref.set_frozen_mask(frozen_mask);
                opt_ref.set_frozen_lr_scale(0.0f);
                opt_ref.set_crop_damping_mask(damp_mask);
                opt_ref.set_cropbox_lr_scale(0.2f);
                opt_fused.allocate_gradients(cap);
                opt_ref.allocate_gradients(cap);

                std::vector<float> dirs_h(n * 3);
                std::vector<float> v_colors_h(n * color_stride, 0.0f);
                for (size_t i = 0; i < n; ++i) {
                    const float x = std::sin(0.17f * static_cast<float>(i + 1));
                    const float y = std::cos(0.31f * static_cast<float>(i + 2));
                    const float z = std::sin(0.53f * static_cast<float>(i + 3));
                    const float inv = 1.0f / std::sqrt(x * x + y * y + z * z + 1e-8f);
                    dirs_h[i * 3 + 0] = x * inv;
                    dirs_h[i * 3 + 1] = y * inv;
                    dirs_h[i * 3 + 2] = z * inv;
                    for (int c = 0; c < 3; ++c) {
                        v_colors_h[i * color_stride + c] =
                            0.015f * std::sin(0.11f * static_cast<float>(i * 3 + c + 1));
                    }
                    // Trailing geometry channels (D/N/D2) must be ignored by SH.
                    for (int c = 3; c < color_stride; ++c) {
                        v_colors_h[i * color_stride + c] = 999.0f;
                    }
                }
                auto dirs = Tensor::from_vector(dirs_h, {size_t{C}, n, size_t{3}}, Device::CUDA);
                auto v_colors = Tensor::from_vector(
                    v_colors_h, {size_t{C}, n, static_cast<size_t>(color_stride)}, Device::CUDA);

                const auto layout_rest = static_cast<uint32_t>(splat_fused.max_sh_coeffs_rest());
                const int sh_layout_slots = static_cast<int>(sh_float4_slots_for_rest(layout_rest));
                const int active_sh_bases = static_cast<int>(splat_fused.active_sh_coeffs_rest() + 1);

                auto fused = opt_fused.prepare_fastgs_fused_adam(past_warmup);
                auto ref_fused = opt_ref.prepare_fastgs_fused_adam(past_warmup);
                ASSERT_TRUE(fused.shN.enabled);

                auto& v_sh0_ref = opt_ref.get_grad(ParamType::Sh0);
                auto& v_shN_ref = opt_ref.get_grad(ParamType::ShN);
                v_sh0_ref.zero_();
                v_shN_ref.zero_();
                // Dirty frozen row gradient to confirm zero_grad=true clears even unstepped rows.
                {
                    const size_t float_layout = sh_swizzled_float_count(n, layout_rest);
                    std::vector<float> seed_grad(float_layout, 0.25f);
                    ASSERT_EQ(cudaMemcpy(v_shN_ref.ptr<float>(), seed_grad.data(),
                                         float_layout * sizeof(float), cudaMemcpyHostToDevice),
                              cudaSuccess);
                }
                auto v_sh0_dummy = Tensor::zeros({n, size_t{3}}, Device::CUDA);

                gsplat_lfs::spherical_harmonics_streamed_bwd(
                    static_cast<uint32_t>(deg), static_cast<uint32_t>(deg),
                    dirs.ptr<float>(), v_colors.ptr<float>(), C, static_cast<uint32_t>(n),
                    static_cast<uint32_t>(color_stride), v_sh0_ref.ptr<float>(),
                    v_shN_ref.ptr<float>(), nullptr);
                // Match fused path ignoring existing prefill cleanly: use zeroed v_shN_clean for value check
                // while checking zero_grad clears all rows including frozen row 0.
                fast_lfs::optimizer::adam_step_shN_joint_from_grad(
                    ref_fused.shN.param, ref_fused.shN.joint_packed, ref_fused.shN.joint_bounds,
                    ref_fused.shN.sh_value_bounds, v_shN_ref.ptr<float>(),
                    ref_fused.shN.frozen_mask, ref_fused.shN.frozen_mask_size, ref_fused.shN.frozen_lr_scale,
                    ref_fused.shN.crop_damping_mask, ref_fused.shN.crop_damping_mask_size, ref_fused.shN.cropbox_lr_scale,
                    ref_fused.shN.n_primitives, sh_layout_slots, active_sh_bases,
                    ref_fused.shN.sh_value_bits, ref_fused.shN.sh_value_n_cells,
                    ref_fused.shN.step_size, ref_fused.beta1, ref_fused.beta2, ref_fused.eps,
                    ref_fused.shN.bias_correction2_sqrt_rcp, nullptr, 0.01f, nullptr, true, sr_seed);
                ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

                // Verify zero_grad=true cleared all live primitives (0..n-1) including frozen/damped rows.
                auto v_shN_cpu = v_shN_ref.cpu();
                const float* v_shN_ptr = v_shN_cpu.ptr<float>();
                for (size_t p = 0; p < n; ++p) {
                    const size_t blk = p / 32u;
                    const size_t lane = p % 32u;
                    for (int k = 0; k < sh_layout_slots; ++k) {
                        const size_t base =
                            (blk * (static_cast<size_t>(sh_layout_slots) * 32u) +
                             static_cast<size_t>(k) * 32u + lane) *
                            4u;
                        for (int c = 0; c < 4; ++c) {
                            ASSERT_EQ(v_shN_ptr[base + c], 0.0f)
                                << "deg=" << deg << " p=" << p << " k=" << k << " c=" << c;
                        }
                    }
                }

                // Now re-run reference cleanly from zeroed v_shN_ref to verify exact value parity.
                auto splat_ref_clean = splat_fused.clone();
                AdamOptimizer opt_clean(splat_ref_clean, make_cfg(cap));
                opt_clean.set_frozen_mask(frozen_mask);
                opt_clean.set_frozen_lr_scale(0.0f);
                opt_clean.set_crop_damping_mask(damp_mask);
                opt_clean.set_cropbox_lr_scale(0.2f);
                opt_clean.allocate_gradients(cap);
                auto ref_clean = opt_clean.prepare_fastgs_fused_adam(past_warmup);
                auto& v_shN_clean = opt_clean.get_grad(ParamType::ShN);
                v_shN_clean.zero_();

                gsplat_lfs::spherical_harmonics_streamed_bwd(
                    static_cast<uint32_t>(deg), static_cast<uint32_t>(deg),
                    dirs.ptr<float>(), v_colors.ptr<float>(), C, static_cast<uint32_t>(n),
                    static_cast<uint32_t>(color_stride), v_sh0_dummy.ptr<float>(),
                    v_shN_clean.ptr<float>(), nullptr);
                fast_lfs::optimizer::adam_step_shN_joint_from_grad(
                    ref_clean.shN.param, ref_clean.shN.joint_packed, ref_clean.shN.joint_bounds,
                    ref_clean.shN.sh_value_bounds, v_shN_clean.ptr<float>(),
                    ref_clean.shN.frozen_mask, ref_clean.shN.frozen_mask_size, ref_clean.shN.frozen_lr_scale,
                    ref_clean.shN.crop_damping_mask, ref_clean.shN.crop_damping_mask_size, ref_clean.shN.cropbox_lr_scale,
                    ref_clean.shN.n_primitives, sh_layout_slots, active_sh_bases,
                    ref_clean.shN.sh_value_bits, ref_clean.shN.sh_value_n_cells,
                    ref_clean.shN.step_size, ref_clean.beta1, ref_clean.beta2, ref_clean.eps,
                    ref_clean.shN.bias_correction2_sqrt_rcp, nullptr, 0.01f, nullptr, true, sr_seed);

                fast_lfs::optimizer::adam_step_shN_joint_from_color(
                    fused.shN.param, fused.shN.joint_packed, fused.shN.joint_bounds,
                    fused.shN.sh_value_bounds, v_colors.ptr<float>(), dirs.ptr<float>(), color_stride,
                    fused.shN.frozen_mask, fused.shN.frozen_mask_size, fused.shN.frozen_lr_scale,
                    fused.shN.crop_damping_mask, fused.shN.crop_damping_mask_size, fused.shN.cropbox_lr_scale,
                    fused.shN.n_primitives, sh_layout_slots, active_sh_bases,
                    fused.shN.sh_value_bits, fused.shN.sh_value_n_cells,
                    fused.shN.step_size, fused.beta1, fused.beta2, fused.eps,
                    fused.shN.bias_correction2_sqrt_rcp, nullptr, 0.01f, nullptr, sr_seed);
                ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

                if (use_q16) {
                    auto vb_a = splat_fused.shN_value_bounds().cpu();
                    auto vb_b = splat_ref_clean.shN_value_bounds().cpu();
                    ASSERT_EQ(vb_a.numel(), vb_b.numel());
                    for (size_t i = 0; i < vb_a.numel(); ++i) {
                        ASSERT_NEAR(vb_a.ptr<float>()[i], vb_b.ptr<float>()[i], 1e-5f);
                    }
                    auto c_a = splat_fused.shN().cpu();
                    auto c_b = splat_ref_clean.shN().cpu();
                    ASSERT_EQ(c_a.numel(), c_b.numel());
                    const auto* pa = reinterpret_cast<const uint16_t*>(c_a.data_ptr());
                    const auto* pb = reinterpret_cast<const uint16_t*>(c_b.data_ptr());
                    for (size_t i = 0; i < c_a.numel(); ++i) {
                        ASSERT_LE(std::abs(static_cast<int>(pa[i]) - static_cast<int>(pb[i])), 2)
                            << "q16 deg=" << deg << " seed=" << sr_seed << " idx=" << i;
                    }
                } else {
                    auto a_cpu = splat_fused.shN().cpu();
                    auto b_cpu = splat_ref_clean.shN().cpu();
                    ASSERT_EQ(a_cpu.numel(), b_cpu.numel());
                    for (size_t i = 0; i < a_cpu.numel(); ++i) {
                        ASSERT_NEAR(a_cpu.ptr<float>()[i], b_cpu.ptr<float>()[i], 1e-6f)
                            << "fp32 deg=" << deg << " seed=" << sr_seed << " idx=" << i;
                    }
                }
                const auto* st_a = opt_fused.get_state_mutable(ParamType::ShN);
                const auto* st_b = opt_clean.get_state_mutable(ParamType::ShN);
                auto jb_a = st_a->joint_bounds.cpu();
                auto jb_b = st_b->joint_bounds.cpu();
                ASSERT_EQ(jb_a.numel(), jb_b.numel());
                for (size_t i = 0; i < jb_a.numel(); ++i) {
                    ASSERT_NEAR(jb_a.ptr<float>()[i], jb_b.ptr<float>()[i], 1e-6f);
                }
            }
        }
        sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
    }
}

TEST(GutJointAdamBatch, FoldedGutRegularizationAndZeroGradMatchUnfused) {
    sh_value::set_sh_value_quant_enabled_for_testing(false);
    constexpr size_t n = 300;
    constexpr size_t cap = 512;
    constexpr int iteration = 1005;

    auto splat_fused = make_mixed_splat(n, 0);
    auto splat_ref = splat_fused.clone();

    std::vector<uint8_t> frozen_h(n, 0);
    frozen_h[0] = 1;
    frozen_h[258] = 1;
    std::vector<float> blend_h(n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        blend_h[i] = (i % 3 == 0) ? 0.0f : 0.4f;
    }
    auto frozen_mask = Tensor::from_vector(
        std::vector<bool>(frozen_h.begin(), frozen_h.end()), {n}, Device::CUDA);
    auto blend_peak = Tensor::from_vector(blend_h, {n}, Device::CUDA);

    AdamOptimizer opt_fused(splat_fused, make_cfg(cap));
    AdamOptimizer opt_ref(splat_ref, make_cfg(cap));
    opt_fused.set_frozen_mask(frozen_mask);
    opt_fused.set_frozen_lr_scale(0.0f);
    opt_ref.set_frozen_mask(frozen_mask);
    opt_ref.set_frozen_lr_scale(0.0f);
    opt_fused.allocate_gradients(cap);
    opt_ref.allocate_gradients(cap);

    fill_contiguous_gradients(opt_fused, 0.012f);
    fill_contiguous_gradients(opt_ref, 0.012f);

    constexpr float kLogScaleW = 0.05f;
    constexpr float kErankW = 0.08f;
    constexpr float kFlattenW = 0.03f;
    constexpr float kDcW = 0.1f;

    opt_fused.set_gut_regularization({
        .blend_peak = blend_peak.ptr<float>(),
        .log_scale_weight = kLogScaleW,
        .erank_weight = kErankW,
        .flatten_weight = kFlattenW,
        .dc_weight = kDcW});
    ASSERT_NO_THROW(opt_fused.step(iteration));

    // Unfused reference: run separate kernels, then step(iteration) without gut_reg.
    auto& s_grad_ref = opt_ref.get_grad(ParamType::Scaling);
    auto& sh0_grad_ref = opt_ref.get_grad(ParamType::Sh0);
    lfs::training::kernels::launch_gut_param_regularization(
        splat_ref.scaling_raw().ptr<float>(), s_grad_ref.ptr<float>(),
        splat_ref.sh0_raw().ptr<float>(), sh0_grad_ref.ptr<float>(),
        3, static_cast<int>(n), kErankW, kDcW, nullptr, nullptr, nullptr,
        lfs::training::kernels::GutShapeRegularization{
            .blend_peak = blend_peak.ptr<float>(),
            .log_scale_weight = kLogScaleW,
            .scale_loss_out = nullptr});
    gsplat_lfs::flatten_scale_grad(
        splat_ref.scaling_raw().ptr<float>(), s_grad_ref.ptr<float>(),
        static_cast<uint32_t>(n), kFlattenW, nullptr, blend_peak.ptr<float>());
    ASSERT_NO_THROW(opt_ref.step(iteration));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    for (auto type : {ParamType::Means, ParamType::Sh0, ParamType::Scaling, ParamType::Rotation, ParamType::Opacity}) {
        auto g_cpu = opt_fused.get_grad(type).cpu();
        for (size_t i = 0; i < g_cpu.numel(); ++i) {
            ASSERT_EQ(g_cpu.ptr<float>()[i], 0.0f) << "grad not zeroed at i=" << i;
        }
    }

    auto check_near = [](const Tensor& a, const Tensor& b, const char* label) {
        auto ac = a.cpu();
        auto bc = b.cpu();
        ASSERT_EQ(ac.numel(), bc.numel());
        for (size_t i = 0; i < ac.numel(); ++i) {
            ASSERT_NEAR(ac.ptr<float>()[i], bc.ptr<float>()[i], 1e-6f) << label << " idx=" << i;
        }
    };
    check_near(splat_fused.means_raw(), splat_ref.means_raw(), "means");
    check_near(splat_fused.sh0_raw(), splat_ref.sh0_raw(), "sh0");
    check_near(splat_fused.scaling_raw(), splat_ref.scaling_raw(), "scaling");
    check_near(splat_fused.rotation_raw(), splat_ref.rotation_raw(), "rotation");
    check_near(splat_fused.opacity_raw(), splat_ref.opacity_raw(), "opacity");
    sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
}
