/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// The GUT path's erank / dc / SH-rest regularizers, which FastGS fuses into its backward.

#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "lfs/kernels/gut_regularization.cuh"
#include "training/optimizer/adam_optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace lfs::core;
using namespace lfs::training;

namespace {

    constexpr int kN = 37;

    struct Params {
        std::vector<float> log_scales, sh0;
        Params() : log_scales(3 * kN), sh0(3 * kN) {
            std::mt19937 rng(7);
            std::uniform_real_distribution<float> scale(-4.0f, -1.0f);
            std::uniform_real_distribution<float> colour(-3.0f, 3.0f);
            for (auto& v : log_scales)
                v = scale(rng);
            // Some needles, so the spike penalty is active.
            for (int i = 0; i < kN; i += 3)
                log_scales[3 * i] += 3.0f;
            for (auto& v : sh0)
                v = colour(rng);
        }
    };

    struct Result {
        float erank_loss, dc_loss;
        std::vector<float> scales_grad, sh0_grad;
    };

    Result evaluate(const Params& p, float erank_weight, float dc_weight) {
        auto scales = Tensor::from_vector(p.log_scales, {kN, 3}, Device::CUDA);
        auto sh0 = Tensor::from_vector(p.sh0, {kN, 1, 3}, Device::CUDA);
        auto scales_grad = Tensor::zeros({kN, 3}, Device::CUDA);
        auto sh0_grad = Tensor::zeros({kN, 1, 3}, Device::CUDA);
        auto erank_loss = Tensor::zeros({1}, Device::CUDA);
        auto dc_loss = Tensor::zeros({1}, Device::CUDA);
        kernels::launch_gut_param_regularization(
            scales.ptr<float>(), scales_grad.ptr<float>(), sh0.ptr<float>(), sh0_grad.ptr<float>(), 3, kN,
            erank_weight, dc_weight, erank_loss.ptr<float>(), dc_loss.ptr<float>());
        return {erank_loss.item<float>(), dc_loss.item<float>(), scales_grad.cpu().to_vector(),
                sh0_grad.cpu().to_vector()};
    }

    class GutRegularizationTest : public ::testing::Test {
    protected:
        void SetUp() override {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
                GTEST_SKIP() << "No CUDA device available";
        }
    };

} // namespace

TEST_F(GutRegularizationTest, EffectiveRankGradientIsTheReportedLossDerivative) {
    Params p;
    constexpr float weight = 0.3f;
    const auto analytic = evaluate(p, weight, 0.0f);
    EXPECT_GT(analytic.erank_loss, 0.0f);
    EXPECT_EQ(analytic.dc_loss, 0.0f);
    // The penalty in double, as in test_fused_reg_loss.cpp; differentiated numerically there too, since
    // the float kernel's own loss is too noisy near needles (-log of a small rank gap) to difference.
    const auto penalty = [](const double* s) {
        const double peak = std::max({s[0], s[1], s[2]});
        double q[3], sum = 0.0;
        for (int d = 0; d < 3; ++d)
            sum += q[d] = std::exp(2.0 * (s[d] - peak));
        double entropy = 0.0;
        for (const double value : q)
            entropy -= value / sum * std::log(value / sum);
        return std::max(-std::log(std::exp(entropy) - 0.99999), 0.0);
    };
    const double coef = static_cast<double>(weight) / kN;
    double expected_loss = 0.0;
    for (int i = 0; i < kN; ++i) {
        double s[3] = {p.log_scales[3 * i], p.log_scales[3 * i + 1], p.log_scales[3 * i + 2]};
        expected_loss += coef * penalty(s);
        for (int axis = 0; axis < 3; ++axis) {
            SCOPED_TRACE(::testing::Message() << "splat=" << i << " axis=" << axis);
            constexpr double h = 1.0e-6;
            double plus[3] = {s[0], s[1], s[2]}, minus[3] = {s[0], s[1], s[2]};
            plus[axis] += h;
            minus[axis] -= h;
            const double grad = coef * (penalty(plus) - penalty(minus)) / (2.0 * h);
            EXPECT_NEAR(analytic.scales_grad[3 * i + axis], grad, 1.0e-6 + 1.0e-2 * std::abs(grad));
        }
    }
    EXPECT_NEAR(analytic.erank_loss, expected_loss, 1.0e-4 * expected_loss);
    for (float g : analytic.sh0_grad)
        EXPECT_EQ(g, 0.0f);
}

TEST_F(GutRegularizationTest, ColourRangeMatchesTheFusedTerms) {
    // FastGS: gradient (2 w / 3) * delta per splat, loss w / (3 N) * sum(delta^2), delta the excess
    // past +-0.5 / C0.
    const Params p;
    constexpr float weight = 0.25f;
    const auto result = evaluate(p, 0.0f, weight);
    const float limit = 0.5f / 0.28209479177387814f;
    double loss = 0.0;
    for (int i = 0; i < 3 * kN; ++i) {
        const float v = p.sh0[i];
        const float delta = v > limit ? v - limit : (v < -limit ? v + limit : 0.0f);
        EXPECT_FLOAT_EQ(result.sh0_grad[i], (2.0f * weight / 3.0f) * delta) << i;
        loss += static_cast<double>(weight) / (3.0 * kN) * delta * delta;
    }
    EXPECT_GT(loss, 0.0);
    EXPECT_NEAR(result.dc_loss, loss, 1.0e-6 + 1.0e-5 * loss);
    for (float g : result.scales_grad)
        EXPECT_EQ(g, 0.0f);
}

TEST_F(GutRegularizationTest, ZeroWeightsLeaveGradientsUntouched) {
    const auto result = evaluate(Params{}, 0.0f, 0.0f);
    EXPECT_EQ(result.erank_loss, 0.0f);
    EXPECT_EQ(result.dc_loss, 0.0f);
    for (float g : result.scales_grad)
        EXPECT_EQ(g, 0.0f);
    for (float g : result.sh0_grad)
        EXPECT_EQ(g, 0.0f);
}

TEST_F(GutRegularizationTest, ShRestRegularizerEqualsItsGradientInTheShNStep) {
    // A step with the regularizer and a zero gradient equals a step whose gradient is the
    // regularizer's own: 2 w / (N * 3 * rest) * value per rest coefficient.
    constexpr size_t n = 300;
    constexpr int sh_degree = 3;
    constexpr int past_warmup = 1001;
    constexpr float weight = 0.4f;
    const size_t rest = sh_degree * (sh_degree + 2);
    std::vector<float> sh0(n * 3, 0.1f), means(n * 3), scaling(n * 3, -2.0f), rotation(n * 4, 0.0f),
        opacity(n, 0.0f), shN(n * rest * 3);
    for (size_t i = 0; i < n; ++i) {
        means[3 * i] = 0.01f * static_cast<float>(i);
        rotation[4 * i] = 1.0f;
    }
    for (size_t i = 0; i < shN.size(); ++i)
        shN[i] = 0.03f * static_cast<float>(static_cast<int>(i % 13) - 6);
    const auto make = [&] {
        return SplatData(sh_degree, Tensor::from_vector(means, {n, 3}, Device::CUDA),
                         Tensor::from_vector(sh0, {n, 1, 3}, Device::CUDA),
                         Tensor::from_vector(shN, {n, rest, 3}, Device::CUDA),
                         Tensor::from_vector(scaling, {n, 3}, Device::CUDA),
                         Tensor::from_vector(rotation, {n, 4}, Device::CUDA),
                         Tensor::from_vector(opacity, {n, 1}, Device::CUDA), 1.0f);
    };
    auto regularized = make();
    auto reference = make();
    ASSERT_FALSE(regularized.shN_value_quantized());
    ASSERT_EQ(regularized.active_sh_coeffs_rest(), rest);

    AdamConfig cfg;
    cfg.lr = 1e-3f;
    cfg.initial_capacity = 512;
    AdamOptimizer opt_regularized(regularized, cfg);
    AdamOptimizer opt_reference(reference, cfg);
    opt_regularized.allocate_gradients(512);
    opt_reference.allocate_gradients(512);

    auto loss = Tensor::zeros({1}, Device::CUDA);
    opt_regularized.get_grad(ParamType::ShN).zero_();
    opt_regularized.set_sh_rest_regularization(weight, loss.ptr<float>());
    // The swizzled layout's pad cells hold zeros, so the gradient is elementwise in the values.
    const float reg_scale = weight / (static_cast<float>(n) * static_cast<float>(rest * 3));
    auto values = reference.shN().cpu().to_vector();
    std::vector<float> grad(values.size());
    double expected_loss = 0.0;
    for (size_t i = 0; i < values.size(); ++i) {
        grad[i] = 2.0f * reg_scale * values[i];
        expected_loss += static_cast<double>(reg_scale) * values[i] * values[i];
    }
    auto& reference_grad = opt_reference.get_grad(ParamType::ShN);
    ASSERT_EQ(reference_grad.numel(), grad.size());
    ASSERT_EQ(cudaMemcpy(reference_grad.ptr<float>(), grad.data(), grad.size() * sizeof(float),
                         cudaMemcpyHostToDevice),
              cudaSuccess);

    ASSERT_NO_THROW(opt_regularized.step(past_warmup));
    ASSERT_NO_THROW(opt_reference.step(past_warmup));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto a = regularized.shN().cpu().to_vector();
    const auto b = reference.shN().cpu().to_vector();
    ASSERT_EQ(a.size(), b.size());
    size_t moved = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        ASSERT_NEAR(a[i], b[i], 1.0e-7f) << i;
        moved += a[i] != values[i];
    }
    EXPECT_GT(moved, a.size() / 2);
    EXPECT_NEAR(loss.item<float>(), expected_loss, 1.0e-6 * expected_loss + 1.0e-9);
}
