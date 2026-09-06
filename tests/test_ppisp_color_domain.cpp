/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/crash_handler.hpp"
#include "core/logger.hpp"
#include "core/tensor.hpp"
#include "lfs/kernels/ppisp.cuh"
#include "training/components/ppisp.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using lfs::core::Device;
using lfs::core::Tensor;
namespace kernels = lfs::training::kernels;

namespace {
    void near(float actual, float expected, float tolerance, const char* label) {
        if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance)
            throw std::runtime_error(std::string(label) + ": " + std::to_string(actual) + " vs " + std::to_string(expected));
    }

    void test_negative_render_colors() {
        // Signed dark pixels from a real GUT eval plus black, tiny positive and
        // HDR cases. Neutral PPISP must not turn any of these into bright colors.
        const std::vector<std::array<float, 3>> samples{
            {.00685103f, -.00409285f, -.06005293f},
            {-.02010575f, .0015682f, -.01317868f},
            {.01204113f, .01160863f, -.02435323f},
            {-.1f, -.2f, -.3f},
            {0.f, 0.f, 0.f},
            {.2f, .3f, .4f},
            {-.01f, .2f, .3f},
            {1e-4f, 2e-4f, 3e-4f},
            {1.2f, .2f, -.1f}};
        const size_t n = samples.size();
        std::vector<float> host(3 * n);
        for (size_t i = 0; i < n; ++i)
            for (size_t c = 0; c < 3; ++c)
                host[c * n + i] = samples[i][c];
        auto rgb = Tensor::from_vector(host, {3, 1, n}, Device::CUDA);
        lfs::training::PPISP ppisp(100);
        ppisp.register_frame(10, 7);
        ppisp.finalize();
        const auto train = ppisp.apply(rgb, 7, 10).cpu();
        const auto eval = ppisp.apply_with_exposure(rgb, 7, 0.f).cpu();
        for (size_t i = 0; i < n; ++i) {
            float intensity = 0.f;
            for (float c : samples[i])
                intensity += std::max(c, 0.f);
            for (size_t c = 0; c < 3; ++c) {
                const float expected = std::clamp(std::max(samples[i][c], 0.f) * intensity / (intensity + 1e-5f), 0.f, 1.f);
                near(train.ptr<float>()[c * n + i], expected, 2e-5f, "neutral training color domain");
                near(eval.ptr<float>()[c * n + i], expected, 2e-5f, "neutral held-out color domain");
            }
        }
    }

    void test_color_vjp() {
        // Isolate exposure/color from CRF clipping, then check RGB, color latent
        // and exposure derivatives with both signs represented in each channel.
        constexpr size_t n = 5;
        auto rgb = Tensor::from_vector({-.02f, .3f, .1f, -.3f, .7f,
                                        .01f, -.04f, .2f, -.2f, .4f,
                                        -.03f, .2f, -.07f, -.1f, .2f},
                                       {3, 1, n}, Device::CUDA);
        auto color = Tensor::from_vector({.12f, -.15f, .08f, .04f, -.06f, .02f, .1f, -.07f}, {8}, Device::CUDA);
        auto exposure = Tensor::full({1}, .35f, Device::CUDA);
        auto vig = Tensor::zeros({15}, Device::CUDA), crf = Tensor::zeros({12}, Device::CUDA);
        auto output = Tensor::zeros(rgb.shape(), Device::CUDA);
        auto grad = Tensor::from_vector({.1f, -.2f, .3f, .2f, .1f,
                                         -.3f, .2f, .1f, -.1f, .2f,
                                         .2f, .1f, -.1f, .3f, -.3f},
                                        rgb.shape(), Device::CUDA);
        auto forward = [&] {
            kernels::launch_ppisp_forward_chw(exposure.ptr<float>(), vig.ptr<float>(), color.ptr<float>(),
                                              crf.ptr<float>(), rgb.ptr<float>(), output.ptr<float>(), 1, n, 1, 1, -1, 0);
            return output.mul(grad).sum().item<float>();
        };
        auto rgb_grad = Tensor::zeros(rgb.shape(), Device::CUDA);
        auto color_grad = Tensor::zeros(color.shape(), Device::CUDA);
        auto exposure_grad = Tensor::zeros(exposure.shape(), Device::CUDA);
        auto vig_grad = Tensor::zeros(vig.shape(), Device::CUDA), crf_grad = Tensor::zeros(crf.shape(), Device::CUDA);
        kernels::launch_ppisp_backward_chw(exposure.ptr<float>(), vig.ptr<float>(), color.ptr<float>(), crf.ptr<float>(),
                                           rgb.ptr<float>(), grad.ptr<float>(), exposure_grad.ptr<float>(), vig_grad.ptr<float>(),
                                           color_grad.ptr<float>(), crf_grad.ptr<float>(), rgb_grad.ptr<float>(), 1, n, 1, 1, -1, 0);
        auto check_fd = [&](Tensor& input, const Tensor& analytic, const char* label) {
            auto host = input.cpu(), g = analytic.cpu();
            for (size_t i = 0; i < host.numel(); ++i) {
                constexpr float eps = 1e-3f;
                const float original = host.ptr<float>()[i];
                host.ptr<float>()[i] = original + eps;
                input.copy_from(host);
                const float plus = forward();
                host.ptr<float>()[i] = original - eps;
                input.copy_from(host);
                const float minus = forward();
                host.ptr<float>()[i] = original;
                input.copy_from(host);
                near(g.ptr<float>()[i], (plus - minus) / (2 * eps), 8e-5f, label);
                if (&input == &rgb && original < 0.f)
                    near(g.ptr<float>()[i], 0.f, 0.f, "negative color has zero input gradient");
            }
        };
        check_fd(rgb, rgb_grad, "RGB finite difference");
        check_fd(color, color_grad, "color latent finite difference");
        check_fd(exposure, exposure_grad, "exposure finite difference");
    }
} // namespace

int main() {
    lfs::core::Logger::get().init();
    struct Shutdown {
        ~Shutdown() { lfs::core::teardown_gpu_before_exit(); }
    } shutdown;
    try {
        test_negative_render_colors();
        test_color_vjp();
        std::cout << "PPISP color-domain tests passed: signed RGB, held-out exposure, RGB/color/exposure gradients\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
