/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/crash_handler.hpp"
#include "core/cuda_error.hpp"
#include "core/logger.hpp"
#include "training/components/sky_background.hpp"
#include "training/rasterization/gsplat_rasterizer.hpp"
#include <cmath>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace lfs::core;
using namespace lfs::training;

static void check(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
static void near(float a, float b, float tolerance, const char* message) {
    if (!std::isfinite(a) || !std::isfinite(b) || std::abs(a - b) > tolerance)
        throw std::runtime_error(std::string(message) + ": " + std::to_string(a) + " vs " + std::to_string(b));
}
static float objective(const Tensor& image, const Tensor& grad, const Tensor& alpha) {
    return image.mul(grad).mul(alpha.mul(-1.f).add(1.f)).sum().item<float>();
}

// Verify the foreground opacity receives the derivative of the complete
// composition, rather than the derivative of a foreground render over black.
static void test_foreground_composition() {
    SkyBackground sky(1024);
    auto r = Tensor::zeros({3, 3}, Device::CPU);
    r.ptr<float>()[0] = r.ptr<float>()[4] = r.ptr<float>()[8] = 1;
    Camera cam(r, Tensor::zeros({3}, Device::CPU), 12, 12, 8, 8, {}, {}, lfs::core::CameraModelType::PINHOLE,
               "test.png", "test.png", {}, 16, 16, 0);
    cam.set_image_dimensions(16, 16);
    auto means = Tensor::zeros({1, 3}, Device::CPU);
    means.ptr<float>()[2] = 2.f;
    auto rotation = Tensor::zeros({1, 4}, Device::CPU);
    rotation.ptr<float>()[0] = 1.f;
    SplatData foreground(0, means.cuda(), Tensor::full({1, 1, 3}, -.5f, Device::CUDA), {},
                         Tensor::full({1, 3}, -1.f, Device::CUDA), rotation.cuda(),
                         Tensor::zeros({1, 1}, Device::CUDA), 1.f);
    AdamOptimizer optimizer(foreground, AdamConfig{});
    auto black = Tensor::zeros({3}, Device::CUDA);
    auto bg = sky.render(cam);
    auto result = gsplat_rasterize_forward(cam, foreground, black, 0, 0, 0, 0, 1.f,
                                           false, GsplatRenderMode::RGB, true, bg);
    check(bool(result), "foreground forward");
    auto original_bg = bg.clone();
    const auto grad = Tensor::full({3, 16, 16}, 1.f / (3 * 16 * 16), Device::CUDA);
    gsplat_rasterize_backward(result->second, grad, {}, foreground, optimizer);
    const float analytic = optimizer.get_grad(ParamType::Opacity).item<float>();
    const float eps = .01f;
    foreground.opacity_raw().fill_(eps);
    const float plus = gsplat_rasterize(cam, foreground, black, 1.f, false, GsplatRenderMode::RGB, true, bg).image.mean().item<float>();
    foreground.opacity_raw().fill_(-eps);
    const float minus = gsplat_rasterize(cam, foreground, black, 1.f, false, GsplatRenderMode::RGB, true, bg).image.mean().item<float>();
    near(analytic, (plus - minus) / (2 * eps), 1e-5f, "foreground opacity composite gradient");
    check(analytic < 0.f, "dark foreground opacity reduces brightness against sky");
    // A second sky render with different colors must not mutate the first result.
    const_cast<Tensor&>(sky.colors()).zero_();
    auto changed = sky.render(cam);
    check(changed.sub(original_bg).abs().max().item<float>() > .1f, "cache test changes image");
    near(bg.sub(original_bg).abs().max().item<float>(), 0, 0, "independent sky image storage");
}

int main() {
    lfs::core::Logger::get().init();
    struct Shutdown {
        ~Shutdown() { lfs::core::teardown_gpu_before_exit(); }
    } shutdown;
    try {
        test_foreground_composition();
        SkyBackground sky(1024);
        auto r = Tensor::zeros({3, 3}, Device::CPU);
        r.ptr<float>()[0] = r.ptr<float>()[4] = r.ptr<float>()[8] = 1;
        for (auto model : {lfs::core::CameraModelType::PINHOLE, lfs::core::CameraModelType::FISHEYE, lfs::core::CameraModelType::THIN_PRISM_FISHEYE}) {
            auto radial = model == lfs::core::CameraModelType::PINHOLE ? Tensor{} : Tensor::zeros({4}, Device::CPU);
            auto tangent = model == lfs::core::CameraModelType::THIN_PRISM_FISHEYE ? Tensor::zeros({4}, Device::CPU) : Tensor{};
            Camera cam(r, Tensor::zeros({3}, Device::CPU), 12, 12, 8, 8, radial, tangent, model,
                       "test.png", "test.png", {}, 16, 16, 0);
            cam.set_image_dimensions(16, 16);
            auto image = sky.render(cam);
            check(image.mean().item<float>() > .7f, "Gaussian sky visible at radius 10000 (far plane)");
            auto alpha = Tensor::full({1, 16, 16}, .3f, Device::CUDA);
            auto grad = Tensor::full({3, 16, 16}, 1.f / (3 * 16 * 16), Device::CUDA);
            auto before = image.clone();
            sky.zero_grad();
            sky.backward(cam, grad.mul(alpha.mul(-1.f).add(1.f)));
            near(image.sub(before).abs().max().item<float>(), 0, 0, "sky result independent of renderer cache");
            auto analytic = sky.gradients().cpu();
            size_t best = 0;
            for (size_t i = 1; i < analytic.numel(); ++i)
                if (analytic.ptr<float>()[i] > analytic.ptr<float>()[best])
                    best = i;
            check(analytic.ptr<float>()[best] > 1e-5f, "sky color receives reconstruction gradient");
            auto host = sky.colors().cpu();
            auto& colors = const_cast<Tensor&>(sky.colors());
            const float original = host.ptr<float>()[best], eps = .01f;
            host.ptr<float>()[best] = original + eps;
            colors.copy_from(host);
            const float plus = objective(sky.render(cam), grad, alpha);
            host.ptr<float>()[best] = original - eps;
            colors.copy_from(host);
            const float minus = objective(sky.render(cam), grad, alpha);
            near(analytic.ptr<float>()[best], (plus - minus) / (2 * eps), 1e-5f, "Gaussian SH0 finite difference");
            host.ptr<float>()[best] = original;
            colors.copy_from(host);
            sky.zero_grad();
            sky.backward(cam, Tensor::zeros({3, 16, 16}, Device::CUDA));
            near(sky.gradients().abs().max().item<float>(), 0, 0, "occluded or invalid sky has zero gradient");
        }
        auto means = sky.model().means().clone(), scales = sky.model().scaling_raw().clone();
        auto rotations = sky.model().rotation_raw().clone(), opacity = sky.model().opacity_raw().clone();
        auto& gradient_state = const_cast<Tensor&>(sky.gradients());
        gradient_state.fill_(.03f);
        sky.optimizer_step(.0025f);
        std::stringstream saved(std::ios::in | std::ios::out | std::ios::binary);
        sky.serialize(saved);
        SkyBackground resumed(1024);
        resumed.deserialize(saved);
        check(resumed.step() == 1, "optimizer step restore");
        near(sky.colors().sub(resumed.colors()).abs().max().item<float>(), 0, 0, "colors restore");
        gradient_state.fill_(.017f);
        const_cast<Tensor&>(resumed.gradients()).fill_(.017f);
        sky.optimizer_step(.0025f);
        resumed.optimizer_step(.0025f);
        near(sky.colors().sub(resumed.colors()).abs().max().item<float>(), 0, 1e-7f, "Adam resume continuation");
        near(means.sub(sky.model().means()).abs().max().item<float>(), 0, 0, "fixed positions");
        near(scales.sub(sky.model().scaling_raw()).abs().max().item<float>(), 0, 0, "fixed scales");
        near(rotations.sub(sky.model().rotation_raw()).abs().max().item<float>(), 0, 0, "fixed rotations");
        near(opacity.sub(sky.model().opacity_raw()).abs().max().item<float>(), 0, 0, "fixed opacity");
        for (const auto& bytes : {std::string("bad"), std::string("\1\0\0\0", 4)}) {
            bool rejected = false;
            std::stringstream bad(bytes);
            try {
                resumed.deserialize(bad);
            } catch (const std::exception&) { rejected = true; }
            check(rejected, "corrupt or cubemap checkpoint must fail");
        }
        lfs::core::param::OptimizationParameters params;
        params.sky_enabled = true;
        params.sky_num_points = 1024;
        const auto json = params.to_json();
        check(json["sky_enabled"].get<bool>() && json["sky_num_points"].get<int>() == 1024, "sky config serialization");
        std::cout << "Gaussian sky tests passed: camera models, gradients, fixed geometry, Adam resume\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
