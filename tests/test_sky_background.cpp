/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/crash_handler.hpp"
#include "core/cuda_error.hpp"
#include "core/logger.hpp"
#include "training/components/sky_background.hpp"
#include "training/components/bilateral_grid.hpp"
#include "training/rasterization/gsplat_rasterizer.hpp"
#include "training/rasterization/fast_rasterizer.hpp"
#include "training/losses/mask_loss.hpp"
#include <algorithm>
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

static void test_mask_supervision() {
    SkyBackground sky(1024);
    auto mask = Tensor::empty({1, 7}, Device::CPU, DataType::UInt8);
    const uint8_t labels[] = {0, 127, 128, 200, 250, 251, 255};
    std::copy(std::begin(labels), std::end(labels), mask.ptr<uint8_t>());
    mask = mask.cuda();
    for (const auto& mask_samples : {mask, mask.to(DataType::Float32) / 255.f}) {
        for (const auto& target : {Tensor::zeros({3, 1, 7}, Device::CUDA),
                                  Tensor::zeros({3, 1, 7}, Device::CUDA, DataType::UInt8)}) {
            auto loss = sky.masked_color_loss(Tensor::full({3, 1, 7}, .6f, Device::CUDA), target, mask_samples);
            near(loss.loss.item<float>(), .6f, 1e-6f, "sky loss is normalized over segment pixels");
            const auto grad = loss.grad_image.cpu();
            for (int c = 0; c < 3; ++c)
                for (int i = 0; i < 7; ++i)
                    near(grad.ptr<float>()[c * 7 + i], i >= 2 && i <= 4 ? 1.f / 9.f : 0.f,
                         1e-6f, "only segment pixels supervise sky color");
        }
        auto foreground_rgb_grad = Tensor::full({3, 1, 7}, 1.f, Device::CUDA);
        sky.mask_foreground_gradient(foreground_rgb_grad, mask_samples);
        auto grad = foreground_rgb_grad.cpu();
        for (int i = 0; i < 7; ++i)
            near(grad.ptr<float>()[i], i >= 5 ? 1.f : 0.f, 0.f, "only keep pixels supervise foreground RGB");

        losses::MaskPreprocessWorkspace ws;
        auto penalty = losses::fuse_mask_opacity_penalty(
            ws, Tensor::full({1, 7}, .4f, Device::CUDA), mask_samples, {}, 2.f, 1.f, true);
        near(penalty.loss.item<float>(), .4f * 3.f / 7.f, 1e-6f, "upstream alpha penalty normalization");
        grad = penalty.grad_alpha.cpu();
        for (int i = 0; i < 7; ++i)
            near(grad.ptr<float>()[i], i >= 2 && i <= 4 ? 1.f / 7.f : 0.f,
                 1e-6f, "segment alpha penalty excludes ignore and keep bands");
        const auto keep = losses::fuse_photometric_mask_weight(ws, mask_samples, {}, true).cpu();
        for (int i = 0; i < 7; ++i)
            near(keep.ptr<float>()[i], i >= 5 ? 1.f : 0.f, 0.f, "upstream foreground photometric bands are unchanged");
    }
    auto empty = sky.masked_color_loss(Tensor::full({3, 1, 7}, .6f, Device::CUDA),
                                      Tensor::zeros({3, 1, 7}, Device::CUDA),
                                      Tensor::zeros({1, 7}, Device::CUDA));
    near(empty.loss.item<float>(), 0.f, 0.f, "no segment pixels gives zero sky loss");
    near(empty.grad_image.abs().max().item<float>(), 0.f, 0.f, "no segment pixels gives zero sky gradient");
    auto roi = Tensor::zeros({1, 7}, Device::CUDA);
    empty = sky.masked_color_loss(Tensor::full({3, 1, 7}, .6f, Device::CUDA),
                                 Tensor::zeros({3, 1, 7}, Device::CUDA), mask, roi);
    near(empty.loss.item<float>(), 0.f, 0.f, "zero ROI suppresses sky loss");
}

static void test_segment_alpha_densification() {
    auto r = Tensor::zeros({3, 3}, Device::CPU);
    r.ptr<float>()[0] = r.ptr<float>()[4] = r.ptr<float>()[8] = 1;
    Camera cam(r, Tensor::zeros({3}, Device::CPU), 12, 12, 8, 8, {}, {}, lfs::core::CameraModelType::PINHOLE,
               "test.png", "test.png", {}, 16, 16, 0);
    cam.set_image_dimensions(16, 16);
    auto make_foreground = [] {
        auto means = Tensor::zeros({1, 3}, Device::CPU);
        means.ptr<float>()[0] = .1f;
        means.ptr<float>()[2] = 2.f;
        auto rotation = Tensor::zeros({1, 4}, Device::CPU);
        rotation.ptr<float>()[0] = 1.f;
        return SplatData(0, means.cuda(), Tensor::full({1, 1, 3}, -.5f, Device::CUDA), {},
                         Tensor::full({1, 3}, -1.f, Device::CUDA), rotation.cuda(),
                         Tensor::zeros({1, 1}, Device::CUDA), 1.f);
    };
    auto black = Tensor::zeros({3}, Device::CUDA);
    auto zero_rgb = Tensor::zeros({3, 16, 16}, Device::CUDA);
    auto alpha_penalty = Tensor::full({16, 16}, 1.f / 256.f, Device::CUDA);
    for (bool exclude : {false, true}) {
        auto foreground = make_foreground();
        foreground._densification_info = Tensor::zeros({2, 1}, Device::CUDA);
        AdamOptimizer optimizer(foreground, AdamConfig{});
        optimizer.allocate_gradients();
        optimizer.zero_grad(0);
        auto result = gsplat_rasterize_forward(cam, foreground, black, 0, 0, 0, 0, 1.f,
                                               false, GsplatRenderMode::RGB, true);
        check(bool(result), "segment alpha forward");
        gsplat_rasterize_backward(result->second, zero_rgb, alpha_penalty, foreground, optimizer,
                                  {}, {}, {}, {}, {}, 0.f, exclude ? alpha_penalty : Tensor{});
        check(optimizer.get_grad(ParamType::Opacity).item<float>() > 0.f, "alpha penalty reaches foreground opacity");
        check(optimizer.get_grad(ParamType::Means).abs().max().item<float>() > 0.f, "upstream alpha geometry gradient is preserved");
        const auto norm = foreground._densification_info.cpu().ptr<float>()[0];
        if (exclude)
            near(norm, 0.f, 1e-7f, "segment alpha penalty does not drive GUT densification");
        else
            check(norm > 0.f, "unmodified upstream densification includes alpha geometry gradient");
    }
    for (bool exclude : {false, true}) {
        auto foreground = make_foreground();
        foreground._densification_info = Tensor::zeros({2, 1}, Device::CUDA);
        AdamOptimizer optimizer(foreground, AdamConfig{});
        optimizer.allocate_gradients();
        optimizer.zero_grad(0);
        auto result = fast_rasterize_forward(cam, foreground, black);
        check(bool(result), "FastGS segment alpha forward");
        fast_rasterize_backward(result->second, zero_rgb, foreground, optimizer, alpha_penalty,
                                {}, DensificationType::None, 1, {}, {}, {}, exclude ? alpha_penalty : Tensor{});
        check(foreground.opacity_raw().item<float>() < 0.f, "FastGS alpha penalty reduces foreground opacity");
        const auto norm = foreground._densification_info.cpu().ptr<float>()[1];
        if (exclude)
            near(norm, 0.f, 1e-6f, "segment alpha penalty does not drive FastGS densification");
        else
            check(norm > 0.f, "unmodified FastGS densification includes alpha geometry gradient");
    }
}

static void test_joint_appearance_accumulation() {
    BilateralGrid combined(1, 2, 2, 2, 10), split(1, 2, 2, 2, 10);
    auto image = Tensor::full({3, 4, 4}, .4f, Device::CUDA);
    auto grad = Tensor::full({3, 4, 4}, .03f, Device::CUDA);
    auto combined_grad = grad.mul(3.f);
    combined.backward_in_place(image, combined_grad, 0);
    auto foreground_grad = grad.clone(), sky_grad = grad.mul(2.f);
    split.backward_in_place(image, foreground_grad, 0);
    split.backward_in_place(image, sky_grad, 0, true);
    near(combined.grad_slice().sub(split.grad_slice()).abs().max().item<float>(), 0.f, 1e-6f,
         "sky appearance backward preserves foreground grid parameter gradients");
    near(combined_grad.sub(foreground_grad.add(sky_grad)).abs().max().item<float>(), 0.f, 1e-6f,
         "joint appearance returns both image gradients");
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
        test_mask_supervision();
        test_segment_alpha_densification();
        test_joint_appearance_accumulation();
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
        std::cout << "Gaussian sky tests passed: mask bands, foreground alpha/densification isolation, camera models, gradients, fixed geometry, Adam resume\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
