/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "sky_background.hpp"
#include "config_serialization.hpp"
#include "core/logger.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "nanoflann.hpp"
#include "training/kernels/sky_background.hpp"
#include "training/rasterization/gsplat_rasterizer.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numbers>
#include <stdexcept>

namespace lfs::training {
    using core::DataType;
    using core::Device;
    using core::Tensor;
    namespace {
        struct Cloud {
            const float* points;
            size_t count;
            size_t kdtree_get_point_count() const { return count; }
            float kdtree_get_pt(size_t i, size_t axis) const { return points[3 * i + axis]; }
            template <class BBox>
            bool kdtree_get_bbox(BBox&) const { return false; }
        };
    } // namespace
    SkyBackground::SkyBackground(int count, float radius, float opacity)
        : count_(count), radius_(radius), opacity_(opacity) {
        if (count < 16 || count > 1000000 || !std::isfinite(radius) || radius <= 0 ||
            !std::isfinite(opacity) || opacity <= 0 || opacity >= 1)
            throw std::invalid_argument("Invalid Gaussian sky geometry: count [16,1000000], radius > 0, opacity (0,1)");
        const size_t n = count;
        auto means = Tensor::empty({n, 3}, Device::CPU);
        auto scales = Tensor::empty({n, 3}, Device::CPU);
        auto rotations = Tensor::zeros({n, 4}, Device::CPU);
        for (size_t i = 0; i < n; ++i) {
            const double z = (i + .5) / n;
            const double phi = i * (std::numbers::pi * (3. - std::sqrt(5.)));
            const double xy = std::sqrt(1. - z * z);
            means.ptr<float>()[3 * i] = radius * xy * std::cos(phi);
            means.ptr<float>()[3 * i + 1] = radius * xy * std::sin(phi);
            means.ptr<float>()[3 * i + 2] = radius * z;
            rotations.ptr<float>()[4 * i] = 1.f;
        }
        Cloud cloud{means.ptr<float>(), n};
        using Tree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, Cloud>, Cloud, 3>;
        Tree tree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        tree.buildIndex();
        for (size_t i = 0; i < n; ++i) {
            Tree::IndexType indices[2];
            float distances[2];
            tree.knnSearch(means.ptr<float>() + 3 * i, 2, indices, distances);
            const float scale = std::log(std::max(std::sqrt(distances[1]), std::max(1e-6f, std::numeric_limits<float>::epsilon() * radius)));
            std::fill_n(scales.ptr<float>() + 3 * i, 3, scale);
        }
        model_ = std::make_unique<core::SplatData>(0, means.cuda(),
                                                   Tensor::full({n, 1, 3}, .5f / .28209479177387814f, Device::CUDA), Tensor{},
                                                   scales.cuda(), rotations.cuda(), Tensor::full({n, 1}, std::log(opacity / (1.f - opacity)), Device::CUDA), 1.f);
        gradients_ = Tensor::zeros({n, 1, 3}, Device::CUDA);
        exp_avg_ = Tensor::zeros(gradients_.shape(), Device::CUDA);
        exp_avg_sq_ = Tensor::zeros(gradients_.shape(), Device::CUDA);
    }

    void SkyBackground::configure(const core::param::TrainingParameters& params,
                                  const std::vector<core::Camera*>& cameras) {
        const auto& opt = params.optimization;
        if (opt.undistort || opt.bg_mode != core::param::BackgroundMode::SolidColor)
            throw std::invalid_argument("Sky requires native camera images (--gut for distorted cameras) and solid background mode");
        if (!std::isfinite(opt.sky_lr) || opt.sky_lr <= 0)
            throw std::invalid_argument("Invalid sky optimizer settings");
        for (const auto* cam : cameras) {
            if (cam->camera_model_type() == core::CameraModelType::ORTHO)
                throw std::invalid_argument("Sky does not support orthographic cameras");
        }
        LOG_INFO("Gaussian sky: {} fixed points, radius {}, opacity {}, SH0 colors; excluded from PLY", count_, radius_, opacity_);
    }

    Tensor SkyBackground::render(const core::Camera& cam) const {
        auto black = Tensor::zeros({3}, Device::CUDA);
        auto result = gsplat_rasterize_forward(const_cast<core::Camera&>(cam), *model_, black,
                                               0, 0, 0, 0, 1.f, false, GsplatRenderMode::RGB, true, {}, 1e10f);
        if (!result)
            throw std::runtime_error(result.error());
        core::GlobalArenaManager::instance().get_arena().end_frame(result->second.frame_id, result->second.stream);
        // Forward output is TLS-owned. A foreground forward must not overwrite it.
        return result->first.image.clone();
    }

    void SkyBackground::backward(const core::Camera& cam, const Tensor& grad_sky) {
        auto black = Tensor::zeros({3}, Device::CUDA);
        auto result = gsplat_rasterize_forward(const_cast<core::Camera&>(cam), *model_, black,
                                               0, 0, 0, 0, 1.f, false, GsplatRenderMode::RGB, true, {}, 1e10f);
        if (!result)
            throw std::runtime_error(result.error());
        gsplat_rasterize_backward_sh0(result->second, grad_sky, *model_, gradients_);
    }
    void SkyBackground::zero_grad() { gradients_.zero_(); }
    void SkyBackground::optimizer_step(float lr) {
        auto& sh0 = model_->sh0();
        core::prepare_inputs_for_stream({&sh0, &gradients_, &exp_avg_, &exp_avg_sq_}, core::getCurrentCUDAStream());
        kernels::sky_adam(sh0.ptr<float>(), gradients_.ptr<float>(), exp_avg_.ptr<float>(),
                          exp_avg_sq_.ptr<float>(), count_, ++step_, lr, core::getCurrentCUDAStream());
    }
    void SkyBackground::serialize(std::ostream& os) const {
        using config_serialization_detail::write_little_endian;
        write_little_endian(os, uint32_t(2), "sky version");
        write_little_endian(os, uint32_t(count_), "sky count");
        write_little_endian(os, radius_, "sky radius");
        write_little_endian(os, opacity_, "sky opacity");
        write_little_endian(os, step_, "sky step");
        // Persist geometry too, so resume never depends on generator changes.
        os << model_->means() << model_->scaling_raw() << model_->rotation_raw() << model_->opacity_raw();
        os << model_->sh0() << exp_avg_ << exp_avg_sq_;
    }
    void SkyBackground::deserialize(std::istream& is) {
        using config_serialization_detail::read_little_endian;
        const auto version = read_little_endian<uint32_t>(is, "sky version");
        if (version == 1)
            throw std::runtime_error("Cubemap sky checkpoint cannot resume as Gaussian sky; start a new training output directory");
        if (version != 2)
            throw std::runtime_error("Invalid Gaussian sky checkpoint version");
        const auto count = read_little_endian<uint32_t>(is, "sky count");
        const auto radius = read_little_endian<float>(is, "sky radius");
        const auto opacity = read_little_endian<float>(is, "sky opacity");
        const auto step = read_little_endian<uint64_t>(is, "sky step");
        if (count != uint32_t(count_) || radius != radius_ || opacity != opacity_)
            throw std::runtime_error("Cannot change Gaussian sky geometry on resume");
        if (step > uint64_t(std::numeric_limits<int>::max()))
            throw std::runtime_error("Invalid sky checkpoint step");
        Tensor means, scales, rotations, opacities, sh0, m, v;
        is >> means >> scales >> rotations >> opacities >> sh0 >> m >> v;
        const size_t n = count;
        const std::pair<const Tensor*, core::TensorShape> tensors[] = {
            {&means, {n, 3}},
            {&scales, {n, 3}},
            {&rotations, {n, 4}},
            {&opacities, {n, 1}},
            {&sh0, {n, 1, 3}},
            {&m, {n, 1, 3}},
            {&v, {n, 1, 3}}};
        for (const auto& [tensor, shape] : tensors) {
            if (!tensor->is_valid() || tensor->shape() != shape || tensor->dtype() != DataType::Float32)
                throw std::runtime_error("Invalid sky checkpoint tensor");
            auto host = tensor->cpu().contiguous();
            for (size_t i = 0; i < host.numel(); ++i)
                if (!std::isfinite(host.ptr<float>()[i]) || (tensor == &v && host.ptr<float>()[i] < 0))
                    throw std::runtime_error("Non-finite sky checkpoint or negative variance");
        }
        auto model = std::make_unique<core::SplatData>(0, means.cuda(), sh0.cuda(), Tensor{},
                                                       scales.cuda(), rotations.cuda(), opacities.cuda(), 1.f);
        auto grad = Tensor::zeros(sh0.shape(), Device::CUDA);
        auto first = m.cuda(), second = v.cuda();
        model_ = std::move(model);
        gradients_ = std::move(grad);
        exp_avg_ = std::move(first);
        exp_avg_sq_ = std::move(second);
        step_ = step;
        LOG_INFO("Restored Gaussian sky: {} points, optimizer step {}", count_, step_);
    }
    void SkyBackground::adopt_checkpoint_state(SkyBackground& loaded) noexcept {
        using std::swap;
        swap(model_, loaded.model_);
        swap(gradients_, loaded.gradients_);
        swap(exp_avg_, loaded.exp_avg_);
        swap(exp_avg_sq_, loaded.exp_avg_sq_);
        swap(count_, loaded.count_);
        swap(radius_, loaded.radius_);
        swap(opacity_, loaded.opacity_);
        swap(step_, loaded.step_);
    }
} // namespace lfs::training
