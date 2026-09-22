/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/camera.hpp"
#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <vector>

namespace lfs::training {
    // Separate, fixed world-space upper-hemisphere Gaussians. Only SH0 colors
    // are optimized. This model is never registered in the foreground scene.
    class SkyBackground {
    public:
        explicit SkyBackground(int count = 100000, float radius = 10000.f, float opacity = .7f);
        void configure(const core::param::TrainingParameters& params,
                       const std::vector<core::Camera*>& cameras);
        core::Tensor render(const core::Camera& camera) const;
        // Must run AFTER foreground backward: the renderer reuses TLS caches.
        void backward(const core::Camera& camera, const core::Tensor& grad_sky);
        void optimizer_step(float learning_rate);
        void zero_grad();
        void serialize(std::ostream& stream) const;
        void deserialize(std::istream& stream);
        void adopt_checkpoint_state(SkyBackground& loaded) noexcept;
        int count() const { return count_; }
        float radius() const { return radius_; }
        float opacity() const { return opacity_; }
        std::uint64_t step() const { return step_; }
        const core::SplatData& model() const { return *model_; }
        const core::Tensor& colors() const { return model_->sh0(); }
        const core::Tensor& gradients() const { return gradients_; }

    private:
        int count_;
        float radius_, opacity_;
        std::uint64_t step_ = 0;
        std::unique_ptr<core::SplatData> model_;
        core::Tensor gradients_, exp_avg_, exp_avg_sq_;
    };
} // namespace lfs::training
