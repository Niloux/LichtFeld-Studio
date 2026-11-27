/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/tensor.hpp"
#include <cstddef>
#include <filesystem>
#include <memory>
#include <unordered_map>
#include <vector>

namespace lfs::training {

    class PhotometricWeightsSidecar {
    public:
        void load(const std::filesystem::path& dataset_path,
                  const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras);
        void clear();

        [[nodiscard]] bool has_weights() const noexcept { return !weights_by_uid_.empty(); }
        [[nodiscard]] size_t count() const noexcept { return weights_by_uid_.size(); }
        [[nodiscard]] lfs::core::Tensor weight_for_camera(int uid) const;

    private:
        std::unordered_map<int, lfs::core::Tensor> weights_by_uid_;
    };

} // namespace lfs::training
