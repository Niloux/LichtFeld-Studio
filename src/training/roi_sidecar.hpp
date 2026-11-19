/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
#include "core/scene.hpp"
#include "training_cropbox_mask.hpp"

#include <filesystem>
#include <glm/glm.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace lfs::training {

    // <dataset>/roi.json sidecar emitted by GeosunPre. Describes a simple closed 2D polygon
    // extruded vertically along [z_min, z_max] in the dataset's coordinate frame.
    struct RoiSidecar {
        std::vector<glm::dvec2> vertices_xy;
        double z_min = 0.0;
        double z_max = 0.0;
        std::string source;
        std::string coordinate_frame;
        double area_m2 = 0.0;
    };

    struct RoiCropPolygonPlacement {
        lfs::core::CropPolygonData data;
        glm::mat4 local_transform{1.0f};
    };

    // Missing file returns std::nullopt; present but unreadable/malformed returns an explicit error.
    [[nodiscard]] lfs::Result<std::optional<RoiSidecar>> read_roi_sidecar(
        const std::filesystem::path& dataset_path);

    [[nodiscard]] RoiCropPolygonPlacement make_roi_crop_polygon(
        const RoiSidecar& sidecar,
        const glm::vec3& training_origin = glm::vec3{0.0f});

    [[nodiscard]] TrainingCropBoxGeometry make_roi_eval_geometry(
        const RoiSidecar& sidecar,
        const glm::vec3& training_origin = glm::vec3{0.0f});

    [[nodiscard]] std::string format_roi_summary(
        const TrainingCropBoxGeometry& geometry,
        std::string_view origin_label = "crop polygon");

} // namespace lfs::training
