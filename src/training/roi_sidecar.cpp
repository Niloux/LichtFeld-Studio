/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "roi_sidecar.hpp"

#include "core/crop_polygon.hpp"
#include "core/native_fs.hpp"
#include "core/path_utils.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <fstream>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <limits>
#include <nlohmann/json.hpp>

namespace lfs::training {

    namespace {
        constexpr std::string_view kSidecarName = "roi.json";

        [[nodiscard]] lfs::Error make_roi_error(
            const lfs::ErrorCode code,
            std::string message,
            const lfs::core::SourceSite site) {
            return lfs::make_error(lfs::ErrorInit{
                .code = code,
                .domain = lfs::ErrorDomain::Training,
                .user_message = message,
                .detail = std::move(message),
                .detection = site,
            });
        }

        [[nodiscard]] double polygon_area_d(const std::vector<glm::dvec2>& vertices) {
            if (vertices.size() < 3) {
                return 0.0;
            }
            double twice = 0.0;
            for (size_t i = 0; i < vertices.size(); ++i) {
                const glm::dvec2& a = vertices[i];
                const glm::dvec2& b = vertices[(i + 1) % vertices.size()];
                twice += a.x * b.y - b.x * a.y;
            }
            return 0.5 * std::abs(twice);
        }
    } // namespace

    lfs::Result<std::optional<RoiSidecar>> read_roi_sidecar(
        const std::filesystem::path& dataset_path) {
        if (dataset_path.empty()) {
            return std::optional<RoiSidecar>{std::nullopt};
        }
        const auto path = dataset_path / kSidecarName;
        std::error_code ec;
        if (!lfs::core::native_fs::exists(path, ec)) {
            if (ec) {
                return make_roi_error(
                    lfs::ErrorCode::DataLoss,
                    std::format("Failed to stat ROI sidecar '{}': {}", lfs::core::path_to_utf8(path), ec.message()),
                    LFS_SOURCE_SITE_CURRENT());
            }
            return std::optional<RoiSidecar>{std::nullopt};
        }

        std::ifstream file;
        if (!lfs::core::open_file_for_read(path, file)) {
            return make_roi_error(
                lfs::ErrorCode::DataLoss,
                std::format("Failed to open ROI sidecar '{}'", lfs::core::path_to_utf8(path)),
                LFS_SOURCE_SITE_CURRENT());
        }

        const auto json = nlohmann::json::parse(file, nullptr, false);
        const std::string path_utf8 = lfs::core::path_to_utf8(path);
        if (json.is_discarded() || !json.is_object()) {
            return make_roi_error(
                lfs::ErrorCode::InvalidArgument,
                std::format("Invalid ROI sidecar '{}': expected JSON object", path_utf8),
                LFS_SOURCE_SITE_CURRENT());
        }

        const auto version_it = json.find("version");
        if (version_it == json.end() || !version_it->is_number_integer() || version_it->get<int>() != 1) {
            return make_roi_error(
                lfs::ErrorCode::InvalidArgument,
                std::format("Invalid ROI sidecar '{}': version must be 1", path_utf8),
                LFS_SOURCE_SITE_CURRENT());
        }

        const auto type_it = json.find("type");
        if (type_it == json.end() || !type_it->is_string() || type_it->get<std::string>() != "polygon_prism") {
            return make_roi_error(
                lfs::ErrorCode::InvalidArgument,
                std::format("Invalid ROI sidecar '{}': type must be 'polygon_prism'", path_utf8),
                LFS_SOURCE_SITE_CURRENT());
        }

        const auto z_min_it = json.find("z_min");
        const auto z_max_it = json.find("z_max");
        if (z_min_it == json.end() || !z_min_it->is_number() ||
            z_max_it == json.end() || !z_max_it->is_number()) {
            return make_roi_error(
                lfs::ErrorCode::InvalidArgument,
                std::format("Invalid ROI sidecar '{}': missing numeric z_min/z_max", path_utf8),
                LFS_SOURCE_SITE_CURRENT());
        }
        const double z_min = z_min_it->get<double>();
        const double z_max = z_max_it->get<double>();
        if (!std::isfinite(z_min) || !std::isfinite(z_max) || !(z_max > z_min + 1e-4)) {
            return make_roi_error(
                lfs::ErrorCode::InvalidArgument,
                std::format("Invalid ROI sidecar '{}': z_max ({}) must exceed z_min ({})", path_utf8, z_max, z_min),
                LFS_SOURCE_SITE_CURRENT());
        }

        const auto verts_it = json.find("vertices_xy");
        if (verts_it == json.end() || !verts_it->is_array() ||
            verts_it->size() < lfs::core::crop_polygon::kMinVertices ||
            verts_it->size() > lfs::core::crop_polygon::kMaxVertices) {
            return make_roi_error(
                lfs::ErrorCode::InvalidArgument,
                std::format("Invalid ROI sidecar '{}': vertices_xy must contain {} to {} vertices",
                            path_utf8, lfs::core::crop_polygon::kMinVertices, lfs::core::crop_polygon::kMaxVertices),
                LFS_SOURCE_SITE_CURRENT());
        }

        RoiSidecar sidecar;
        sidecar.z_min = z_min;
        sidecar.z_max = z_max;
        if (const auto it = json.find("source"); it != json.end() && it->is_string()) {
            sidecar.source = it->get<std::string>();
        }
        if (const auto it = json.find("coordinate_frame"); it != json.end() && it->is_string()) {
            sidecar.coordinate_frame = it->get<std::string>();
        }

        sidecar.vertices_xy.reserve(verts_it->size());
        for (const auto& item : *verts_it) {
            if (!item.is_array() || item.size() != 2 || !item[0].is_number() || !item[1].is_number()) {
                return make_roi_error(
                    lfs::ErrorCode::InvalidArgument,
                    std::format("Invalid ROI sidecar '{}': vertices_xy entries must be [x, y] number pairs", path_utf8),
                    LFS_SOURCE_SITE_CURRENT());
            }
            const double x = item[0].get<double>();
            const double y = item[1].get<double>();
            if (!std::isfinite(x) || !std::isfinite(y)) {
                return make_roi_error(
                    lfs::ErrorCode::InvalidArgument,
                    std::format("Invalid ROI sidecar '{}': non-finite vertex coordinates", path_utf8),
                    LFS_SOURCE_SITE_CURRENT());
            }
            sidecar.vertices_xy.emplace_back(x, y);
        }

        sidecar.area_m2 = polygon_area_d(sidecar.vertices_xy);
        const auto placement = make_roi_crop_polygon(sidecar);
        if (!lfs::core::crop_polygon::isValidOutline(placement.data.vertices) ||
            !lfs::core::crop_polygon::isValidHalfDepth(placement.data.half_depth) ||
            !(sidecar.area_m2 > 1e-4)) {
            return make_roi_error(
                lfs::ErrorCode::InvalidArgument,
                std::format("Invalid ROI sidecar '{}': degenerate polygon outline or half depth", path_utf8),
                LFS_SOURCE_SITE_CURRENT());
        }

        return std::optional<RoiSidecar>{std::move(sidecar)};
    }

    RoiCropPolygonPlacement make_roi_crop_polygon(
        const RoiSidecar& sidecar,
        const glm::vec3& training_origin) {
        double min_x = std::numeric_limits<double>::infinity();
        double max_x = -std::numeric_limits<double>::infinity();
        double min_y = std::numeric_limits<double>::infinity();
        double max_y = -std::numeric_limits<double>::infinity();
        for (const auto& v : sidecar.vertices_xy) {
            min_x = std::min(min_x, v.x);
            max_x = std::max(max_x, v.x);
            min_y = std::min(min_y, v.y);
            max_y = std::max(max_y, v.y);
        }
        const double cx = 0.5 * (min_x + max_x);
        const double cy = 0.5 * (min_y + max_y);
        const double cz = 0.5 * (sidecar.z_min + sidecar.z_max);

        RoiCropPolygonPlacement placement;
        placement.data.vertices.reserve(sidecar.vertices_xy.size());
        for (const auto& v : sidecar.vertices_xy) {
            placement.data.vertices.emplace_back(
                static_cast<float>(v.x - cx),
                static_cast<float>(v.y - cy));
        }
        placement.data.half_depth = static_cast<float>(0.5 * (sidecar.z_max - sidecar.z_min));
        placement.data.inverse = false;
        placement.data.enabled = true;

        const glm::vec3 center(
            static_cast<float>(cx - static_cast<double>(training_origin.x)),
            static_cast<float>(cy - static_cast<double>(training_origin.y)),
            static_cast<float>(cz - static_cast<double>(training_origin.z)));
        placement.local_transform = glm::translate(glm::mat4{1.0f}, center);
        return placement;
    }

    TrainingCropBoxGeometry make_roi_eval_geometry(
        const RoiSidecar& sidecar,
        const glm::vec3& training_origin) {
        const auto placement = make_roi_crop_polygon(sidecar, training_origin);
        const glm::vec2 extents = lfs::core::crop_polygon::outlineHalfExtents(placement.data.vertices);
        const glm::mat4 world_to_polygon = glm::inverse(placement.local_transform);
        return TrainingCropBoxGeometry{
            .min = glm::vec3(-extents, -placement.data.half_depth),
            .max = glm::vec3(extents, placement.data.half_depth),
            .world_to_cropbox = world_to_polygon,
            .model_to_cropbox = world_to_polygon,
            .inverse = false,
            .polygon = placement.data.vertices};
    }

    std::string format_roi_summary(
        const TrainingCropBoxGeometry& geometry,
        const std::string_view origin_label) {
        if (geometry.is_polygon()) {
            double twice_area = 0.0;
            const size_t n = geometry.polygon.size();
            for (size_t i = 0; i < n; ++i) {
                const glm::vec2 a = geometry.polygon[i];
                const glm::vec2 b = geometry.polygon[(i + 1) % n];
                twice_area += static_cast<double>(a.x) * b.y - static_cast<double>(b.x) * a.y;
            }
            const double area_m2 = 0.5 * std::abs(twice_area);
            const glm::mat4 crop_to_world = glm::inverse(geometry.world_to_cropbox);
            float z_min = std::numeric_limits<float>::max();
            float z_max = std::numeric_limits<float>::lowest();
            for (int mask = 0; mask < 8; ++mask) {
                const glm::vec3 corner{
                    (mask & 1) ? geometry.max.x : geometry.min.x,
                    (mask & 2) ? geometry.max.y : geometry.min.y,
                    (mask & 4) ? geometry.max.z : geometry.min.z};
                const glm::vec4 world_p = crop_to_world * glm::vec4(corner, 1.0f);
                z_min = std::min(z_min, world_p.z);
                z_max = std::max(z_max, world_p.z);
            }
            return std::format("{} ({} vertices, z=[{:.2f}, {:.2f}], area={:.1f} m^2{})",
                               origin_label, n, z_min, z_max, area_m2,
                               geometry.inverse ? ", inverted" : "");
        }
        return std::format("{}{}", origin_label, geometry.inverse ? " (inverted)" : "");
    }

} // namespace lfs::training
