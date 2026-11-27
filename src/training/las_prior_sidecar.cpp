/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "las_prior_sidecar.hpp"

#include "core/logger.hpp"
#include "core/path_utils.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string_view>

namespace lfs::training {

    namespace {
        constexpr std::string_view kSidecarName = "supervision.json";
        constexpr std::string_view kLinearDepthEncoding = "linear_[min_d,max_d]_to_[0.02,1.0]*65535";

        [[nodiscard]] std::string string_field(const nlohmann::json& json, const char* key) {
            const auto it = json.find(key);
            return it != json.end() && it->is_string() ? it->get<std::string>() : std::string{};
        }

        [[nodiscard]] std::string normalize_rel_name(std::string name) {
            std::replace(name.begin(), name.end(), '\\', '/');
            return name;
        }
    } // namespace

    std::optional<LasPriorSidecar> read_las_prior_sidecar(const std::filesystem::path& dataset_path) {
        const auto path = dataset_path / kSidecarName;
        std::error_code ec;
        if (!lfs::core::native_fs::exists(path, ec)) {
            return std::nullopt;
        }
        std::ifstream file;
        if (!lfs::core::open_file_for_read(path, file)) {
            LOG_WARN("Could not open {}", lfs::core::path_to_utf8(path));
            return std::nullopt;
        }
        const auto json = nlohmann::json::parse(file, nullptr, false);
        if (json.is_discarded() || !json.is_object()) {
            LOG_WARN("Ignoring {}: not a JSON object", lfs::core::path_to_utf8(path));
            return std::nullopt;
        }
        LasPriorSidecar sidecar;
        const auto depth_format = string_field(json, "depth_format");
        const auto depth_source = string_field(json, "depth_source");
        const auto source = string_field(json, "source");
        sidecar.ssi_depth =
            depth_format == "uint16_png_ssi" ||
            depth_source == "moge2_ssi" ||
            source == "moge2_ssi";
        sidecar.ray_distance = string_field(json, "depth_type") == "ray_distance";
        sidecar.normals_camera_opencv = string_field(json, "normal_frame").starts_with("opencv_camera");
        sidecar.depth_alignment = string_field(json, "depth_alignment");
        const auto depth_encoding = string_field(json, "depth_encoding");
        if (sidecar.depth_alignment.starts_with("las_") && depth_encoding == kLinearDepthEncoding) {
            if (const auto imgs_it = json.find("images");
                imgs_it != json.end() && imgs_it->is_array()) {
                sidecar.metric_depth_ranges.reserve(imgs_it->size());
                for (const auto& entry : *imgs_it) {
                    if (!entry.is_object()) {
                        continue;
                    }
                    const auto name_it = entry.find("name");
                    const auto mono_it = entry.find("monocular");
                    if (name_it == entry.end() || !name_it->is_string() ||
                        mono_it == entry.end() || !mono_it->is_object()) {
                        continue;
                    }
                    const auto min_it = mono_it->find("min_distance_m");
                    const auto max_it = mono_it->find("max_distance_m");
                    if (min_it == mono_it->end() || !min_it->is_number() ||
                        max_it == mono_it->end() || !max_it->is_number()) {
                        continue;
                    }
                    const double min_d = min_it->get<double>();
                    const double max_d = max_it->get<double>();
                    if (!std::isfinite(min_d) || !std::isfinite(max_d) ||
                        min_d <= 0.0 || max_d <= min_d + 1.0e-4) {
                        continue;
                    }
                    sidecar.metric_depth_ranges.emplace(
                        normalize_rel_name(name_it->get<std::string>()),
                        LasPriorSidecar::MetricDepthRange{
                            static_cast<float>(min_d),
                            static_cast<float>(max_d)});
                }
                sidecar.metric_aligned = !sidecar.metric_depth_ranges.empty();
            }
        }
        if (!sidecar.ssi_depth && !sidecar.normals_camera_opencv && !sidecar.metric_aligned) {
            return std::nullopt;
        }
        return sidecar;
    }

    std::vector<std::string> apply_las_prior_defaults(
        lfs::core::param::OptimizationParameters& opt,
        const LasPriorSidecar& sidecar) {
        using lfs::core::param::NormalLossSpace;
        std::vector<std::string> changes;
        if (sidecar.ssi_depth) {
            if (opt.depth_auto_generate) {
                opt.depth_auto_generate = false;
                changes.emplace_back("depth_auto_generate off");
            }
        }
        if (sidecar.normals_camera_opencv) {
            if (opt.normal_loss_space == NormalLossSpace::Auto) {
                opt.normal_loss_space = NormalLossSpace::CameraOpenCV;
                changes.emplace_back("normal_loss_space auto -> camera-opencv");
            }
            if (opt.normal_auto_generate) {
                opt.normal_auto_generate = false;
                changes.emplace_back("normal_auto_generate off");
            }
        }
        return changes;
    }

} // namespace lfs::training
