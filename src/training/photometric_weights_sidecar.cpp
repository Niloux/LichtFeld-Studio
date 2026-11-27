/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "photometric_weights_sidecar.hpp"

#include "core/logger.hpp"
#include "core/native_fs.hpp"
#include "core/path_utils.hpp"

#include <cstring>
#include <fstream>
#include <memory>
#include <stb_image.h>

namespace lfs::training {

    namespace {
        constexpr size_t kMaxCudaResidentWeightBytes = 256ULL * 1024 * 1024;

        [[nodiscard]] std::filesystem::path resolve_weight_path(
            const std::filesystem::path& weights_dir,
            const std::string& image_name) {
            const auto rel = lfs::core::utf8_to_path(image_name);
            auto candidate = weights_dir / lfs::core::utf8_to_path(image_name + ".png");
            std::error_code ec;
            if (lfs::core::native_fs::is_regular_file(candidate, ec))
                return candidate;

            auto stem_rel = rel;
            stem_rel.replace_extension(".png");
            candidate = weights_dir / stem_rel;
            if (lfs::core::native_fs::is_regular_file(candidate, ec))
                return candidate;

            return {};
        }

        [[nodiscard]] lfs::core::Tensor decode_u8_grayscale_png(
            const std::filesystem::path& path) {
            std::ifstream file;
            if (!lfs::core::open_file_for_read(path, std::ios::binary, file))
                return {};
            file.seekg(0, std::ios::end);
            const auto size = file.tellg();
            if (size <= 0)
                return {};
            file.seekg(0, std::ios::beg);
            std::vector<stbi_uc> bytes(static_cast<size_t>(size));
            if (!file.read(reinterpret_cast<char*>(bytes.data()), size))
                return {};

            int w = 0, h = 0, comp = 0;
            const std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> decoded(
                stbi_load_from_memory(bytes.data(), static_cast<int>(bytes.size()), &w, &h, &comp, 1),
                stbi_image_free);
            if (!decoded || w <= 0 || h <= 0)
                return {};

            auto tensor = lfs::core::Tensor::empty(
                {static_cast<size_t>(h), static_cast<size_t>(w)},
                lfs::core::Device::CPU,
                lfs::core::DataType::UInt8);
            std::memcpy(tensor.ptr<uint8_t>(), decoded.get(), static_cast<size_t>(w) * static_cast<size_t>(h));
            return tensor;
        }
    } // namespace

    void PhotometricWeightsSidecar::load(
        const std::filesystem::path& dataset_path,
        const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras) {
        clear();
        if (dataset_path.empty() || cameras.empty())
            return;

        const auto weights_dir = dataset_path / "weights";
        std::error_code ec;
        if (!lfs::core::native_fs::is_directory(weights_dir, ec))
            return;

        size_t total_cuda_bytes = 0;
        size_t cpu_fallback_count = 0;
        for (const auto& cam : cameras) {
            if (!cam || cam->is_undistort_prepared())
                continue;
            const auto path = resolve_weight_path(weights_dir, cam->image_name());
            if (path.empty())
                continue;
            // Decode on CPU first when approaching the 256 MiB VRAM resident cap.
            const bool try_cuda = total_cuda_bytes < kMaxCudaResidentWeightBytes;
            auto tensor = decode_u8_grayscale_png(path);
            if (!tensor.is_valid()) {
                LOG_WARN("Failed to decode photometric weight map '{}'", lfs::core::path_to_utf8(path));
                continue;
            }
            const size_t bytes = tensor.numel();
            if (try_cuda && total_cuda_bytes + bytes <= kMaxCudaResidentWeightBytes) {
                tensor = tensor.cuda();
                total_cuda_bytes += bytes;
            } else {
                ++cpu_fallback_count;
            }
            weights_by_uid_.emplace(cam->uid(), std::move(tensor));
        }

        if (!weights_by_uid_.empty()) {
            LOG_INFO("Photometric diffuse weights loaded for {}/{} training cameras ({:.1f} MiB CUDA, {} CPU)",
                     weights_by_uid_.size(), cameras.size(),
                     static_cast<double>(total_cuda_bytes) / (1024.0 * 1024.0),
                     cpu_fallback_count);
        }
    }

    void PhotometricWeightsSidecar::clear() {
        weights_by_uid_.clear();
    }

    lfs::core::Tensor PhotometricWeightsSidecar::weight_for_camera(const int uid) const {
        const auto it = weights_by_uid_.find(uid);
        return it != weights_by_uid_.end() ? it->second : lfs::core::Tensor{};
    }

} // namespace lfs::training
