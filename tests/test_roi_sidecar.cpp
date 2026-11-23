/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/image_io.hpp"
#include "core/image_loader.hpp"
#include "core/parameters.hpp"
#include "core/tensor.hpp"
#include "io/cache_image_loader.hpp"
#include "training/metrics/metrics.hpp"
#include "training/roi_sidecar.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cuda_runtime.h>
#include <filesystem>
#include <format>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

    using lfs::core::Camera;
    using lfs::core::CameraModelType;
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::Tensor;

    class TempDir {
    public:
        explicit TempDir(const std::string_view prefix = "lfs_roi_sidecar_") {
            static std::atomic<uint64_t> seq{0};
            path_ = fs::temp_directory_path() /
                    std::format("{}_{}_{}", prefix,
                                std::chrono::steady_clock::now().time_since_epoch().count(),
                                seq.fetch_add(1, std::memory_order_relaxed));
            fs::create_directories(path_);
        }
        ~TempDir() {
            std::error_code ec;
            fs::remove_all(path_, ec);
        }
        TempDir(const TempDir&) = delete;
        TempDir& operator=(const TempDir&) = delete;
        [[nodiscard]] const fs::path& path() const { return path_; }

    private:
        fs::path path_;
    };

    void write_text(const fs::path& path, const std::string& text) {
        std::ofstream file(path, std::ios::binary);
        file << text;
    }

    bool cuda_available() {
        int device_count = 0;
        return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
    }

    void ensure_image_loader() {
        static bool initialized = false;
        if (initialized) {
            return;
        }
        lfs::io::CacheLoader::getInstance(false);
        lfs::core::set_image_loader([](const lfs::core::ImageLoadParams& p) {
            return lfs::io::CacheLoader::getInstance().load_cached_image(
                p.path,
                {.resize_factor = p.resize_factor,
                 .max_width = p.max_width,
                 .cuda_stream = p.stream,
                 .output_uint8 = p.output_uint8,
                 .mask_grayscale = p.mask_grayscale});
        });
        initialized = true;
    }

} // namespace

TEST(RoiSidecarTest, MissingFileReturnsEmpty) {
    TempDir dir;
    const auto res = lfs::training::read_roi_sidecar(dir.path());
    ASSERT_TRUE(res.has_value()) << res.error().user_message();
    EXPECT_FALSE(res->has_value());
}

TEST(RoiSidecarTest, ParsesCentersAndAppliesOrigin) {
    TempDir dir;
    write_text(dir.path() / "roi.json", R"({
        "version": 1,
        "type": "polygon_prism",
        "source": "las_camera_seeded_occupancy",
        "coordinate_frame": "geocentric_local",
        "z_min": 8.0,
        "z_max": 20.0,
        "vertices_xy": [
            [100.0, 200.0],
            [110.0, 200.0],
            [110.0, 206.0],
            [104.0, 206.0],
            [104.0, 210.0],
            [100.0, 210.0]
        ]
    })");

    const auto res = lfs::training::read_roi_sidecar(dir.path());
    ASSERT_TRUE(res.has_value()) << res.error().user_message();
    ASSERT_TRUE(res->has_value());
    const auto& roi = **res;
    EXPECT_EQ(roi.vertices_xy.size(), 6u);
    EXPECT_NEAR(roi.area_m2, 76.0, 1e-6);

    const glm::vec3 origin{100.0f, 200.0f, 10.0f};
    const auto placement = lfs::training::make_roi_crop_polygon(roi, origin);
    EXPECT_FLOAT_EQ(placement.data.half_depth, 6.0f);
    EXPECT_TRUE(placement.data.enabled);
    EXPECT_FALSE(placement.data.inverse);

    const auto geom = lfs::training::make_roi_eval_geometry(roi, origin);
    ASSERT_TRUE(geom.is_polygon());

    // Inside concave L-shape after shifting by origin: (102, 208, 12) -> (2, 8, 2)
    const glm::vec3 inside_world{2.0f, 8.0f, 2.0f};
    EXPECT_TRUE(lfs::training::training_crop_region_contains(
        geom, glm::vec3(geom.world_to_cropbox * glm::vec4(inside_world, 1.0f))));

    // In the concave notch (inside AABB, outside polygon): (108, 208, 12) -> (8, 8, 2)
    const glm::vec3 notch_world{8.0f, 8.0f, 2.0f};
    EXPECT_FALSE(lfs::training::training_crop_region_contains(
        geom, glm::vec3(geom.world_to_cropbox * glm::vec4(notch_world, 1.0f))));

    // Above z_max after origin shift: z_max - origin.z = 10
    const glm::vec3 above_world{2.0f, 8.0f, 10.5f};
    EXPECT_FALSE(lfs::training::training_crop_region_contains(
        geom, glm::vec3(geom.world_to_cropbox * glm::vec4(above_world, 1.0f))));

    const auto summary = lfs::training::format_roi_summary(geom, "roi.json polygon");
    EXPECT_NE(summary.find("6 vertices"), std::string::npos);
    EXPECT_NE(summary.find("76.0 m^2"), std::string::npos);
}

TEST(RoiSidecarTest, RejectsMalformedOrInvalidSidecars) {
    const std::array<const char*, 5> bad_payloads{
        "{not valid json",
        R"({"version": 2, "type": "polygon_prism", "z_min": 0, "z_max": 5, "vertices_xy": [[0,0],[1,0],[0,1]]})",
        R"({"version": 1, "type": "box", "z_min": 0, "z_max": 5, "vertices_xy": [[0,0],[1,0],[0,1]]})",
        R"({"version": 1, "type": "polygon_prism", "z_min": 5, "z_max": 5, "vertices_xy": [[0,0],[1,0],[0,1]]})",
        R"({"version": 1, "type": "polygon_prism", "z_min": 0, "z_max": 5, "vertices_xy": [[0,0],[1,0]]})"};
    for (const char* payload : bad_payloads) {
        TempDir dir;
        write_text(dir.path() / "roi.json", payload);
        const auto res = lfs::training::read_roi_sidecar(dir.path());
        EXPECT_FALSE(res.has_value()) << payload;
    }
}

TEST(EvalMaskCropPolygonTest, IntersectsRoiStaticMaskAndRenderedSurface) {
    if (!cuda_available()) {
        GTEST_SKIP() << "No CUDA device available";
    }
    ensure_image_loader();

    TempDir dir("lfs_eval_poly_");
    constexpr int w = 8;
    constexpr int h = 8;
    const auto img_path = dir.path() / "view.png";
    const auto mask_path = dir.path() / "view_mask.png";

    std::vector<uint8_t> rgb(static_cast<size_t>(w) * h * 3, 180);
    ASSERT_TRUE(lfs::core::save_png(img_path, rgb.data(), w, h, 3, 8, 0));

    // segment_and_ignore semantics:
    // y < 2: 128 (sky, excluded), y in [2, 5]: 255 (static foreground, kept), y >= 6: 0 (dynamic/ignore, excluded).
    std::vector<uint8_t> mask_vals(static_cast<size_t>(w) * h, 0);
    for (int y = 0; y < h; ++y) {
        const uint8_t v = y < 2 ? 128 : (y < 6 ? 255 : 0);
        for (int x = 0; x < w; ++x)
            mask_vals[static_cast<size_t>(y) * w + x] = v;
    }
    ASSERT_TRUE(lfs::core::save_png(mask_path, mask_vals.data(), w, h, 1, 8, 0));

    // Camera at world origin looking down +Z with fx=fy=100, cx=cy=4.0.
    // Ray direction at pixel center (x, y) is ((x + 0.5 - 4)/100, (y + 0.5 - 4)/100, 1).
    auto R = Tensor::eye(3, Device::CPU);
    auto T = Tensor::zeros({3}, Device::CPU);
    Camera camera(R, T, 100.0f, 100.0f, 4.0f, 4.0f,
                  Tensor(), Tensor(), CameraModelType::PINHOLE,
                  "view.png", img_path, mask_path, w, h, 0, 0);

    // Polygon covers x in [-0.01, 1.0], y in [-1.0, 1.0], z in [-1.0, 5.0]:
    // Camera is at origin (inside ROI).
    // At d=2.0m: x<=1 gives p.x <= -0.05 (outside ROI); x>=4 gives p.x >= +0.01 (inside ROI).
    lfs::training::RoiSidecar sidecar{
        .vertices_xy = {{-0.01, -1.0}, {1.0, -1.0}, {1.0, 1.0}, {-0.01, 1.0}},
        .z_min = -1.0,
        .z_max = 5.0,
        .source = "test",
        .coordinate_frame = "local",
        .area_m2 = 2.02};
    const auto crop_region = lfs::training::make_roi_eval_geometry(sidecar);

    lfs::core::param::TrainingParameters params;
    params.optimization.mask_mode = lfs::core::param::MaskMode::SegmentAndIgnore;
    params.optimization.eval_mask = "cropbox";
    params.optimization.enable_eval = true;

    // Rendered surface: alpha=1, depth=2.0m across rows 0..7 EXCEPT:
    // - (y=3, x=7): depth=20.0m (> z_max=5.0m -> outside ROI Z)
    // - (y=4, x=6): alpha=0, depth=0 (empty/transparent pixel inside ROI -> excluded)
    std::vector<float> r_depth(static_cast<size_t>(w) * h, 2.0f);
    std::vector<float> r_alpha(static_cast<size_t>(w) * h, 1.0f);
    r_depth[static_cast<size_t>(3) * w + 7] = 20.0f;
    r_depth[static_cast<size_t>(4) * w + 6] = 0.0f;
    r_alpha[static_cast<size_t>(4) * w + 6] = 0.0f;

    const lfs::training::EvaluationRenderFn render_fn =
        [&](Camera&, float) -> lfs::Result<lfs::training::EvaluationRenderResult> {
        return lfs::training::EvaluationRenderResult{
            .output = lfs::training::RenderOutput{
                .image = Tensor::full({3, h, w}, 0.7f, Device::CUDA),
                .alpha = Tensor::from_vector(r_alpha, {h, w}, Device::CUDA),
                .depth = Tensor::from_vector(r_depth, {h, w}, Device::CUDA),
                .width = w,
                .height = h},
            .raw_image = {}};
    };

    const auto prepared = lfs::training::prepare_evaluation_view(
        camera, params, render_fn, nullptr, nullptr,
        lfs::training::EvaluationMaskSources{.crop_region = &crop_region});
    ASSERT_TRUE(prepared.has_value()) << prepared.error().detail();
    EXPECT_TRUE(prepared->user_mask_applied);
    ASSERT_TRUE(prepared->metric_mask.is_valid());

    const auto mask_out = prepared->metric_mask.cpu().contiguous().to_vector_uint8();
    ASSERT_EQ(mask_out.size(), static_cast<size_t>(w) * h);
    auto at = [&](const int y, const int x) { return static_cast<int>(mask_out[static_cast<size_t>(y) * w + x]); };

    // Sky (mask 128) & ignore (mask 0) excluded even inside ROI
    EXPECT_EQ(at(1, 6), 0);
    EXPECT_EQ(at(6, 6), 0);
    // Outside polygon in XY even though mask==255 and d=2.0
    EXPECT_EQ(at(2, 0), 0);
    EXPECT_EQ(at(3, 1), 0);
    // Inside XY & mask==255, but rendered depth 20m > z_max=5m -> outside ROI Z
    EXPECT_EQ(at(3, 7), 0);
    // Solid surface inside ROI & static foreground mask==255 -> included
    EXPECT_EQ(at(2, 6), 1);
    EXPECT_EQ(at(4, 5), 1);
    // Transparent/empty rendered pixel inside camera -> excluded
    EXPECT_EQ(at(4, 6), 0);
}
