/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/native_fs.hpp"
#include "rasterization/gsplat/TsdfFusion.h"
#include "splat_mesh.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <numbers>
#include <vector>

using lfs::training::SplatMesh;

namespace {
    uint64_t key_of(const int x, const int y, const int z) {
        return uint64_t(x + gsplat_lfs::kTsdfKeyOffset) | (uint64_t(y + gsplat_lfs::kTsdfKeyOffset) << gsplat_lfs::kTsdfKeyBits) |
               (uint64_t(z + gsplat_lfs::kTsdfKeyOffset) << (2 * gsplat_lfs::kTsdfKeyBits));
    }

    // Truncated signed distance of a unit sphere, positive outside, on sorted keys.
    void sphere_voxels(const float voxel, std::vector<uint64_t>& keys, std::vector<float>& tsdf) {
        const int extent = int(1.3f / voxel);
        std::vector<std::pair<uint64_t, float>> voxels;
        const float trunc = 4.f * voxel;
        for (int z = -extent; z <= extent; ++z)
            for (int y = -extent; y <= extent; ++y)
                for (int x = -extent; x <= extent; ++x) {
                    const float sdf = std::hypot(x * voxel, y * voxel, z * voxel) - 1.f;
                    if (std::abs(sdf) < trunc)
                        voxels.emplace_back(key_of(x, y, z), sdf / trunc);
                }
        std::ranges::sort(voxels);
        for (const auto& [key, value] : voxels) {
            keys.push_back(key);
            tsdf.push_back(value);
        }
    }

    SplatMesh sphere_mesh(const float voxel) {
        std::vector<uint64_t> keys;
        std::vector<float> tsdf;
        sphere_voxels(voxel, keys, tsdf);
        SplatMesh mesh;
        lfs::training::surface_nets(keys, tsdf, voxel, mesh);
        return mesh;
    }
} // namespace

TEST(SplatMesh, SurfaceNetsSitsOnTheSphereFacingOutward) {
    constexpr float voxel = .04f;
    auto mesh = sphere_mesh(voxel);
    ASSERT_FALSE(mesh.faces.empty());
    lfs::training::compute_splat_mesh_normals(mesh);
    EXPECT_NEAR(mesh.area / (4. * std::numbers::pi), 1., .1);
    size_t outward = 0;
    for (size_t i = 0; i < mesh.vertices.size(); ++i) {
        const auto& p = mesh.vertices[i];
        const auto& n = mesh.normals[i];
        EXPECT_NEAR(std::hypot(p[0], p[1], p[2]), 1.f, voxel / 2);
        outward += n[0] * p[0] + n[1] * p[1] + n[2] * p[2] > 0;
    }
    EXPECT_GT(double(outward) / mesh.vertices.size(), .999);
}

TEST(SplatMesh, JumpsAcrossTheBandAreNotSurfaces) {
    const std::vector<uint64_t> keys{key_of(0, 0, 0), key_of(1, 0, 0)};
    const std::vector<float> tsdf{.9f, -.9f};
    SplatMesh mesh;
    lfs::training::surface_nets(keys, tsdf, 1.f, mesh);
    EXPECT_TRUE(mesh.faces.empty());
}

TEST(SplatMesh, SmallComponentsGoAndLargeOnesStay) {
    SplatMesh mesh;
    mesh.vertices.resize(7);
    for (size_t i = 0; i < 7; ++i)
        mesh.vertices[i] = {float(i), 0.f, 0.f};
    mesh.faces = {{0, 1, 2}, {0, 2, 3}, {0, 3, 1}, {4, 5, 6}};
    lfs::training::remove_small_mesh_components(mesh, 2);
    EXPECT_EQ(mesh.vertices.size(), 4u);
    EXPECT_EQ(mesh.faces.size(), 3u);
    EXPECT_EQ(mesh.components_kept, 1u);
    EXPECT_EQ(mesh.components_total, 2u);
}

namespace {
    // A flat grid of 2 cm cells facing +z, with the listed cells left out.
    SplatMesh grid_without(const std::vector<std::pair<int, int>>& missing) {
        constexpr int n = 41;
        SplatMesh mesh;
        for (int y = 0; y < n; ++y)
            for (int x = 0; x < n; ++x)
                mesh.vertices.push_back({x * .02f, y * .02f, 0.f});
        for (int y = 0; y + 1 < n; ++y)
            for (int x = 0; x + 1 < n; ++x) {
                if (std::ranges::find(missing, std::pair{x, y}) != missing.end())
                    continue;
                const int a = y * n + x;
                mesh.faces.push_back({a, a + 1, a + n + 1});
                mesh.faces.push_back({a, a + n + 1, a + n});
            }
        return mesh;
    }

    size_t rim_edges(const SplatMesh& mesh) {
        std::vector<std::pair<int32_t, int32_t>> edges;
        for (const auto& f : mesh.faces)
            for (int i = 0; i < 3; ++i)
                edges.emplace_back(f[i], f[(i + 1) % 3]);
        std::ranges::sort(edges);
        return std::ranges::count_if(edges, [&](const auto& e) { return !std::ranges::binary_search(edges, std::pair{e.second, e.first}); });
    }
} // namespace

TEST(SplatMesh, FillClosesSmallHolesFacingTheSameWay) {
    auto mesh = grid_without({{10, 10}, {11, 10}, {30, 30}});
    const auto outer_rim = rim_edges(grid_without({}));
    const auto before = mesh.faces.size();
    EXPECT_EQ(lfs::training::fill_small_mesh_holes(mesh, .3f), 2u);
    EXPECT_GT(mesh.faces.size(), before);
    EXPECT_EQ(rim_edges(mesh), outer_rim);
    lfs::training::compute_splat_mesh_normals(mesh);
    EXPECT_NEAR(mesh.area, .8 * .8, 1e-4);
    for (const auto& n : mesh.normals)
        EXPECT_GT(n[2], .99f);
}

TEST(SplatMesh, FillLeavesLargeHolesAndTheOuterRimOpen) {
    std::vector<std::pair<int, int>> missing;
    for (int y = 10; y < 20; ++y)
        for (int x = 10; x < 20; ++x)
            missing.emplace_back(x, y);
    auto mesh = grid_without(missing);
    const auto faces = mesh.faces.size();
    EXPECT_EQ(lfs::training::fill_small_mesh_holes(mesh, .3f), 0u);
    EXPECT_EQ(mesh.faces.size(), faces);
    EXPECT_EQ(lfs::training::fill_small_mesh_holes(mesh, 0.f), 0u);
}

TEST(SplatMesh, SimplifyStaysOnTheSurfaceWithFewerFaces) {
    constexpr float voxel = .04f, tolerance = .01f;
    auto mesh = sphere_mesh(voxel);
    mesh.colors.assign(mesh.vertices.size(), {10, 20, 30});
    const size_t before = mesh.faces.size();
    lfs::training::simplify_splat_mesh(mesh, tolerance, 1.f);
    EXPECT_LT(mesh.faces.size(), before / 2);
    EXPECT_EQ(mesh.faces_before_simplify, before);
    ASSERT_EQ(mesh.normals.size(), mesh.vertices.size());
    ASSERT_EQ(mesh.colors.size(), mesh.vertices.size());
    EXPECT_NEAR(mesh.area / (4. * std::numbers::pi), 1., .1);
    std::vector<char> used(mesh.vertices.size(), 0);
    for (const auto& face : mesh.faces)
        for (const auto vertex : face) {
            ASSERT_GE(vertex, 0);
            ASSERT_LT(size_t(vertex), mesh.vertices.size());
            used[vertex] = 1;
        }
    EXPECT_EQ(std::ranges::count(used, char{0}), 0);
    for (size_t i = 0; i < mesh.vertices.size(); ++i) {
        const auto& p = mesh.vertices[i];
        EXPECT_NEAR(std::hypot(p[0], p[1], p[2]), 1.f, voxel / 2 + 2 * tolerance);
        EXPECT_EQ(mesh.colors[i], (std::array<uint8_t, 3>{10, 20, 30}));
    }
}

TEST(SplatMesh, SimplifyWithoutToleranceKeepsEveryFace) {
    auto mesh = sphere_mesh(.04f);
    const auto faces = mesh.faces;
    const auto vertices = mesh.vertices;
    lfs::training::simplify_splat_mesh(mesh, 0.f, 1.f);
    EXPECT_EQ(mesh.faces, faces);
    EXPECT_EQ(mesh.vertices, vertices);
}

TEST(SplatMesh, SimplifyInPiecesLeavesNoCracksAndStaysOnTheSurface) {
    constexpr float voxel = .04f, tolerance = .01f;
    auto mesh = sphere_mesh(voxel);
    mesh.colors.assign(mesh.vertices.size(), {10, 20, 30});
    ASSERT_EQ(rim_edges(mesh), 0u);
    const size_t before = mesh.faces.size();
    lfs::training::simplify_splat_mesh(mesh, tolerance, 1.f, 0, before / 16);
    EXPECT_LT(mesh.faces.size(), before * 3 / 4);
    EXPECT_EQ(rim_edges(mesh), 0u);
    ASSERT_EQ(mesh.normals.size(), mesh.vertices.size());
    ASSERT_EQ(mesh.colors.size(), mesh.vertices.size());
    EXPECT_NEAR(mesh.area / (4. * std::numbers::pi), 1., .1);
    for (const auto& p : mesh.vertices)
        EXPECT_NEAR(std::hypot(p[0], p[1], p[2]), 1.f, 2.f * tolerance + voxel);
    for (const auto& c : mesh.colors)
        EXPECT_EQ(c, (std::array<uint8_t, 3>{10, 20, 30}));
}

TEST(SplatMesh, SimplifyKeepsMoreFacesWhereTheColorChanges) {
    // A flat grid has no shape to keep, so only the color stripes can hold faces back.
    const auto stripes = [] {
        constexpr int n = 101;
        SplatMesh mesh;
        for (int y = 0; y < n; ++y)
            for (int x = 0; x < n; ++x) {
                mesh.vertices.push_back({x * .02f, y * .02f, 0.f});
                const auto level = static_cast<uint8_t>((x / 5) % 2 ? 255 : 0);
                mesh.colors.push_back({level, level, level});
            }
        for (int y = 0; y + 1 < n; ++y)
            for (int x = 0; x + 1 < n; ++x) {
                const int a = y * n + x;
                mesh.faces.push_back({a, a + 1, a + n + 1});
                mesh.faces.push_back({a, a + n + 1, a + n});
            }
        return mesh;
    };
    auto plain = stripes(), colored = stripes();
    lfs::training::simplify_splat_mesh(plain, .01f, 0.f);
    lfs::training::simplify_splat_mesh(colored, .01f, 20.f);
    EXPECT_LT(plain.faces.size() * 4, colored.faces.size());
}

TEST(SplatMesh, PlyHasTheHeaderCountsAndRecordSizes) {
    SplatMesh mesh;
    mesh.vertices = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    mesh.normals = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}};
    mesh.colors = {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}};
    mesh.faces = {{0, 1, 2}};
    const auto path = std::filesystem::temp_directory_path() / "lfs_splat_mesh_test.ply";
    lfs::training::write_splat_mesh_ply(mesh, path);
    std::ifstream in(path, std::ios::binary);
    std::string text((std::istreambuf_iterator<char>(in)), {});
    in.close();
    std::filesystem::remove(path);
    const auto end = text.find("end_header\n");
    ASSERT_NE(end, std::string::npos);
    EXPECT_NE(text.find("element vertex 3\n"), std::string::npos);
    EXPECT_NE(text.find("element face 1\n"), std::string::npos);
    EXPECT_EQ(text.size() - (end + 11), 3u * 27u + 13u);
}

namespace {
    class SplatMeshGpu : public ::testing::Test {
        void SetUp() override {
            int n = 0;
            if (cudaGetDeviceCount(&n) != cudaSuccess || !n)
                GTEST_SKIP() << "CUDA device unavailable";
        }
    };

    template <class T>
    struct Device {
        T* data = nullptr;
        explicit Device(const size_t n) {
            EXPECT_EQ(cudaMalloc(&data, std::max<size_t>(n, 1) * sizeof(T)), cudaSuccess);
            EXPECT_EQ(cudaMemset(data, 0, std::max<size_t>(n, 1) * sizeof(T)), cudaSuccess);
        }
        ~Device() { (void)cudaFree(data); }
    };

    constexpr uint32_t kSize = 200;

    gsplat_lfs::TsdfView pinhole_at(const float camera_z) {
        gsplat_lfs::TsdfView view;
        view.width = view.height = kSize;
        view.fx = view.fy = 200.f;
        view.cx = view.cy = 100.f;
        view.viewmat[0] = view.viewmat[5] = view.viewmat[10] = view.viewmat[15] = 1.f;
        view.viewmat[11] = -camera_z;
        return view;
    }

    // Fuses flat walls at z = surface seen straight on; returns voxels on sorted keys.
    struct Wall {
        float camera_z;
        float surface_z;
        bool floater = false; // a patch in the middle of the view reads as halfway to the wall
    };
    void fuse(const std::vector<Wall>& walls, const gsplat_lfs::TsdfGrid& grid, const uint32_t min_views,
              std::vector<uint64_t>& out_keys, std::vector<float>& out_tsdf) {
        const size_t samples = 3 * size_t((kSize + 1) / 2) * ((kSize + 1) / 2);
        std::vector<float> rays(size_t(kSize) * kSize * 3);
        for (uint32_t y = 0; y < kSize; ++y)
            for (uint32_t x = 0; x < kSize; ++x) {
                rays[3 * (y * kSize + x)] = (x + .5f - 100.f) / 200.f;
                rays[3 * (y * kSize + x) + 1] = (y + .5f - 100.f) / 200.f;
                rays[3 * (y * kSize + x) + 2] = 1.f;
            }
        Device<float> rays_d(rays.size()), depth_d(size_t(kSize) * kSize);
        ASSERT_EQ(cudaMemcpy(rays_d.data, rays.data(), rays.size() * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
        const auto upload = [&](const Wall& wall) {
            std::vector<float> depth(size_t(kSize) * kSize, wall.surface_z - wall.camera_z);
            if (wall.floater)
                for (uint32_t y = 90; y < 110; ++y)
                    for (uint32_t x = 90; x < 110; ++x)
                        depth[y * kSize + x] = .5f * (wall.surface_z - wall.camera_z);
            ASSERT_EQ(cudaMemcpy(depth_d.data, depth.data(), depth.size() * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
        };
        Device<uint64_t> keys(samples * walls.size()), scratch(samples * walls.size()), counter(1);
        for (size_t w = 0; w < walls.size(); ++w) {
            upload(walls[w]);
            gsplat_lfs::tsdf_touch_keys(depth_d.data, rays_d.data, pinhole_at(walls[w].camera_z), grid, keys.data + w * samples, nullptr);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        }
        const size_t blocks = gsplat_lfs::tsdf_sort_unique(keys.data, scratch.data, samples * walls.size(), counter.data, nullptr);
        ASSERT_GT(blocks, 0u);
        const size_t voxels = blocks * grid.block * grid.block * grid.block;
        Device<float> sum(voxels), weight(voxels), carve(blocks);
        Device<uint16_t> views(voxels);
        Device<uint32_t> near_blocks(blocks), near_count(1);
        const size_t tiles = size_t(gsplat_lfs::tsdf_tile_count(kSize)) * gsplat_lfs::tsdf_tile_count(kSize);
        Device<float> raw_low(tiles), raw_high(tiles), low(tiles), high(tiles);
        for (const auto& wall : walls) {
            upload(wall);
            const auto view = pinhole_at(wall.camera_z);
            gsplat_lfs::tsdf_tile_bounds(depth_d.data, kSize, kSize, raw_low.data, raw_high.data, low.data, high.data, nullptr);
            const uint32_t near_found = gsplat_lfs::tsdf_classify_blocks(keys.data, uint32_t(blocks), view, grid, low.data, high.data,
                                                                         carve.data, near_blocks.data, near_count.data, nullptr);
            gsplat_lfs::tsdf_integrate(keys.data, near_blocks.data, near_found, view, grid, depth_d.data, sum.data, weight.data, views.data, nullptr);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        }
        const uint64_t count = gsplat_lfs::tsdf_collect_voxels(keys.data, uint32_t(blocks), 0, uint32_t(blocks), grid, sum.data, weight.data, views.data, carve.data,
                                                               min_views, nullptr, nullptr, counter.data, nullptr);
        ASSERT_GT(count, 0u);
        Device<uint64_t> found_keys(count);
        Device<float> found_tsdf(count);
        (void)gsplat_lfs::tsdf_collect_voxels(keys.data, uint32_t(blocks), 0, uint32_t(blocks), grid, sum.data, weight.data, views.data, carve.data,
                                              min_views, found_keys.data, found_tsdf.data, counter.data, nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        std::vector<uint64_t> host_keys(count);
        std::vector<float> host_tsdf(count);
        ASSERT_EQ(cudaMemcpy(host_keys.data(), found_keys.data, count * 8, cudaMemcpyDeviceToHost), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(host_tsdf.data(), found_tsdf.data, count * 4, cudaMemcpyDeviceToHost), cudaSuccess);
        std::vector<uint32_t> order(count);
        for (uint32_t i = 0; i < count; ++i)
            order[i] = i;
        std::ranges::sort(order, [&](const uint32_t a, const uint32_t b) { return host_keys[a] < host_keys[b]; });
        for (const auto i : order) {
            out_keys.push_back(host_keys[i]);
            out_tsdf.push_back(host_tsdf[i]);
        }
    }

    float central_surface_z(const std::vector<uint64_t>& keys, const std::vector<float>& tsdf, const float voxel) {
        SplatMesh mesh;
        lfs::training::surface_nets(keys, tsdf, voxel, mesh);
        std::vector<float> z;
        for (const auto& p : mesh.vertices)
            if (std::abs(p[0]) < .1f && std::abs(p[1]) < .1f)
                z.push_back(p[2]);
        EXPECT_FALSE(z.empty());
        if (z.empty())
            return NAN;
        std::ranges::nth_element(z, z.begin() + z.size() / 2);
        return z[z.size() / 2];
    }
} // namespace

TEST_F(SplatMeshGpu, ASingleViewPutsTheSurfaceOnTheWall) {
    std::vector<uint64_t> keys;
    std::vector<float> tsdf;
    fuse({{-1.f, .1f}}, {.voxel = .02f, .block = 8, .trunc = .08f, .full_weight_range = 0.f}, 1, keys, tsdf);
    EXPECT_NEAR(central_surface_z(keys, tsdf, .02f), .1f, .002f);
}

TEST_F(SplatMeshGpu, RangeWeightFollowsTheCloseView) {
    const std::vector<Wall> walls{{-1.f, 0.f}, {-5.f, .04f}};
    std::vector<uint64_t> keys;
    std::vector<float> tsdf;
    fuse(walls, {.voxel = .02f, .block = 8, .trunc = .08f, .full_weight_range = 0.f}, 1, keys, tsdf);
    const float equal = central_surface_z(keys, tsdf, .02f);
    EXPECT_GT(equal, .015f);
    EXPECT_LT(equal, .025f);
    keys.clear();
    tsdf.clear();
    fuse(walls, {.voxel = .02f, .block = 8, .trunc = .08f, .full_weight_range = 1.f}, 1, keys, tsdf);
    EXPECT_LT(central_surface_z(keys, tsdf, .02f), .005f);
}

TEST_F(SplatMeshGpu, AOneViewFloaterDoesNotSurvive) {
    // The first view reports a surface halfway to the wall; the others look through it.
    std::vector<uint64_t> keys;
    std::vector<float> tsdf;
    fuse({{-1.f, 0.f, true}, {-1.f, 0.f}, {-1.f, 0.f}}, {.voxel = .02f, .block = 8, .trunc = .08f, .full_weight_range = 0.f}, 2, keys, tsdf);
    SplatMesh mesh;
    lfs::training::surface_nets(keys, tsdf, .02f, mesh);
    ASSERT_FALSE(mesh.faces.empty());
    for (const auto& p : mesh.vertices)
        EXPECT_GT(p[2], -.15f);
}
