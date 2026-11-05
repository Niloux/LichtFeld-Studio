/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "components/ppisp.hpp"
#include "core/native_fs.hpp"
#include "core/tensor.hpp"
#include "rasterization/gsplat/TsdfFusion.h"
#include "splat_mesh.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <numbers>
#include <string>
#include <vector>
#include <zlib.h>

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

TEST(SplatMesh, ReadsBackTheMeshItWrites) {
    SplatMesh mesh;
    mesh.vertices = {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}};
    mesh.colors = {{10, 20, 30}, {40, 50, 60}, {70, 80, 90}};
    mesh.faces = {{0, 1, 2}};
    lfs::training::compute_splat_mesh_normals(mesh);
    const auto path = std::filesystem::temp_directory_path() / "lfs_test_read.ply";
    lfs::training::write_splat_mesh_ply(mesh, path);
    const auto read = lfs::training::read_splat_mesh_ply(path);
    std::filesystem::remove(path);
    EXPECT_EQ(read.vertices, mesh.vertices);
    EXPECT_EQ(read.colors, mesh.colors);
    EXPECT_EQ(read.faces, mesh.faces);
    EXPECT_EQ(read.normals, mesh.normals);
    EXPECT_FLOAT_EQ(float(read.area), .5f);
}

TEST(SplatMesh, ReadsTextPlyOfPolygonsWithoutColors) {
    const auto path = std::filesystem::temp_directory_path() / "lfs_test_read_text.ply";
    {
        std::ofstream out(path);
        out << "ply\nformat ascii 1.0\ncomment by hand\nelement vertex 4\nproperty double x\nproperty float y\nproperty float z\n"
               "element face 1\nproperty list uchar uint vertex_index\nelement extra 1\nproperty int value\nend_header\n"
               "0 0 0\n1 0 0\n1 1 0\n0 1 0\n4 0 1 2 3\n7\n";
    }
    const auto read = lfs::training::read_splat_mesh_ply(path);
    std::filesystem::remove(path);
    ASSERT_EQ(read.vertices.size(), 4u);
    ASSERT_EQ(read.faces.size(), 2u);
    EXPECT_EQ(read.faces[1], (std::array<int32_t, 3>{0, 2, 3}));
    EXPECT_EQ(read.colors[0], (std::array<uint8_t, 3>{128, 128, 128}));
}

TEST(SplatMesh, RefusesPlyWithFacesBeyondItsVertices) {
    const auto path = std::filesystem::temp_directory_path() / "lfs_test_read_bad.ply";
    {
        std::ofstream out(path);
        out << "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\n"
               "element face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n3 0 1 2\n";
    }
    EXPECT_ANY_THROW((void)lfs::training::read_splat_mesh_ply(path));
    std::filesystem::remove(path);
}

TEST(SplatMesh, OsgbFilesHoldOneSceneWhoseBracketsAddUp) {
    lfs::training::TiledMesh tiled;
    auto& mesh = tiled.combined;
    mesh.vertices = {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}};
    mesh.normals.assign(6, {0.f, 0.f, 1.f});
    mesh.uvs = {{0.f, 0.f}, {1.f, 0.f}, {0.f, 1.f}, {0.f, 0.f}, {1.f, 0.f}, {0.f, 1.f}};
    mesh.faces = {{0, 1, 2}, {3, 4, 5}};
    mesh.page_first_face = {0, 1, 2};
    mesh.page_size = 8;
    mesh.pages.assign(2, std::vector<uint8_t>(8 * 8 * 3, 128));
    lfs::training::TileLevel fine;
    fine.page_count = 1;
    fine.vertex_count = 3;
    fine.texel = .01f;
    fine.radius = 1.f;
    auto coarse = fine;
    coarse.level = 1;
    coarse.first_page = 1;
    coarse.first_vertex = 3;
    coarse.texel = .02f;
    tiled.levels = {fine, coarse};

    const auto folder = std::filesystem::temp_directory_path() / "lfs_test_osgb";
    std::filesystem::remove_all(folder);
    lfs::training::write_splat_mesh_osgb(tiled, folder);
    const auto unpack = [](const std::filesystem::path& path) {
        std::ifstream file(path, std::ios::binary);
        std::string data((std::istreambuf_iterator<char>(file)), {});
        EXPECT_GT(data.size(), 28u);
        EXPECT_EQ(data.substr(0, 8), std::string("\xa1\x0e\x91\x6c\x45\x45\xfb\x1a", 8));
        EXPECT_EQ(data.substr(24, 4), "zlib");
        std::string scene(size_t{1} << 20, '\0');
        z_stream stream{};
        EXPECT_EQ(inflateInit2(&stream, 15 + 16), Z_OK);
        stream.next_in = reinterpret_cast<Bytef*>(data.data() + 28);
        stream.avail_in = static_cast<uInt>(data.size() - 28);
        stream.next_out = reinterpret_cast<Bytef*>(scene.data());
        stream.avail_out = static_cast<uInt>(scene.size());
        EXPECT_EQ(inflate(&stream, Z_FINISH), Z_STREAM_END);
        scene.resize(stream.total_out);
        inflateEnd(&stream);
        return scene;
    };
    // The outermost bracket spans the file: its size counts itself and all that follows it.
    const auto outer_bracket_fits = [](const std::string& scene) {
        uint32_t name = 0;
        std::memcpy(&name, scene.data(), 4);
        uint64_t size = 0;
        std::memcpy(&size, scene.data() + 4 + name, 8);
        return size == scene.size() - 4 - name;
    };
    const auto opening = unpack(folder / "Data" / "Tile_+000_+000" / "Tile_+000_+000.osgb");
    EXPECT_EQ(opening.substr(4, 13), "osg::PagedLOD");
    EXPECT_NE(opening.find("Tile_+000_+000_L1.osgb"), std::string::npos);
    EXPECT_NE(opening.find("Tile_+000_+000_0.jpg"), std::string::npos);
    EXPECT_NE(opening.find("\xff\xd8\xff"), std::string::npos);
    EXPECT_TRUE(outer_bracket_fits(opening));
    const auto finest = unpack(folder / "Data" / "Tile_+000_+000" / "Tile_+000_+000_L1.osgb");
    EXPECT_EQ(finest.substr(4, 10), "osg::Geode");
    EXPECT_EQ(finest.find("PagedLOD"), std::string::npos);
    EXPECT_TRUE(outer_bracket_fits(finest));
    EXPECT_TRUE(std::filesystem::exists(folder / "metadata.xml"));
    std::filesystem::remove_all(folder);
}

TEST(SplatMesh, LevelingClosesTheStepBetweenViewsAndKeepsThemElsewhere) {
    // A row of squares, one view on each half, meeting along a line at x = 10.
    SplatMesh mesh;
    constexpr int kColumns = 21;
    for (int x = 0; x < kColumns; ++x)
        for (int y = 0; y < 2; ++y)
            mesh.vertices.push_back({float(x), float(y), 0.f});
    std::vector<int32_t> labels;
    for (int x = 0; x + 1 < kColumns; ++x) {
        mesh.faces.push_back({2 * x, 2 * x + 2, 2 * x + 1});
        mesh.faces.push_back({2 * x + 1, 2 * x + 2, 2 * x + 3});
        labels.insert(labels.end(), 2, x < 10 ? 3 : 7);
    }
    auto leveling = lfs::training::plan_color_leveling(mesh, labels);
    // Two points on the line, each read from both views.
    ASSERT_EQ(leveling.sample_labels.size(), 4u);
    std::vector<float> colors;
    for (const auto label : leveling.sample_labels)
        colors.insert(colors.end(), 3, label == 3 ? .3f : .5f);
    lfs::training::solve_color_leveling(leveling, mesh, colors);

    const auto color_at = [&](const size_t face, const std::array<float, 3> position, const float base) {
        float offset[3];
        leveling.offset_at(mesh, static_cast<uint32_t>(face), position.data(), offset);
        return base + offset[0];
    };
    // Both sides of the line now agree, each moved by half the step.
    const float left = color_at(19, {10.f, 1.f, 0.f}, .3f), right = color_at(20, {10.f, 1.f, 0.f}, .5f);
    EXPECT_NEAR(left, right, .01f);
    EXPECT_NEAR(left, .4f, .01f);
    // A step dies away with distance, and never overshoots.
    const float far_left = color_at(0, {0.f, 0.f, 0.f}, .3f);
    EXPECT_GT(far_left, .3f);
    EXPECT_LE(far_left, left + .001f);
}

TEST(SplatMesh, LevelingIgnoresViewsThatDoNotSeeTheSeam) {
    SplatMesh mesh;
    mesh.vertices = {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {1.f, 1.f, 0.f}};
    mesh.faces = {{0, 1, 2}, {2, 1, 3}};
    const std::vector<int32_t> labels{0, 1};
    auto leveling = lfs::training::plan_color_leveling(mesh, labels);
    std::vector<float> colors(leveling.sample_labels.size() * 3, -1.f);
    for (size_t s = 0; s < leveling.sample_labels.size(); ++s)
        if (leveling.sample_labels[s] == 0)
            for (size_t c = 0; c < 3; ++c)
                colors[3 * s + c] = .9f;
    lfs::training::solve_color_leveling(leveling, mesh, colors);
    for (const float offset : leveling.offsets)
        EXPECT_EQ(offset, 0.f);
}

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
    lfs::training::simplify_splat_mesh(mesh, tolerance, 1.f, before / 16);
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

TEST(SplatMesh, SimplifySplitsMeshesTooLargeForItsToleranceWithoutCracks) {
    // At this tolerance the sphere is wider than one piece may be, whatever its face count.
    constexpr float voxel = .04f, tolerance = .0002f;
    auto mesh = sphere_mesh(voxel);
    mesh.colors.assign(mesh.vertices.size(), {10, 20, 30});
    lfs::training::simplify_splat_mesh(mesh, tolerance, 1.f);
    EXPECT_EQ(rim_edges(mesh), 0u);
    for (const auto& p : mesh.vertices)
        EXPECT_NEAR(std::hypot(p[0], p[1], p[2]), 1.f, voxel);
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

TEST(SplatMesh, UnwrapKeepsTheFacesAndGivesThemAreaOnThePage) {
    constexpr float voxel = .04f, texel = .005f;
    constexpr int side = 2048;
    auto mesh = sphere_mesh(voxel);
    lfs::training::compute_splat_mesh_normals(mesh);
    mesh.colors.assign(mesh.vertices.size(), {10, 20, 30});
    const size_t faces = mesh.faces.size();
    const double area = mesh.area;
    lfs::training::unwrap_splat_mesh(mesh, texel, side, {}, 4000);
    // Surface nets makes a few faces without area on the symmetry planes; only those go.
    EXPECT_GT(mesh.faces.size(), faces * 99 / 100);
    auto copy = mesh;
    lfs::training::compute_splat_mesh_normals(copy);
    EXPECT_NEAR(copy.area / area, 1., 1e-4);
    ASSERT_EQ(mesh.uvs.size(), mesh.vertices.size());
    EXPECT_EQ(mesh.normals.size(), mesh.vertices.size());
    EXPECT_EQ(mesh.colors.size(), mesh.vertices.size());
    EXPECT_GT(mesh.charts, 1u);
    ASSERT_GE(mesh.page_first_face.size(), 2u);
    EXPECT_EQ(mesh.page_first_face.back(), mesh.faces.size());
    for (const auto& uv : mesh.uvs) {
        EXPECT_GE(uv[0], 0.f);
        EXPECT_LE(uv[0], 1.f);
        EXPECT_GE(uv[1], 0.f);
        EXPECT_LE(uv[1], 1.f);
    }
    for (const auto& p : mesh.vertices)
        EXPECT_NEAR(std::hypot(p[0], p[1], p[2]), 1.f, voxel / 2);

    size_t texels = 0;
    for (size_t page = 0; page + 1 < mesh.page_first_face.size(); ++page) {
        const auto found = lfs::training::rasterize_mesh_texels(mesh, page);
        texels += found.pixels.size();
        ASSERT_EQ(found.positions.size(), 3 * found.pixels.size());
        // Every texel lies on a face, which lies within a voxel of the sphere.
        for (size_t t = 0; t < found.pixels.size(); t += 97)
            ASSERT_NEAR(std::hypot(found.positions[3 * t], found.positions[3 * t + 1], found.positions[3 * t + 2]), 1.f, voxel);
        EXPECT_EQ(found.fallback[0], 10);
        EXPECT_EQ(found.fallback[2], 30);
    }
    // At least what the surface takes at this density, and not much more for the margins.
    const double expected = area / (texel * texel);
    EXPECT_GT(double(texels), .95 * expected);
    EXPECT_LT(double(texels), 1.5 * expected);
}

TEST(SplatMesh, UnwrapGroupsKeepTheirOwnPagesAndEveryFace) {
    constexpr float voxel = .04f, texel = .005f;
    auto single = sphere_mesh(voxel), grouped = single;
    lfs::training::compute_splat_mesh_normals(single);
    lfs::training::compute_splat_mesh_normals(grouped);
    const double area = single.area;
    lfs::training::unwrap_splat_mesh(single, texel, 2048, {}, 2000);
    lfs::training::unwrap_splat_mesh(grouped, texel, 2048, {}, 2000, 6000);
    EXPECT_GT(grouped.page_first_face.size(), single.page_first_face.size());
    EXPECT_EQ(grouped.faces.size(), single.faces.size());
    EXPECT_EQ(grouped.page_first_face.back(), grouped.faces.size());
    lfs::training::compute_splat_mesh_normals(grouped);
    EXPECT_NEAR(grouped.area / area, 1., 1e-4);
    // A vertex keeps its position in the group's output, and each page stays on the page.
    for (const auto& p : grouped.vertices)
        EXPECT_NEAR(std::hypot(p[0], p[1], p[2]), 1.f, voxel / 2);
    for (const auto& uv : grouped.uvs) {
        EXPECT_GE(uv[0], 0.f);
        EXPECT_LE(uv[1], 1.f);
    }
}

namespace {
    // A strip of n quads along x, with the faces of quad i at indices 2i and 2i + 1.
    SplatMesh strip_mesh(const int n) {
        SplatMesh mesh;
        for (int i = 0; i <= n; ++i) {
            mesh.vertices.push_back({float(i), 0.f, 0.f});
            mesh.vertices.push_back({float(i), 1.f, 0.f});
        }
        for (int i = 0; i < n; ++i) {
            mesh.faces.push_back({2 * i, 2 * i + 2, 2 * i + 1});
            mesh.faces.push_back({2 * i + 1, 2 * i + 2, 2 * i + 3});
        }
        return mesh;
    }
} // namespace

TEST(SplatMesh, TilesKeepTheirBordersOnEveryLevelAndEveryFaceOfTheFinest) {
    constexpr float voxel = .04f, texel = .005f;
    auto mesh = sphere_mesh(voxel);
    lfs::training::compute_splat_mesh_normals(mesh);
    mesh.colors.assign(mesh.vertices.size(), {10, 20, 30});
    const size_t faces = mesh.faces.size();
    const auto tiled = lfs::training::build_tiled_mesh(mesh, texel, 2048, 4, .005f, {}, 5000);
    const auto& all = tiled.combined;
    size_t tiles = 0, finest = 0, reduced = 0;
    for (const auto& level : tiled.levels) {
        tiles = std::max(tiles, level.tile + 1);
        const size_t count = all.page_first_face[level.first_page + level.page_count] - all.page_first_face[level.first_page];
        (level.level == 0 ? finest : reduced) += count;
        EXPECT_GT(level.radius, 0.f);
    }
    EXPECT_GT(tiles, 1u);
    EXPECT_GT(reduced, 0u);
    // Unwrapping drops only faces without area.
    EXPECT_GT(finest, faces * 99 / 100);
    EXPECT_LE(finest, faces);
    EXPECT_EQ(all.page_sizes.size() + 1, all.page_first_face.size());
    EXPECT_EQ(all.face_error.size(), all.faces.size());
    // A tile's outer rim is the same set of points on every level: edges used by one face only, against the finest.
    const auto rim_of = [&](const lfs::training::TileLevel& level) {
        std::vector<std::array<int64_t, 6>> edges;
        const auto key = [](const std::array<float, 3>& p) { return std::array<int64_t, 3>{std::llround(p[0] * 1e5), std::llround(p[1] * 1e5), std::llround(p[2] * 1e5)}; };
        for (size_t f = all.page_first_face[level.first_page]; f < all.page_first_face[level.first_page + level.page_count]; ++f)
            for (int c = 0; c < 3; ++c) {
                auto a = key(all.vertices[size_t(all.faces[f][size_t(c)])]), b = key(all.vertices[size_t(all.faces[f][size_t((c + 1) % 3)])]);
                if (b < a)
                    std::swap(a, b);
                edges.push_back({a[0], a[1], a[2], b[0], b[1], b[2]});
            }
        std::ranges::sort(edges);
        std::vector<std::array<int64_t, 3>> rim;
        for (size_t e = 0; e < edges.size();) {
            size_t end = e;
            while (end < edges.size() && edges[end] == edges[e])
                ++end;
            if (end - e == 1) {
                rim.push_back({edges[e][0], edges[e][1], edges[e][2]});
                rim.push_back({edges[e][3], edges[e][4], edges[e][5]});
            }
            e = end;
        }
        std::ranges::sort(rim);
        rim.erase(std::unique(rim.begin(), rim.end()), rim.end());
        return rim;
    };
    for (const auto& level : tiled.levels) {
        if (level.level == 0)
            continue;
        const auto finest_level = std::ranges::find_if(tiled.levels, [&](const auto& other) { return other.tile == level.tile && other.level == 0; });
        ASSERT_NE(finest_level, tiled.levels.end());
        const auto outer = rim_of(*finest_level), coarse = rim_of(level);
        std::vector<std::array<int64_t, 3>> missing;
        std::ranges::set_difference(outer, coarse, std::back_inserter(missing));
        EXPECT_TRUE(missing.empty()) << missing.size() << " rim points lost on level " << level.level;
    }
}

TEST(SplatMesh, CandidatesKeepTheBestViewsInOrder) {
    lfs::training::FaceViewCandidates candidates(1);
    for (int view = 0; view < 20; ++view) {
        const float score = float(view % 7) + 1.f + float(view) * .01f;
        candidates.add(view, std::span(&score, 1));
    }
    for (size_t slot = 0; slot + 1 < lfs::training::FaceViewCandidates::kPerFace; ++slot)
        EXPECT_GE(candidates.scores[slot], candidates.scores[slot + 1]);
    EXPECT_EQ(candidates.views[0], 13);
}

TEST(SplatMesh, OneViewTakesEveryFaceItSeesWellEnough) {
    // Each view is the very best on its own half, and sees the other half at 60%: one must take all, or the middle is a seam.
    const auto mesh = strip_mesh(50);
    lfs::training::FaceViewCandidates candidates(mesh.faces.size());
    std::vector<float> first(mesh.faces.size()), second(mesh.faces.size());
    for (size_t f = 0; f < mesh.faces.size(); ++f) {
        first[f] = f < mesh.faces.size() / 2 ? 1.f : .6f;
        second[f] = f < mesh.faces.size() / 2 ? .6f : 1.f;
    }
    candidates.add(3, first);
    candidates.add(5, second);
    const auto labels = lfs::training::choose_face_views(mesh, candidates);
    EXPECT_TRUE(std::ranges::all_of(labels, [&](const int32_t label) { return label == labels[0]; }));
    EXPECT_TRUE(labels[0] == 3 || labels[0] == 5);
}

TEST(SplatMesh, ViewsSplitOnlyWhereOneCannotServeAndUnseenFacesStayUnset) {
    const auto mesh = strip_mesh(50);
    const size_t faces = mesh.faces.size();
    lfs::training::FaceViewCandidates candidates(faces);
    std::vector<float> first(faces, 0.f), second(faces, 0.f);
    for (size_t f = 0; f + 2 < faces; ++f)
        (f < faces / 2 ? first : second)[f] = 1.f;
    candidates.add(3, first);
    candidates.add(5, second);
    const auto labels = lfs::training::choose_face_views(mesh, candidates);
    EXPECT_EQ(labels[0], 3);
    EXPECT_EQ(labels[faces - 3], 5);
    EXPECT_EQ(labels[faces - 1], -1);
}

TEST(SplatMesh, ASpeckJoinsTheViewAroundIt) {
    // A face whose own best view differs, but which the surrounding view sees at 40%.
    const auto mesh = strip_mesh(50);
    const size_t faces = mesh.faces.size();
    lfs::training::FaceViewCandidates candidates(faces);
    std::vector<float> around(faces, 1.f), own(faces, 0.f);
    around[40] = .4f;
    own[40] = 1.f;
    candidates.add(3, around);
    candidates.add(5, own);
    const auto labels = lfs::training::choose_face_views(mesh, candidates);
    EXPECT_EQ(labels[40], 3);
}

TEST(SplatMesh, ComposingAPageFillsOutwardFromTheCharts) {
    lfs::training::MeshTexels texels;
    constexpr int side = 64;
    texels.pixels = {uint32_t(20 * side + 20), uint32_t(20 * side + 21)};
    texels.positions.assign(6, 0.f);
    texels.fallback = {0, 0, 0, 200, 100, 50};
    const std::vector<float> colors{1.f, .5f, 0.f, -1.f, -1.f, -1.f};
    const auto image = lfs::training::compose_mesh_page(texels, colors, side);
    const auto at = [&](const int x, const int y, const int c) { return int(image[3 * (size_t(y) * side + x) + c]); };
    EXPECT_EQ(at(20, 20, 0), 255);
    EXPECT_EQ(at(20, 20, 1), 128);
    // Unobserved texels use the vertex colors.
    EXPECT_EQ(at(21, 20, 0), 200);
    // The gutter continues the chart, and the far page holds its mean.
    EXPECT_EQ(at(19, 20, 0), 255);
    EXPECT_EQ(at(60, 60, 1), (128 + 100) / 2);
}

TEST(SplatMesh, FilesStayConnectedAcrossChartSeams) {
    // Two charts sharing an edge: its two points are repeated in the output, with different texture coordinates.
    SplatMesh mesh;
    mesh.vertices = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}};
    mesh.normals.assign(6, {0.f, 0.f, 1.f});
    mesh.colors.assign(6, {1, 2, 3});
    mesh.faces = {{0, 1, 2}, {3, 5, 4}};
    mesh.uvs = {{0.f, 0.f}, {.1f, 0.f}, {0.f, .1f}, {.5f, .5f}, {.5f, .6f}, {.6f, .6f}};
    mesh.uv_source = {10, 20, 30, 20, 30, 40};
    mesh.page_first_face = {0, 2};
    mesh.page_size = 8;
    mesh.pages = {std::vector<uint8_t>(3 * 64, 128)};
    const auto welded = lfs::training::weld_chart_seams(mesh);
    EXPECT_EQ(welded.originals.size(), 4u);
    EXPECT_EQ(welded.index[1], welded.index[3]);
    EXPECT_EQ(welded.index[2], welded.index[4]);

    const auto folder = std::filesystem::temp_directory_path() / "lfs_splat_mesh_weld_test";
    std::filesystem::create_directories(folder);
    lfs::training::write_splat_mesh_obj(mesh, folder / "scene.obj");
    lfs::training::write_splat_mesh_ply(mesh, folder / "scene.ply");
    std::ifstream obj_in(folder / "scene.obj", std::ios::binary), ply_in(folder / "scene.ply", std::ios::binary);
    const std::string obj((std::istreambuf_iterator<char>(obj_in)), {}), ply((std::istreambuf_iterator<char>(ply_in)), {});
    obj_in.close();
    ply_in.close();
    std::filesystem::remove_all(folder);
    const auto count = [&](const std::string_view prefix) {
        size_t found = 0;
        for (size_t at = 0; (at = obj.find(prefix, at)) != std::string::npos; at += prefix.size())
            ++found;
        return found;
    };
    EXPECT_EQ(count("\nv "), 4u);
    EXPECT_EQ(count("\nvn "), 4u);
    EXPECT_EQ(count("\nvt "), 6u);
    // The second face shares the points of the first, through its own texture coordinates.
    EXPECT_NE(obj.find("f 1/1/1 2/2/2 3/3/3\nf 2/4/2 4/6/4 3/5/3\n"), std::string::npos);
    EXPECT_NE(ply.find("element vertex 4\n"), std::string::npos);
    EXPECT_NE(ply.find("element face 2\n"), std::string::npos);
}

TEST(SplatMesh, ObjNamesItsTexturesAndFlipsV) {
    SplatMesh mesh;
    mesh.vertices = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    mesh.normals = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}};
    mesh.faces = {{0, 1, 2}};
    mesh.uvs = {{0.f, .25f}, {1.f, .25f}, {0.f, 1.f}};
    mesh.page_first_face = {0, 1};
    mesh.page_size = 8;
    mesh.pages = {std::vector<uint8_t>(3 * 64, 128)};
    const auto folder = std::filesystem::temp_directory_path() / "lfs_splat_mesh_obj_test";
    std::filesystem::create_directories(folder);
    lfs::training::write_splat_mesh_obj(mesh, folder / "scene.obj");
    std::ifstream in(folder / "scene.obj", std::ios::binary);
    const std::string text((std::istreambuf_iterator<char>(in)), {});
    in.close();
    std::ifstream material(folder / "scene.mtl", std::ios::binary);
    const std::string mtl((std::istreambuf_iterator<char>(material)), {});
    material.close();
    const bool image_written = std::filesystem::file_size(folder / "scene_0.jpg") > 0;
    std::filesystem::remove_all(folder);
    EXPECT_TRUE(image_written);
    EXPECT_NE(text.find("mtllib scene.mtl\n"), std::string::npos);
    EXPECT_NE(text.find("vt 0 0.75\n"), std::string::npos);
    EXPECT_NE(text.find("usemtl page_0\nf 1/1/1 2/2/2 3/3/3\n"), std::string::npos);
    EXPECT_NE(mtl.find("map_Kd scene_0.jpg\n"), std::string::npos);
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

TEST_F(SplatMeshGpu, ColorsLeanOnCloseViews) {
    const std::vector<float> vertex{0.f, 0.f, 0.f};
    Device<float> vertices(3), depth(size_t(kSize) * kSize), photo(size_t(kSize) * kSize * 3), sum(3), weight(1);
    ASSERT_EQ(cudaMemcpy(vertices.data, vertex.data(), 12, cudaMemcpyHostToDevice), cudaSuccess);
    const auto add = [&](const float camera_z, const float level) {
        std::vector<float> d(size_t(kSize) * kSize, -camera_z), c(size_t(kSize) * kSize * 3, level);
        ASSERT_EQ(cudaMemcpy(depth.data, d.data(), d.size() * 4, cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(photo.data, c.data(), c.size() * 4, cudaMemcpyHostToDevice), cudaSuccess);
        gsplat_lfs::mesh_color_accumulate(vertices.data, 1, pinhole_at(camera_z), depth.data, photo.data, .1f, sum.data, weight.data, nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    };
    add(-1.f, 1.f);
    add(-4.f, 0.f);
    float s = 0.f, w = 0.f;
    ASSERT_EQ(cudaMemcpy(&s, sum.data, 4, cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&w, weight.data, 4, cudaMemcpyDeviceToHost), cudaSuccess);
    // Weights 1 and 1/4.
    EXPECT_NEAR(s / w, .8f, .001f);
}

TEST_F(SplatMeshGpu, TextureSamplesPhotosBetweenPixelsAndSkipsOccludingEdges) {
    constexpr size_t pixels = size_t(kSize) * kSize;
    const auto view = pinhole_at(-1.f);
    // A texel at world x = .5025 lands on image x = 200 * .5025 + 100 = 200.5, past the last pixel; these land inside.
    const std::vector<float> position{0.f, 0.f, 0.f, .1025f, 0.f, 0.f};
    Device<float> texels(6), depth(pixels), photo(pixels * 3), sum(6), weight(2), high(6), best(2);
    Device<int32_t> labels(2);
    Device<float> tolerances(2);
    const float tolerance_values[2] = {.1f, .1f};
    ASSERT_EQ(cudaMemcpy(tolerances.data, tolerance_values, 8, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(texels.data, position.data(), 24, cudaMemcpyHostToDevice), cudaSuccess);
    std::vector<float> d(pixels, 1.f), c(pixels * 3, 0.f);
    for (size_t i = 0; i < pixels; ++i)
        c[i] = float(i % kSize) / float(kSize); // red rises along x by 1 / kSize per pixel
    // The first texel samples pixels 99 and 100; the second 120 and 121. Occlude 121 in one row.
    const size_t row = 100;
    d[row * kSize + 121] = .5f;
    ASSERT_EQ(cudaMemcpy(depth.data, d.data(), pixels * 4, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(photo.data, c.data(), c.size() * 4, cudaMemcpyHostToDevice), cudaSuccess);
    gsplat_lfs::mesh_texture_accumulate(texels.data, 2, view, 0, labels.data, depth.data, photo.data, photo.data, tolerances.data, sum.data, weight.data, high.data, best.data, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    float s[6], w[2];
    ASSERT_EQ(cudaMemcpy(s, sum.data, 24, cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(w, weight.data, 8, cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_GT(w[0], 0.f);
    // Image x = 100 lies between the centers of pixels 99 and 100.
    EXPECT_NEAR(s[0] / w[0], 99.5f / float(kSize), 1e-4f);
    EXPECT_EQ(w[1], 0.f);
}

TEST_F(SplatMeshGpu, TextureColorComesFromTheChosenViewAndLevelsFromAll) {
    constexpr size_t pixels = size_t(kSize) * kSize;
    const std::vector<float> position{0.f, 0.f, 0.f};
    Device<float> texels(3), depth(pixels), photo(pixels * 3), low_photo(pixels * 3), sum(3), weight(1), high(3), best(1);
    Device<int32_t> labels(1);
    Device<float> tolerances(1);
    const float tolerance_value = .1f;
    ASSERT_EQ(cudaMemcpy(tolerances.data, &tolerance_value, 4, cudaMemcpyHostToDevice), cudaSuccess);
    const int32_t chosen = 7;
    ASSERT_EQ(cudaMemcpy(labels.data, &chosen, 4, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(texels.data, position.data(), 12, cudaMemcpyHostToDevice), cudaSuccess);
    const auto add = [&](const int32_t label, const float camera_z, const float low_level, const float detail) {
        std::vector<float> d(pixels, -camera_z), low(pixels * 3, low_level), c(pixels * 3, low_level + detail);
        ASSERT_EQ(cudaMemcpy(depth.data, d.data(), pixels * 4, cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(photo.data, c.data(), c.size() * 4, cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(low_photo.data, low.data(), low.size() * 4, cudaMemcpyHostToDevice), cudaSuccess);
        gsplat_lfs::mesh_texture_accumulate(texels.data, 1, pinhole_at(camera_z), label, labels.data, depth.data, photo.data, low_photo.data, tolerances.data, sum.data, weight.data, high.data, best.data, nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    };
    add(7, -4.f, .8f, .2f);
    add(8, -1.f, .4f, -.1f);
    add(9, -2.f, .6f, .3f);
    float s[3], w = 0.f, h[3];
    ASSERT_EQ(cudaMemcpy(s, sum.data, 12, cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&w, weight.data, 4, cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h, high.data, 12, cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_NEAR(s[0] / w, (.25f * .8f + 1.f * .4f + .5f * .6f) / 1.75f, 1e-4f);
    // The chosen view is the farthest, closer ones do not take over, and it keeps its whole color.
    EXPECT_NEAR(h[0], 1.f, 1e-4f);
}

TEST_F(SplatMeshGpu, TextureLowPassLeavesInvalidPixelsOut) {
    constexpr size_t pixels = size_t(kSize) * kSize;
    Device<float> depth(pixels), photo(pixels * 3), scratch(pixels * 3), weights(pixels * 2), low(pixels * 3);
    std::vector<float> d(pixels, 1.f), c(pixels * 3, .5f);
    // A bright person, masked out of the depth.
    for (size_t y = 90; y < 110; ++y)
        for (size_t x = 90; x < 110; ++x) {
            d[y * kSize + x] = 0.f;
            c[y * kSize + x] = 1.f;
        }
    ASSERT_EQ(cudaMemcpy(depth.data, d.data(), pixels * 4, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(photo.data, c.data(), c.size() * 4, cudaMemcpyHostToDevice), cudaSuccess);
    gsplat_lfs::mesh_photo_low_pass(photo.data, depth.data, kSize, kSize, 8, scratch.data, weights.data, low.data, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<float> out(pixels * 3);
    ASSERT_EQ(cudaMemcpy(out.data(), low.data, out.size() * 4, cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_NEAR(out[88 * kSize + 100], .5f, 1e-4f);
    EXPECT_NEAR(out[100 * kSize + 111], .5f, 1e-4f);
}

TEST_F(SplatMeshGpu, TrainedAppearanceInvertsBackToTheRender) {
    lfs::training::PPISP ppisp(1);
    ppisp.register_frame(7, 3);
    ppisp.finalize();
    const std::vector<float> exposure{.4f}, color{3.f, -2.f, 1.f, 2.f, -1.f, 2.f, 4.f, -3.f};
    ASSERT_EQ(cudaMemcpy(ppisp.exposure_params().ptr<float>(), exposure.data(), 4, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(ppisp.color_params().ptr<float>(), color.data(), 32, cudaMemcpyHostToDevice), cudaSuccess);
    constexpr size_t side = 16;
    std::vector<float> render(3 * side * side);
    for (size_t i = 0; i < render.size(); ++i)
        render[i] = .1f + .4f * float(i % 7) / 6.f;
    auto rgb = lfs::core::Tensor::from_vector(render, {3, side, side}, lfs::core::Device::CUDA);
    const auto photo = ppisp.apply(rgb, 3, 7);
    auto back = ppisp.invert(photo, 3, 7).cpu().contiguous();
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    for (size_t i = 0; i < render.size(); ++i)
        EXPECT_NEAR(back.ptr<float>()[i], render[i], .002f);
}

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
