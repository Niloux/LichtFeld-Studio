/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/progress.hpp"
#include "core/splat_mesh_settings.hpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <stop_token>
#include <vector>

namespace lfs::core {
    class Camera;
    class SplatData;
    struct MeshData;
} // namespace lfs::core

namespace lfs::training {

    struct SplatMesh {
        std::vector<std::array<float, 3>> vertices;
        std::vector<std::array<float, 3>> normals;
        std::vector<std::array<uint8_t, 3>> colors; // sRGB, as in the photos
        std::vector<std::array<int32_t, 3>> faces;

        // What each stage found, for reports.
        size_t views = 0;
        size_t blocks = 0;
        uint64_t band_voxels = 0;
        size_t raw_faces = 0;
        size_t components_kept = 0;
        size_t components_total = 0;
        double area = 0.;
        float colored_vertex_share = 0.f;
        double seconds_render = 0.;
        double seconds_fusion = 0.;
        double seconds_extract = 0.;
        double seconds_color = 0.;
    };

    // Renders transmittance median depth from the cameras, fuses it into a sparse TSDF, extracts
    // the surface with naive surface nets and colors it from the photos. Mesh is in model space.
    // Camera image sizes are changed while it runs and restored afterwards.
    [[nodiscard]] SplatMesh extract_splat_mesh(core::SplatData& model,
                                               std::span<core::Camera* const> cameras,
                                               const core::SplatMeshSettings& settings,
                                               std::stop_token stop = {}, const core::ProgressCallback& progress = {});

    // Sorted sparse voxel keys (21 bits per axis, gsplat_lfs::kTsdfKeyOffset) and their TSDF
    // values. Fills vertices and faces with outward-facing triangles.
    void surface_nets(std::span<const uint64_t> keys, std::span<const float> tsdf, float voxel, SplatMesh& mesh,
                      float max_step = 1.5f);
    // Removes connected components with fewer faces and the vertices they leave unused.
    void remove_small_mesh_components(SplatMesh& mesh, int min_faces);
    // Area-weighted vertex normals and the mesh area.
    void compute_splat_mesh_normals(SplatMesh& mesh);

    void write_splat_mesh_ply(const SplatMesh& mesh, const std::filesystem::path& path);
    [[nodiscard]] core::MeshData splat_mesh_to_mesh_data(const SplatMesh& mesh);

} // namespace lfs::training
