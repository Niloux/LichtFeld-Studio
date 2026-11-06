/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/logger.hpp"
#include "splat_mesh.hpp"
#include "splat_mesh_parallel.hpp"
#include <algorithm>
#include <bit>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <meshoptimizer.h>

namespace lfs::training {
    namespace {
        constexpr int kTileThreads = 8;
        // Unwrapping the tiles is most of the work here: 16 against 8 took 228 s against 268 s on an outdoor scene, and 24 no less.
        constexpr int kTileWorkThreads = 16;
        constexpr float kPieceShareOfPage = .7f;
        // Each level allows this much more surface deviation than the one below it, and doubles its texel.
        constexpr float kLevelErrorGrowth = 4.f;
        constexpr int kMinPageSize = 256;
        // Chart padding and packing waste, on top of the texels a chunk's area asks for.
        constexpr double kPagePackingSlack = 1.8;
        // A coarser level is not worth a file unless it is this much smaller, or below this size at all.
        constexpr double kMinLevelReduction = .85;
        constexpr size_t kMinLevelFaces = 1000;

        struct TileChunk {
            std::vector<SplatMesh> levels; // finest first
            std::vector<float> errors;     // surface deviation of each level from the finest
            std::array<float, 3> center{0.f, 0.f, 0.f};
            float radius = 0.f;
        };

        // Removes vertices no face uses.
        void compact(SplatMesh& mesh, std::vector<uint8_t>* locks = nullptr) {
            std::vector<int32_t> remap(mesh.vertices.size(), -1);
            std::vector<std::array<float, 3>> vertices;
            std::vector<std::array<uint8_t, 3>> colors;
            std::vector<uint8_t> kept_locks;
            vertices.reserve(mesh.vertices.size());
            colors.reserve(mesh.colors.size());
            for (auto& face : mesh.faces)
                for (auto& vertex : face) {
                    auto& target = remap[size_t(vertex)];
                    if (target < 0) {
                        target = static_cast<int32_t>(vertices.size());
                        vertices.push_back(mesh.vertices[size_t(vertex)]);
                        if (!mesh.colors.empty())
                            colors.push_back(mesh.colors[size_t(vertex)]);
                        if (locks)
                            kept_locks.push_back((*locks)[size_t(vertex)]);
                    }
                    vertex = target;
                }
            mesh.vertices = std::move(vertices);
            mesh.colors = std::move(colors);
            if (locks)
                *locks = std::move(kept_locks);
        }

        // Reduces faces while the surface stays within about max_error of it. Locked vertices keep their place.
        [[nodiscard]] SplatMesh reduce_level(const SplatMesh& finer, std::vector<uint8_t>& locks, const float max_error) {
            SplatMesh coarser;
            coarser.vertices = finer.vertices;
            coarser.colors = finer.colors;
            std::vector<unsigned int> indices(finer.faces.size() * 3);
            std::memcpy(indices.data(), finer.faces.data(), indices.size() * sizeof(unsigned int));
            float result_error = 0.f;
            const size_t kept = meshopt_simplifyWithUpdate(indices.data(), indices.size(), &coarser.vertices[0][0], coarser.vertices.size(), sizeof(float) * 3, nullptr, 0, nullptr, 0,
                                                           locks.data(), 0, max_error, meshopt_SimplifyErrorAbsolute, &result_error);
            coarser.faces.resize(kept / 3);
            std::memcpy(coarser.faces.data(), indices.data(), kept * sizeof(unsigned int));
            compact(coarser, &locks);
            return coarser;
        }

        [[nodiscard]] int page_side_for(const double area, const float texel, const int max_page) {
            const double side = std::sqrt(kPagePackingSlack * area / (double(texel) * texel));
            const auto power = std::bit_ceil(static_cast<uint32_t>(std::max(side, double(kMinPageSize))));
            return std::min(static_cast<int>(power), max_page);
        }
    } // namespace

    TiledMesh build_tiled_mesh(SplatMesh& base, const float texel_size, const int max_page, const int max_levels, const float finest_error, const std::stop_token stop, const size_t tile_faces) {
        const float max_extent = kPieceShareOfPage * float(max_page) * texel_size;
        const auto ranges = split_mesh_into_pieces(base, tile_faces, max_extent, kTileThreads);

        // Vertices that tiles share stay put on every level, so that no level of a tile opens a crack against its neighbor.
        std::vector<int32_t> owner(base.vertices.size(), -1);
        std::vector<uint8_t> shared(base.vertices.size(), 0);
        for (size_t r = 0; r < ranges.size(); ++r)
            for (size_t f = ranges[r].first; f < ranges[r].second; ++f)
                for (const auto vertex : base.faces[f]) {
                    if (owner[size_t(vertex)] < 0)
                        owner[size_t(vertex)] = static_cast<int32_t>(r);
                    else if (owner[size_t(vertex)] != static_cast<int32_t>(r))
                        shared[size_t(vertex)] = 1;
                }
        owner = {};

        std::vector<TileChunk> tiles(ranges.size());
        detail::parallel_for(ranges.size(), kTileWorkThreads, [&](const size_t r) {
            auto& tile = tiles[r];
            SplatMesh finest;
            std::vector<uint8_t> locks;
            // Only the tile's own vertices, in the order its faces reach them: a copy of the whole mesh per tile costs gigabytes.
            {
                std::vector<int32_t> remap(base.vertices.size(), -1);
                finest.faces.assign(base.faces.begin() + std::ptrdiff_t(ranges[r].first), base.faces.begin() + std::ptrdiff_t(ranges[r].second));
                for (auto& face : finest.faces)
                    for (auto& vertex : face) {
                        auto& target = remap[size_t(vertex)];
                        if (target < 0) {
                            target = static_cast<int32_t>(finest.vertices.size());
                            finest.vertices.push_back(base.vertices[size_t(vertex)]);
                            if (!base.colors.empty())
                                finest.colors.push_back(base.colors[size_t(vertex)]);
                            locks.push_back(shared[size_t(vertex)]);
                        }
                        vertex = target;
                    }
            }
            std::array<double, 3> low{DBL_MAX, DBL_MAX, DBL_MAX}, high{-DBL_MAX, -DBL_MAX, -DBL_MAX};
            for (const auto& p : finest.vertices)
                for (int c = 0; c < 3; ++c) {
                    low[size_t(c)] = std::min(low[size_t(c)], double(p[size_t(c)]));
                    high[size_t(c)] = std::max(high[size_t(c)], double(p[size_t(c)]));
                }
            for (int c = 0; c < 3; ++c)
                tile.center[size_t(c)] = float(.5 * (low[size_t(c)] + high[size_t(c)]));
            tile.radius = float(.5 * std::hypot(high[0] - low[0], high[1] - low[1], high[2] - low[2]));

            tile.levels.push_back(std::move(finest));
            tile.errors.push_back(finest_error);
            float error = finest_error;
            for (int level = 1; level < max_levels && tile.levels.back().faces.size() > kMinLevelFaces; ++level) {
                error *= kLevelErrorGrowth;
                auto coarser = reduce_level(tile.levels.back(), locks, error);
                if (coarser.faces.empty() || double(coarser.faces.size()) > kMinLevelReduction * double(tile.levels.back().faces.size()))
                    break;
                tile.levels.push_back(std::move(coarser));
                tile.errors.push_back(error);
            }
            float texel = texel_size;
            for (auto& mesh : tile.levels) {
                if (stop.stop_requested())
                    throw core::OperationCancelled();
                compute_splat_mesh_normals(mesh);
                unwrap_splat_mesh(mesh, texel, page_side_for(mesh.area, texel, max_page), stop);
                texel *= 2.f;
            }
        });

        TiledMesh tiled;
        auto& all = tiled.combined;
        size_t vertices = 0, faces = 0, pages = 0;
        for (const auto& tile : tiles)
            for (const auto& mesh : tile.levels) {
                vertices += mesh.vertices.size();
                faces += mesh.faces.size();
                pages += mesh.page_first_face.size() - 1;
            }
        all.vertices.reserve(vertices);
        all.normals.reserve(vertices);
        all.colors.reserve(vertices);
        all.uvs.reserve(vertices);
        all.uv_source.reserve(vertices);
        all.faces.reserve(faces);
        all.face_error.reserve(faces);
        all.page_first_face.reserve(pages + 1);
        all.page_sizes.reserve(pages);
        all.page_first_face.push_back(0);
        int32_t source_base = 0;
        for (size_t t = 0; t < tiles.size(); ++t) {
            auto& tile = tiles[t];
            for (size_t l = 0; l < tile.levels.size(); ++l) {
                auto& mesh = tile.levels[l];
                TileLevel entry;
                entry.tile = t;
                entry.level = l;
                entry.first_page = all.page_first_face.size() - 1;
                entry.page_count = mesh.page_first_face.size() - 1;
                entry.texel = texel_size * float(1u << l);
                entry.center = tile.center;
                entry.radius = tile.radius;
                entry.first_vertex = all.vertices.size();
                entry.vertex_count = mesh.vertices.size();
                tiled.levels.push_back(entry);
                const auto base_vertex = static_cast<int32_t>(all.vertices.size());
                all.vertices.insert(all.vertices.end(), mesh.vertices.begin(), mesh.vertices.end());
                all.normals.insert(all.normals.end(), mesh.normals.begin(), mesh.normals.end());
                all.colors.insert(all.colors.end(), mesh.colors.begin(), mesh.colors.end());
                all.uvs.insert(all.uvs.end(), mesh.uvs.begin(), mesh.uvs.end());
                int32_t top = 0;
                for (const auto source : mesh.uv_source) {
                    all.uv_source.push_back(source_base + source);
                    top = std::max(top, source);
                }
                source_base += top + 1;
                for (size_t p = 0; p + 1 < mesh.page_first_face.size(); ++p) {
                    for (size_t f = mesh.page_first_face[p]; f < mesh.page_first_face[p + 1]; ++f) {
                        const auto& face = mesh.faces[f];
                        all.faces.push_back({base_vertex + face[0], base_vertex + face[1], base_vertex + face[2]});
                        all.face_error.push_back(tile.errors[l]);
                    }
                    all.page_first_face.push_back(all.faces.size());
                    all.page_sizes.push_back(mesh.page_size);
                }
                all.charts += mesh.charts;
                mesh = {};
            }
        }
        all.page_size = max_page;
        size_t finest_faces = 0;
        for (size_t i = 0; i < tiled.levels.size(); ++i)
            if (tiled.levels[i].level == 0)
                finest_faces += (all.page_first_face[tiled.levels[i].first_page + tiled.levels[i].page_count] - all.page_first_face[tiled.levels[i].first_page]);
        LOG_INFO("Mesh: {} tiles, {} tile levels, {} faces on the finest level and {} in all, on {} pages", tiles.size(), tiled.levels.size(), finest_faces, all.faces.size(), all.page_sizes.size());
        return tiled;
    }

} // namespace lfs::training
