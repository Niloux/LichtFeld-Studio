/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/collision_volume.hpp"
#include "core/logger.hpp"
#include "core/native_fs.hpp"
#include "core/path_utils.hpp"
#include "splat_mesh.hpp"
#include "splat_mesh_parallel.hpp"
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
#include <memory>
#include <numeric>
#include <span>
#include <stb_image_write.h>
#include <xatlas/xatlas.h>

namespace lfs::training {
    namespace {
        using core::collisionError;

        constexpr uint32_t kPagePadding = 4;
        // A chart in a piece this share of a page across fits a page along its diagonal.
        constexpr float kPieceShareOfPage = .7f;
        constexpr int kUnwrapThreads = 8;
        constexpr int kTextureJpegQuality = 95;
        // Reaches what bilinear sampling reads from the edge of a chart.
        constexpr double kCoverageMarginTexels = .75;
        constexpr int kFillRings = 8;

        struct AtlasDeleter {
            void operator()(xatlas::Atlas* atlas) const { xatlas::Destroy(atlas); }
        };

        bool keep_going(xatlas::ProgressCategory, int, void* user) {
            return !static_cast<const std::stop_token*>(user)->stop_requested();
        }
    } // namespace

    namespace {
        // The charts of a group of pieces, packed onto pages of their own.
        struct UnwrappedGroup {
            std::vector<int32_t> sources;                                // per output vertex, the mesh vertex it repeats
            std::vector<std::array<float, 2>> uvs;                       // per output vertex
            std::vector<std::vector<std::array<int32_t, 3>>> page_faces; // into the output vertices
            size_t charts = 0;
            double seconds_charts = 0., seconds_packing = 0.;
        };

        [[nodiscard]] UnwrappedGroup unwrap_group(const SplatMesh& mesh, const std::span<const std::pair<size_t, size_t>> ranges, const float texel_size,
                                                  const int page_size, const std::stop_token& stop) {
            using Clock = std::chrono::steady_clock;
            const auto seconds = [](const Clock::time_point a, const Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); };
            std::unique_ptr<xatlas::Atlas, AtlasDeleter> atlas(xatlas::Create());
            xatlas::SetProgressCallback(atlas.get(), keep_going, const_cast<std::stop_token*>(&stop));
            // For each piece, which of the mesh's vertices its input vertices are.
            std::vector<std::vector<int32_t>> piece_vertices(ranges.size());
            for (size_t r = 0; r < ranges.size(); ++r) {
                const auto [first, last] = ranges[r];
                auto& used = piece_vertices[r];
                used.reserve((last - first) * 3);
                for (size_t f = first; f < last; ++f)
                    used.insert(used.end(), mesh.faces[f].begin(), mesh.faces[f].end());
                std::ranges::sort(used);
                used.erase(std::unique(used.begin(), used.end()), used.end());
                std::vector<uint32_t> indices;
                indices.reserve((last - first) * 3);
                for (size_t f = first; f < last; ++f)
                    for (const auto vertex : mesh.faces[f])
                        indices.push_back(static_cast<uint32_t>(std::ranges::lower_bound(used, vertex) - used.begin()));
                std::vector<std::array<float, 3>> positions(used.size());
                for (size_t v = 0; v < used.size(); ++v)
                    positions[v] = mesh.vertices[used[v]];
                xatlas::MeshDecl decl;
                decl.vertexPositionData = positions.data();
                decl.vertexPositionStride = sizeof(positions[0]);
                decl.vertexCount = static_cast<uint32_t>(positions.size());
                decl.indexData = indices.data();
                decl.indexCount = static_cast<uint32_t>(indices.size());
                decl.indexFormat = xatlas::IndexFormat::UInt32;
                if (const auto error = xatlas::AddMesh(atlas.get(), decl, static_cast<uint32_t>(ranges.size())); error != xatlas::AddMeshError::Success)
                    throw collisionError("mesh.unwrap_failed", "Could not unwrap the mesh", ErrorCode::Internal, xatlas::StringForEnum(error));
            }
            xatlas::AddMeshJoin(atlas.get());
            if (stop.stop_requested())
                throw core::OperationCancelled();
            const auto charts_start = Clock::now();
            xatlas::ComputeCharts(atlas.get(), xatlas::ChartOptions{});
            const auto pack_start = Clock::now();
            if (stop.stop_requested())
                throw core::OperationCancelled();
            xatlas::PackOptions pack;
            pack.texelsPerUnit = 1.f / texel_size;
            pack.resolution = static_cast<uint32_t>(page_size);
            pack.padding = kPagePadding;
            pack.bilinear = true;
            xatlas::PackCharts(atlas.get(), pack);
            if (stop.stop_requested())
                throw core::OperationCancelled();
            const uint32_t pages = atlas->atlasCount;
            if (pages == 0 || atlas->meshCount != ranges.size() || atlas->width != uint32_t(page_size) || atlas->height != uint32_t(page_size))
                throw collisionError("mesh.unwrap_failed", "Could not unwrap the mesh", ErrorCode::Internal,
                                     std::format("{} pages of {}x{} from {} pieces", pages, atlas->width, atlas->height, atlas->meshCount));

            UnwrappedGroup group;
            group.charts = atlas->chartCount;
            group.seconds_charts = seconds(charts_start, pack_start);
            group.seconds_packing = seconds(pack_start, Clock::now());
            group.page_faces.resize(pages);
            for (uint32_t m = 0; m < atlas->meshCount; ++m) {
                const auto& output = atlas->meshes[m];
                const auto base = static_cast<int32_t>(group.sources.size());
                for (uint32_t v = 0; v < output.vertexCount; ++v) {
                    const auto& vertex = output.vertexArray[v];
                    group.sources.push_back(piece_vertices[m][vertex.xref]);
                    group.uvs.push_back({vertex.uv[0] / float(page_size), vertex.uv[1] / float(page_size)});
                }
                for (uint32_t i = 0; i + 2 < output.indexCount; i += 3) {
                    const uint32_t a = output.indexArray[i], b = output.indexArray[i + 1], c = output.indexArray[i + 2];
                    const int32_t page = output.vertexArray[a].atlasIndex;
                    // Faces without area stay out of the charts.
                    if (page >= 0 && output.vertexArray[b].atlasIndex == page && output.vertexArray[c].atlasIndex == page)
                        group.page_faces[size_t(page)].push_back({base + int32_t(a), base + int32_t(b), base + int32_t(c)});
                }
            }
            return group;
        }
    } // namespace

    void unwrap_splat_mesh(SplatMesh& mesh, const float texel_size, const int page_size, const std::stop_token stop, const size_t piece_faces, const size_t group_faces) {
        const float max_extent = kPieceShareOfPage * float(page_size) * texel_size;
        const auto ranges = split_mesh_into_pieces(mesh, piece_faces, max_extent, kUnwrapThreads);

        // The pieces come in the order of the splits, so neighbors in the list are neighbors in space.
        std::vector<std::span<const std::pair<size_t, size_t>>> group_ranges;
        for (size_t first = 0; first < ranges.size();) {
            size_t last = first + 1;
            while (last < ranges.size() && ranges[last].second - ranges[first].first <= group_faces)
                ++last;
            group_ranges.emplace_back(ranges.data() + first, last - first);
            first = last;
        }
        std::vector<UnwrappedGroup> groups(group_ranges.size());
        detail::parallel_for(groups.size(), kUnwrapThreads, [&](const size_t g) { groups[g] = unwrap_group(mesh, group_ranges[g], texel_size, page_size, stop); });

        size_t vertex_count = 0, face_count = 0, charts = 0;
        double seconds_charts = 0., seconds_packing = 0.;
        for (const auto& group : groups) {
            vertex_count += group.sources.size();
            charts += group.charts;
            seconds_charts += group.seconds_charts;
            seconds_packing += group.seconds_packing;
            for (const auto& faces : group.page_faces)
                face_count += faces.size();
        }
        const bool has_normals = mesh.normals.size() == mesh.vertices.size();
        const bool has_colors = mesh.colors.size() == mesh.vertices.size();
        std::vector<std::array<float, 3>> vertices, normals;
        std::vector<std::array<uint8_t, 3>> colors;
        std::vector<std::array<float, 2>> uvs;
        std::vector<std::array<int32_t, 3>> faces;
        std::vector<int32_t> uv_source;
        uv_source.reserve(vertex_count);
        vertices.reserve(vertex_count);
        normals.reserve(has_normals ? vertex_count : 0);
        colors.reserve(has_colors ? vertex_count : 0);
        uvs.reserve(vertex_count);
        faces.reserve(face_count);
        std::vector<size_t> page_first_face{0};
        for (auto& group : groups) {
            const auto base = static_cast<int32_t>(vertices.size());
            for (const auto source : group.sources) {
                uv_source.push_back(source);
                vertices.push_back(mesh.vertices[source]);
                if (has_normals)
                    normals.push_back(mesh.normals[source]);
                if (has_colors)
                    colors.push_back(mesh.colors[source]);
            }
            uvs.insert(uvs.end(), group.uvs.begin(), group.uvs.end());
            for (const auto& page : group.page_faces) {
                for (const auto& face : page)
                    faces.push_back({base + face[0], base + face[1], base + face[2]});
                page_first_face.push_back(faces.size());
            }
            group = {};
        }
        LOG_INFO("Mesh: {} faces in {} charts on {} pages from {} groups; {:.1f} s of charts, {:.1f} s of packing over all groups; {} faces without area left out", faces.size(), charts,
                 page_first_face.size() - 1, groups.size(), seconds_charts, seconds_packing, mesh.faces.size() - faces.size());
        mesh.vertices = std::move(vertices);
        mesh.normals = std::move(normals);
        mesh.colors = std::move(colors);
        mesh.faces = std::move(faces);
        mesh.uvs = std::move(uvs);
        mesh.uv_source = std::move(uv_source);
        mesh.page_first_face = std::move(page_first_face);
        mesh.page_size = page_size;
        mesh.charts = charts;
    }

    MeshTexels rasterize_mesh_texels(const SplatMesh& mesh, const size_t page) {
        const auto side = size_t(mesh.page_side(page));
        const bool has_colors = mesh.colors.size() == mesh.vertices.size();
        std::vector<float> positions(side * side * 3);
        std::vector<uint8_t> fallback(side * side * 3), state(side * side, 0);
        std::vector<uint32_t> owner(side * side);
        const size_t first = mesh.page_first_face[page], last = mesh.page_first_face[page + 1];
        // Texels inside a face first, so that a margin never takes a texel from the face it lies on.
        for (const double margin : {0., kCoverageMarginTexels}) {
            for (size_t f = first; f < last; ++f) {
                const auto& face = mesh.faces[f];
                double u[3], v[3];
                for (int c = 0; c < 3; ++c) {
                    u[c] = double(mesh.uvs[face[c]][0]) * double(side);
                    v[c] = double(mesh.uvs[face[c]][1]) * double(side);
                }
                const double doubled_area = (u[1] - u[0]) * (v[2] - v[0]) - (u[2] - u[0]) * (v[1] - v[0]);
                if (doubled_area == 0.)
                    continue;
                const double sign = doubled_area > 0. ? 1. : -1.;
                const double edge_length[3] = {std::hypot(u[2] - u[1], v[2] - v[1]), std::hypot(u[0] - u[2], v[0] - v[2]), std::hypot(u[1] - u[0], v[1] - v[0])};
                const auto low_u = long(std::floor(std::min({u[0], u[1], u[2]}) - margin)), high_u = long(std::ceil(std::max({u[0], u[1], u[2]}) + margin));
                const auto low_v = long(std::floor(std::min({v[0], v[1], v[2]}) - margin)), high_v = long(std::ceil(std::max({v[0], v[1], v[2]}) + margin));
                for (long y = std::max(low_v, 0l); y <= std::min(high_v, long(side) - 1); ++y) {
                    for (long x = std::max(low_u, 0l); x <= std::min(high_u, long(side) - 1); ++x) {
                        const size_t at = size_t(y) * side + size_t(x);
                        if (state[at])
                            continue;
                        const double px = double(x) + .5, py = double(y) + .5;
                        double w[3];
                        bool covered = true;
                        for (int c = 0; c < 3; ++c) {
                            const int a = (c + 1) % 3, b = (c + 2) % 3;
                            const double edge = sign * ((u[b] - u[a]) * (py - v[a]) - (v[b] - v[a]) * (px - u[a]));
                            w[c] = edge / std::abs(doubled_area);
                            covered = covered && edge >= -margin * edge_length[c];
                        }
                        if (!covered)
                            continue;
                        const double sum = std::max(w[0], 0.) + std::max(w[1], 0.) + std::max(w[2], 0.);
                        for (int c = 0; c < 3; ++c)
                            w[c] = std::max(w[c], 0.) / sum;
                        state[at] = 1;
                        owner[at] = static_cast<uint32_t>(f);
                        for (int k = 0; k < 3; ++k) {
                            double position = 0., color = 128.;
                            for (int c = 0; c < 3; ++c)
                                position += w[c] * double(mesh.vertices[face[c]][k]);
                            if (has_colors) {
                                color = 0.;
                                for (int c = 0; c < 3; ++c)
                                    color += w[c] * double(mesh.colors[face[c]][k]);
                            }
                            positions[3 * at + k] = float(position);
                            fallback[3 * at + k] = static_cast<uint8_t>(std::lround(color));
                        }
                    }
                }
            }
        }
        MeshTexels texels;
        const auto count = size_t(std::ranges::count(state, uint8_t{1}));
        texels.pixels.reserve(count);
        texels.positions.reserve(3 * count);
        texels.fallback.reserve(3 * count);
        texels.faces.reserve(count);
        for (size_t at = 0; at < state.size(); ++at) {
            if (!state[at])
                continue;
            texels.pixels.push_back(static_cast<uint32_t>(at));
            texels.faces.push_back(owner[at]);
            texels.positions.insert(texels.positions.end(), positions.begin() + std::ptrdiff_t(3 * at), positions.begin() + std::ptrdiff_t(3 * at + 3));
            texels.fallback.insert(texels.fallback.end(), fallback.begin() + std::ptrdiff_t(3 * at), fallback.begin() + std::ptrdiff_t(3 * at + 3));
        }
        return texels;
    }

    void FaceViewCandidates::add(const int32_t view, const std::span<const float> face_scores) {
        for (size_t f = 0; f < face_scores.size(); ++f) {
            const float score = face_scores[f];
            float* s = &scores[f * kPerFace];
            int32_t* v = &views[f * kPerFace];
            if (score <= s[kPerFace - 1])
                continue;
            size_t slot = kPerFace - 1;
            for (; slot > 0 && score > s[slot - 1]; --slot) {
                s[slot] = s[slot - 1];
                v[slot] = v[slot - 1];
            }
            s[slot] = score;
            v[slot] = view;
        }
    }

    std::vector<int32_t> choose_face_views(const SplatMesh& mesh, const FaceViewCandidates& candidates) {
        constexpr size_t k = FaceViewCandidates::kPerFace;
        constexpr float kServes = .5f;   // of the best footprint: detail at most 1.4 times coarser
        constexpr float kMayJoin = .25f; // twice as coarse, for faces surrounded by another view
        constexpr int kJoinSweeps = 5;
        const size_t face_count = mesh.faces.size();
        const int32_t view_count = candidates.views.empty() ? 0 : *std::ranges::max_element(candidates.views) + 1;
        std::vector<double> served(static_cast<size_t>(view_count), 0.);
        std::vector<std::vector<uint32_t>> faces_of_view(static_cast<size_t>(view_count));
        std::vector<double> areas(face_count);
        const auto eligible = [&](const size_t f, const size_t slot, const float share) {
            const float score = candidates.scores[f * k + slot];
            return score > 0.f && score >= share * candidates.scores[f * k];
        };
        for (size_t f = 0; f < face_count; ++f) {
            const auto& a = mesh.vertices[mesh.faces[f][0]];
            const auto& b = mesh.vertices[mesh.faces[f][1]];
            const auto& c = mesh.vertices[mesh.faces[f][2]];
            const double u[3] = {double(b[0]) - a[0], double(b[1]) - a[1], double(b[2]) - a[2]};
            const double v[3] = {double(c[0]) - a[0], double(c[1]) - a[1], double(c[2]) - a[2]};
            areas[f] = .5 * std::hypot(u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]);
            for (size_t slot = 0; slot < k; ++slot)
                if (eligible(f, slot, kServes)) {
                    const auto view = size_t(candidates.views[f * k + slot]);
                    served[view] += areas[f];
                    faces_of_view[view].push_back(static_cast<uint32_t>(f));
                }
        }
        std::vector<int32_t> labels(face_count, -1);
        for (int32_t round = 0; round < view_count; ++round) {
            const auto best = size_t(std::ranges::max_element(served) - served.begin());
            if (faces_of_view[best].empty())
                break;
            for (const auto f : faces_of_view[best]) {
                if (labels[f] >= 0)
                    continue;
                labels[f] = int32_t(best);
                for (size_t slot = 0; slot < k; ++slot)
                    if (eligible(f, slot, kServes))
                        served[size_t(candidates.views[size_t(f) * k + slot])] -= areas[f];
            }
            faces_of_view[best] = {};
            served[best] = -1.;
        }

        // Neighbors across each edge, by the surface's own vertices: charts repeat vertices along their seams.
        const auto source = [&](const int32_t vertex) { return mesh.uv_source.empty() ? vertex : mesh.uv_source[size_t(vertex)]; };
        struct Edge {
            uint64_t key;
            uint32_t face;
        };
        std::vector<Edge> edges;
        edges.reserve(face_count * 3);
        for (size_t f = 0; f < face_count; ++f)
            for (int c = 0; c < 3; ++c) {
                const auto p = uint32_t(source(mesh.faces[f][c])), q = uint32_t(source(mesh.faces[f][(c + 1) % 3]));
                edges.push_back({(uint64_t(std::min(p, q)) << 32) | std::max(p, q), static_cast<uint32_t>(f)});
            }
        std::ranges::sort(edges, {}, &Edge::key);
        constexpr uint32_t kNone = ~uint32_t{0};
        std::vector<std::array<uint32_t, 3>> neighbors(face_count, {kNone, kNone, kNone});
        const auto link = [&](const uint32_t a, const uint32_t b) {
            for (auto& slot : neighbors[a])
                if (slot == kNone) {
                    slot = b;
                    return;
                }
        };
        for (size_t e = 0; e + 1 < edges.size(); ++e)
            if (edges[e].key == edges[e + 1].key) {
                link(edges[e].face, edges[e + 1].face);
                link(edges[e + 1].face, edges[e].face);
            }
        edges = {};
        for (int sweep = 0; sweep < kJoinSweeps; ++sweep) {
            size_t changed = 0;
            for (size_t f = 0; f < face_count; ++f) {
                const auto& n = neighbors[f];
                int32_t majority = -1;
                for (int a = 0; a < 3 && majority < 0; ++a) {
                    if (n[size_t(a)] == kNone)
                        continue;
                    const int32_t label = labels[n[size_t(a)]];
                    int same = 0;
                    for (const auto other : n)
                        same += other != kNone && labels[other] == label;
                    if (label >= 0 && same >= 2)
                        majority = label;
                }
                if (majority < 0 || majority == labels[f])
                    continue;
                for (size_t slot = 0; slot < k; ++slot)
                    if (candidates.views[f * k + slot] == majority && eligible(f, slot, kMayJoin)) {
                        labels[f] = majority;
                        ++changed;
                        break;
                    }
            }
            if (changed == 0)
                break;
        }
        return labels;
    }

    ColorLeveling plan_color_leveling(const SplatMesh& mesh, const std::span<const int32_t> face_views) {
        // Points of the surface by position: levels of detail and tiles repeat them, and leveling them together keeps
        // tile borders closed.
        std::vector<uint32_t> order(mesh.vertices.size());
        std::iota(order.begin(), order.end(), 0u);
        std::ranges::sort(order, {}, [&](const uint32_t v) { return mesh.vertices[v]; });
        std::vector<uint32_t> point_of(mesh.vertices.size());
        uint32_t points = 0;
        for (size_t i = 0; i < order.size(); ++i) {
            if (i > 0 && mesh.vertices[order[i]] != mesh.vertices[order[i - 1]])
                ++points;
            point_of[order[i]] = points;
        }
        order = {};

        struct Corner {
            uint64_t key; // point, then label
            uint32_t corner;
        };
        std::vector<Corner> corners(mesh.faces.size() * 3);
        for (size_t f = 0; f < mesh.faces.size(); ++f)
            for (size_t c = 0; c < 3; ++c)
                corners[3 * f + c] = {(uint64_t(point_of[size_t(mesh.faces[f][c])]) << 32) | uint32_t(face_views[f] + 1), static_cast<uint32_t>(3 * f + c)};
        std::ranges::sort(corners, {}, &Corner::key);

        ColorLeveling leveling;
        leveling.corner_unknowns.resize(corners.size());
        uint32_t unknown = 0;
        for (size_t i = 0; i < corners.size();) {
            const uint32_t point = uint32_t(corners[i].key >> 32);
            size_t next_point = i;
            uint32_t labels_here = 0;
            for (size_t j = i; j < corners.size() && uint32_t(corners[j].key >> 32) == point; ++j, ++next_point)
                labels_here += j == i || corners[j].key != corners[j - 1].key;
            for (size_t j = i; j < next_point; ++j) {
                const bool first = j == i || corners[j].key != corners[j - 1].key;
                if (first && j != i)
                    ++unknown;
                leveling.corner_unknowns[corners[j].corner] = unknown;
                if (first && labels_here > 1) {
                    const uint32_t face = corners[j].corner / 3;
                    const auto& position = mesh.vertices[size_t(mesh.faces[face][corners[j].corner % 3])];
                    leveling.sample_labels.push_back(face_views[face]);
                    leveling.sample_positions.insert(leveling.sample_positions.end(), position.begin(), position.end());
                    leveling.sample_faces.push_back(face);
                    leveling.sample_unknowns.push_back(unknown);
                    leveling.sample_points.push_back(point);
                }
            }
            ++unknown;
            i = next_point;
        }
        leveling.offsets.assign(size_t(unknown) * 3, 0.f);
        return leveling;
    }

    void solve_color_leveling(ColorLeveling& leveling, const SplatMesh& mesh, const std::span<const float> colors) {
        // Texrecon's weights: a step across a seam counts ten times a difference along the surface.
        constexpr float kSmooth = .1f;
        // Keeps a photo as it is far from any seam; a step spreads over about sqrt(kSmooth / kPull) = 100 vertices.
        constexpr float kPull = 1e-5f;
        constexpr int kMaxIterations = 2000;
        constexpr double kTolerance = 1e-3;
        const size_t unknowns = leveling.offsets.size() / 3;
        struct Entry {
            uint64_t key;
            float value;
        };
        std::vector<Entry> entries;
        entries.reserve(mesh.faces.size() * 6);
        std::vector<float> diagonal(unknowns, kPull), right(unknowns * 3, 0.f);
        const auto couple = [&](const uint32_t a, const uint32_t b, const float weight) {
            if (a == b)
                return;
            diagonal[a] += weight;
            diagonal[b] += weight;
            entries.push_back({(uint64_t(a) << 32) | b, -weight});
            entries.push_back({(uint64_t(b) << 32) | a, -weight});
        };
        // Each edge inside a region comes twice, once from each face.
        for (size_t f = 0; f < mesh.faces.size(); ++f)
            for (size_t c = 0; c < 3; ++c)
                couple(leveling.corner_unknowns[3 * f + c], leveling.corner_unknowns[3 * f + (c + 1) % 3], .5f * kSmooth);
        size_t steps = 0;
        double step_sum = 0.;
        for (size_t first = 0; first < leveling.sample_points.size();) {
            size_t last = first;
            while (last < leveling.sample_points.size() && leveling.sample_points[last] == leveling.sample_points[first])
                ++last;
            for (size_t a = first; a < last; ++a)
                for (size_t b = a + 1; b < last; ++b) {
                    if (colors[3 * a] < 0.f || colors[3 * b] < 0.f)
                        continue;
                    const uint32_t ua = leveling.sample_unknowns[a], ub = leveling.sample_unknowns[b];
                    couple(ua, ub, 1.f);
                    ++steps;
                    for (size_t c = 0; c < 3; ++c) {
                        const float step = colors[3 * a + c] - colors[3 * b + c];
                        step_sum += std::abs(step);
                        right[3 * size_t(ua) + c] -= step;
                        right[3 * size_t(ub) + c] += step;
                    }
                }
            first = last;
        }
        std::ranges::sort(entries, {}, &Entry::key);
        std::vector<size_t> row_first(unknowns + 1, 0);
        std::vector<uint32_t> columns;
        std::vector<float> values;
        columns.reserve(entries.size());
        values.reserve(entries.size());
        for (size_t e = 0; e < entries.size(); ++e) {
            if (e > 0 && entries[e].key == entries[e - 1].key) {
                values.back() += entries[e].value;
                continue;
            }
            columns.push_back(uint32_t(entries[e].key));
            values.push_back(entries[e].value);
            ++row_first[size_t(entries[e].key >> 32) + 1];
        }
        entries = {};
        std::partial_sum(row_first.begin(), row_first.end(), row_first.begin());

        // Conjugate gradients with the diagonal as preconditioner, on the three channels at once.
        constexpr size_t kChunk = size_t{1} << 15;
        const size_t chunks = (unknowns + kChunk - 1) / kChunk;
        const auto over_unknowns = [&](auto&& body) {
            detail::parallel_for(chunks, kUnwrapThreads, [&](const size_t chunk) {
                for (size_t i = chunk * kChunk; i < std::min((chunk + 1) * kChunk, unknowns); ++i)
                    body(i);
            });
        };
        auto& x = leveling.offsets;
        std::vector<float> residual = right, direction(unknowns * 3), product(unknowns * 3);
        const auto dot = [&](const std::vector<float>& a, const std::vector<float>& b, const bool scaled) {
            std::array<double, 3> total{0., 0., 0.};
            for (size_t i = 0; i < unknowns; ++i) {
                const double scale = scaled ? 1. / double(diagonal[i]) : 1.;
                for (size_t c = 0; c < 3; ++c)
                    total[c] += scale * double(a[3 * i + c]) * double(b[3 * i + c]);
            }
            return total;
        };
        over_unknowns([&](const size_t i) {
            for (size_t c = 0; c < 3; ++c)
                direction[3 * i + c] = residual[3 * i + c] / diagonal[i];
        });
        auto rho = dot(residual, residual, true);
        const auto start = dot(right, right, false);
        int iteration = 0;
        for (; iteration < kMaxIterations; ++iteration) {
            const auto norm = dot(residual, residual, false);
            bool done = true;
            for (size_t c = 0; c < 3; ++c)
                done = done && norm[c] <= kTolerance * kTolerance * start[c];
            if (done)
                break;
            over_unknowns([&](const size_t i) {
                float sum[3];
                for (size_t c = 0; c < 3; ++c)
                    sum[c] = diagonal[i] * direction[3 * i + c];
                for (size_t e = row_first[i]; e < row_first[i + 1]; ++e)
                    for (size_t c = 0; c < 3; ++c)
                        sum[c] += values[e] * direction[3 * size_t(columns[e]) + c];
                for (size_t c = 0; c < 3; ++c)
                    product[3 * i + c] = sum[c];
            });
            const auto curvature = dot(direction, product, false);
            std::array<float, 3> alpha;
            for (size_t c = 0; c < 3; ++c)
                alpha[c] = curvature[c] > 0. ? float(rho[c] / curvature[c]) : 0.f;
            over_unknowns([&](const size_t i) {
                for (size_t c = 0; c < 3; ++c) {
                    x[3 * i + c] += alpha[c] * direction[3 * i + c];
                    residual[3 * i + c] -= alpha[c] * product[3 * i + c];
                }
            });
            const auto next_rho = dot(residual, residual, true);
            std::array<float, 3> beta;
            for (size_t c = 0; c < 3; ++c)
                beta[c] = rho[c] > 0. ? float(next_rho[c] / rho[c]) : 0.f;
            over_unknowns([&](const size_t i) {
                for (size_t c = 0; c < 3; ++c)
                    direction[3 * i + c] = residual[3 * i + c] / diagonal[i] + beta[c] * direction[3 * i + c];
            });
            rho = next_rho;
        }
        leveling.iterations = size_t(iteration);
        double offset_sum = 0.;
        for (const float value : x)
            offset_sum += std::abs(value);
        LOG_INFO("Mesh: leveling {} unknowns, {} steps of {:.1f} / 255 on average, {} iterations, mean offset {:.1f} / 255", unknowns, steps,
                 255. * step_sum / double(std::max<size_t>(3 * steps, 1)), iteration, 255. * offset_sum / double(std::max<size_t>(x.size(), 1)));
    }

    void ColorLeveling::release_samples() {
        sample_labels = {};
        sample_positions = {};
        sample_faces = {};
        sample_unknowns = {};
        sample_points = {};
    }

    void ColorLeveling::offset_at(const SplatMesh& mesh, const uint32_t face, const float* position, float* offset) const {
        const auto& corner = mesh.faces[face];
        const auto& a = mesh.vertices[size_t(corner[0])];
        const auto& b = mesh.vertices[size_t(corner[1])];
        const auto& c = mesh.vertices[size_t(corner[2])];
        const float u[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
        const float v[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
        const float p[3] = {position[0] - a[0], position[1] - a[1], position[2] - a[2]};
        const auto dot = [](const float* x, const float* y) { return x[0] * y[0] + x[1] * y[1] + x[2] * y[2]; };
        const float uu = dot(u, u), uv = dot(u, v), vv = dot(v, v), pu = dot(p, u), pv = dot(p, v);
        const float area = uu * vv - uv * uv;
        const float wb = area > 0.f ? std::clamp((vv * pu - uv * pv) / area, 0.f, 1.f) : 0.f;
        const float wc = area > 0.f ? std::clamp((uu * pv - uv * pu) / area, 0.f, 1.f - wb) : 0.f;
        const float wa = 1.f - wb - wc;
        const uint32_t* unknown = &corner_unknowns[3 * size_t(face)];
        for (size_t k = 0; k < 3; ++k)
            offset[k] = wa * offsets[3 * size_t(unknown[0]) + k] + wb * offsets[3 * size_t(unknown[1]) + k] + wc * offsets[3 * size_t(unknown[2]) + k];
    }

    std::vector<uint8_t> compose_mesh_page(const MeshTexels& texels, const std::vector<float>& colors, const int page_size) {
        const auto side = size_t(page_size);
        std::vector<uint8_t> image(side * side * 3, 0), state(side * side, 0);
        std::array<double, 3> total{0., 0., 0.};
        for (size_t t = 0; t < texels.pixels.size(); ++t) {
            const size_t at = texels.pixels[t];
            const bool observed = colors[3 * t] >= 0.f;
            state[at] = 1;
            for (int c = 0; c < 3; ++c) {
                const uint8_t value = observed ? static_cast<uint8_t>(std::clamp(std::lround(255.f * colors[3 * t + c]), 0l, 255l)) : texels.fallback[3 * t + c];
                image[3 * at + c] = value;
                total[c] += value;
            }
        }
        std::vector<uint32_t> frontier;
        const auto neighbors = [&](const size_t at, auto&& visit) {
            const size_t x = at % side, y = at / side;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    if ((dx == 0 && dy == 0) || (dx < 0 && x == 0) || (dy < 0 && y == 0) || (dx > 0 && x + 1 == side) || (dy > 0 && y + 1 == side))
                        continue;
                    visit(size_t(ptrdiff_t(y) + dy) * side + size_t(ptrdiff_t(x) + dx));
                }
        };
        for (const auto at : texels.pixels)
            neighbors(at, [&](const size_t other) {
                if (state[other] == 0) {
                    state[other] = 2;
                    frontier.push_back(static_cast<uint32_t>(other));
                }
            });
        std::vector<uint32_t> next;
        for (int ring = 0; ring < kFillRings && !frontier.empty(); ++ring) {
            next.clear();
            for (const auto at : frontier) {
                int sum[3] = {0, 0, 0}, count = 0;
                neighbors(at, [&](const size_t other) {
                    if (state[other] == 1) {
                        ++count;
                        for (int c = 0; c < 3; ++c)
                            sum[c] += image[3 * other + c];
                    } else if (state[other] == 0) {
                        state[other] = 2;
                        next.push_back(static_cast<uint32_t>(other));
                    }
                });
                for (int c = 0; c < 3; ++c)
                    image[3 * size_t(at) + c] = static_cast<uint8_t>((sum[c] + count / 2) / std::max(count, 1));
            }
            for (const auto at : frontier)
                state[at] = 1;
            frontier.swap(next);
        }
        const double count = double(std::max<size_t>(texels.pixels.size(), 1));
        const uint8_t mean[3] = {static_cast<uint8_t>(total[0] / count), static_cast<uint8_t>(total[1] / count), static_cast<uint8_t>(total[2] / count)};
        for (size_t at = 0; at < state.size(); ++at)
            if (state[at] != 1)
                std::memcpy(&image[3 * at], mean, 3);
        return image;
    }

    namespace {
        void write_or_throw(std::ofstream& out, const std::filesystem::path& path) {
            out.close();
            if (!out)
                throw collisionError("mesh.write_failed", "Could not write the mesh; check disk space and permissions", ErrorCode::Unavailable, core::path_to_utf8(path));
        }

        void write_jpeg(const std::vector<uint8_t>& rgb, const int side, const std::filesystem::path& path) {
            std::ofstream out(core::native_fs::native_path(path), std::ios::binary | std::ios::trunc);
            const auto append = [](void* context, void* data, const int size) { static_cast<std::ofstream*>(context)->write(static_cast<const char*>(data), size); };
            if (!stbi_write_jpg_to_func(append, &out, side, side, 3, rgb.data(), kTextureJpegQuality))
                throw collisionError("mesh.write_failed", "Could not write the mesh texture", ErrorCode::Unavailable, core::path_to_utf8(path));
            write_or_throw(out, path);
        }

        struct LineBuffer {
            std::ofstream& out;
            std::vector<char> data = std::vector<char>(size_t{1} << 24);
            size_t used = 0;
            void reserve(const size_t bytes) {
                if (used + bytes > data.size()) {
                    out.write(data.data(), std::streamsize(used));
                    used = 0;
                }
            }
            void flush() { reserve(data.size()); }
            void text(const std::string_view value) {
                reserve(value.size());
                std::memcpy(data.data() + used, value.data(), value.size());
                used += value.size();
            }
            void number(const auto value) {
                reserve(64);
                used = size_t(std::to_chars(data.data() + used, data.data() + data.size(), value).ptr - data.data());
            }
        };
    } // namespace

    WeldedVertices weld_chart_seams(const SplatMesh& mesh) {
        WeldedVertices welded;
        const size_t count = mesh.vertices.size();
        welded.index.resize(count);
        if (mesh.uv_source.empty()) {
            welded.originals.resize(count);
            std::iota(welded.index.begin(), welded.index.end(), 0);
            std::iota(welded.originals.begin(), welded.originals.end(), 0);
            return welded;
        }
        auto sources = mesh.uv_source;
        std::ranges::sort(sources);
        sources.erase(std::unique(sources.begin(), sources.end()), sources.end());
        welded.originals.assign(sources.size(), -1);
        for (size_t v = 0; v < count; ++v) {
            const auto merged = static_cast<int32_t>(std::ranges::lower_bound(sources, mesh.uv_source[v]) - sources.begin());
            welded.index[v] = merged;
            if (welded.originals[size_t(merged)] < 0)
                welded.originals[size_t(merged)] = static_cast<int32_t>(v);
        }
        return welded;
    }

    void write_splat_mesh_obj(const SplatMesh& mesh, const std::filesystem::path& path) {
        if (mesh.uvs.size() != mesh.vertices.size() || mesh.page_first_face.size() != mesh.pages.size() + 1)
            throw collisionError("mesh.write_failed", "The mesh has no texture to write", ErrorCode::InvalidArgument);
        const auto stem = core::path_to_utf8(path.stem());
        const auto folder = path.parent_path();
        const auto mtl_name = stem + ".mtl";
        {
            std::ofstream out(core::native_fs::native_path(folder / core::utf8_to_path(mtl_name)), std::ios::binary | std::ios::trunc);
            for (size_t p = 0; p < mesh.pages.size(); ++p)
                out << std::format("newmtl page_{}\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nillum 1\nmap_Kd {}_{}.jpg\n\n", p, stem, p);
            write_or_throw(out, folder / core::utf8_to_path(mtl_name));
        }
        detail::parallel_for(mesh.pages.size(), kUnwrapThreads, [&](const size_t p) {
            write_jpeg(mesh.pages[p], mesh.page_size, folder / core::utf8_to_path(std::format("{}_{}.jpg", stem, p)));
        });

        const auto welded = weld_chart_seams(mesh);
        const auto temporary = std::filesystem::path(path).concat(".tmp");
        {
            std::ofstream out(core::native_fs::native_path(temporary), std::ios::binary | std::ios::trunc);
            LineBuffer lines{out};
            lines.text(std::format("mtllib {}\n", mtl_name));
            for (const auto original : welded.originals) {
                const auto& p = mesh.vertices[size_t(original)];
                lines.text("v ");
                lines.number(p[0]);
                lines.text(" ");
                lines.number(p[1]);
                lines.text(" ");
                lines.number(p[2]);
                lines.text("\n");
            }
            // OBJ counts v up from the bottom of the image.
            for (const auto& uv : mesh.uvs) {
                lines.text("vt ");
                lines.number(uv[0]);
                lines.text(" ");
                lines.number(1.f - uv[1]);
                lines.text("\n");
            }
            for (const auto original : welded.originals) {
                const auto& n = mesh.normals[size_t(original)];
                lines.text("vn ");
                lines.number(n[0]);
                lines.text(" ");
                lines.number(n[1]);
                lines.text(" ");
                lines.number(n[2]);
                lines.text("\n");
            }
            for (size_t p = 0; p < mesh.pages.size(); ++p) {
                lines.text(std::format("usemtl page_{}\n", p));
                for (size_t f = mesh.page_first_face[p]; f < mesh.page_first_face[p + 1]; ++f) {
                    lines.text("f");
                    for (const auto vertex : mesh.faces[f]) {
                        const int32_t position = welded.index[size_t(vertex)] + 1;
                        lines.text(" ");
                        lines.number(position);
                        lines.text("/");
                        lines.number(vertex + 1);
                        lines.text("/");
                        lines.number(position);
                    }
                    lines.text("\n");
                }
            }
            lines.flush();
            write_or_throw(out, path);
        }
        std::error_code error;
        core::native_fs::rename(temporary, path, error);
        if (error)
            throw collisionError("mesh.write_failed", error.message(), ErrorCode::Unavailable, core::path_to_utf8(path));
    }

} // namespace lfs::training
