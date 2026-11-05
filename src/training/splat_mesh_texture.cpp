/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/collision_volume.hpp"
#include "core/logger.hpp"
#include "core/native_fs.hpp"
#include "core/path_utils.hpp"
#include "splat_mesh.hpp"
#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <format>
#include <fstream>
#include <memory>
#include <numeric>
#include <span>
#include <stb_image_write.h>
#include <thread>
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

        void parallel_for(const size_t count, const int thread_limit, auto&& body) {
            const int threads = std::clamp(int(std::thread::hardware_concurrency()), 1, thread_limit);
            std::atomic<size_t> next{0};
            std::vector<std::exception_ptr> failures(static_cast<size_t>(threads));
            const auto work = [&](const int thread) {
                try {
                    for (size_t i = next++; i < count; i = next++)
                        body(i);
                } catch (...) {
                    failures[size_t(thread)] = std::current_exception();
                    next = count;
                }
            };
            std::vector<std::thread> pool;
            for (int t = 1; t < threads; ++t)
                pool.emplace_back(work, t);
            work(0);
            for (auto& thread : pool)
                thread.join();
            for (const auto& failure : failures)
                if (failure)
                    std::rethrow_exception(failure);
        }
    } // namespace

    namespace {
        // The charts of a group of pieces, packed onto pages of their own.
        struct UnwrappedGroup {
            std::vector<int32_t> sources;              // per output vertex, the mesh vertex it repeats
            std::vector<std::array<float, 2>> uvs;     // per output vertex
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
        parallel_for(groups.size(), kUnwrapThreads, [&](const size_t g) { groups[g] = unwrap_group(mesh, group_ranges[g], texel_size, page_size, stop); });

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
        const auto side = size_t(mesh.page_size);
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
        parallel_for(mesh.pages.size(), kUnwrapThreads, [&](const size_t p) {
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
