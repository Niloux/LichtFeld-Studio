/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "splat_mesh.hpp"

#include "components/rolling_shutter.hpp"
#include "core/camera.hpp"
#include "core/checked_arithmetic.hpp"
#include "core/collision_volume.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/logger.hpp"
#include "core/memory_pressure.hpp"
#include "core/mesh_data.hpp"
#include "core/native_fs.hpp"
#include "core/path_utils.hpp"
#include "core/splat_data.hpp"
#include "core/tensor/internal/cub_workspace.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "rasterization/gsplat/GeometryFeatures.h"
#include "rasterization/gsplat/TsdfFusion.h"
#include "rasterization/gsplat_rasterizer.hpp"
#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <exception>
#include <format>
#include <fstream>
#include <meshoptimizer.h>
#include <mutex>
#include <new>
#include <numeric>
#include <thread>

namespace lfs::training {
    namespace {
        using core::collisionError;
        using core::Device;
        using core::Tensor;
        using Clock = std::chrono::steady_clock;

        constexpr uint64_t kAxisStep[3] = {uint64_t{1}, uint64_t{1} << gsplat_lfs::kTsdfKeyBits,
                                           uint64_t{1} << (2 * gsplat_lfs::kTsdfKeyBits)};
        constexpr uint64_t kAxisMask = (uint64_t{1} << gsplat_lfs::kTsdfKeyBits) - 1;
        // Measured on a 30k indoor scene at 2 cm voxels and 1 cm tolerance: a weight of 20 loses 0.16 dB
        // of color against the photos, and a weight 400 times smaller loses 1 dB.
        constexpr float kSimplifyColorWeightPerVoxel = 1000.f;
        constexpr double kTsdfShareOfFreeMemory = 0.8;
        constexpr int kSimplifyThreads = 8;

        template <class T>
        struct DeviceArray {
            core::tensor_ops::ScopedDeviceBuffer storage;
            size_t count = 0;
            DeviceArray() = default;
            DeviceArray(const size_t n, cudaStream_t stream, const bool zeroed = false)
                : storage(core::checked_product(std::max<size_t>(n, 1), sizeof(T), "Mesh allocation"), stream, "splat_mesh"), count(n) {
                if (zeroed)
                    LFS_CUDA_CHECK(cudaMemsetAsync(data(), 0, count * sizeof(T), stream));
            }
            [[nodiscard]] T* data() const { return storage.template as<T>(); }
        };

        [[nodiscard]] double seconds_since(const Clock::time_point start) {
            return std::chrono::duration<double>(Clock::now() - start).count();
        }

        void validate(const core::SplatMeshSettings& s) {
            const bool ok = std::isfinite(s.voxel_size) && s.voxel_size > 0 && s.block_voxels >= 2 && s.block_voxels <= 32 &&
                            std::isfinite(s.truncation) && s.truncation >= 0 && std::isfinite(s.fill_hole_perimeter) &&
                            s.fill_hole_perimeter >= 0 && std::isfinite(s.full_weight_range) &&
                            s.full_weight_range >= 0 && s.view_stride >= 1 && s.color_stride >= 1 && s.min_views >= 1 &&
                            s.min_views <= 65535 && s.min_component_faces >= 0 && s.max_width >= 16 &&
                            std::isfinite(s.simplify_error) && s.simplify_error >= 0 && std::isfinite(s.max_range) && s.max_range >= 0;
            if (!ok)
                throw collisionError("mesh.settings_invalid", "Invalid mesh settings", ErrorCode::InvalidArgument,
                                     std::format("{}; {}", s.voxel_size, s.truncation));
        }

        void device_buffer_preflight(const size_t bytes, const char* operation) {
            if (!core::MemoryPressureCoordinator::instance().preflight({.operation = operation, .temporary_device_bytes = bytes}).ok)
                throw collisionError("mesh.memory", "Not enough GPU memory for the mesh; free memory or use a larger voxel size",
                                     ErrorCode::ResourceExhausted, std::format("{:.1f} GB", static_cast<double>(bytes) / (1 << 30)));
        }

        template <size_t N>
        void copy_coefficients(const Tensor& source, float (&target)[N], const size_t limit = N) {
            if (!source.is_valid() || source.numel() == 0)
                return;
            auto host = source.cpu().to(core::DataType::Float32).contiguous();
            std::memcpy(target, host.ptr<float>(), std::min({source.numel(), limit, N}) * sizeof(float));
        }

        void copy_matrix(const std::array<float, 16>& source, float (&target)[16]) {
            std::copy(source.begin(), source.end(), target);
        }

        // Mirrors what gsplat_rasterize_forward uses, so voxels land on the pixels that rendered them.
        [[nodiscard]] gsplat_lfs::TsdfView make_view(const core::Camera& camera) {
            gsplat_lfs::TsdfView view;
            const bool undistorted = camera.is_undistort_prepared();
            view.model = undistorted ? CameraModelType::PINHOLE : static_cast<CameraModelType>(static_cast<int>(camera.camera_model_type()));
            view.width = static_cast<uint32_t>(camera.image_width());
            view.height = static_cast<uint32_t>(camera.image_height());
            std::tie(view.fx, view.fy, view.cx, view.cy) = camera.get_intrinsics();
            if (!undistorted) {
                copy_coefficients(camera.radial_distortion(), view.radial);
                if (view.model == CameraModelType::THIN_PRISM_FISHEYE) {
                    copy_coefficients(camera.tangential_distortion(), view.thin_prism);
                } else if (view.model == CameraModelType::PINHOLE) {
                    copy_coefficients(camera.tangential_distortion(), view.tangential);
                    view.has_distortion = camera.radial_distortion().is_valid() && camera.radial_distortion().numel() > 0;
                    view.has_distortion |= camera.tangential_distortion().is_valid() && camera.tangential_distortion().numel() > 0;
                }
            }
            auto matrix = camera.world_view_transform().cpu().contiguous();
            std::memcpy(view.viewmat, matrix.ptr<float>(), sizeof(view.viewmat));

            const auto& motion = camera.rolling_shutter();
            if (motion && !undistorted && view.model != CameraModelType::EQUIRECTANGULAR) {
                const bool rows = motion->readout_rows;
                const int extent = static_cast<int>(rows ? view.height : view.width);
                if (extent > 1) {
                    auto R = camera.R().cpu().contiguous();
                    auto T = camera.T().cpu().contiguous();
                    std::array<float, 9> rotation{};
                    std::array<float, 3> translation{};
                    std::memcpy(rotation.data(), R.ptr<float>(), sizeof(rotation));
                    std::memcpy(translation.data(), T.ptr<float>(), sizeof(translation));
                    const auto poses = rolling_shutter_window_poses(*motion, rotation, translation, extent, 0, extent);
                    copy_matrix(poses.first, view.readout0);
                    copy_matrix(poses.last, view.readout1);
                    view.shutter = rows ? ShutterType::ROLLING_TOP_TO_BOTTOM : ShutterType::ROLLING_LEFT_TO_RIGHT;
                }
            }
            return view;
        }

        void camera_rays(const gsplat_lfs::TsdfView& view, float* rays, cudaStream_t stream) {
            const float K[9] = {view.fx, 0.f, view.cx, 0.f, view.fy, view.cy, 0.f, 0.f, 1.f};
            auto K_device = Tensor::from_vector(std::vector<float>(K, K + 9), {9}, Device::CUDA);
            const bool radial = view.model == CameraModelType::FISHEYE || view.model == CameraModelType::THIN_PRISM_FISHEYE || view.has_distortion;
            const bool tangential = view.model == CameraModelType::PINHOLE && view.has_distortion;
            auto radial_device = Tensor::from_vector(std::vector<float>(view.radial, view.radial + 6), {6}, Device::CUDA);
            auto tangential_device = Tensor::from_vector(std::vector<float>(view.tangential, view.tangential + 2), {2}, Device::CUDA);
            auto prism_device = Tensor::from_vector(std::vector<float>(view.thin_prism, view.thin_prism + 4), {4}, Device::CUDA);
            const bool prism = view.model == CameraModelType::THIN_PRISM_FISHEYE;
            gsplat_lfs::geometry_camera_rays(rays, view.width, view.height, K_device.ptr<float>(), view.model,
                                             radial ? radial_device.ptr<float>() : nullptr,
                                             tangential ? tangential_device.ptr<float>() : nullptr,
                                             prism ? prism_device.ptr<float>() : nullptr, stream);
            LFS_ENSURE_CUDA_SUCCESS(cudaStreamSynchronize(stream), "Mesh camera rays");
        }

        // Restores the cameras' image sizes however the extraction ends.
        struct CameraSizes {
            struct Saved {
                core::Camera* camera;
                int width, height;
                bool loaded;
            };
            std::vector<Saved> saved;
            ~CameraSizes() {
                for (const auto& s : saved)
                    s.camera->restore_image_dimensions(s.width, s.height, s.loaded);
            }
        };

        // Renders a view's median depth, masked. The result stays valid until the next render.
        [[nodiscard]] Tensor render_median_depth(core::Camera& camera, core::SplatData& model,
                                                 const core::SplatMeshSettings& settings, cudaStream_t stream) {
            const size_t width = camera.image_width(), height = camera.image_height();
            auto depth = Tensor::empty({height, width}, Device::CUDA);
            depth.set_stream(stream);
            gsplat_lfs::FreeSpaceRender request;
            request.median_depth = depth.ptr<float>();
            Tensor background;
            auto forward = gsplat_rasterize_forward(camera, model, background, 0, 0, 0, 0, 1.f, false,
                                                    GsplatRenderMode::D, true, {}, request);
            if (!forward)
                throw collisionError("mesh.render_failed", forward.error(), ErrorCode::Internal);
            const cudaStream_t render_stream = forward->second.stream;
            core::GlobalArenaManager::instance().get_arena().end_frame(forward->second.frame_id, render_stream);
            // A worker without a current stream renders on the model's.
            LFS_ENSURE_CUDA_SUCCESS(cudaStreamSynchronize(render_stream), "Mesh render");
            if (camera.has_mask()) {
                auto mask = camera.load_mask_file(camera.mask_path(), 1, settings.max_width, settings.invert_mask,
                                                  settings.mask_threshold, true, false)
                                .to(Device::CUDA, stream);
                if (mask.numel() == width * height) {
                    depth = depth.mul(mask.reshape({int(height), int(width)}).to(core::DataType::Float32));
                } else {
                    LOG_WARN("Mesh: mask size of {} does not match the render; using the view unmasked", camera.image_name());
                }
            }
            return depth.contiguous();
        }

        struct Tsdf {
            gsplat_lfs::TsdfGrid grid;
            DeviceArray<uint64_t> keys;
            DeviceArray<float> sum, weight, carve;
            DeviceArray<uint16_t> views;
            DeviceArray<uint32_t> near_blocks, near_count;
            DeviceArray<float> raw_low, raw_high, low, high;
            uint32_t block_count = 0;
            size_t tiles = 0;
        };

        // Adds keys to a sorted unique set. Returns the new set; count is its size.
        [[nodiscard]] DeviceArray<uint64_t> merge_unique_keys(const DeviceArray<uint64_t>& existing, const size_t existing_count,
                                                              const DeviceArray<uint64_t>& added, size_t& count, cudaStream_t stream) {
            const size_t total = existing_count + added.count;
            DeviceArray<uint64_t> merged(total, stream), scratch(total, stream), counter(1, stream);
            if (existing_count > 0)
                LFS_CUDA_CHECK(cudaMemcpyAsync(merged.data(), existing.data(), existing_count * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream));
            LFS_CUDA_CHECK(cudaMemcpyAsync(merged.data() + existing_count, added.data(), added.count * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream));
            count = gsplat_lfs::tsdf_sort_unique(merged.data(), scratch.data(), total, counter.data(), stream);
            return merged;
        }

        [[nodiscard]] size_t tsdf_block_bytes(const gsplat_lfs::TsdfGrid& grid) {
            return size_t(grid.block) * grid.block * grid.block * 10 + 12;
        }

        // A run of whole block layers along z, the slowest axis of the sorted keys, with the
        // layers next to it for the neighbors of its surface voxels.
        struct TsdfChunk {
            size_t first = 0, last = 0;           // blocks whose voxels this chunk reports
            size_t held_first = 0, held_last = 0; // blocks it holds
        };

        [[nodiscard]] std::vector<TsdfChunk> plan_chunks(const std::vector<uint64_t>& keys, const size_t max_blocks) {
            const auto layer_of = [](const uint64_t key) { return key >> (2 * gsplat_lfs::kTsdfKeyBits); };
            std::vector<size_t> starts;
            for (size_t b = 0; b < keys.size(); ++b)
                if (b == 0 || layer_of(keys[b]) != layer_of(keys[b - 1]))
                    starts.push_back(b);
            starts.push_back(keys.size());
            const size_t layers = starts.size() - 1;
            std::vector<TsdfChunk> chunks;
            for (size_t layer = 0; layer < layers;) {
                const size_t held_first = starts[layer > 0 ? layer - 1 : 0];
                size_t end = layer + 1;
                const auto held_last = [&](const size_t after) { return starts[std::min(after + 1, layers)]; };
                while (end < layers && held_last(end + 1) - held_first <= max_blocks)
                    ++end;
                if (held_last(end) - held_first > max_blocks)
                    throw collisionError("mesh.memory", "Not enough GPU memory for the mesh; free memory or use a larger voxel size",
                                         ErrorCode::ResourceExhausted, std::format("{} blocks in 3 layers, room for {}", held_last(end) - held_first, max_blocks));
                chunks.push_back({starts[layer], starts[end], held_first, held_last(end)});
                layer = end;
            }
            return chunks;
        }

        void allocate_tsdf(Tsdf& tsdf, DeviceArray<uint64_t> keys, const size_t count, cudaStream_t stream) {
            const size_t voxels_per_block = size_t(tsdf.grid.block) * tsdf.grid.block * tsdf.grid.block;
            const size_t voxels = core::checked_product(count, voxels_per_block, "TSDF voxels");
            device_buffer_preflight(core::checked_product(count, tsdf_block_bytes(tsdf.grid), "TSDF bytes"), "Mesh TSDF");
            tsdf.keys = std::move(keys);
            tsdf.block_count = static_cast<uint32_t>(count);
            tsdf.sum = DeviceArray<float>(voxels, stream, true);
            tsdf.weight = DeviceArray<float>(voxels, stream, true);
            tsdf.views = DeviceArray<uint16_t>(voxels, stream, true);
            tsdf.carve = DeviceArray<float>(count, stream, true);
            tsdf.near_blocks = DeviceArray<uint32_t>(count, stream);
            tsdf.near_count = DeviceArray<uint32_t>(1, stream);
        }

        void integrate_view(Tsdf& tsdf, const gsplat_lfs::TsdfView& view, const Tensor& depth, cudaStream_t stream) {
            const size_t tiles = size_t(gsplat_lfs::tsdf_tile_count(view.width)) * gsplat_lfs::tsdf_tile_count(view.height);
            if (tiles != tsdf.tiles) {
                tsdf.tiles = tiles;
                for (auto* bounds : {&tsdf.raw_low, &tsdf.raw_high, &tsdf.low, &tsdf.high})
                    *bounds = DeviceArray<float>(tiles, stream);
            }
            gsplat_lfs::tsdf_tile_bounds(depth.ptr<float>(), view.width, view.height, tsdf.raw_low.data(), tsdf.raw_high.data(),
                                         tsdf.low.data(), tsdf.high.data(), stream);
            const uint32_t near_count = gsplat_lfs::tsdf_classify_blocks(tsdf.keys.data(), tsdf.block_count, view, tsdf.grid,
                                                                         tsdf.low.data(), tsdf.high.data(), tsdf.carve.data(),
                                                                         tsdf.near_blocks.data(), tsdf.near_count.data(), stream);
            gsplat_lfs::tsdf_integrate(tsdf.keys.data(), tsdf.near_blocks.data(), near_count, view, tsdf.grid, depth.ptr<float>(),
                                       tsdf.sum.data(), tsdf.weight.data(), tsdf.views.data(), stream);
        }

        struct SparseVoxels {
            std::vector<uint64_t> keys;
            std::vector<float> tsdf;
        };

        // Appends a chunk's voxels, sorted. Chunks arrive in key order, so all of them stay sorted.
        void collect_voxels(const Tsdf& tsdf, const uint32_t first_block, const uint32_t last_block, const uint32_t min_views, SparseVoxels& all,
                            cudaStream_t stream) {
            DeviceArray<uint64_t> counter(1, stream);
            const auto collect = [&](uint64_t* keys, float* values) {
                return gsplat_lfs::tsdf_collect_voxels(tsdf.keys.data(), tsdf.block_count, first_block, last_block, tsdf.grid, tsdf.sum.data(),
                                                       tsdf.weight.data(), tsdf.views.data(), tsdf.carve.data(),
                                                       min_views, keys, values, counter.data(), stream);
            };
            const uint64_t count = collect(nullptr, nullptr);
            if (count == 0)
                return;
            device_buffer_preflight(count * 12, "Mesh voxels");
            DeviceArray<uint64_t> keys(count, stream);
            DeviceArray<float> values(count, stream);
            (void)collect(keys.data(), values.data());
            std::vector<uint64_t> host_keys(count);
            std::vector<float> host_values(count);
            LFS_CUDA_CHECK(cudaMemcpyAsync(host_keys.data(), keys.data(), count * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
            LFS_CUDA_CHECK(cudaMemcpyAsync(host_values.data(), values.data(), count * sizeof(float), cudaMemcpyDeviceToHost, stream));
            LFS_ENSURE_CUDA_SUCCESS(cudaStreamSynchronize(stream), "Mesh voxel download");
            std::vector<uint32_t> order(count);
            std::iota(order.begin(), order.end(), 0u);
            std::sort(order.begin(), order.end(), [&](const uint32_t a, const uint32_t b) { return host_keys[a] < host_keys[b]; });
            const size_t base = all.keys.size();
            all.keys.resize(base + count);
            all.tsdf.resize(base + count);
            for (size_t i = 0; i < count; ++i) {
                all.keys[base + i] = host_keys[order[i]];
                all.tsdf[base + i] = host_values[order[i]];
            }
        }
    } // namespace

    void surface_nets(const std::span<const uint64_t> keys, const std::span<const float> tsdf, const float voxel, SplatMesh& mesh,
                      const float max_step) {
        // Calls found(a, b, axis) for each voxel pair along an axis where the surface crosses, in key order per axis.
        const auto for_each_crossing = [&](const int axis, auto&& found) {
            const uint64_t step = kAxisStep[axis];
            // key + step is sorted too, so one walk finds every neighbor.
            size_t other = 0;
            for (size_t a = 0; a < keys.size(); ++a) {
                const uint64_t target = keys[a] + step;
                while (other < keys.size() && keys[other] < target)
                    ++other;
                if (other == keys.size())
                    break;
                if (keys[other] != target)
                    continue;
                const float va = tsdf[a], vb = tsdf[other];
                // A jump from free space to the far end of the band is where two observations meet, not a surface.
                // Grazing views steepen real crossings well past one voxel's share of the band, so this stays permissive.
                if ((va < 0) == (vb < 0) || !(std::abs(va - vb) < max_step))
                    continue;
                found(a, va, vb);
            }
        };
        // The cells around a crossing, in the order that faces it outward.
        const auto cell_offsets = [](const int axis, uint64_t (&offsets)[4]) {
            const uint64_t step_j = kAxisStep[(axis + 1) % 3], step_k = kAxisStep[(axis + 2) % 3];
            offsets[0] = 0;
            offsets[1] = step_j;
            offsets[2] = step_j + step_k;
            offsets[3] = step_k;
        };

        // Every cell touched, sorted and unique. A vertex per cell.
        std::vector<uint64_t> cells;
        size_t quad_count = 0;
        for (int axis = 0; axis < 3; ++axis) {
            uint64_t offsets[4];
            cell_offsets(axis, offsets);
            const size_t before = cells.size();
            for_each_crossing(axis, [&](const size_t a, float, float) {
                for (const auto offset : offsets)
                    cells.push_back(keys[a] - offset);
                ++quad_count;
            });
            if (cells.size() != before) {
                std::sort(cells.begin() + std::ptrdiff_t(before), cells.end());
                std::inplace_merge(cells.begin(), cells.begin() + std::ptrdiff_t(before), cells.end());
                cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
            }
        }
        cells.shrink_to_fit();
        if (cells.empty())
            return;

        // Each vertex averages the crossings of its cell.
        std::vector<std::array<double, 3>> sums(cells.size(), {0., 0., 0.});
        std::vector<uint8_t> crossings(cells.size(), 0);
        mesh.faces.clear();
        mesh.faces.reserve(quad_count * 2);
        for (int axis = 0; axis < 3; ++axis) {
            uint64_t offsets[4];
            cell_offsets(axis, offsets);
            // The keys rise along the walk, so each corner's cell only moves forward.
            size_t cursor[4] = {0, 0, 0, 0};
            for_each_crossing(axis, [&](const size_t a, const float va, const float vb) {
                float p[3] = {float(int64_t(keys[a] & kAxisMask) - gsplat_lfs::kTsdfKeyOffset),
                              float(int64_t((keys[a] >> gsplat_lfs::kTsdfKeyBits) & kAxisMask) - gsplat_lfs::kTsdfKeyOffset),
                              float(int64_t((keys[a] >> (2 * gsplat_lfs::kTsdfKeyBits)) & kAxisMask) - gsplat_lfs::kTsdfKeyOffset)};
                p[axis] += va / (va - vb);
                const float x = p[0] * voxel, y = p[1] * voxel, z = p[2] * voxel;
                int32_t corner[4];
                for (int c = 0; c < 4; ++c) {
                    const uint64_t cell = keys[a] - offsets[c];
                    while (cells[cursor[c]] < cell)
                        ++cursor[c];
                    const size_t vertex = cursor[c];
                    corner[c] = static_cast<int32_t>(vertex);
                    sums[vertex][0] += x;
                    sums[vertex][1] += y;
                    sums[vertex][2] += z;
                    ++crossings[vertex];
                }
                if (va >= 0)
                    std::reverse(corner, corner + 4);
                mesh.faces.push_back({corner[0], corner[1], corner[2]});
                mesh.faces.push_back({corner[0], corner[2], corner[3]});
            });
        }
        cells = {};
        mesh.vertices.resize(sums.size());
        for (size_t v = 0; v < sums.size(); ++v) {
            const double n = crossings[v];
            mesh.vertices[v] = {float(sums[v][0] / n), float(sums[v][1] / n), float(sums[v][2] / n)};
        }
        mesh.raw_faces = mesh.faces.size();
    }

    void remove_small_mesh_components(SplatMesh& mesh, const int min_faces) {
        std::vector<uint32_t> parent(mesh.vertices.size());
        std::iota(parent.begin(), parent.end(), 0u);
        const auto find = [&](uint32_t v) {
            while (parent[v] != v) {
                parent[v] = parent[parent[v]];
                v = parent[v];
            }
            return v;
        };
        for (const auto& f : mesh.faces) {
            parent[find(f[1])] = find(f[0]);
            parent[find(f[2])] = find(f[0]);
        }
        std::vector<uint32_t> sizes(mesh.vertices.size(), 0);
        for (const auto& f : mesh.faces)
            ++sizes[find(f[0])];
        mesh.components_total = std::ranges::count_if(sizes, [](const uint32_t s) { return s > 0; });
        mesh.components_kept = std::ranges::count_if(sizes, [&](const uint32_t s) { return s > 0 && int64_t(s) >= min_faces; });
        std::vector<int32_t> remap(mesh.vertices.size(), -1);
        std::vector<std::array<float, 3>> vertices;
        std::vector<std::array<int32_t, 3>> faces;
        faces.reserve(mesh.faces.size());
        for (const auto& f : mesh.faces) {
            if (int64_t(sizes[find(f[0])]) < min_faces)
                continue;
            std::array<int32_t, 3> mapped{};
            for (int i = 0; i < 3; ++i) {
                auto& target = remap[f[i]];
                if (target < 0) {
                    target = static_cast<int32_t>(vertices.size());
                    vertices.push_back(mesh.vertices[f[i]]);
                }
                mapped[i] = target;
            }
            faces.push_back(mapped);
        }
        mesh.vertices = std::move(vertices);
        mesh.faces = std::move(faces);
    }

    size_t fill_small_mesh_holes(SplatMesh& mesh, const float max_perimeter) {
        if (max_perimeter <= 0.f)
            return 0;
        const auto key = [](const int32_t from, const int32_t to) { return (uint64_t(uint32_t(from)) << 32) | uint32_t(to); };
        std::vector<uint64_t> directed;
        directed.reserve(mesh.faces.size() * 3);
        for (const auto& f : mesh.faces)
            for (int i = 0; i < 3; ++i)
                directed.push_back(key(f[i], f[(i + 1) % 3]));
        std::ranges::sort(directed);
        // A rim edge has no triangle on its other side. Sorted by start vertex, so a walk can find what follows.
        std::vector<uint64_t> rim;
        for (const auto edge : directed)
            if (!std::ranges::binary_search(directed, (edge << 32) | (edge >> 32)))
                rim.push_back(edge);
        directed = {};
        const auto rim_start = [&](const int32_t vertex) { return std::ranges::lower_bound(rim, key(vertex, 0)) - rim.begin(); };
        std::vector<char> used(rim.size(), 0);
        const auto edge_length = [&](const int32_t a, const int32_t b) {
            const auto& p = mesh.vertices[a];
            const auto& q = mesh.vertices[b];
            return std::hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
        };
        size_t filled = 0;
        std::vector<int32_t> loop;
        for (size_t first = 0; first < rim.size(); ++first) {
            if (used[first])
                continue;
            used[first] = 1;
            const auto start = static_cast<int32_t>(rim[first] >> 32);
            auto current = static_cast<int32_t>(rim[first] & 0xffffffffu);
            float perimeter = edge_length(start, current);
            loop.assign({start});
            bool closed = false;
            while (perimeter <= max_perimeter) {
                if (current == start) {
                    closed = true;
                    break;
                }
                loop.push_back(current);
                size_t next = rim_start(current);
                while (next < rim.size() && static_cast<int32_t>(rim[next] >> 32) == current && used[next])
                    ++next;
                if (next == rim.size() || static_cast<int32_t>(rim[next] >> 32) != current)
                    break;
                used[next] = 1;
                const auto following = static_cast<int32_t>(rim[next] & 0xffffffffu);
                perimeter += edge_length(current, following);
                current = following;
            }
            if (!closed || loop.size() < 3)
                continue;
            // The rim runs along existing triangles, so the new ones run against it.
            if (loop.size() == 3) {
                mesh.faces.push_back({loop[2], loop[1], loop[0]});
            } else {
                std::array<double, 3> center{0., 0., 0.};
                for (const auto vertex : loop)
                    for (int i = 0; i < 3; ++i)
                        center[i] += mesh.vertices[vertex][i];
                const auto middle = static_cast<int32_t>(mesh.vertices.size());
                const double count = static_cast<double>(loop.size());
                mesh.vertices.push_back({float(center[0] / count), float(center[1] / count), float(center[2] / count)});
                for (size_t i = 0; i < loop.size(); ++i)
                    mesh.faces.push_back({loop[(i + 1) % loop.size()], loop[i], middle});
            }
            ++filled;
        }
        return filled;
    }

    void compute_splat_mesh_normals(SplatMesh& mesh) {
        std::vector<std::array<double, 3>> sum(mesh.vertices.size(), {0., 0., 0.});
        double area = 0.;
        for (const auto& f : mesh.faces) {
            const auto& a = mesh.vertices[f[0]];
            const auto& b = mesh.vertices[f[1]];
            const auto& c = mesh.vertices[f[2]];
            const double u[3] = {double(b[0]) - a[0], double(b[1]) - a[1], double(b[2]) - a[2]};
            const double v[3] = {double(c[0]) - a[0], double(c[1]) - a[1], double(c[2]) - a[2]};
            const double n[3] = {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
            area += 0.5 * std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
            for (const auto vertex : f)
                for (int i = 0; i < 3; ++i)
                    sum[vertex][i] += n[i];
        }
        mesh.area = area;
        mesh.normals.resize(mesh.vertices.size());
        for (size_t i = 0; i < sum.size(); ++i) {
            const auto& n = sum[i];
            const double length = std::max(std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]), 1e-30);
            mesh.normals[i] = {float(n[0] / length), float(n[1] / length), float(n[2] / length)};
        }
    }

    namespace {
        // Orders mesh.faces so that each returned range of up to max_faces is a compact piece of space.
        void split_faces(SplatMesh& mesh, const size_t first, const size_t last, const size_t max_faces, const int threads_left,
                         std::vector<std::pair<size_t, size_t>>& ranges, std::mutex& ranges_mutex) {
            if (last - first <= max_faces) {
                std::lock_guard lock(ranges_mutex);
                ranges.emplace_back(first, last);
                return;
            }
            std::array<float, 3> low{FLT_MAX, FLT_MAX, FLT_MAX}, high{-FLT_MAX, -FLT_MAX, -FLT_MAX};
            for (size_t f = first; f < last; ++f) {
                const auto& p = mesh.vertices[mesh.faces[f][0]];
                for (int i = 0; i < 3; ++i) {
                    low[i] = std::min(low[i], p[i]);
                    high[i] = std::max(high[i], p[i]);
                }
            }
            int axis = 0;
            for (int i = 1; i < 3; ++i)
                if (high[i] - low[i] > high[axis] - low[axis])
                    axis = i;
            const size_t middle = first + (last - first) / 2;
            std::nth_element(mesh.faces.begin() + std::ptrdiff_t(first), mesh.faces.begin() + std::ptrdiff_t(middle), mesh.faces.begin() + std::ptrdiff_t(last),
                             [&](const auto& a, const auto& b) { return mesh.vertices[a[0]][axis] < mesh.vertices[b[0]][axis]; });
            if (threads_left > 1) {
                std::exception_ptr failure;
                std::thread other([&] {
                    try {
                        split_faces(mesh, first, middle, max_faces, threads_left / 2, ranges, ranges_mutex);
                    } catch (...) {
                        failure = std::current_exception();
                    }
                });
                try {
                    split_faces(mesh, middle, last, max_faces, threads_left - threads_left / 2, ranges, ranges_mutex);
                } catch (...) {
                    other.join();
                    throw;
                }
                other.join();
                if (failure)
                    std::rethrow_exception(failure);
            } else {
                split_faces(mesh, first, middle, max_faces, 1, ranges, ranges_mutex);
                split_faces(mesh, middle, last, max_faces, 1, ranges, ranges_mutex);
            }
        }

        // Simplifies each piece on its own, so RAM follows the piece size. Vertices that pieces share stay put,
        // so no cracks open between them, and the seams keep their density. Leaves unused vertices for the caller.
        void simplify_in_pieces(SplatMesh& mesh, const float max_error, const float color_weight, const size_t piece_faces) {
            const int threads = std::clamp(int(std::thread::hardware_concurrency()), 1, kSimplifyThreads);
            std::vector<std::pair<size_t, size_t>> ranges;
            std::mutex ranges_mutex;
            split_faces(mesh, 0, mesh.faces.size(), piece_faces, threads, ranges, ranges_mutex);
            std::ranges::sort(ranges);

            std::vector<int32_t> owner(mesh.vertices.size(), -1);
            std::vector<uint8_t> shared(mesh.vertices.size(), 0);
            for (size_t r = 0; r < ranges.size(); ++r)
                for (size_t f = ranges[r].first; f < ranges[r].second; ++f)
                    for (const auto vertex : mesh.faces[f]) {
                        if (owner[vertex] < 0)
                            owner[vertex] = static_cast<int32_t>(r);
                        else if (owner[vertex] != static_cast<int32_t>(r))
                            shared[vertex] = 1;
                    }
            owner = {};

            const bool has_colors = mesh.colors.size() == mesh.vertices.size();
            std::vector<size_t> kept(ranges.size(), 0);
            std::vector<float> errors(ranges.size(), 0.f);
            std::atomic<size_t> next{0};
            std::vector<std::exception_ptr> failures(static_cast<size_t>(threads));
            const auto work = [&](const int thread) {
                try {
                    for (size_t r = next++; r < ranges.size(); r = next++) {
                        const auto [first, last] = ranges[r];
                        std::vector<int32_t> used;
                        used.reserve((last - first) * 3);
                        for (size_t f = first; f < last; ++f)
                            used.insert(used.end(), mesh.faces[f].begin(), mesh.faces[f].end());
                        std::ranges::sort(used);
                        used.erase(std::unique(used.begin(), used.end()), used.end());
                        std::vector<unsigned int> indices;
                        indices.reserve((last - first) * 3);
                        for (size_t f = first; f < last; ++f)
                            for (const auto vertex : mesh.faces[f])
                                indices.push_back(static_cast<unsigned int>(std::ranges::lower_bound(used, vertex) - used.begin()));
                        std::vector<float> positions(used.size() * 3), attributes(has_colors ? used.size() * 3 : 0);
                        std::vector<unsigned char> lock(used.size());
                        for (size_t v = 0; v < used.size(); ++v) {
                            std::memcpy(&positions[3 * v], mesh.vertices[used[v]].data(), 12);
                            lock[v] = shared[used[v]];
                            if (has_colors)
                                for (int c = 0; c < 3; ++c)
                                    attributes[3 * v + c] = float(mesh.colors[used[v]][c]) / 255.f;
                        }
                        const float weight = color_weight / std::max(meshopt_simplifyScale(positions.data(), used.size(), 12), 1e-12f);
                        const float weights[3] = {weight, weight, weight};
                        const size_t index_count = meshopt_simplifyWithUpdate(
                            indices.data(), indices.size(), positions.data(), used.size(), 12, has_colors ? attributes.data() : nullptr, 12, weights,
                            has_colors ? 3 : 0, lock.data(), 0, max_error, meshopt_SimplifyErrorAbsolute | meshopt_SimplifyPrune, &errors[r]);
                        // A vertex that moved belongs to this piece alone.
                        for (size_t v = 0; v < used.size(); ++v) {
                            if (lock[v])
                                continue;
                            std::memcpy(mesh.vertices[used[v]].data(), &positions[3 * v], 12);
                            if (has_colors)
                                for (int c = 0; c < 3; ++c)
                                    mesh.colors[used[v]][c] = static_cast<uint8_t>(std::clamp(std::lround(255.f * attributes[3 * v + c]), 0l, 255l));
                        }
                        for (size_t i = 0; i < index_count / 3; ++i)
                            for (int c = 0; c < 3; ++c)
                                mesh.faces[first + i][c] = used[indices[3 * i + c]];
                        kept[r] = index_count / 3;
                    }
                } catch (...) {
                    failures[size_t(thread)] = std::current_exception();
                    next = ranges.size();
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

            size_t out = 0;
            for (size_t r = 0; r < ranges.size(); ++r) {
                std::copy_n(mesh.faces.begin() + std::ptrdiff_t(ranges[r].first), kept[r], mesh.faces.begin() + std::ptrdiff_t(out));
                out += kept[r];
            }
            mesh.faces.resize(out);
            mesh.simplify_result_error = errors.empty() ? 0.f : *std::ranges::max_element(errors);
        }
    } // namespace

    void simplify_splat_mesh(SplatMesh& mesh, const float max_error, const float color_weight, const size_t max_whole_faces, const size_t piece_faces) {
        mesh.faces_before_simplify = mesh.faces.size();
        if (max_error <= 0.f || mesh.faces.empty())
            return;
        const bool has_colors = mesh.colors.size() == mesh.vertices.size();
        std::vector<float> attributes;
        if (mesh.faces.size() > max_whole_faces) {
            simplify_in_pieces(mesh, max_error, color_weight, piece_faces);
        } else {
            // The simplifier measures positions in units of the mesh extent.
            const float weight = color_weight / std::max(meshopt_simplifyScale(mesh.vertices.front().data(), mesh.vertices.size(), sizeof(float) * 3), 1e-12f);
            const float weights[3] = {weight, weight, weight};
            // Always carried, so the colors of the vertices that move follow them.
            if (has_colors) {
                attributes.resize(mesh.vertices.size() * 3);
                for (size_t v = 0; v < mesh.vertices.size(); ++v)
                    for (int c = 0; c < 3; ++c)
                        attributes[3 * v + c] = float(mesh.colors[v][c]) / 255.f;
            }
            static_assert(sizeof(std::array<int32_t, 3>) == 3 * sizeof(unsigned int));
            auto* indices = reinterpret_cast<unsigned int*>(mesh.faces.data());
            float* positions = mesh.vertices.front().data();
            const size_t index_count = meshopt_simplifyWithUpdate(
                indices, mesh.faces.size() * 3, positions, mesh.vertices.size(), sizeof(float) * 3,
                has_colors ? attributes.data() : nullptr, sizeof(float) * 3, weights, has_colors ? 3 : 0, nullptr,
                0, max_error, meshopt_SimplifyErrorAbsolute | meshopt_SimplifyPrune, &mesh.simplify_result_error);
            mesh.faces.resize(index_count / 3);
        }

        std::vector<int32_t> remap(mesh.vertices.size(), -1);
        std::vector<std::array<float, 3>> vertices;
        std::vector<std::array<uint8_t, 3>> colors;
        for (auto& face : mesh.faces)
            for (auto& vertex : face) {
                auto& target = remap[vertex];
                if (target < 0) {
                    target = static_cast<int32_t>(vertices.size());
                    vertices.push_back(mesh.vertices[vertex]);
                    if (has_colors) {
                        auto& color = colors.emplace_back();
                        if (attributes.empty()) {
                            color = mesh.colors[vertex];
                        } else {
                            for (int c = 0; c < 3; ++c)
                                color[c] = static_cast<uint8_t>(std::clamp(std::lround(255.f * attributes[3 * size_t(vertex) + c]), 0l, 255l));
                        }
                    }
                }
                vertex = target;
            }
        mesh.vertices = std::move(vertices);
        mesh.colors = std::move(colors);
        compute_splat_mesh_normals(mesh);
    }

    SplatMesh extract_splat_mesh(core::SplatData& model, const std::span<core::Camera* const> cameras,
                                 const core::SplatMeshSettings& settings, const std::stop_token stop, const core::ProgressCallback& progress) try {
        validate(settings);
        const auto stage = [](const std::string_view key, const char* diagnostic) { return std::string(core::collisionStage(key, diagnostic)); };
        std::vector<core::Camera*> views;
        for (size_t i = 0; i < cameras.size(); i += settings.view_stride)
            if (cameras[i])
                views.push_back(cameras[i]);
        if (views.empty())
            throw collisionError("mesh.no_cameras", "The mesh needs the cameras of the dataset; load the project or dataset the model was trained on");
        if (model.size() == 0)
            throw collisionError("collision.geometry_invalid", "The model has no splats", ErrorCode::InvalidArgument);

        const auto stream = core::getCurrentCUDAStream();
        SplatMesh mesh;
        mesh.views = views.size();
        Tsdf tsdf;
        tsdf.grid = {.voxel = settings.voxel_size, .block = static_cast<uint32_t>(settings.block_voxels),
                     .trunc = settings.band(), .full_weight_range = settings.full_weight_range, .max_range = settings.max_range};

        CameraSizes sizes;
        for (auto* camera : views)
            sizes.saved.push_back({camera, camera->image_width(), camera->image_height(), camera->image_size_loaded()});

        // Depth stays in RAM between the touch and the integrate passes, which need the block set first.
        auto step = Clock::now();
        std::vector<std::vector<float>> depths(views.size());
        DeviceArray<uint64_t> block_keys;
        Tensor rays;
        size_t block_count = 0;
        double seconds_touch = 0.;
        for (size_t i = 0; i < views.size(); ++i) {
            core::reportProgress(stop, progress, .5f * float(i) / views.size(), stage("mesh.stage_render", "Rendering depth"));
            auto& camera = *views[i];
            camera.load_image_size(1, settings.max_width);
            const auto view = make_view(camera);
            auto depth = render_median_depth(camera, model, settings, stream);
            const auto touch_start = Clock::now();
            const size_t pixels = size_t(view.width) * view.height;
            if (!rays.is_valid() || rays.numel() != pixels * 3 || i == 0 || views[i - 1]->camera_id() != camera.camera_id()) {
                rays = Tensor::empty({pixels * 3}, Device::CUDA);
                rays.set_stream(stream);
                camera_rays(view, rays.ptr<float>(), stream);
            }
            const size_t samples = size_t((view.width + 1) / 2) * ((view.height + 1) / 2);
            DeviceArray<uint64_t> added(samples * 3, stream);
            gsplat_lfs::tsdf_touch_keys(depth.ptr<float>(), rays.ptr<float>(), view, tsdf.grid, added.data(), stream);
            block_keys = merge_unique_keys(block_keys, block_count, added, block_count, stream);
            seconds_touch += seconds_since(touch_start);
            depths[i].resize(pixels);
            LFS_CUDA_CHECK(cudaMemcpyAsync(depths[i].data(), depth.ptr<float>(), pixels * sizeof(float), cudaMemcpyDeviceToHost, stream));
            LFS_ENSURE_CUDA_SUCCESS(cudaStreamSynchronize(stream), "Mesh depth download");
        }
        rays = {};
        mesh.seconds_render = seconds_since(step) - seconds_touch;
        if (block_count == 0)
            throw collisionError("mesh.empty", "No surface was rendered from these cameras");
        if (block_count > (size_t{1} << 31))
            throw collisionError("mesh.memory", "Too many TSDF blocks; use a larger voxel size", ErrorCode::ResourceExhausted);

        step = Clock::now();
        mesh.blocks = block_count;
        std::vector<uint64_t> host_blocks(block_count);
        LFS_CUDA_CHECK(cudaMemcpyAsync(host_blocks.data(), block_keys.data(), block_count * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
        LFS_ENSURE_CUDA_SUCCESS(cudaStreamSynchronize(stream), "Mesh block download");
        block_keys = {};
        // What the renders left behind is the largest thing in the way of the TSDF.
        core::GlobalArenaManager::instance().get_arena().release_idle_for_pressure();
        core::Tensor::trim_memory_pool();
        const auto room = core::MemoryPressureCoordinator::instance().preflight({.operation = "Mesh TSDF"});
        const size_t available = room.effective_free_bytes > room.required_peak_bytes ? room.effective_free_bytes - room.required_peak_bytes : 0;
        // The rest holds a depth map, the normals, the voxels on their way to RAM and whatever else runs on the GPU.
        const size_t max_blocks = std::min(size_t(double(available) * kTsdfShareOfFreeMemory) / tsdf_block_bytes(tsdf.grid), size_t{1} << 31);
        const auto chunks = plan_chunks(host_blocks, max_blocks);
        LOG_INFO("Mesh: {} blocks, {:.2f} GB TSDF in {} chunks, {:.2f} GB free", block_count,
                 static_cast<double>(block_count * tsdf_block_bytes(tsdf.grid)) / (1 << 30), chunks.size(), static_cast<double>(available) / (1 << 30));
        SparseVoxels voxels;
        const float view_steps = float(chunks.size()) * float(views.size());
        for (size_t c = 0; c < chunks.size(); ++c) {
            const auto& chunk = chunks[c];
            {
                const auto grid = tsdf.grid;
                tsdf = {};
                tsdf.grid = grid;
            }
            core::Tensor::trim_memory_pool();
            const size_t held = chunk.held_last - chunk.held_first;
            DeviceArray<uint64_t> held_keys(held, stream);
            LFS_CUDA_CHECK(cudaMemcpyAsync(held_keys.data(), host_blocks.data() + chunk.held_first, held * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
            allocate_tsdf(tsdf, std::move(held_keys), held, stream);
            for (size_t i = 0; i < views.size(); ++i) {
                core::reportProgress(stop, progress, .5f + .3f * (float(c) * float(views.size()) + float(i)) / view_steps, stage("mesh.stage_fuse", "Fusing depth"));
                const auto view = make_view(*views[i]);
                auto depth = Tensor::empty({size_t(view.height), size_t(view.width)}, Device::CUDA);
                depth.set_stream(stream);
                LFS_CUDA_CHECK(cudaMemcpyAsync(depth.ptr<float>(), depths[i].data(), depths[i].size() * sizeof(float), cudaMemcpyHostToDevice, stream));
                integrate_view(tsdf, view, depth, stream);
                LFS_ENSURE_CUDA_SUCCESS(cudaStreamSynchronize(stream), "Mesh fusion");
            }
            collect_voxels(tsdf, static_cast<uint32_t>(chunk.first - chunk.held_first), static_cast<uint32_t>(chunk.last - chunk.held_first),
                           static_cast<uint32_t>(settings.min_views), voxels, stream);
        }
        host_blocks = {};
        tsdf = {};
        core::Tensor::trim_memory_pool();
        mesh.seconds_fusion = seconds_since(step);

        step = Clock::now();
        core::reportProgress(stop, progress, .8f, stage("mesh.stage_extract", "Extracting surface"));
        mesh.band_voxels = voxels.keys.size();
        surface_nets(voxels.keys, voxels.tsdf, settings.voxel_size, mesh);
        voxels = {};
        remove_small_mesh_components(mesh, settings.min_component_faces);
        if (mesh.faces.empty())
            throw collisionError("mesh.empty", "No surface survived; lower the view count or use a larger voxel size");
        mesh.holes_filled = fill_small_mesh_holes(mesh, settings.fill_hole_perimeter);
        compute_splat_mesh_normals(mesh);
        mesh.seconds_extract = seconds_since(step);

        step = Clock::now();
        const auto vertex_count = static_cast<uint32_t>(mesh.vertices.size());
        DeviceArray<float> vertices_device(size_t(vertex_count) * 3, stream);
        LFS_CUDA_CHECK(cudaMemcpyAsync(vertices_device.data(), mesh.vertices.data(), size_t(vertex_count) * 3 * sizeof(float), cudaMemcpyHostToDevice, stream));
        DeviceArray<float> color_sum(size_t(vertex_count) * 3, stream, true);
        DeviceArray<uint32_t> color_count(vertex_count, stream, true);
        const float tolerance = 2.5f * settings.voxel_size;
        for (size_t i = 0; i < views.size(); i += settings.color_stride) {
            core::reportProgress(stop, progress, .85f + .1f * float(i) / views.size(), stage("mesh.stage_color", "Coloring"));
            auto& camera = *views[i];
            if (!camera.has_image())
                continue;
            const auto view = make_view(camera);
            auto photo = camera.load_and_get_image(1, settings.max_width, false, false);
            const size_t pixels = size_t(view.width) * view.height;
            if (photo.numel() != pixels * 3) {
                LOG_WARN("Mesh: photo size of {} does not match the render; skipping its colors", camera.image_name());
                continue;
            }
            photo = photo.to(Device::CUDA, stream).to(core::DataType::Float32).contiguous();
            auto depth = Tensor::empty({size_t(view.height), size_t(view.width)}, Device::CUDA);
            depth.set_stream(stream);
            LFS_CUDA_CHECK(cudaMemcpyAsync(depth.ptr<float>(), depths[i].data(), depths[i].size() * sizeof(float), cudaMemcpyHostToDevice, stream));
            gsplat_lfs::mesh_color_accumulate(vertices_device.data(), vertex_count, view, depth.ptr<float>(), photo.ptr<float>(), tolerance,
                                              color_sum.data(), color_count.data(), stream);
            LFS_ENSURE_CUDA_SUCCESS(cudaStreamSynchronize(stream), "Mesh color");
        }
        std::vector<float> sums(size_t(vertex_count) * 3);
        std::vector<uint32_t> counts(vertex_count);
        LFS_CUDA_CHECK(cudaMemcpyAsync(sums.data(), color_sum.data(), sums.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
        LFS_CUDA_CHECK(cudaMemcpyAsync(counts.data(), color_count.data(), counts.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
        LFS_ENSURE_CUDA_SUCCESS(cudaStreamSynchronize(stream), "Mesh color download");
        mesh.colors.resize(vertex_count);
        size_t colored = 0;
        for (size_t v = 0; v < vertex_count; ++v) {
            colored += counts[v] > 0;
            for (int c = 0; c < 3; ++c) {
                const float value = counts[v] > 0 ? 255.f * sums[3 * v + c] / float(counts[v]) : 128.f;
                mesh.colors[v][c] = static_cast<uint8_t>(std::clamp(std::lround(value), 0l, 255l));
            }
        }
        mesh.colored_vertex_share = float(colored) / float(vertex_count);
        mesh.seconds_color = seconds_since(step);
        // Reducing faces is the peak of RAM use; it needs none of these, and recomputes the normals.
        depths = {};
        sums = {};
        counts = {};
        if (settings.simplify_error > 0.f)
            mesh.normals = {};
        vertices_device = {};
        color_sum = {};
        color_count = {};
        core::Tensor::trim_memory_pool();
        core::reportProgress(stop, progress, .95f, stage("mesh.stage_simplify", "Reducing faces"));
        step = Clock::now();
        simplify_splat_mesh(mesh, settings.simplify_error, kSimplifyColorWeightPerVoxel * settings.voxel_size);
        mesh.seconds_simplify = seconds_since(step);
        core::reportProgress(stop, progress, 1.f, stage("mesh.stage_simplify", "Reducing faces"));
        return mesh;
    } catch (const std::bad_alloc&) {
        throw collisionError("mesh.ram", "Not enough RAM for the mesh; use a larger voxel size or a maximum range", ErrorCode::ResourceExhausted);
    }

    void write_splat_mesh_ply(const SplatMesh& mesh, const std::filesystem::path& path) {
        const auto temporary = std::filesystem::path(path).concat(".tmp");
        {
            std::ofstream out(core::native_fs::native_path(temporary), std::ios::binary | std::ios::trunc);
            out << std::format("ply\nformat binary_little_endian 1.0\nelement vertex {}\n"
                               "property float x\nproperty float y\nproperty float z\n"
                               "property float nx\nproperty float ny\nproperty float nz\n"
                               "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                               "element face {}\nproperty list uchar int vertex_indices\nend_header\n",
                               mesh.vertices.size(), mesh.faces.size());
            static_assert(std::endian::native == std::endian::little);
            std::vector<char> buffer;
            buffer.reserve(size_t{1} << 24);
            const auto flush = [&] {
                out.write(buffer.data(), std::streamsize(buffer.size()));
                buffer.clear();
            };
            const auto append = [&](const void* data, const size_t bytes) {
                const auto* begin = static_cast<const char*>(data);
                buffer.insert(buffer.end(), begin, begin + bytes);
                if (buffer.size() >= (size_t{1} << 24) - 64)
                    flush();
            };
            for (size_t v = 0; v < mesh.vertices.size(); ++v) {
                append(mesh.vertices[v].data(), 12);
                append(mesh.normals[v].data(), 12);
                append(mesh.colors[v].data(), 3);
            }
            const uint8_t corners = 3;
            for (const auto& f : mesh.faces) {
                append(&corners, 1);
                append(f.data(), 12);
            }
            flush();
            out.close();
            if (!out)
                throw collisionError("mesh.write_failed", "Could not write the mesh; check disk space and permissions", ErrorCode::Unavailable,
                                     core::path_to_utf8(path));
        }
        std::error_code error;
        core::native_fs::rename(temporary, path, error);
        if (error)
            throw collisionError("mesh.write_failed", error.message(), ErrorCode::Unavailable, core::path_to_utf8(path));
    }

    core::MeshData splat_mesh_to_mesh_data(const SplatMesh& mesh) {
        const size_t v = mesh.vertices.size(), f = mesh.faces.size();
        auto vertices = Tensor::empty({v, size_t{3}}, Device::CPU);
        auto normals = Tensor::empty({v, size_t{3}}, Device::CPU);
        auto colors = Tensor::empty({v, size_t{4}}, Device::CPU);
        auto indices = Tensor::empty({f, size_t{3}}, Device::CPU, core::DataType::Int32);
        std::memcpy(vertices.ptr<float>(), mesh.vertices.data(), v * 12);
        std::memcpy(normals.ptr<float>(), mesh.normals.data(), v * 12);
        std::memcpy(indices.ptr<int32_t>(), mesh.faces.data(), f * 12);
        auto* out = colors.ptr<float>();
        const auto linear = [](const uint8_t value) {
            const float s = float(value) / 255.f;
            return s <= .04045f ? s / 12.92f : std::pow((s + .055f) / 1.055f, 2.4f);
        };
        for (size_t i = 0; i < v; ++i) {
            for (int c = 0; c < 3; ++c)
                out[4 * i + c] = linear(mesh.colors[i][c]);
            out[4 * i + 3] = 1.f;
        }
        core::MeshData data(std::move(vertices), std::move(indices));
        data.normals = std::move(normals);
        data.colors = std::move(colors);
        return data;
    }

} // namespace lfs::training
