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
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
#include <meshoptimizer.h>
#include <numeric>

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
                            std::isfinite(s.trunc_voxels) && s.trunc_voxels > 0 && std::isfinite(s.full_weight_range) &&
                            s.full_weight_range >= 0 && s.view_stride >= 1 && s.color_stride >= 1 && s.min_views >= 1 &&
                            s.min_views <= 65535 && s.min_component_faces >= 0 && s.max_width >= 16 &&
                            std::isfinite(s.simplify_error) && s.simplify_error >= 0;
            if (!ok)
                throw collisionError("mesh.settings_invalid", "Invalid mesh settings", ErrorCode::InvalidArgument,
                                     std::format("{}; {}", s.voxel_size, s.trunc_voxels));
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

        void allocate_tsdf(Tsdf& tsdf, DeviceArray<uint64_t> keys, const size_t count, cudaStream_t stream) {
            const size_t voxels_per_block = size_t(tsdf.grid.block) * tsdf.grid.block * tsdf.grid.block;
            const size_t voxels = core::checked_product(count, voxels_per_block, "TSDF voxels");
            const size_t bytes = core::checked_product(voxels, size_t{10}, "TSDF bytes") + count * 12;
            device_buffer_preflight(bytes, "Mesh TSDF");
            LOG_INFO("Mesh: {} blocks, {:.2f} GB TSDF", count, static_cast<double>(bytes) / (1 << 30));
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

        [[nodiscard]] SparseVoxels collect_voxels(const Tsdf& tsdf, const uint32_t min_views, cudaStream_t stream) {
            DeviceArray<uint64_t> counter(1, stream);
            const auto collect = [&](uint64_t* keys, float* values) {
                return gsplat_lfs::tsdf_collect_voxels(tsdf.keys.data(), tsdf.block_count, tsdf.grid, tsdf.sum.data(),
                                                       tsdf.weight.data(), tsdf.views.data(), tsdf.carve.data(),
                                                       min_views, keys, values, counter.data(), stream);
            };
            const uint64_t count = collect(nullptr, nullptr);
            SparseVoxels out;
            if (count == 0)
                return out;
            device_buffer_preflight(count * 12, "Mesh voxels");
            DeviceArray<uint64_t> keys(count, stream);
            DeviceArray<float> values(count, stream);
            (void)collect(keys.data(), values.data());
            out.keys.resize(count);
            out.tsdf.resize(count);
            LFS_CUDA_CHECK(cudaMemcpyAsync(out.keys.data(), keys.data(), count * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
            LFS_CUDA_CHECK(cudaMemcpyAsync(out.tsdf.data(), values.data(), count * sizeof(float), cudaMemcpyDeviceToHost, stream));
            LFS_ENSURE_CUDA_SUCCESS(cudaStreamSynchronize(stream), "Mesh voxel download");
            std::vector<uint32_t> order(count);
            std::iota(order.begin(), order.end(), 0u);
            std::sort(order.begin(), order.end(), [&](const uint32_t a, const uint32_t b) { return out.keys[a] < out.keys[b]; });
            SparseVoxels sorted;
            sorted.keys.resize(count);
            sorted.tsdf.resize(count);
            for (size_t i = 0; i < count; ++i) {
                sorted.keys[i] = out.keys[order[i]];
                sorted.tsdf[i] = out.tsdf[order[i]];
            }
            return sorted;
        }
    } // namespace

    void surface_nets(const std::span<const uint64_t> keys, const std::span<const float> tsdf, const float voxel, SplatMesh& mesh,
                      const float max_step) {
        struct Corner {
            uint64_t cell;
            uint32_t crossing;
        };
        struct Crossing {
            float x, y, z;
        };
        std::vector<Crossing> crossings;
        std::vector<Corner> corners;
        std::vector<std::array<uint32_t, 4>> quads;
        for (int axis = 0; axis < 3; ++axis) {
            const uint64_t step = kAxisStep[axis];
            const uint64_t step_j = kAxisStep[(axis + 1) % 3], step_k = kAxisStep[(axis + 2) % 3];
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
                float p[3] = {float(int64_t(keys[a] & kAxisMask) - gsplat_lfs::kTsdfKeyOffset),
                              float(int64_t((keys[a] >> gsplat_lfs::kTsdfKeyBits) & kAxisMask) - gsplat_lfs::kTsdfKeyOffset),
                              float(int64_t((keys[a] >> (2 * gsplat_lfs::kTsdfKeyBits)) & kAxisMask) - gsplat_lfs::kTsdfKeyOffset)};
                p[axis] += va / (va - vb);
                const auto id = static_cast<uint32_t>(crossings.size());
                crossings.push_back({p[0] * voxel, p[1] * voxel, p[2] * voxel});
                uint64_t cells[4] = {keys[a], keys[a] - step_j, keys[a] - step_j - step_k, keys[a] - step_k};
                if (va >= 0)
                    std::reverse(cells, cells + 4);
                for (const auto cell : cells)
                    corners.push_back({cell, id});
            }
        }
        if (crossings.empty())
            return;
        const auto quad_count = corners.size() / 4;
        // Sort by cell; each cell's vertex averages the crossings of its corners.
        std::vector<uint32_t> order(corners.size());
        std::iota(order.begin(), order.end(), 0u);
        std::sort(order.begin(), order.end(), [&](const uint32_t a, const uint32_t b) { return corners[a].cell < corners[b].cell; });
        std::vector<uint32_t> corner_vertex(corners.size());
        mesh.vertices.clear();
        for (size_t i = 0; i < order.size();) {
            const uint64_t cell = corners[order[i]].cell;
            double sum[3] = {0, 0, 0};
            size_t j = i;
            for (; j < order.size() && corners[order[j]].cell == cell; ++j) {
                const auto& c = crossings[corners[order[j]].crossing];
                sum[0] += c.x;
                sum[1] += c.y;
                sum[2] += c.z;
                corner_vertex[order[j]] = static_cast<uint32_t>(mesh.vertices.size());
            }
            const double n = static_cast<double>(j - i);
            mesh.vertices.push_back({float(sum[0] / n), float(sum[1] / n), float(sum[2] / n)});
            i = j;
        }
        mesh.faces.clear();
        mesh.faces.reserve(quad_count * 2);
        for (size_t q = 0; q < quad_count; ++q) {
            const auto a = static_cast<int32_t>(corner_vertex[4 * q]), b = static_cast<int32_t>(corner_vertex[4 * q + 1]);
            const auto c = static_cast<int32_t>(corner_vertex[4 * q + 2]), d = static_cast<int32_t>(corner_vertex[4 * q + 3]);
            mesh.faces.push_back({a, b, c});
            mesh.faces.push_back({a, c, d});
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

    void simplify_splat_mesh(SplatMesh& mesh, const float max_error, const float color_weight) {
        mesh.faces_before_simplify = mesh.faces.size();
        if (max_error <= 0.f || mesh.faces.empty())
            return;
        const bool has_colors = mesh.colors.size() == mesh.vertices.size();
        std::vector<float> attributes;
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
                        for (int c = 0; c < 3; ++c)
                            color[c] = static_cast<uint8_t>(std::clamp(std::lround(255.f * attributes[3 * size_t(vertex) + c]), 0l, 255l));
                    }
                }
                vertex = target;
            }
        mesh.vertices = std::move(vertices);
        mesh.colors = std::move(colors);
        compute_splat_mesh_normals(mesh);
    }

    SplatMesh extract_splat_mesh(core::SplatData& model, const std::span<core::Camera* const> cameras,
                                 const core::SplatMeshSettings& settings, const std::stop_token stop, const core::ProgressCallback& progress) {
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
                     .trunc = settings.voxel_size * settings.trunc_voxels, .full_weight_range = settings.full_weight_range};

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
        allocate_tsdf(tsdf, std::move(block_keys), block_count, stream);
        for (size_t i = 0; i < views.size(); ++i) {
            core::reportProgress(stop, progress, .5f + .3f * float(i) / views.size(), stage("mesh.stage_fuse", "Fusing depth"));
            const auto view = make_view(*views[i]);
            auto depth = Tensor::empty({size_t(view.height), size_t(view.width)}, Device::CUDA);
            depth.set_stream(stream);
            LFS_CUDA_CHECK(cudaMemcpyAsync(depth.ptr<float>(), depths[i].data(), depths[i].size() * sizeof(float), cudaMemcpyHostToDevice, stream));
            integrate_view(tsdf, view, depth, stream);
            LFS_ENSURE_CUDA_SUCCESS(cudaStreamSynchronize(stream), "Mesh fusion");
        }
        mesh.seconds_fusion = seconds_since(step);

        step = Clock::now();
        core::reportProgress(stop, progress, .8f, stage("mesh.stage_extract", "Extracting surface"));
        auto voxels = collect_voxels(tsdf, static_cast<uint32_t>(settings.min_views), stream);
        tsdf = {};
        core::Tensor::trim_memory_pool();
        mesh.band_voxels = voxels.keys.size();
        surface_nets(voxels.keys, voxels.tsdf, settings.voxel_size, mesh);
        voxels = {};
        remove_small_mesh_components(mesh, settings.min_component_faces);
        if (mesh.faces.empty())
            throw collisionError("mesh.empty", "No surface survived; lower the view count or use a larger voxel size");
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
        core::reportProgress(stop, progress, .95f, stage("mesh.stage_simplify", "Reducing faces"));
        step = Clock::now();
        simplify_splat_mesh(mesh, settings.simplify_error, kSimplifyColorWeightPerVoxel * settings.voxel_size);
        mesh.seconds_simplify = seconds_since(step);
        core::reportProgress(stop, progress, 1.f, stage("mesh.stage_simplify", "Reducing faces"));
        return mesh;
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
