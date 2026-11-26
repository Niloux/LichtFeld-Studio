/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "Rasterization.h"
#include "Common.h"
#include "GeometryFeatures.h"
#include "GutProfile.h"
#include "Intersect.h"
#include "Ops.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace gsplat_lfs {

    // The micro-tile forward drops the pairs that blend no pixel of a sub-tile from the
    // keys the backward walks; LFS_GUT_PRUNE=0 keeps every pair.
    static bool use_contributor_pruning() {
        static const bool enabled = [] {
            const char* value = std::getenv("LFS_GUT_PRUNE");
            return value == nullptr || value[0] != '0';
        }();
        return enabled;
    }

    static bool use_micro_forward_enabled() {
        static const bool enabled = [] {
            const char* value = std::getenv("LFS_GUT_MICRO_FWD");
            return value == nullptr || value[0] != '0';
        }();
        return enabled;
    }

    //=========================================================================
    // Forward rasterization dispatcher
    //=========================================================================

    void rasterize_to_pixels_from_world_3dgs_fwd(
        const float* means,
        const float* quats,
        const float* scales,
        const float* colors,
        const float* opacities,
        const float* backgrounds,
        const float* bg_images,
        const bool* masks,
        const float* compensations,
        const float* dilations,
        uint32_t C,
        uint32_t N,
        uint32_t n_isects,
        uint32_t channels,
        uint32_t image_width,
        uint32_t image_height,
        uint32_t tile_size,
        const float* viewmats0,
        const float* viewmats1,
        const float* Ks,
        CameraModelType camera_model,
        const UnscentedTransformParameters& ut_params,
        ShutterType rs_type,
        const float* radial_coeffs,
        const float* tangential_coeffs,
        const float* thin_prism_coeffs,
        const int32_t* tile_offsets,
        const int32_t* flatten_ids,
        float* renders,
        float* alphas,
        int32_t* last_ids,
        cudaStream_t stream, TileRange tiles,
        float* median_depth,
        const ScreenFootprint& footprint) {
        gsplat_lfs::debug_validate_cuda_pointer(means, "means");
        gsplat_lfs::debug_validate_cuda_pointer(quats, "quats");
        gsplat_lfs::debug_validate_cuda_pointer(scales, "scales");
        gsplat_lfs::debug_validate_cuda_pointer(colors, "colors");
        gsplat_lfs::debug_validate_cuda_pointer(opacities, "opacities");
        gsplat_lfs::debug_validate_cuda_pointer(renders, "renders");
        gsplat_lfs::debug_validate_cuda_pointer(alphas, "alphas");
        gsplat_lfs::debug_validate_cuda_pointer(last_ids, "last_ids");

#define __LAUNCH_KERNEL__(CDIM)                                      \
    case CDIM:                                                       \
        launch_rasterize_to_pixels_from_world_3dgs_fwd_kernel<CDIM>( \
            means, quats, scales, colors, opacities,                 \
            backgrounds, bg_images, masks, compensations, dilations, \
            C, N, n_isects,                                          \
            image_width, image_height, tile_size,                    \
            viewmats0, viewmats1, Ks, camera_model,                  \
            ut_params, rs_type,                                      \
            radial_coeffs, tangential_coeffs, thin_prism_coeffs,     \
            tile_offsets, flatten_ids,                               \
            renders, alphas, last_ids, stream, tiles, median_depth,  \
            footprint);                                              \
        break;

        switch (channels) {
            __LAUNCH_KERNEL__(1)
            __LAUNCH_KERNEL__(2)
            __LAUNCH_KERNEL__(3)
            __LAUNCH_KERNEL__(4)
            __LAUNCH_KERNEL__(5)
            __LAUNCH_KERNEL__(8)
            __LAUNCH_KERNEL__(9)
            __LAUNCH_KERNEL__(16)
            __LAUNCH_KERNEL__(17)
            __LAUNCH_KERNEL__(32)
            __LAUNCH_KERNEL__(33)
            __LAUNCH_KERNEL__(64)
            __LAUNCH_KERNEL__(65)
            __LAUNCH_KERNEL__(128)
            __LAUNCH_KERNEL__(129)
            __LAUNCH_KERNEL__(256)
            __LAUNCH_KERNEL__(257)
            __LAUNCH_KERNEL__(512)
            __LAUNCH_KERNEL__(513)
        default:
            fprintf(stderr, "GSPLAT ERROR: Unsupported number of channels: %u\n", channels);
            assert(false && "Unsupported number of channels");
        }
#undef __LAUNCH_KERNEL__
    }

    //=========================================================================
    // Backward rasterization dispatcher
    //=========================================================================

    void rasterize_to_pixels_from_world_3dgs_bwd(
        const float* means,
        const float* quats,
        const float* scales,
        const float* colors,
        const float* opacities,
        const float* backgrounds,
        const float* bg_images,
        const bool* masks,
        const float* compensations,
        const float* dilations,
        uint32_t C,
        uint32_t N,
        uint32_t n_isects,
        uint32_t channels,
        uint32_t image_width,
        uint32_t image_height,
        uint32_t tile_size,
        const float* viewmats0,
        const float* viewmats1,
        const float* Ks,
        CameraModelType camera_model,
        const UnscentedTransformParameters& ut_params,
        ShutterType rs_type,
        const float* radial_coeffs,
        const float* tangential_coeffs,
        const float* thin_prism_coeffs,
        const int32_t* tile_offsets,
        const int32_t* flatten_ids,
        const float* render_alphas,
        const int32_t* last_ids,
        const float* v_render_colors,
        const float* v_render_alphas,
        float* v_means,
        float* v_quats,
        float* v_scales,
        float* v_colors,
        float* v_opacities,
        float* densification_weight,
        float* densification_error,
        const float* densification_error_map,
        const void* edge_weight_map,
        float* edge_score_out,
        cudaStream_t stream, TileRange tiles,
        const ScreenFootprint& footprint,
        const ColorVarianceRender& color_variance,
        float* blend_peak_out,
        const RenderGradPlanes& grad_planes,
        const bool edge_weight_fp16) {
        gsplat_lfs::debug_validate_cuda_pointer(means, "means");
        gsplat_lfs::debug_validate_cuda_pointer(quats, "quats");
        gsplat_lfs::debug_validate_cuda_pointer(scales, "scales");
        gsplat_lfs::debug_validate_cuda_pointer(colors, "colors");
        gsplat_lfs::debug_validate_cuda_pointer(opacities, "opacities");
        gsplat_lfs::debug_validate_cuda_pointer(v_means, "v_means");
        gsplat_lfs::debug_validate_cuda_pointer(v_quats, "v_quats");
        gsplat_lfs::debug_validate_cuda_pointer(v_scales, "v_scales");
        gsplat_lfs::debug_validate_cuda_pointer(v_colors, "v_colors");
        gsplat_lfs::debug_validate_cuda_pointer(v_opacities, "v_opacities");

        if (n_isects == 0) {
            // Skip kernel launch if no intersections
            return;
        }

#define __LAUNCH_KERNEL__(CDIM)                                      \
    case CDIM:                                                       \
        launch_rasterize_to_pixels_from_world_3dgs_bwd_kernel<CDIM>( \
            means, quats, scales, colors, opacities,                 \
            backgrounds, bg_images, masks, compensations, dilations, \
            C, N, n_isects,                                          \
            image_width, image_height, tile_size,                    \
            viewmats0, viewmats1, Ks, camera_model,                  \
            ut_params, rs_type,                                      \
            radial_coeffs, tangential_coeffs, thin_prism_coeffs,     \
            tile_offsets, flatten_ids,                               \
            render_alphas, last_ids,                                 \
            v_render_colors, v_render_alphas,                        \
            v_means, v_quats, v_scales, v_colors, v_opacities,       \
            densification_weight, densification_error, densification_error_map,    \
            edge_weight_map, edge_score_out, stream, tiles, footprint,             \
            color_variance, blend_peak_out, grad_planes, edge_weight_fp16);        \
        break;

        switch (channels) {
            __LAUNCH_KERNEL__(1)
            __LAUNCH_KERNEL__(2)
            __LAUNCH_KERNEL__(3)
            __LAUNCH_KERNEL__(4)
            __LAUNCH_KERNEL__(5)
            __LAUNCH_KERNEL__(8)
            __LAUNCH_KERNEL__(9)
            __LAUNCH_KERNEL__(16)
            __LAUNCH_KERNEL__(17)
            __LAUNCH_KERNEL__(32)
            __LAUNCH_KERNEL__(33)
            __LAUNCH_KERNEL__(64)
            __LAUNCH_KERNEL__(65)
            __LAUNCH_KERNEL__(128)
            __LAUNCH_KERNEL__(129)
            __LAUNCH_KERNEL__(256)
            __LAUNCH_KERNEL__(257)
            __LAUNCH_KERNEL__(512)
            __LAUNCH_KERNEL__(513)
        default:
            fprintf(stderr, "GSPLAT ERROR: Unsupported number of channels: %u\n", channels);
            assert(false && "Unsupported number of channels");
        }
#undef __LAUNCH_KERNEL__
    }

    //=========================================================================
    // High-level fused forward with SH evaluation
    //=========================================================================

    void rasterize_from_world_with_sh_fwd(
        const float* means,
        const float* quats,
        const float* scales,
        const float* opacities,
        const float* sh0,
        const ShRestSource& shN,
        uint32_t sh_degree,
        uint32_t sh_layout_degree,
        const float* backgrounds,
        const float* bg_images,
        const bool* masks,
        uint32_t N,
        uint32_t C,
        uint32_t K,
        uint32_t image_width,
        uint32_t image_height,
        uint32_t tile_size,
        const float* viewmats0,
        const float* viewmats1,
        const float* Ks,
        CameraModelType camera_model,
        float eps2d,
        float near_plane,
        float far_plane,
        float radius_clip,
        float scaling_modifier,
        bool calc_compensations,
        int render_mode,
        const UnscentedTransformParameters& ut_params,
        ShutterType rs_type,
        const float* radial_coeffs,
        const float* tangential_coeffs,
        const float* thin_prism_coeffs,
        RasterizeWithSHResult& result,
        cudaStream_t stream,
        const float* reference_viewmats,
        float* median_depth,
        const uint32_t* tile_sub_masks,
        const float* cached_unit_rays) {
        gsplat_lfs::debug_validate_cuda_pointer(means, "means");
        gsplat_lfs::debug_validate_cuda_pointer(quats, "quats");
        gsplat_lfs::debug_validate_cuda_pointer(scales, "scales");
        gsplat_lfs::debug_validate_cuda_pointer(opacities, "opacities");
        gsplat_lfs::debug_validate_cuda_pointer(sh0, "sh0");

        const uint32_t tile_width = (image_width + tile_size - 1) / tile_size;
        const uint32_t tile_height = (image_height + tile_size - 1) / tile_size;

        // Determine output channels based on render mode
        // render_mode: 0=RGB, 1=D, 2=ED, 3=RGB_D, 4=RGB_ED
        uint32_t channels = 3; // Default RGB
        if (render_mode == 1 || render_mode == 2) {
            channels = 1; // Depth only
        } else if (render_mode == 3 || render_mode == 4) {
            channels = 4; // RGB + Depth
        } else if (render_mode == 5) {
            channels = 8;
        }

        // Use scales directly (scaling_modifier should be applied by caller if needed)
        const float* scaled_scales = scales;
        const float* const reference = reference_viewmats ? reference_viewmats : viewmats0;

        GutSubstageProfiler::instance().begin_fwd(N, stream);

        // Step 1: Projection
        {
            GutProfScope prof_proj(GutProfStage::ProjUT, stream);
            projection_ut_3dgs_fused(
                means, quats, scaled_scales, opacities,
                viewmats0, viewmats1, Ks,
                N, C, image_width, image_height,
                eps2d, near_plane, far_plane, radius_clip,
                calc_compensations, camera_model,
                ut_params, rs_type,
                radial_coeffs, tangential_coeffs, thin_prism_coeffs,
                result.radii, result.means2d, result.depths, result.conics,
                result.compensations, result.dilations, stream);
        }

        // Step 2: Compute viewing directions and evaluate SH. Nothing here needs the tile
        // pairs, so it is queued before the intersection waits on the host for their count.
        {
            GutProfScope prof_sh(GutProfStage::ShGeomFwd, stream);
            if (render_mode == 0 || render_mode >= 3) {
                if (sh_degree > 0) {
                    compute_view_dirs(means, reference, C, N, result.dirs, stream);
                }
                spherical_harmonics_streamed_fwd(
                    sh_degree, sh_layout_degree, sh_degree > 0 ? result.dirs : nullptr,
                    sh0, shN, C, N, result.colors, channels, stream, result.radii);
            }
            if (channels == 8) {
                geometry_features_fwd(means, quats, scales, reference, result.colors,
                                      N, C, channels, camera_model, stream, result.radii);
            }
        }

        const float* render_backgrounds = (render_mode == 1 || render_mode == 2) ? nullptr : backgrounds;
        const float* render_bg_images = (render_mode == 1 || render_mode == 2) ? nullptr : bg_images;

        // A tile batch is intersected again for its backward, so pruning its keys would be lost.
        auto render = [&](const IntersectTileResult& batch, TileRange tiles) {
            GutProfScope prof_rfwd(GutProfStage::RasterFwd, stream);
            const bool use_keys32 = batch.isect_keys32 != nullptr;
            const bool prune = batch.sub_tile_masks && tiles.end == UINT32_MAX && use_contributor_pruning();
            const bool micro_ok = tiles.end == UINT32_MAX && C == 1 &&
                                  tile_size % kCullSubTile == 0 && channels <= 8 &&
                                  use_micro_forward_enabled();
            result.isect_ids = batch.isect_ids;
            result.isect_keys32 = batch.isect_keys32;
            result.flatten_ids = batch.flatten_ids;
            result.n_sort = batch.n_sort;
            result.sub_tile_masks = batch.sub_tile_masks;
            result.tile_order = batch.tile_order;
            const uint32_t raster_n_isects = static_cast<uint32_t>(batch.n_sort);
            rasterize_to_pixels_from_world_3dgs_fwd(
                means, quats, scaled_scales, result.colors, opacities,
                render_backgrounds, render_bg_images, masks, result.compensations, result.dilations,
                C, N, raster_n_isects, channels,
                image_width, image_height, tile_size,
                viewmats0, viewmats1, Ks, camera_model,
                ut_params, rs_type,
                radial_coeffs, tangential_coeffs, thin_prism_coeffs,
                result.tile_offsets, result.flatten_ids,
                result.render_colors, result.render_alphas, result.last_ids,
                stream, tiles, median_depth,
                ScreenFootprint{.means2d = result.means2d, .conics = result.conics, .radii = result.radii,
                                .tile_keys = (!use_keys32 && batch.sub_tile_masks) ? batch.isect_ids : nullptr,
                                .tile_keys32 = (use_keys32 && batch.sub_tile_masks) ? batch.isect_keys32 : nullptr,
                                .prune_tile_keys = (!use_keys32 && prune) ? batch.isect_ids : nullptr,
                                .prune_tile_keys32 = (use_keys32 && prune) ? batch.isect_keys32 : nullptr,
                                .tile_sub_masks = tile_sub_masks,
                                .unit_rays = cached_unit_rays,
                                .tile_order = micro_ok ? batch.tile_order : nullptr});
        };

        // Step 3: Tile intersection. The first call retains the existing speculative warm-cache path.
        auto intersect = [&](TileRange tiles, const IntersectQueuedFn& before_count_wait = {}) {
            return intersect_tile(result.means2d, result.radii, result.depths,
                                  nullptr, nullptr, C, N, tile_size, tile_width, tile_height,
                                  true, result.tiles_per_gauss, stream, result.tile_offsets, tiles,
                                  result.conics, tile_sub_masks, before_count_wait);
        };
        // A warm cache queues its pairs before the host waits for their count; rendering them
        // right away keeps the GPU busy through that wait. A list that turns out short is redone.
        IntersectTileResult speculative{};
        const auto whole = intersect({}, [&](const IntersectTileResult& queued) {
            speculative = queued;
            render(queued, {});
        });
        result.n_isects = whole.n_isects;
        result.batches.clear();
        if (whole.n_isects == 0 || whole.n_sort > 0) {
            const bool rendered = (speculative.isect_ids != nullptr || speculative.isect_keys32 != nullptr) &&
                                  speculative.isect_ids == whole.isect_ids &&
                                  speculative.isect_keys32 == whole.isect_keys32 &&
                                  speculative.n_sort == whole.n_sort;
            if (!rendered) {
                render(whole, {});
            }
            return;
        }
        // Only the exceptional path counts additional ranges. Recursion depth
        // is at most 31; every successful leaf fits the bounded pair cache.
        auto visit = [&](auto&& self, TileRange tiles) -> void {
            const auto batch = intersect(tiles);
            if (batch.n_isects > 0 && batch.n_sort == 0) {
                LFS_ASSERT(tiles.end - tiles.begin > 1);
                const uint32_t mid = tiles.begin + (tiles.end - tiles.begin) / 2;
                self(self, {tiles.begin, mid});
                self(self, {mid, tiles.end});
                return;
            }
            result.batches.push_back({tiles, batch.n_isects});
            render(batch, tiles);
        };
        const uint32_t mid = tile_width * tile_height / 2;
        visit(visit, {0, mid});
        visit(visit, {mid, tile_width * tile_height});
        // All lists have been consumed. Backward replays the saved leaves.
        result.isect_ids = nullptr;
        result.isect_keys32 = nullptr;
        result.flatten_ids = nullptr;
        result.n_sort = 0;
        result.sub_tile_masks = false;
        result.tile_order = nullptr;
    }

    //=========================================================================
    // High-level fused backward with SH evaluation
    //=========================================================================

    void rasterize_from_world_with_sh_bwd(
        const float* means,
        const float* quats,
        const float* scales,
        const float* opacities,
        uint32_t sh_degree,
        uint32_t sh_layout_degree,
        const float* backgrounds,
        const float* bg_images,
        const bool* masks,
        uint32_t N,
        uint32_t C,
        uint32_t image_width,
        uint32_t image_height,
        uint32_t tile_size,
        const float* viewmats0,
        const float* viewmats1,
        const float* Ks,
        CameraModelType camera_model,
        float eps2d,
        float near_plane,
        float far_plane,
        float radius_clip,
        float scaling_modifier,
        bool calc_compensations,
        int render_mode,
        const UnscentedTransformParameters& ut_params,
        ShutterType rs_type,
        const float* radial_coeffs,
        const float* tangential_coeffs,
        const float* thin_prism_coeffs,
        const float* render_alphas,
        const int32_t* last_ids,
        const int32_t* tile_offsets,
        const int32_t* flatten_ids,
        uint32_t n_isects,
        const float* colors,
        const float* dirs,
        const int32_t* radii,
        const float* means2d,
        const float* depths,
        const float* compensations,
        const float* dilations,
        const float* v_render_colors,
        const float* v_render_alphas,
        float* v_means,
        float* v_quats,
        float* v_scales,
        float* v_opacities,
        float* v_sh0,
        float* v_shN,
        float* densification_weight,
        float* densification_error,
        const float* densification_error_map,
        const void* edge_weight_map,
        float* edge_score_out,
        cudaStream_t stream,
        const std::vector<TileBatch>& batches, int32_t* tiles_per_gauss,
        const float* reference_viewmats,
        const float* conics,
        const int64_t* tile_keys,
        const uint32_t* tile_sub_masks,
        const ColorVarianceRender& color_variance,
        float* blend_peak_out,
        float* color_grad_workspace,
        const float* cached_unit_rays,
        const RenderGradPlanes& grad_planes,
        const std::function<void(const float*, const float*, uint32_t, cudaStream_t)>& on_sh_colors_ready,
        const uint32_t* tile_keys32,
        const bool edge_weight_fp16) {
        // Determine output channels
        uint32_t channels = 3;
        if (render_mode == 1 || render_mode == 2) {
            channels = 1;
        } else if (render_mode == 3 || render_mode == 4) {
            channels = 4;
        } else if (render_mode == 5) {
            channels = 8;
        }

        const float* render_backgrounds = (render_mode == 1 || render_mode == 2) ? nullptr : backgrounds;
        const float* render_bg_images = (render_mode == 1 || render_mode == 2) ? nullptr : bg_images;
        const float* const reference = reference_viewmats ? reference_viewmats : viewmats0;

        const size_t color_values = checked_multiply(
            checked_multiply(static_cast<size_t>(C), static_cast<size_t>(N),
                             "gsplat backward color elements"),
            static_cast<size_t>(channels), "gsplat backward color elements");
        const size_t color_bytes = checked_bytes(
            color_values, sizeof(float), "gsplat backward color gradients");
        float* const v_colors = color_grad_workspace
                                    ? color_grad_workspace
                                    : static_cast<float*>(ensure_gsplat_color_grad_workspace(color_bytes, stream));
        LFS_CUDA_CHECK_MSG(
            cudaMemsetAsync(v_colors, 0, color_bytes, stream),
            "gsplat backward color-gradient initialization");

        const uint32_t* tile_keys32_ptr = tile_keys32;
        auto render_backward = [&](TileRange tiles) {
            rasterize_to_pixels_from_world_3dgs_bwd(
                means, quats, scales, colors, opacities,
                render_backgrounds, render_bg_images, masks, compensations, dilations,
                C, N, n_isects, channels,
                image_width, image_height, tile_size,
                viewmats0, viewmats1, Ks, camera_model,
                ut_params, rs_type,
                radial_coeffs, tangential_coeffs, thin_prism_coeffs,
                tile_offsets, flatten_ids,
                render_alphas, last_ids,
                v_render_colors, v_render_alphas,
                v_means, v_quats, v_scales, v_colors, v_opacities,
                densification_weight, densification_error, densification_error_map,
                edge_weight_map, edge_score_out,
                stream, tiles,
                ScreenFootprint{.means2d = means2d, .conics = conics, .radii = radii,
                                .tile_keys = tile_keys,
                                .tile_keys32 = tile_keys32_ptr,
                                .tile_sub_masks = tile_sub_masks,
                                .unit_rays = cached_unit_rays},
                color_variance, blend_peak_out, grad_planes, edge_weight_fp16);
        };
        GutSubstageProfiler::instance().begin_bwd(stream);
        {
            GutProfScope prof_rbwd(GutProfStage::RasterBwd, stream);
            if (batches.empty()) {
                render_backward({});
            } else {
                const uint32_t tile_width = (image_width + tile_size - 1) / tile_size;
                const uint32_t tile_height = (image_height + tile_size - 1) / tile_size;
                for (const auto& saved : batches) {
                    if (saved.count == 0)
                        continue;
                    const auto batch = intersect_tile(means2d, radii, depths, nullptr, nullptr,
                                                      C, N, tile_size, tile_width, tile_height, true,
                                                      tiles_per_gauss, stream, const_cast<int32_t*>(tile_offsets), saved.tiles,
                                                      conics, tile_sub_masks);
                    LFS_ASSERT(batch.n_isects == saved.count && batch.n_sort > 0);
                    flatten_ids = batch.flatten_ids;
                    tile_keys = (!batch.isect_keys32 && batch.sub_tile_masks) ? batch.isect_ids : nullptr;
                    tile_keys32_ptr = batch.sub_tile_masks ? batch.isect_keys32 : nullptr;
                    n_isects = static_cast<uint32_t>(batch.n_sort);
                    render_backward(saved.tiles);
                }
            }
        }

        {
            GutProfScope prof_gsh(GutProfStage::GeomShBwd, stream);
            if (channels == 8) {
                geometry_features_bwd(means, quats, scales, reference, v_colors, v_means, v_quats,
                                      N, C, channels, camera_model, stream);
            }

            // Backward through SH, straight into the optimizer's gradients
            if (render_mode == 0 || render_mode >= 3) {
                spherical_harmonics_streamed_bwd(
                    sh_degree, sh_layout_degree, dirs, v_colors, C, N, channels,
                    v_sh0, v_shN, stream);
                if (on_sh_colors_ready) {
                    on_sh_colors_ready(v_colors, dirs, channels, stream);
                }
            }
        }
        GutSubstageProfiler::instance().end_bwd();

        // Apply scaling modifier to scale gradients
        if (scaling_modifier != 1.0f) {
            // TODO: Scale v_scales by scaling_modifier
        }
    }

} // namespace gsplat_lfs
