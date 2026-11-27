/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>

namespace lfs::training::kernels {

    // Below this alpha the expected depth means little, as in the normal consistency loss.
    constexpr float kDepthDistortionMinAlpha = 0.5f;
    // Relative depth variance softening floor (meters) for near pixels.
    constexpr float kDepthDistortionDepthFloor = 0.80f;
    constexpr float kDepthDistortionMinExpectedDepth = 0.05f;
    // Dimensionless relative std-dev floor and smooth logarithmic saturation scale.
    constexpr float kDepthDistortionRelEpsilon = 0.002f;
    constexpr float kDepthDistortionRelSaturation = 0.12f;
    // Mean-median log consistency Charbonnier epsilon.
    constexpr float kDepthDistortionMeanMedianEpsilon = 0.005f;
    // Skip distortion/mean-median at target-depth discontinuities (>3.5% neighbor jump) or invalid boundary.
    constexpr float kDepthDistortionEdgeRelJump = 0.035f;
    constexpr float kDepthDistortionMinTargetCode = 0.018f;

    // Penalizes dimensionless relative depth variance along each ray with smooth saturation
    // (avoiding hard cutoffs on wide multi-layer separations), optionally combined with
    // log(mean_depth) vs log(median_depth) consistency and target-depth edge masking.
    void launch_depth_distortion_loss(
        const float* rendered_depth_accum,    // [H*W] accumulated depth
        const float* rendered_depth_sq_accum, // [H*W] accumulated squared depth
        const float* rendered_alpha,          // [H*W]
        float* grad_depth_accum,              // [H*W], accumulated into
        float* grad_depth_sq_accum,           // [H*W], written
        float* grad_alpha,                    // [H*W], accumulated into
        int width,
        int height,
        float weight,
        cudaStream_t stream = nullptr,
        const float* pixel_weight = nullptr,
        const float* median_depth = nullptr,
        float* grad_median_depth = nullptr,
        float mean_median_weight = 0.0f,
        const float* target_depth = nullptr,
        float anchor_scale = 0.0f,
        float anchor_shift = 0.0f);

} // namespace lfs::training::kernels
