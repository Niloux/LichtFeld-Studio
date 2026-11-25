/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>

namespace lfs::training::kernels {

    // Below this alpha the expected depth means little, as in the normal consistency loss.
    constexpr float kDepthDistortionMinAlpha = 0.5f;
    // A wider spread is two surfaces at a depth edge, which no splat placement can make thin.
    constexpr float kDepthDistortionMaxStdDev = 0.15f;
    // Keeps the gradient finite on a pixel that is already one thin layer.
    constexpr float kDepthDistortionEpsilon = 0.001f;

    // Penalizes a stack of splats whose mean depth is right but which spreads along the ray.
    // With s1 = sum(w d), s2 = sum(w d^2) and alpha = sum(w) composited per pixel:
    //   var = s2 / alpha - (s1 / alpha)^2
    //   L = weight * sum(pixel_weight * sqrt(var + eps^2)) / (width * height)
    // grad_depth_accum and grad_alpha are accumulated into, as they also carry the other depth
    // terms; grad_depth_sq is written.
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
        const float* pixel_weight = nullptr);

} // namespace lfs::training::kernels
