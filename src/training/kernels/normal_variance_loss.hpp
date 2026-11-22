/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>

namespace lfs::training::kernels {

    // Below this alpha the expected normal mean is unreliable, matching depth distortion.
    constexpr float kNormalVarianceMinAlpha = 0.5f;
    // Skip degenerate pixels where opposing normals cancel nearly to zero.
    constexpr float kNormalVarianceMinNorm = 1e-4f;

    // Penalizes per-splat normal dispersion along each ray without extra render channels.
    // With N = sum(w_i n_i) (|n_i| = 1) and alpha = sum(w_i) composited per pixel:
    //   var_n = 1 - |N|^2 / alpha^2
    //   L = weight * sum(pixel_weight * var_n) / (width * height)
    // Both grad_normal and grad_alpha are accumulated into (+=).
    void launch_normal_variance_loss(
        const float* rendered_normal, // [3*H*W] CHW accumulated normals
        const float* rendered_alpha,  // [H*W]
        float* grad_normal,           // [3*H*W] CHW, accumulated into
        float* grad_alpha,            // [H*W], accumulated into
        int width,
        int height,
        float weight,
        cudaStream_t stream = nullptr,
        const float* pixel_weight = nullptr);

} // namespace lfs::training::kernels
