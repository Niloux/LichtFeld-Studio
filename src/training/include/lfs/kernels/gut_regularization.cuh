/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>

namespace lfs::training::kernels {

    // The per-splat regularizers that FastGS folds into its fused Adam, for the GUT path, whose backward leaves
    // plain optimizer gradients: the spike-shape (effective rank) penalty on the log scales and the base-colour
    // range penalty on sh0. Same terms, normalization and loss reporting as the fused kernel; the colour gate is
    // not applied (every channel regularized). Gradients accumulate into the optimizer's; losses add to
    // caller-owned device scalars (either may be null). A zero weight skips its term.
    void launch_gut_param_regularization(
        const float* log_scales, // [N, 3]
        float* scales_grad,      // [N, 3]
        const float* sh0,        // [N, sh0_stride], first three are the DC colour coefficients
        float* sh0_grad,         // [N, sh0_stride]
        int sh0_stride,
        int n,
        float erank_weight,
        float dc_weight,
        float* erank_loss_out,
        float* dc_loss_out,
        cudaStream_t stream = nullptr);

} // namespace lfs::training::kernels
