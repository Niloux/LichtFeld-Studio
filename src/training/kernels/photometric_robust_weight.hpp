/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/tensor.hpp"
#include <cstddef>
#include <cuda_runtime.h>

namespace lfs::training::kernels {

    struct LFS_CUDA_API PhotometricRobustWorkspace {
        lfs::core::Tensor diffuse_weight; // [H, W] Float32 (with optional online outlier guard)
        lfs::core::Tensor static_weight;  // [H, W] Float32 (base * sidecar only)
        lfs::core::Tensor low_res_a;      // [gh, gw] Float32
        lfs::core::Tensor low_res_b;      // [gh, gw] Float32
        lfs::core::Tensor sidecar_cuda;   // [H_w, W_w] UInt8 staging if sidecar lives on CPU
        size_t allocated_h = 0;
        size_t allocated_w = 0;
        int allocated_gh = 0;
        int allocated_gw = 0;

        void ensure_size(size_t H, size_t W, bool need_low_res);
    };

    struct PhotometricRobustWeights {
        lfs::core::Tensor loss_weight;   // [H, W] Float32: base * sidecar * optional symmetric outlier guard
        lfs::core::Tensor static_weight; // [H, W] Float32: base * sidecar only (for densification error map)
    };

    /// Computes [H, W] Float32 photometric confidence maps combining:
    /// 1) optional base mask/ROI weight (empty -> 1.0 everywhere),
    /// 2) optional bilinearly-sampled uint8 sidecar confidence map [H_w, W_w],
    /// 3) optional online low-frequency (8x8 block average + 3x3 Gaussian) symmetric Cauchy
    ///    guard against gross photometric outliers (glare/shadows) without starving densification.
    LFS_CUDA_API PhotometricRobustWeights compute_photometric_robust_weight(
        const lfs::core::Tensor& base_weight,
        const lfs::core::Tensor& sidecar_u8,
        const lfs::core::Tensor& corrected_image,
        const lfs::core::Tensor& gt_image,
        bool online_suppress,
        PhotometricRobustWorkspace& workspace,
        cudaStream_t stream = nullptr);

} // namespace lfs::training::kernels
