/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
namespace lfs::training::kernels {
    void sky_masked_l1(const float* image, const void* target, bool target_u8,
                       const void* mask, bool mask_u8, const float* roi_weight,
                       float* grad_image, float* loss_stats, float* reduce_temp,
                       int pixels, cudaStream_t stream);
    void sky_mask_foreground_gradient(float* gradient, const void* mask, bool mask_u8,
                                      int pixels, cudaStream_t stream);
    void sky_adam(float* sh0, float* gradient, float* first, float* second,
                  int count, std::uint64_t step, float lr, cudaStream_t stream);
}
