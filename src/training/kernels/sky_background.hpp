/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
namespace lfs::training::kernels {
    void sky_adam(float* sh0, float* gradient, float* first, float* second,
                  int count, std::uint64_t step, float lr, cudaStream_t stream);
}
