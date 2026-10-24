/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <cstdint>

namespace gsplat_lfs {
    // Linear, half-open range in the original full-frame tile grid.
    struct TileRange {
        uint32_t begin = 0;
        uint32_t end = UINT32_MAX;
    };
    // The forward's screen-space footprint of each projected element, which lets
    // a kernel skip elements whose ellipse misses its pixels.
    struct ScreenFootprint {
        const float* means2d = nullptr; // [C, N, 2]
        const float* conics = nullptr;  // [C, N, 3]
        const int32_t* radii = nullptr; // [C, N, 2]
        [[nodiscard]] bool valid() const { return means2d && conics && radii; }
    };
    struct TileBatch {
        TileRange tiles;
        int64_t count;
    };
} // namespace gsplat_lfs
