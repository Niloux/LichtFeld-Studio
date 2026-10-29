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
    // Rasterizers cull a tile's list per square sub-tile of this side.
    inline constexpr uint32_t kCullSubTile = 8;
    // Whether a pair key can hold one bit per sub-tile of a tile in its low 32 bits.
    constexpr bool tile_keys_fit_sub_masks(const uint32_t tile_size) {
        return tile_size % kCullSubTile == 0 &&
               (tile_size / kCullSubTile) * (tile_size / kCullSubTile) <= 32;
    }

    // The forward's screen-space footprint of each projected element, which lets
    // a kernel skip elements whose ellipse misses its pixels.
    struct ScreenFootprint {
        const float* means2d = nullptr; // [C, N, 2]
        const float* conics = nullptr;  // [C, N, 3]
        const int32_t* radii = nullptr; // [C, N, 2]
        // Sorted pair keys whose low 32 bits are the sub-tile masks of the pairs,
        // row-major within the tile; null when the keys still carry depths.
        const int64_t* tile_keys = nullptr;
        // The same keys, writable: the micro-tile forward clears the sub-tile bit of each
        // pair that blends no pixel of that sub-tile, so the backward never walks it.
        int64_t* prune_tile_keys = nullptr;
        [[nodiscard]] bool valid() const { return means2d && conics && radii; }
    };
    struct TileBatch {
        TileRange tiles;
        int64_t count;
    };
} // namespace gsplat_lfs
