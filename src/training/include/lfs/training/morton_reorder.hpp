/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

/**
 * Periodic 3D Morton reordering of the live Gaussian training set.
 *
 * Densification appends rows in arbitrary order; reordering by Morton code of
 * the current AABB restores spatial locality for depth-sorted raster gathers.
 * q16 SH codes and joint Adam moments use per-256-splat block bounds, so a
 * raw row gather is invalid — this path gather-decodes per dest 256-block,
 * re-encodes with fresh dest bounds, and copies back into the live region.
 */

#include "core/splat_data.hpp"
#include "core/tensor.hpp"

#include <algorithm>
#include <cstddef>
#include <cuda_runtime.h>
#include <functional>

namespace lfs::training {
    class AdamOptimizer;
}

namespace lfs::training::morton {

    [[nodiscard]] inline bool should_reorder(
        const int iter,
        const std::size_t interval,
        const std::size_t stop_refine) noexcept {
        if (interval == 0 || iter <= 0) {
            return false;
        }
        if (static_cast<std::size_t>(iter) > stop_refine) {
            return false;
        }
        return static_cast<std::size_t>(iter) % interval == 0;
    }

    inline constexpr std::size_t kGrowthMortonMinSplats = 65'536;

    /// Triggers an extra Morton reorder right after a refining step when
    /// densification has substantially grown the model (or first reached max_cap)
    /// since the last reorder. Setting `interval == 0` disables it along with
    /// periodic reorders.
    [[nodiscard]] inline bool should_reorder_after_refine(
        const int iter,
        const std::size_t interval,
        const std::size_t stop_refine,
        const std::size_t current_size,
        const std::size_t last_morton_size,
        const std::size_t max_cap) noexcept {
        if (interval == 0 || iter <= 0 || static_cast<std::size_t>(iter) > stop_refine) {
            return false;
        }
        if (current_size < kGrowthMortonMinSplats) {
            return false;
        }
        if (last_morton_size == 0) {
            return true;
        }
        if (max_cap > 0 && current_size >= max_cap && last_morton_size < max_cap) {
            return true;
        }
        const std::size_t threshold =
            last_morton_size + std::max(kGrowthMortonMinSplats, last_morton_size / 4);
        return current_size >= threshold;
    }

    struct ReorderResult {
        bool applied = false;
        lfs::core::Tensor permutation; // Int64 [N], dest i <- src perm[i]
    };

    /// Hands each extra row-indexed tensor to the visitor; a reorder calls it
    /// once to size its scratch and once to permute the rows with the model.
    using RowVisitor = std::function<void(lfs::core::Tensor&)>;
    using RowSource = std::function<void(const RowVisitor&)>;

    /// dest[i] <- src[perm[i]] for shN. q16 gather-decodes each dest 256-block
    /// in one launch (no full fp32 expansion). fp32 / IEEE-f16 keep the
    /// swizzled gather path.
    void permute_shN(
        core::SplatData& splat,
        const lfs::core::Tensor& perm,
        cudaStream_t stream = nullptr);

    /// Apply Morton ordering to all per-Gaussian model parameters, optimizer
    /// moments, row-indexed aux tensors on `splat` and `extra_rows`. Skips
    /// (and logs at debug) when FrozenRange spans are active, and leaves
    /// everything untouched when its buffers do not fit in VRAM.
    [[nodiscard]] ReorderResult apply_morton_reorder(
        lfs::core::SplatData& splat,
        AdamOptimizer* optimizer = nullptr,
        cudaStream_t stream = nullptr,
        const RowSource& extra_rows = {});

} // namespace lfs::training::morton
