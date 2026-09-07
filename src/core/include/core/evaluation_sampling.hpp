// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace lfs::core {
    // Source order is stable (camera UID order). Sampling each physical camera
    // independently avoids aliasing an interleaved stereo sequence with stride 8.
    inline std::vector<size_t> sample_evaluation_views(
        const std::vector<int>& camera_ids, int every) {
        if (every <= 0)
            throw std::invalid_argument("Evaluation sampling stride must be positive");
        std::unordered_map<int, size_t> counts;
        std::vector<size_t> indices;
        for (size_t i = 0; i < camera_ids.size(); ++i) {
            if (counts[camera_ids[i]]++ % static_cast<size_t>(every) == 0)
                indices.push_back(i);
        }
        return indices;
    }
}
