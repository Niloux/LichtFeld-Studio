/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/assert.hpp"
#include "core/cuda_error.hpp"
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>

namespace lfs::training {

    // A frame can reuse one trailing allocation for scratch that dies between phases.
    // A forward prefix explicitly retained by the rasterizer stays live through backward.
    class RasterizerPhaseArena {
    public:
        enum class Phase : unsigned char { Forward = 1,
                                           Loss = 2,
                                           Backward = 3 };

        RasterizerPhaseArena(std::function<char*(size_t)> backing, cudaStream_t stream, bool poison = false)
            : backing_(std::move(backing)), stream_(stream), poison_(poison) {}

        void begin_phase(Phase phase, size_t minimum_bytes) {
            LFS_ASSERT_MSG(phase != Phase::Forward || !started_,
                           "rasterizer forward phase may only be begun once");
            if (started_ && minimum_bytes > std::numeric_limits<size_t>::max() - retained_bytes_)
                throw std::overflow_error("rasterizer phase slice size overflow");
            const size_t requested = align_size(
                started_ ? retained_bytes_ + minimum_bytes : minimum_bytes);
            if (!started_) {
                base_ = requested > 0 ? backing_(requested) : nullptr;
                if (requested > 0 && !base_)
                    throw std::runtime_error("OUT_OF_MEMORY: Failed to allocate rasterizer phase slice");
                capacity_ = requested;
                retained_bytes_ = 0;
                started_ = true;
            } else {
                if (requested > capacity_) {
                    char* extension = backing_(requested - capacity_);
                    if (!extension)
                        throw std::runtime_error("OUT_OF_MEMORY: rasterizer phase slice extension was not contiguous");
                    if (base_ == nullptr) {
                        base_ = extension;
                    } else if (extension != base_ + capacity_) {
                        throw std::runtime_error("OUT_OF_MEMORY: rasterizer phase slice extension was not contiguous");
                    }
                    capacity_ = requested;
                }
                if (poison_ && capacity_ > retained_bytes_) {
                    // 0xff is a quiet NaN for IEEE float/float2/float3/float4
                    // storage and is cheap to enqueue on the frame stream.
                    LFS_CUDA_CHECK_MSG(
                        cudaMemsetAsync(base_ + retained_bytes_, 0xff,
                                        capacity_ - retained_bytes_, stream_),
                        "cudaMemsetAsync(rasterizer phase poison)");
                }
            }
            phase_ = phase;
            cursor_ = retained_bytes_;
        }

        std::function<char*(size_t)> allocator(Phase expected_phase) {
            return [this, expected_phase](size_t size) {
                return allocate(size, expected_phase);
            };
        }

        char* retain_prefix(const void* source, size_t size) {
            LFS_DEBUG_ASSERT_MSG(started_ && phase_ == Phase::Forward,
                                 "rasterizer retained prefix requires the forward phase");
            const size_t aligned = align_allocation_size(size);
            LFS_ASSERT_MSG(aligned <= cursor_, "rasterizer retained prefix exceeds the phase high-water");
            if (size > 0 && source != base_) {
                const auto source_address = reinterpret_cast<std::uintptr_t>(source);
                const auto destination_address = reinterpret_cast<std::uintptr_t>(base_);
                LFS_ASSERT_MSG(source_address >= destination_address &&
                                   source_address - destination_address >= size,
                               "rasterizer retained prefix overlaps its source");
                LFS_CUDA_CHECK_MSG(
                    cudaMemcpyAsync(base_, source, size, cudaMemcpyDeviceToDevice, stream_),
                    "cudaMemcpyAsync(rasterizer retained prefix)");
            }
            retained_bytes_ = aligned;
            cursor_ = aligned;
            return base_;
        }

    private:
        static size_t align_size(size_t size) {
            constexpr size_t alignment = 256;
            if (size > std::numeric_limits<size_t>::max() - alignment + 1)
                throw std::overflow_error("rasterizer phase slice size overflow");
            return (size + alignment - 1) & ~(alignment - 1);
        }

        char* allocate(size_t size, Phase expected_phase) {
            LFS_DEBUG_ASSERT_MSG(started_ && phase_ == expected_phase,
                                 "rasterizer phased allocation obtained outside its active phase");
            if (size == 0)
                return nullptr;
            const size_t aligned = align_allocation_size(size);
            if (cursor_ > capacity_ || aligned > capacity_ - cursor_) {
                if (cursor_ > std::numeric_limits<size_t>::max() - aligned)
                    throw std::overflow_error("rasterizer phase allocation size overflow");
                const size_t required_capacity = align_size(cursor_ + aligned);
                const size_t extension_bytes = required_capacity > capacity_
                                                   ? required_capacity - capacity_
                                                   : 0;
                if (extension_bytes == 0)
                    return nullptr;
                char* extension = backing_(extension_bytes);
                if (!extension)
                    return nullptr;
                if (base_ == nullptr) {
                    base_ = extension;
                } else if (extension != base_ + capacity_) {
                    throw std::runtime_error("OUT_OF_MEMORY: rasterizer phase allocation extension was not contiguous");
                }
                capacity_ = required_capacity;
            }
            char* result = base_ + cursor_;
            cursor_ += aligned;
            return result;
        }

        static size_t align_allocation_size(size_t size) {
            // The phase slice also contains CUB workspaces whose contract is
            // 256-byte alignment.  Every suballocation must therefore keep
            // the cursor on that boundary; aligning only the arena growth
            // and the workspace's relative offset is insufficient when an
            // earlier phase buffer has an odd 128-byte-sized footprint.
            constexpr size_t alignment = 256;
            if (size > std::numeric_limits<size_t>::max() - alignment + 1)
                throw std::overflow_error("rasterizer phase allocation size overflow");
            return (size + alignment - 1) & ~(alignment - 1);
        }

        std::function<char*(size_t)> backing_;
        cudaStream_t stream_ = nullptr;
        char* base_ = nullptr;
        size_t capacity_ = 0;
        size_t cursor_ = 0;
        size_t retained_bytes_ = 0;
        Phase phase_ = Phase::Forward;
        bool started_ = false;
        bool poison_ = false;
    };

} // namespace lfs::training
