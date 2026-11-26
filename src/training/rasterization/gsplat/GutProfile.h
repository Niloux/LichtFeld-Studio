/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <mutex>
#include <vector>

namespace gsplat_lfs {

    inline bool gut_profile_enabled() {
        static const bool enabled = [] {
            const char* v = std::getenv("LFS_GUT_PROFILE");
            return v != nullptr && v[0] != '\0' && v[0] != '0';
        }();
        return enabled;
    }

    enum class GutProfStage : int {
        ProjUT = 0,
        ShGeomFwd,
        IsectDepthSort,
        IsectCountScan,
        IsectFillSort,
        RasterFwd,
        RasterBwd,
        GeomShBwd,
        Count
    };

    class GutSubstageProfiler {
    public:
        static GutSubstageProfiler& instance() {
            static GutSubstageProfiler inst;
            return inst;
        }

        void begin_fwd(const uint32_t N, cudaStream_t stream) {
            if (!gut_profile_enabled())
                return;
            std::lock_guard lock(mu_);
            ensure_init();
            ++fwd_calls_;
            const int slot_idx = static_cast<int>((fwd_calls_ - 1) % kRingSize);
            if (slots_[slot_idx].occupied) {
                harvest_slot(slots_[slot_idx]);
            }
            active_slot_ = slot_idx;
            auto& s = slots_[slot_idx];
            s.occupied = true;
            s.warmup = (fwd_calls_ <= kWarmupCalls);
            s.N = N;
            s.stream = stream;
            s.mask = 0;
            s.n_isects = 0;
        }

        void begin_bwd(cudaStream_t stream) {
            if (!gut_profile_enabled())
                return;
            std::lock_guard lock(mu_);
            if (active_slot_ < 0 || !slots_[active_slot_].occupied)
                return;
            slots_[active_slot_].stream = stream;
        }

        void end_bwd() {
            if (!gut_profile_enabled())
                return;
            std::lock_guard lock(mu_);
            if (fwd_calls_ > kWarmupCalls && (fwd_calls_ % kReportStride == 0)) {
                // Harvest all completed slots except the last few in flight.
                for (auto& s : slots_) {
                    if (s.occupied && s.mask != 0 && cudaEventQuery(s.last_event) == cudaSuccess) {
                        harvest_slot(s);
                    }
                }
                print_summary("periodic");
            }
        }

        void mark_start(GutProfStage stage, cudaStream_t stream) {
            if (!gut_profile_enabled())
                return;
            std::lock_guard lock(mu_);
            if (active_slot_ < 0)
                return;
            auto& s = slots_[active_slot_];
            const int idx = static_cast<int>(stage);
            cudaEventRecord(s.start[idx], stream);
        }

        void mark_stop(GutProfStage stage, cudaStream_t stream) {
            if (!gut_profile_enabled())
                return;
            std::lock_guard lock(mu_);
            if (active_slot_ < 0)
                return;
            auto& s = slots_[active_slot_];
            const int idx = static_cast<int>(stage);
            cudaEventRecord(s.stop[idx], stream);
            s.last_event = s.stop[idx];
            s.mask |= (1u << idx);
        }

        void sample_tiles(const int32_t* tile_offsets, uint32_t n_tiles, int64_t n_isects,
                          cudaStream_t stream) {
            if (!gut_profile_enabled() || tile_offsets == nullptr || n_tiles == 0)
                return;
            std::lock_guard lock(mu_);
            if (active_slot_ >= 0) {
                slots_[active_slot_].n_isects = n_isects;
            }
            if (fwd_calls_ <= kWarmupCalls || (fwd_calls_ % 100 != 0))
                return;
            std::vector<int32_t> h_offsets(static_cast<size_t>(n_tiles) + 1u);
            if (cudaMemcpyAsync(h_offsets.data(), tile_offsets,
                                (static_cast<size_t>(n_tiles) + 1u) * sizeof(int32_t),
                                cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
                return;
            }
            if (cudaStreamSynchronize(stream) != cudaSuccess)
                return;
            std::vector<int32_t> occ;
            occ.reserve(n_tiles);
            int64_t total_pairs = 0;
            for (uint32_t t = 0; t < n_tiles; ++t) {
                const int32_t c = h_offsets[t + 1] - h_offsets[t];
                if (c > 0) {
                    occ.push_back(c);
                    total_pairs += c;
                }
            }
            if (occ.empty())
                return;
            std::sort(occ.begin(), occ.end());
            const size_t k = occ.size();
            const int32_t med = occ[k / 2];
            const int32_t p95 = occ[(k * 95) / 100];
            const int32_t mx = occ.back();
            const double mean = static_cast<double>(total_pairs) / static_cast<double>(k);
            ++tile_samples_;
            sum_occ_tiles_ += static_cast<double>(k);
            sum_total_tiles_ += static_cast<double>(n_tiles);
            sum_pairs_ += static_cast<double>(total_pairs);
            sum_tile_med_ += static_cast<double>(med);
            sum_tile_mean_ += mean;
            sum_tile_p95_ += static_cast<double>(p95);
            sum_tile_max_ += static_cast<double>(mx);
            max_tile_max_ = std::max(max_tile_max_, mx);
        }

    private:
        static constexpr int kStages = static_cast<int>(GutProfStage::Count);
        static constexpr int kRingSize = 32;
        static constexpr uint64_t kWarmupCalls = 300;
        static constexpr uint64_t kReportStride = 300;

        struct Slot {
            bool occupied = false;
            bool warmup = false;
            uint32_t N = 0;
            uint32_t mask = 0;
            int64_t n_isects = 0;
            cudaStream_t stream = nullptr;
            cudaEvent_t last_event = nullptr;
            std::array<cudaEvent_t, kStages> start{};
            std::array<cudaEvent_t, kStages> stop{};
        };

        GutSubstageProfiler() = default;
        ~GutSubstageProfiler() {
            if (!initialized_)
                return;
            for (auto& s : slots_) {
                if (s.occupied) {
                    harvest_slot(s);
                }
                for (int i = 0; i < kStages; ++i) {
                    if (s.start[i])
                        (void)cudaEventDestroy(s.start[i]);
                    if (s.stop[i])
                        (void)cudaEventDestroy(s.stop[i]);
                }
            }
            if (steady_frames_ > 0) {
                print_summary("final");
            }
        }

        void ensure_init() {
            if (initialized_)
                return;
            for (auto& s : slots_) {
                for (int i = 0; i < kStages; ++i) {
                    (void)cudaEventCreate(&s.start[i]);
                    (void)cudaEventCreate(&s.stop[i]);
                }
            }
            initialized_ = true;
        }

        void harvest_slot(Slot& s) {
            if (!s.occupied)
                return;
            if (!s.warmup && s.mask != 0) {
                (void)cudaEventSynchronize(s.last_event);
                ++steady_frames_;
                sum_N_ += static_cast<double>(s.N);
                sum_isects_ += static_cast<double>(s.n_isects);
                for (int i = 0; i < kStages; ++i) {
                    if ((s.mask & (1u << i)) != 0) {
                        float ms = 0.0f;
                        if (cudaEventElapsedTime(&ms, s.start[i], s.stop[i]) == cudaSuccess) {
                            sum_ms_[i] += static_cast<double>(ms);
                            ++count_ms_[i];
                        }
                    }
                }
            }
            s.occupied = false;
            s.mask = 0;
        }

        void print_summary(const char* tag) {
            if (steady_frames_ == 0)
                return;
            static constexpr const char* kNames[kStages] = {
                "proj_ut", "sh_geom_fwd", "isect_depth_sort", "isect_count_scan",
                "isect_fill_sort", "raster_fwd", "raster_bwd", "geom_sh_bwd"};
            const double inv_f = 1.0 / static_cast<double>(steady_frames_);
            std::fprintf(stderr, "\n[LFS_GUT_PROFILE:%s] steady_frames=%llu avg_N=%.0f avg_n_isects=%.0f\n",
                         tag, static_cast<unsigned long long>(steady_frames_),
                         sum_N_ * inv_f, sum_isects_ * inv_f);
            double fwd_sum = 0.0, bwd_sum = 0.0;
            for (int i = 0; i < kStages; ++i) {
                const double avg = count_ms_[i] > 0 ? sum_ms_[i] / static_cast<double>(count_ms_[i]) : 0.0;
                std::fprintf(stderr, "  %-18s: %7.3f ms (n=%llu)\n",
                             kNames[i], avg, static_cast<unsigned long long>(count_ms_[i]));
                if (i <= static_cast<int>(GutProfStage::RasterFwd))
                    fwd_sum += avg;
                else
                    bwd_sum += avg;
            }
            std::fprintf(stderr, "  %-18s: %7.3f ms | %-18s: %7.3f ms\n",
                         "sum_fwd_substages", fwd_sum, "sum_bwd_substages", bwd_sum);
            if (tile_samples_ > 0) {
                const double inv_t = 1.0 / static_cast<double>(tile_samples_);
                std::fprintf(stderr,
                             "  tiles(samples=%llu): occ=%.0f/%.0f pairs=%.0f med=%.1f mean=%.1f p95=%.1f avg_max=%.1f abs_max=%d\n",
                             static_cast<unsigned long long>(tile_samples_),
                             sum_occ_tiles_ * inv_t, sum_total_tiles_ * inv_t,
                             sum_pairs_ * inv_t, sum_tile_med_ * inv_t,
                             sum_tile_mean_ * inv_t, sum_tile_p95_ * inv_t,
                             sum_tile_max_ * inv_t, max_tile_max_);
            }
            std::fflush(stderr);
        }

        std::mutex mu_;
        bool initialized_ = false;
        std::array<Slot, kRingSize> slots_{};
        int active_slot_ = -1;
        uint64_t fwd_calls_ = 0;
        uint64_t steady_frames_ = 0;
        double sum_N_ = 0.0;
        double sum_isects_ = 0.0;
        std::array<double, kStages> sum_ms_{};
        std::array<uint64_t, kStages> count_ms_{};

        uint64_t tile_samples_ = 0;
        double sum_occ_tiles_ = 0.0;
        double sum_total_tiles_ = 0.0;
        double sum_pairs_ = 0.0;
        double sum_tile_med_ = 0.0;
        double sum_tile_mean_ = 0.0;
        double sum_tile_p95_ = 0.0;
        double sum_tile_max_ = 0.0;
        int32_t max_tile_max_ = 0;
    };

    struct GutProfScope {
        GutProfStage stage;
        cudaStream_t stream;
        bool active;
        GutProfScope(GutProfStage s, cudaStream_t st)
            : stage(s), stream(st), active(gut_profile_enabled()) {
            if (active)
                GutSubstageProfiler::instance().mark_start(stage, stream);
        }
        ~GutProfScope() {
            if (active)
                GutSubstageProfiler::instance().mark_stop(stage, stream);
        }
        GutProfScope(const GutProfScope&) = delete;
        GutProfScope& operator=(const GutProfScope&) = delete;
    };

} // namespace gsplat_lfs
