/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <thread>
#include <vector>

namespace lfs::training::detail {

    // Runs body(i) for every i below count on up to thread_limit threads, this one among them. The first failure stops
    // the others from taking more work and is thrown here.
    template <class Body>
    void parallel_for(const size_t count, const int thread_limit, Body&& body) {
        const int threads = std::clamp(int(std::thread::hardware_concurrency()), 1, std::max(thread_limit, 1));
        std::atomic<size_t> next{0};
        std::vector<std::exception_ptr> failures(static_cast<size_t>(threads));
        const auto work = [&](const int thread) {
            try {
                for (size_t i = next++; i < count; i = next++)
                    body(i);
            } catch (...) {
                failures[size_t(thread)] = std::current_exception();
                next = count;
            }
        };
        std::vector<std::thread> pool;
        for (int t = 1; t < threads; ++t)
            pool.emplace_back(work, t);
        work(0);
        for (auto& thread : pool)
            thread.join();
        for (const auto& failure : failures)
            if (failure)
                std::rethrow_exception(failure);
    }

} // namespace lfs::training::detail
