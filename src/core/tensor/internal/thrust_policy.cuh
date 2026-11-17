/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>
#include <thrust/execution_policy.h>

namespace lfs::core::tensor_ops {

    // ============================================================================
    // THRUST POLICY HELPER - Stream support for Thrust operations
    // ============================================================================

    template <typename Func>
    inline void run_with_thrust_policy(cudaStream_t stream, Func&& func) {
        // thrust::cuda::par is synchronous regardless of .on(stream): the calling
        // thread waits for the stream before the algorithm returns. .on(stream)
        // only selects which stream to enqueue on; it does not make the call async.
        // par_nosync skips that blocking wait. Callers get stream-order guarantees
        // only — later work on the same stream (or an explicit sync) must consume
        // the output. Do not use this helper for reduce/scan/sort or any algorithm
        // that returns a host-side result.
        func(thrust::cuda::par_nosync.on(stream));
    }

} // namespace lfs::core::tensor_ops
