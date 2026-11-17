/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Explicit template instantiations for DLL export on Windows.
// When consumer .cpp and .cu files evaluate tensor expressions (e.g. tensor * 2.0f),
// the expression template evaluator calls launch_unary_op_generic / launch_binary_op_generic.
// These are template functions defined in tensor_generic_ops.cuh — kept out of tensor_ops.hpp
// so consumer .cu units link to these single-TU exports instead of re-instantiating every kernel.
// On Linux this also guarantees visibility under -fvisibility=hidden.

#include "core/export.hpp"
#include "internal/tensor_functors.hpp"
#include "internal/tensor_generic_ops.cuh"
#include <cstdint>
#include <cuda_fp16.h>

namespace lfs::core::tensor_ops {

    namespace {
        struct ieee_round_float_op {
            __host__ __device__ float operator()(const float value) const {
                return ops::round_op{}(value);
            }
        };

        struct ieee_maximum_float_op {
            __host__ __device__ float operator()(const float lhs, const float rhs) const {
                return ops::maximum_op{}(lhs, rhs);
            }
        };

        struct ieee_minimum_float_op {
            __host__ __device__ float operator()(const float lhs, const float rhs) const {
                return ops::minimum_op{}(lhs, rhs);
            }
        };
    } // namespace

    LFS_CORE_API void launch_ieee_round_float(const float* input, float* output,
                                              size_t n, cudaStream_t stream) {
        launch_unary_op_generic(input, output, n, ieee_round_float_op{}, stream);
    }

    LFS_CORE_API void launch_ieee_maximum_float(const float* lhs, const float* rhs, float* output,
                                                size_t n, cudaStream_t stream) {
        launch_binary_op_generic(lhs, rhs, output, n, ieee_maximum_float_op{}, stream);
    }

    LFS_CORE_API void launch_ieee_minimum_float(const float* lhs, const float* rhs, float* output,
                                                size_t n, cudaStream_t stream) {
        launch_binary_op_generic(lhs, rhs, output, n, ieee_minimum_float_op{}, stream);
    }

    // Type aliases to avoid comma-in-macro issues with multi-parameter templates
    using sr_add_f = ops::scalar_right_op<ops::add_op, float>;
    using sr_sub_f = ops::scalar_right_op<ops::sub_op, float>;
    using sr_mul_f = ops::scalar_right_op<ops::mul_op, float>;
    using sr_div_f = ops::scalar_right_op<ops::div_op, float>;
    using sr_pow_f = ops::scalar_right_op<ops::pow_op, float>;
    using sr_mod_f = ops::scalar_right_op<ops::mod_op, float>;
    using sr_max_f = ops::scalar_right_op<ops::maximum_op, float>;
    using sr_min_f = ops::scalar_right_op<ops::minimum_op, float>;

    using sr_add_i32 = ops::scalar_right_op<ops::add_op, int32_t>;
    using sr_sub_i32 = ops::scalar_right_op<ops::sub_op, int32_t>;
    using sr_mul_i32 = ops::scalar_right_op<ops::mul_op, int32_t>;
    using sr_div_i32 = ops::scalar_right_op<ops::div_op, int32_t>;
    using sr_pow_i32 = ops::scalar_right_op<ops::pow_op, int32_t>;
    using sr_mod_i32 = ops::scalar_right_op<ops::mod_op, int32_t>;
    using sr_max_i32 = ops::scalar_right_op<ops::maximum_op, int32_t>;
    using sr_min_i32 = ops::scalar_right_op<ops::minimum_op, int32_t>;

    using sl_add_f = ops::scalar_left_op<ops::add_op, float>;
    using sl_sub_f = ops::scalar_left_op<ops::sub_op, float>;
    using sl_mul_f = ops::scalar_left_op<ops::mul_op, float>;
    using sl_div_f = ops::scalar_left_op<ops::div_op, float>;

    using sr_eq_f = ops::scalar_right_op<ops::equal_op, float>;
    using sr_ne_f = ops::scalar_right_op<ops::not_equal_op, float>;
    using sr_lt_f = ops::scalar_right_op<ops::less_op, float>;
    using sr_le_f = ops::scalar_right_op<ops::less_equal_op, float>;
    using sr_gt_f = ops::scalar_right_op<ops::greater_op, float>;
    using sr_ge_f = ops::scalar_right_op<ops::greater_equal_op, float>;

    using sr_eq_i32 = ops::scalar_right_op<ops::equal_op, int32_t>;
    using sr_ne_i32 = ops::scalar_right_op<ops::not_equal_op, int32_t>;
    using sr_lt_i32 = ops::scalar_right_op<ops::less_op, int32_t>;
    using sr_le_i32 = ops::scalar_right_op<ops::less_equal_op, int32_t>;
    using sr_gt_i32 = ops::scalar_right_op<ops::greater_op, int32_t>;
    using sr_ge_i32 = ops::scalar_right_op<ops::greater_equal_op, int32_t>;

    // Composed unary ops found in tests
    using composed_exp_mul = ops::composed_unary_op<ops::exp_op, sr_mul_f>;
    using composed_mul_abs = ops::composed_unary_op<sr_mul_f, ops::abs_op>;
    using composed_mul_relu = ops::composed_unary_op<sr_mul_f, ops::relu_op>;

    // ============================================================================
    // Helper macros for systematic instantiation
    // ============================================================================

#define EXPORT_UNARY_SAME(Op)                                                                              \
    template LFS_CORE_API void launch_unary_op_generic<int, int, Op>(const int*, int*, size_t, Op,         \
                                                                     cudaStream_t);                        \
    template LFS_CORE_API void launch_unary_op_generic<float, float, Op>(const float*, float*, size_t, Op, \
                                                                         cudaStream_t);

#define EXPORT_UNARY_BOOL(Op)                                                                                      \
    template LFS_CORE_API void launch_unary_op_generic<uint8_t, uint8_t, Op>(const uint8_t*, uint8_t*, size_t, Op, \
                                                                             cudaStream_t);                        \
    template LFS_CORE_API void launch_unary_op_generic<int, uint8_t, Op>(const int*, uint8_t*, size_t, Op,         \
                                                                         cudaStream_t);                            \
    template LFS_CORE_API void launch_unary_op_generic<float, uint8_t, Op>(const float*, uint8_t*, size_t, Op,     \
                                                                           cudaStream_t);

#define EXPORT_BINARY_SAME(Op)                                                                                                      \
    template LFS_CORE_API void launch_binary_op_generic<float, float, Op>(const float*, const float*, float*, size_t, Op,           \
                                                                          cudaStream_t);                                            \
    template LFS_CORE_API void launch_binary_op_generic<int, int, Op>(const int*, const int*, int*, size_t, Op,                     \
                                                                      cudaStream_t);                                                \
    template LFS_CORE_API void launch_binary_op_generic<__half, __half, Op>(const __half*, const __half*, __half*, size_t, Op,      \
                                                                            cudaStream_t);                                          \
    template LFS_CORE_API void launch_binary_op_generic<int64_t, int64_t, Op>(const int64_t*, const int64_t*, int64_t*, size_t, Op, \
                                                                              cudaStream_t);                                        \
    template LFS_CORE_API void launch_binary_op_generic<uint8_t, uint8_t, Op>(const uint8_t*, const uint8_t*, uint8_t*, size_t, Op, \
                                                                              cudaStream_t);

#define EXPORT_BINARY_BOOL(Op)                                                                                                        \
    template LFS_CORE_API void launch_binary_op_generic<float, uint8_t, Op>(const float*, const float*, uint8_t*, size_t, Op,         \
                                                                            cudaStream_t);                                            \
    template LFS_CORE_API void launch_binary_op_generic<int, uint8_t, Op>(const int*, const int*, uint8_t*, size_t, Op,               \
                                                                          cudaStream_t);                                              \
    template LFS_CORE_API void launch_binary_op_generic<uint8_t, uint8_t, Op>(const uint8_t*, const uint8_t*, uint8_t*, size_t, Op,   \
                                                                              cudaStream_t);                                          \
    template LFS_CORE_API void launch_binary_op_generic<int64_t, uint8_t, Op>(const int64_t*, const int64_t*, uint8_t*, size_t, Op,   \
                                                                              cudaStream_t);                                          \
    template LFS_CORE_API void launch_binary_op_generic<uint32_t, uint8_t, Op>(const uint32_t*, const uint32_t*, uint8_t*, size_t, Op,\
                                                                               cudaStream_t);                                         \
    template LFS_CORE_API void launch_binary_op_generic<__half, uint8_t, Op>(const __half*, const __half*, uint8_t*, size_t, Op,      \
                                                                             cudaStream_t);

#define EXPORT_SCALAR_FLOAT(Op)                                                                              \
    template LFS_CORE_API void launch_scalar_op_generic<float, float, Op>(const float*, float, float*,       \
                                                                          size_t, Op, cudaStream_t);

    // ============================================================================
    // Unary: scalar_right_op arithmetic (tensor OP scalar)
    // ============================================================================
    EXPORT_UNARY_SAME(sr_add_f)
    EXPORT_UNARY_SAME(sr_sub_f)
    EXPORT_UNARY_SAME(sr_mul_f)
    EXPORT_UNARY_SAME(sr_div_f)
    EXPORT_UNARY_SAME(sr_pow_f)
    EXPORT_UNARY_SAME(sr_mod_f)
    EXPORT_UNARY_SAME(sr_max_f)
    EXPORT_UNARY_SAME(sr_min_f)

    EXPORT_UNARY_SAME(sr_add_i32)
    EXPORT_UNARY_SAME(sr_sub_i32)
    EXPORT_UNARY_SAME(sr_mul_i32)
    EXPORT_UNARY_SAME(sr_div_i32)
    EXPORT_UNARY_SAME(sr_pow_i32)
    EXPORT_UNARY_SAME(sr_mod_i32)
    EXPORT_UNARY_SAME(sr_max_i32)
    EXPORT_UNARY_SAME(sr_min_i32)

    // ============================================================================
    // Unary: Composed ops
    // ============================================================================
    EXPORT_UNARY_SAME(composed_exp_mul)
    EXPORT_UNARY_SAME(composed_mul_abs)
    EXPORT_UNARY_SAME(composed_mul_relu)

    // ============================================================================
    // Unary: scalar_left_op arithmetic (scalar OP tensor)
    // ============================================================================
    EXPORT_UNARY_SAME(sl_add_f)
    EXPORT_UNARY_SAME(sl_sub_f)
    EXPORT_UNARY_SAME(sl_mul_f)
    EXPORT_UNARY_SAME(sl_div_f)

    // ============================================================================
    // Unary: scalar_right_op comparisons (return bool/uint8_t)
    // ============================================================================
    EXPORT_UNARY_BOOL(sr_eq_f)
    EXPORT_UNARY_BOOL(sr_ne_f)
    EXPORT_UNARY_BOOL(sr_lt_f)
    EXPORT_UNARY_BOOL(sr_le_f)
    EXPORT_UNARY_BOOL(sr_gt_f)
    EXPORT_UNARY_BOOL(sr_ge_f)

    EXPORT_UNARY_BOOL(sr_eq_i32)
    EXPORT_UNARY_BOOL(sr_ne_i32)
    EXPORT_UNARY_BOOL(sr_lt_i32)
    EXPORT_UNARY_BOOL(sr_le_i32)
    EXPORT_UNARY_BOOL(sr_gt_i32)
    EXPORT_UNARY_BOOL(sr_ge_i32)

    // ============================================================================
    // Unary: direct unary ops
    // ============================================================================
    EXPORT_UNARY_SAME(ops::neg_op)
    EXPORT_UNARY_SAME(ops::abs_op)
    EXPORT_UNARY_SAME(ops::exp_op)
    EXPORT_UNARY_SAME(ops::log_op)
    EXPORT_UNARY_SAME(ops::sqrt_op)
    EXPORT_UNARY_SAME(ops::rsqrt_op)
    EXPORT_UNARY_SAME(ops::square_op)
    EXPORT_UNARY_SAME(ops::reciprocal_op)
    EXPORT_UNARY_SAME(ops::sigmoid_op)
    EXPORT_UNARY_SAME(ops::relu_op)
    EXPORT_UNARY_SAME(ops::sign_op)
    EXPORT_UNARY_SAME(ops::floor_op)
    EXPORT_UNARY_SAME(ops::ceil_op)
    EXPORT_UNARY_SAME(ops::round_op)
    EXPORT_UNARY_SAME(ops::sin_op)
    EXPORT_UNARY_SAME(ops::cos_op)
    EXPORT_UNARY_SAME(ops::tan_op)
    EXPORT_UNARY_SAME(ops::asin_op)
    EXPORT_UNARY_SAME(ops::acos_op)
    EXPORT_UNARY_SAME(ops::atan_op)
    EXPORT_UNARY_SAME(ops::sinh_op)
    EXPORT_UNARY_SAME(ops::cosh_op)
    EXPORT_UNARY_SAME(ops::tanh_op)
    EXPORT_UNARY_SAME(ops::log1p_op)
    EXPORT_UNARY_SAME(ops::exp2_op)
    EXPORT_UNARY_SAME(ops::log2_op)
    EXPORT_UNARY_SAME(ops::log10_op)
    EXPORT_UNARY_SAME(ops::trunc_op)
    EXPORT_UNARY_SAME(ops::gelu_op)
    EXPORT_UNARY_SAME(ops::swish_op)

    // ============================================================================
    // Unary: bool-returning direct ops
    // ============================================================================
    EXPORT_UNARY_BOOL(ops::isnan_op)
    EXPORT_UNARY_BOOL(ops::isinf_op)
    EXPORT_UNARY_BOOL(ops::isfinite_op)
    EXPORT_UNARY_BOOL(ops::logical_not_op)

    // ============================================================================
    // Binary: element-wise arithmetic
    // ============================================================================
    EXPORT_BINARY_SAME(ops::add_op)
    EXPORT_BINARY_SAME(ops::sub_op)
    EXPORT_BINARY_SAME(ops::mul_op)
    EXPORT_BINARY_SAME(ops::div_op)
    EXPORT_BINARY_SAME(ops::pow_op)
    EXPORT_BINARY_SAME(ops::mod_op)
    EXPORT_BINARY_SAME(ops::maximum_op)
    EXPORT_BINARY_SAME(ops::minimum_op)

    // ============================================================================
    // Binary: element-wise comparisons and logical operations
    // ============================================================================
    EXPORT_BINARY_BOOL(ops::equal_op)
    EXPORT_BINARY_BOOL(ops::not_equal_op)
    EXPORT_BINARY_BOOL(ops::less_op)
    EXPORT_BINARY_BOOL(ops::less_equal_op)
    EXPORT_BINARY_BOOL(ops::greater_op)
    EXPORT_BINARY_BOOL(ops::greater_equal_op)
    EXPORT_BINARY_BOOL(ops::logical_and_op)
    EXPORT_BINARY_BOOL(ops::logical_or_op)
    EXPORT_BINARY_BOOL(ops::logical_xor_op)

    // ============================================================================
    // Scalar: direct scalar binary ops
    // ============================================================================
    EXPORT_SCALAR_FLOAT(ops::add_op)
    EXPORT_SCALAR_FLOAT(ops::sub_op)
    EXPORT_SCALAR_FLOAT(ops::mul_op)
    EXPORT_SCALAR_FLOAT(ops::div_op)

#undef EXPORT_UNARY_SAME
#undef EXPORT_UNARY_BOOL
#undef EXPORT_BINARY_SAME
#undef EXPORT_BINARY_BOOL
#undef EXPORT_SCALAR_FLOAT

} // namespace lfs::core::tensor_ops
