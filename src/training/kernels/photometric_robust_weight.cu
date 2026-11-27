/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "photometric_robust_weight.hpp"

#include "core/assert.hpp"
#include "core/cuda_error.hpp"
#include "kernel_stream.hpp"

#include <algorithm>
#include <cstdint>

namespace lfs::training::kernels {
    namespace {

        constexpr int kBlock = 256;
        constexpr int kLowBlock = 8;
        constexpr float kOutlierTau = 0.18f;
        constexpr float kOutlierScale = 0.14f;
        constexpr float kOutlierFloor = 0.20f;
        constexpr float kSidecarOnlineMin = 0.05f;

        template <typename BaseT>
        __device__ __forceinline__ float read_base_weight(const BaseT* base, const int idx) {
            if (!base)
                return 1.0f;
            if constexpr (std::is_same_v<BaseT, float>)
                return base[idx];
            else
                return base[idx] != 0 ? 1.0f : 0.0f;
        }

        template <typename GtT>
        __device__ __forceinline__ float read_gt_channel(const GtT* gt, const int offset) {
            if constexpr (std::is_same_v<GtT, uint8_t>)
                return static_cast<float>(gt[offset]) * (1.0f / 255.0f);
            else
                return gt[offset];
        }

        __device__ __forceinline__ float sample_u8_bilinear(
            const uint8_t* __restrict__ map,
            const int sw,
            const int sh,
            const float u,
            const float v) {
            const float uc = fminf(fmaxf(u, 0.0f), static_cast<float>(sw - 1));
            const float vc = fminf(fmaxf(v, 0.0f), static_cast<float>(sh - 1));
            const int x0 = static_cast<int>(floorf(uc));
            const int y0 = static_cast<int>(floorf(vc));
            const int x1 = min(x0 + 1, sw - 1);
            const int y1 = min(y0 + 1, sh - 1);
            const float fx = uc - static_cast<float>(x0);
            const float fy = vc - static_cast<float>(y0);
            const float v00 = static_cast<float>(map[y0 * sw + x0]);
            const float v01 = static_cast<float>(map[y0 * sw + x1]);
            const float v10 = static_cast<float>(map[y1 * sw + x0]);
            const float v11 = static_cast<float>(map[y1 * sw + x1]);
            const float top = v00 + fx * (v01 - v00);
            const float bot = v10 + fx * (v11 - v10);
            return (top + fy * (bot - top)) * (1.0f / 255.0f);
        }

        __device__ __forceinline__ float sample_f32_bilinear(
            const float* __restrict__ map,
            const int sw,
            const int sh,
            const float u,
            const float v) {
            const float uc = fminf(fmaxf(u, 0.0f), static_cast<float>(sw - 1));
            const float vc = fminf(fmaxf(v, 0.0f), static_cast<float>(sh - 1));
            const int x0 = static_cast<int>(floorf(uc));
            const int y0 = static_cast<int>(floorf(vc));
            const int x1 = min(x0 + 1, sw - 1);
            const int y1 = min(y0 + 1, sh - 1);
            const float fx = uc - static_cast<float>(x0);
            const float fy = vc - static_cast<float>(y0);
            const float v00 = map[y0 * sw + x0];
            const float v01 = map[y0 * sw + x1];
            const float v10 = map[y1 * sw + x0];
            const float v11 = map[y1 * sw + x1];
            const float top = v00 + fx * (v01 - v00);
            const float bot = v10 + fx * (v11 - v10);
            return top + fy * (bot - top);
        }

        template <typename BaseT, typename GtT>
        __global__ void low_res_residual_block_kernel(
            const BaseT* __restrict__ base_weight,
            const uint8_t* __restrict__ sidecar,
            const float* __restrict__ corrected,
            const GtT* __restrict__ gt,
            float* __restrict__ low_res_out,
            const int H,
            const int W,
            const int sh,
            const int sw,
            const int gw) {
            const int bx = static_cast<int>(blockIdx.x);
            const int by = static_cast<int>(blockIdx.y);
            const int x = bx * kLowBlock + static_cast<int>(threadIdx.x);
            const int y = by * kLowBlock + static_cast<int>(threadIdx.y);
            const int tid = static_cast<int>(threadIdx.y) * kLowBlock + static_cast<int>(threadIdx.x);

            float diff = 0.0f;
            int valid = 0;
            if (x < W && y < H) {
                const int idx = y * W + x;
                const float wb = read_base_weight(base_weight, idx);
                float ws = 1.0f;
                if (sidecar && sw > 0 && sh > 0) {
                    const float u = (static_cast<float>(x) + 0.5f) * (static_cast<float>(sw) / static_cast<float>(W)) - 0.5f;
                    const float v = (static_cast<float>(y) + 0.5f) * (static_cast<float>(sh) / static_cast<float>(H)) - 0.5f;
                    ws = sample_u8_bilinear(sidecar, sw, sh, u, v);
                }
                if (wb > 0.0f && ws > kSidecarOnlineMin) {
                    const int n = H * W;
                    const float yc = 0.2126f * corrected[idx] +
                                     0.7152f * corrected[n + idx] +
                                     0.0722f * corrected[2 * n + idx];
                    const float yg = 0.2126f * read_gt_channel(gt, idx) +
                                     0.7152f * read_gt_channel(gt, n + idx) +
                                     0.0722f * read_gt_channel(gt, 2 * n + idx);
                    diff = yg - yc;
                    valid = 1;
                }
            }

            __shared__ float s_diff[kLowBlock * kLowBlock];
            __shared__ int s_cnt[kLowBlock * kLowBlock];
            s_diff[tid] = diff;
            s_cnt[tid] = valid;
            __syncthreads();

            for (int stride = (kLowBlock * kLowBlock) / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    s_diff[tid] += s_diff[tid + stride];
                    s_cnt[tid] += s_cnt[tid + stride];
                }
                __syncthreads();
            }

            if (tid == 0) {
                low_res_out[by * gw + bx] = s_cnt[0] > 0 ? (s_diff[0] / static_cast<float>(s_cnt[0])) : 0.0f;
            }
        }

        __global__ void smooth_low_res_3x3_kernel(
            const float* __restrict__ in,
            float* __restrict__ out,
            const int gw,
            const int gh) {
            const int x = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
            const int y = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.y) + static_cast<int>(threadIdx.y);
            if (x >= gw || y >= gh)
                return;

            constexpr float k1d[3] = {0.25f, 0.5f, 0.25f};
            float sum = 0.0f;
            for (int dy = -1; dy <= 1; ++dy) {
                const int yy = min(max(y + dy, 0), gh - 1);
                const float wy = k1d[dy + 1];
                for (int dx = -1; dx <= 1; ++dx) {
                    const int xx = min(max(x + dx, 0), gw - 1);
                    sum += wy * k1d[dx + 1] * in[yy * gw + xx];
                }
            }
            out[y * gw + x] = sum;
        }

        template <typename BaseT>
        __global__ void compose_robust_weight_kernel(
            const BaseT* __restrict__ base_weight,
            const uint8_t* __restrict__ sidecar,
            const float* __restrict__ low_res_residual,
            float* __restrict__ out,
            float* __restrict__ static_out,
            const int H,
            const int W,
            const int sh,
            const int sw,
            const int gh,
            const int gw) {
            const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
            const int n = H * W;
            if (idx >= n)
                return;

            float w = read_base_weight(base_weight, idx);
            if (w <= 0.0f) {
                if (static_out)
                    static_out[idx] = 0.0f;
                out[idx] = 0.0f;
                return;
            }

            const int y = idx / W;
            const int x = idx % W;

            if (sidecar && sw > 0 && sh > 0) {
                const float u = (static_cast<float>(x) + 0.5f) * (static_cast<float>(sw) / static_cast<float>(W)) - 0.5f;
                const float v = (static_cast<float>(y) + 0.5f) * (static_cast<float>(sh) / static_cast<float>(H)) - 0.5f;
                w *= sample_u8_bilinear(sidecar, sw, sh, u, v);
                if (w <= 0.0f) {
                    if (static_out)
                        static_out[idx] = 0.0f;
                    out[idx] = 0.0f;
                    return;
                }
            }

            if (static_out)
                static_out[idx] = w;

            if (low_res_residual && gw > 0 && gh > 0) {
                const float fx = (static_cast<float>(x) + 0.5f) * (1.0f / static_cast<float>(kLowBlock)) - 0.5f;
                const float fy = (static_cast<float>(y) + 0.5f) * (1.0f / static_cast<float>(kLowBlock)) - 0.5f;
                const float dmu = sample_f32_bilinear(low_res_residual, gw, gh, fx, fy);
                const float admu = fabsf(dmu);
                if (admu > kOutlierTau) {
                    const float u = (admu - kOutlierTau) / kOutlierScale;
                    w *= fmaxf(kOutlierFloor, 1.0f / (1.0f + u * u));
                }
            }

            out[idx] = w;
        }

        template <typename BaseT>
        void launch_compose(
            const BaseT* base_ptr,
            const uint8_t* sidecar_ptr,
            const float* low_res_ptr,
            float* out_ptr,
            float* static_out_ptr,
            const int H,
            const int W,
            const int sh,
            const int sw,
            const int gh,
            const int gw,
            cudaStream_t stream) {
            const int n = H * W;
            const int blocks = (n + kBlock - 1) / kBlock;
            compose_robust_weight_kernel<BaseT><<<blocks, kBlock, 0, stream>>>(
                base_ptr, sidecar_ptr, low_res_ptr, out_ptr, static_out_ptr, H, W, sh, sw, gh, gw);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.photometric_robust_weight.compose");
        }

        template <typename BaseT, typename GtT>
        void launch_low_res_residual(
            const BaseT* base_ptr,
            const uint8_t* sidecar_ptr,
            const float* corrected_ptr,
            const GtT* gt_ptr,
            float* low_a_ptr,
            float* low_b_ptr,
            const int H,
            const int W,
            const int sh,
            const int sw,
            const int gh,
            const int gw,
            cudaStream_t stream) {
            const dim3 block_lr(kLowBlock, kLowBlock);
            const dim3 grid_lr(gw, gh);
            low_res_residual_block_kernel<BaseT, GtT><<<grid_lr, block_lr, 0, stream>>>(
                base_ptr, sidecar_ptr, corrected_ptr, gt_ptr, low_a_ptr, H, W, sh, sw, gw);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.photometric_robust_weight.low_res_block");

            const dim3 block_sm(16, 16);
            const dim3 grid_sm((gw + 15) / 16, (gh + 15) / 16);
            smooth_low_res_3x3_kernel<<<grid_sm, block_sm, 0, stream>>>(low_a_ptr, low_b_ptr, gw, gh);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.photometric_robust_weight.low_res_smooth");
        }

    } // namespace

    void PhotometricRobustWorkspace::ensure_size(const size_t H, const size_t W, const bool need_low_res) {
        if (H == 0 || W == 0)
            return;
        const bool dims_changed = (allocated_h != H || allocated_w != W);
        if (dims_changed || !diffuse_weight.is_valid()) {
            diffuse_weight = lfs::core::Tensor::empty(
                {H, W}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        }
        if (need_low_res) {
            if (dims_changed || !static_weight.is_valid()) {
                static_weight = lfs::core::Tensor::empty(
                    {H, W}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
            }
            const int gh = static_cast<int>((H + kLowBlock - 1) / kLowBlock);
            const int gw = static_cast<int>((W + kLowBlock - 1) / kLowBlock);
            if (allocated_gh != gh || allocated_gw != gw || !low_res_a.is_valid() || !low_res_b.is_valid()) {
                low_res_a = lfs::core::Tensor::empty(
                    {static_cast<size_t>(gh), static_cast<size_t>(gw)},
                    lfs::core::Device::CUDA, lfs::core::DataType::Float32);
                low_res_b = lfs::core::Tensor::empty(
                    {static_cast<size_t>(gh), static_cast<size_t>(gw)},
                    lfs::core::Device::CUDA, lfs::core::DataType::Float32);
                allocated_gh = gh;
                allocated_gw = gw;
            }
        }
        allocated_h = H;
        allocated_w = W;
    }

    PhotometricRobustWeights compute_photometric_robust_weight(
        const lfs::core::Tensor& base_weight,
        const lfs::core::Tensor& sidecar_u8,
        const lfs::core::Tensor& corrected_image,
        const lfs::core::Tensor& gt_image,
        const bool online_suppress,
        PhotometricRobustWorkspace& workspace,
        cudaStream_t stream) {
        using namespace lfs::core;
        stream = resolve_stream(stream);

        LFS_ASSERT_MSG(corrected_image.is_valid() && corrected_image.device() == Device::CUDA &&
                           corrected_image.dtype() == DataType::Float32 &&
                           corrected_image.ndim() == 3 && corrected_image.shape()[0] == 3,
                       "corrected_image must be CUDA Float32 [3, H, W]");
        const size_t H = corrected_image.shape()[1];
        const size_t W = corrected_image.shape()[2];
        LFS_ASSERT_MSG(H > 0 && W > 0, "image dimensions must be positive");

        const bool has_base = base_weight.is_valid() && base_weight.numel() > 0;
        if (has_base) {
            LFS_ASSERT_MSG(base_weight.device() == Device::CUDA && base_weight.ndim() == 2 &&
                               base_weight.shape()[0] == H && base_weight.shape()[1] == W &&
                               base_weight.is_contiguous() &&
                               (base_weight.dtype() == DataType::Float32 ||
                                base_weight.dtype() == DataType::UInt8 ||
                                base_weight.dtype() == DataType::Bool),
                           "base_weight must be contiguous CUDA Float32/UInt8/Bool [H, W]");
        }

        const bool has_sidecar = sidecar_u8.is_valid() && sidecar_u8.numel() > 0;
        const uint8_t* sidecar_ptr = nullptr;
        int sh = 0, sw = 0;
        if (has_sidecar) {
            LFS_ASSERT_MSG(sidecar_u8.ndim() == 2 && sidecar_u8.dtype() == DataType::UInt8 &&
                               sidecar_u8.is_contiguous(),
                           "sidecar_u8 must be contiguous UInt8 [H_w, W_w]");
            sh = static_cast<int>(sidecar_u8.shape()[0]);
            sw = static_cast<int>(sidecar_u8.shape()[1]);
            LFS_ASSERT_MSG(sh > 0 && sw > 0, "sidecar_u8 dimensions must be positive");
            if (sidecar_u8.device() == Device::CUDA) {
                sidecar_ptr = sidecar_u8.ptr<uint8_t>();
            } else {
                if (!workspace.sidecar_cuda.is_valid() || workspace.sidecar_cuda.shape() != sidecar_u8.shape()) {
                    workspace.sidecar_cuda = Tensor::empty(sidecar_u8.shape(), Device::CUDA, DataType::UInt8);
                }
                workspace.sidecar_cuda.set_stream(stream);
                LFS_CUDA_CHECK(cudaMemcpyAsync(
                    workspace.sidecar_cuda.ptr<uint8_t>(),
                    sidecar_u8.ptr<uint8_t>(),
                    sidecar_u8.numel(),
                    cudaMemcpyHostToDevice,
                    stream));
                sidecar_ptr = workspace.sidecar_cuda.ptr<uint8_t>();
            }
        }

        if (online_suppress) {
            LFS_ASSERT_MSG(gt_image.is_valid() && gt_image.device() == Device::CUDA &&
                               gt_image.ndim() == 3 && gt_image.shape()[0] == 3 &&
                               gt_image.shape()[1] == H && gt_image.shape()[2] == W &&
                               gt_image.is_contiguous() &&
                               (gt_image.dtype() == DataType::Float32 || gt_image.dtype() == DataType::UInt8),
                           "gt_image must be contiguous CUDA Float32/UInt8 [3, H, W]");
        }

        const auto corr_contig = corrected_image.contiguous();
        const auto gt_contig = online_suppress ? gt_image.contiguous() : Tensor{};
        workspace.ensure_size(H, W, online_suppress);
        workspace.diffuse_weight.set_stream(stream);

        const int iH = static_cast<int>(H);
        const int iW = static_cast<int>(W);
        int gh = 0, gw = 0;
        const float* low_res_ptr = nullptr;
        if (online_suppress) {
            workspace.low_res_a.set_stream(stream);
            workspace.low_res_b.set_stream(stream);
            gh = workspace.allocated_gh;
            gw = workspace.allocated_gw;
            auto dispatch_gt = [&](const auto* base_ptr) {
                if (gt_contig.dtype() == DataType::UInt8) {
                    launch_low_res_residual(
                        base_ptr, sidecar_ptr, corr_contig.ptr<float>(), gt_contig.ptr<uint8_t>(),
                        workspace.low_res_a.ptr<float>(), workspace.low_res_b.ptr<float>(),
                        iH, iW, sh, sw, gh, gw, stream);
                } else {
                    launch_low_res_residual(
                        base_ptr, sidecar_ptr, corr_contig.ptr<float>(), gt_contig.ptr<float>(),
                        workspace.low_res_a.ptr<float>(), workspace.low_res_b.ptr<float>(),
                        iH, iW, sh, sw, gh, gw, stream);
                }
            };
            if (!has_base)
                dispatch_gt(static_cast<const float*>(nullptr));
            else if (base_weight.dtype() == DataType::Float32)
                dispatch_gt(base_weight.ptr<float>());
            else
                dispatch_gt(base_weight.ptr<uint8_t>());
            low_res_ptr = workspace.low_res_b.ptr<float>();
        }

        float* static_out_ptr = nullptr;
        if (online_suppress) {
            workspace.static_weight.set_stream(stream);
            static_out_ptr = workspace.static_weight.ptr<float>();
        }

        if (!has_base) {
            launch_compose(
                static_cast<const float*>(nullptr), sidecar_ptr, low_res_ptr,
                workspace.diffuse_weight.ptr<float>(), static_out_ptr, iH, iW, sh, sw, gh, gw, stream);
        } else if (base_weight.dtype() == DataType::Float32) {
            launch_compose(
                base_weight.ptr<float>(), sidecar_ptr, low_res_ptr,
                workspace.diffuse_weight.ptr<float>(), static_out_ptr, iH, iW, sh, sw, gh, gw, stream);
        } else {
            launch_compose(
                base_weight.ptr<uint8_t>(), sidecar_ptr, low_res_ptr,
                workspace.diffuse_weight.ptr<float>(), static_out_ptr, iH, iW, sh, sw, gh, gw, stream);
        }

        return {
            workspace.diffuse_weight,
            online_suppress ? workspace.static_weight : workspace.diffuse_weight};
    }

} // namespace lfs::training::kernels
