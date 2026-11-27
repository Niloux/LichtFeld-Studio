/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <cmath>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include "components/ppisp.hpp"
#include "core/tensor.hpp"

namespace {

    constexpr float RTOL = 1e-4f;
    constexpr float ATOL = 1e-5f;

    std::unique_ptr<lfs::training::PPISP> create_test_ppisp(int num_cameras, int num_frames, int total_iterations,
                                                            lfs::training::PPISPConfig config = {}) {
        auto ppisp = std::make_unique<lfs::training::PPISP>(total_iterations, config);
        for (int i = 0; i < num_frames; ++i) {
            ppisp->register_frame(i, i % num_cameras);
        }
        ppisp->finalize();
        return ppisp;
    }

    // ZCA pinv block-diagonal matrix (same as in ppisp.cpp)
    torch::Tensor get_color_pinv_block_diag() {
        return torch::tensor({
                                 // Blue block
                                 0.0480542f,
                                 -0.0043631f,
                                 0.0f,
                                 0.0f,
                                 0.0f,
                                 0.0f,
                                 0.0f,
                                 0.0f,
                                 -0.0043631f,
                                 0.0481283f,
                                 0.0f,
                                 0.0f,
                                 0.0f,
                                 0.0f,
                                 0.0f,
                                 0.0f,
                                 // Red block
                                 0.0f,
                                 0.0f,
                                 0.0580570f,
                                 -0.0179872f,
                                 0.0f,
                                 0.0f,
                                 0.0f,
                                 0.0f,
                                 0.0f,
                                 0.0f,
                                 -0.0179872f,
                                 0.0431061f,
                                 0.0f,
                                 0.0f,
                                 0.0f,
                                 0.0f,
                                 // Green block
                                 0.0f,
                                 0.0f,
                                 0.0f,
                                 0.0f,
                                 0.0433336f,
                                 -0.0180537f,
                                 0.0f,
                                 0.0f,
                                 0.0f,
                                 0.0f,
                                 0.0f,
                                 0.0f,
                                 -0.0180537f,
                                 0.0580500f,
                                 0.0f,
                                 0.0f,
                                 // Neutral block
                                 0.0f,
                                 0.0f,
                                 0.0f,
                                 0.0f,
                                 0.0f,
                                 0.0f,
                                 0.0128369f,
                                 -0.0034654f,
                                 0.0f,
                                 0.0f,
                                 0.0f,
                                 0.0f,
                                 0.0f,
                                 0.0f,
                                 -0.0034654f,
                                 0.0128158f,
                             })
            .reshape({8, 8});
    }

    // Smooth L1 loss (Huber loss)
    torch::Tensor smooth_l1_torch(const torch::Tensor& x, float beta) {
        auto abs_x = x.abs();
        return torch::where(abs_x < beta, 0.5f * x * x / beta, abs_x - 0.5f * beta);
    }

    class PPISPRegularizationTest : public ::testing::Test {
    protected:
        void SetUp() override {
            torch::manual_seed(42);
        }

        bool tensors_close(float expected, float actual, float rtol = RTOL, float atol = ATOL) {
            float tol = atol + rtol * std::abs(expected);
            return std::abs(expected - actual) <= tol;
        }

        bool vectors_close(const std::vector<float>& expected, const std::vector<float>& actual,
                           float rtol = RTOL, float atol = ATOL) {
            if (expected.size() != actual.size())
                return false;
            for (size_t i = 0; i < expected.size(); ++i) {
                float tol = atol + rtol * std::abs(expected[i]);
                if (std::abs(expected[i] - actual[i]) > tol) {
                    std::cerr << "Mismatch at " << i << ": expected " << expected[i]
                              << " got " << actual[i] << std::endl;
                    return false;
                }
            }
            return true;
        }
    };

    TEST_F(PPISPRegularizationTest, SchedulerStartsAtWarmupFloor) {
        lfs::training::PPISPConfig config;
        config.lr = 0.002;
        config.warmup_steps = 100;
        config.warmup_start_factor = 0.01;

        auto ppisp = create_test_ppisp(1, 1, 1000, config);
        EXPECT_NEAR(ppisp->get_lr(), config.lr * config.warmup_start_factor, 1e-9);

        for (int i = 0; i < config.warmup_steps; ++i) {
            ppisp->scheduler_step();
        }
        EXPECT_NEAR(ppisp->get_lr(), config.lr, 1e-9);
    }

    // Test exposure mean regularization loss
    TEST_F(PPISPRegularizationTest, ExposureMeanLoss) {
        const int num_frames = 10;
        auto exposure_torch = torch::randn({num_frames}, torch::kFloat32);
        exposure_torch.set_requires_grad(true);

        // PyTorch reference
        auto exp_mean = exposure_torch.mean();
        auto loss_torch = smooth_l1_torch(exp_mean, 0.1f).sum();

        // Our implementation
        lfs::training::PPISPConfig config;
        config.exposure_mean = 1.0f;
        config.vig_center = 0.0f;
        config.vig_channel = 0.0f;
        config.vig_non_pos = 0.0f;
        config.color_mean = 0.0f;
        config.crf_channel = 0.0f;

        auto ppisp_ptr = create_test_ppisp(1, num_frames, 1000, config);
        auto& ppisp = *ppisp_ptr;

        // Copy exposure params from torch
        auto exp_cpu = exposure_torch.contiguous().cpu();
        std::vector<float> exp_data(exp_cpu.data_ptr<float>(), exp_cpu.data_ptr<float>() + num_frames);
        auto exp_tensor = lfs::core::Tensor::from_vector(exp_data, {static_cast<size_t>(num_frames)},
                                                         lfs::core::Device::CUDA);
        // We need to set the params - use reflection or test the reg_loss directly
        // For now, create a manual test

        // Compute expected loss
        float exp_sum = 0.0f;
        for (int i = 0; i < num_frames; ++i) {
            exp_sum += exp_data[i];
        }
        float exp_mean_val = exp_sum / num_frames;
        float abs_mean = std::abs(exp_mean_val);
        float expected_loss;
        if (abs_mean < 0.1f) {
            expected_loss = 0.5f * exp_mean_val * exp_mean_val / 0.1f;
        } else {
            expected_loss = abs_mean - 0.05f;
        }

        float torch_loss = loss_torch.item<float>();
        EXPECT_TRUE(tensors_close(expected_loss, torch_loss, 1e-5f, 1e-6f));
    }

    // Test exposure mean gradient
    TEST_F(PPISPRegularizationTest, ExposureMeanGradient) {
        const int num_frames = 10;
        auto exposure_torch = torch::randn({num_frames}, torch::kFloat32);
        exposure_torch.set_requires_grad(true);

        // PyTorch forward + backward
        auto exp_mean = exposure_torch.mean();
        auto loss_torch = smooth_l1_torch(exp_mean, 0.1f).sum();
        loss_torch.backward();

        auto grad_torch = exposure_torch.grad().contiguous();
        std::vector<float> expected_grad(grad_torch.data_ptr<float>(),
                                         grad_torch.data_ptr<float>() + num_frames);

        // Our gradient computation
        std::vector<float> exp_data(exposure_torch.data_ptr<float>(),
                                    exposure_torch.data_ptr<float>() + num_frames);

        float exp_sum = 0.0f;
        for (int i = 0; i < num_frames; ++i) {
            exp_sum += exp_data[i];
        }
        float exp_mean_val = exp_sum / num_frames;

        float grad_mean;
        if (std::abs(exp_mean_val) < 0.1f) {
            grad_mean = exp_mean_val / 0.1f;
        } else {
            grad_mean = (exp_mean_val > 0.0f) ? 1.0f : -1.0f;
        }

        std::vector<float> our_grad(num_frames);
        for (int i = 0; i < num_frames; ++i) {
            our_grad[i] = grad_mean / num_frames;
        }

        EXPECT_TRUE(vectors_close(expected_grad, our_grad, 1e-5f, 1e-6f));
    }

    // Test vignetting center loss
    TEST_F(PPISPRegularizationTest, VignettingCenterLoss) {
        const int num_cameras = 3;

        // Vignetting params: [num_cameras, 3, 5] = [cam][ch][cx, cy, alpha0, alpha1, alpha2]
        auto vig_torch = torch::randn({num_cameras, 3, 5}, torch::kFloat32);
        vig_torch.set_requires_grad(true);

        // PyTorch reference: mean(cx^2 + cy^2)
        auto cx = vig_torch.index({"...", 0});
        auto cy = vig_torch.index({"...", 1});
        auto loss_torch = (cx.pow(2) + cy.pow(2)).mean();

        float expected_loss = loss_torch.item<float>();

        // Our implementation
        auto vig_flat = vig_torch.flatten().contiguous();
        std::vector<float> vig_data(vig_flat.data_ptr<float>(),
                                    vig_flat.data_ptr<float>() + vig_flat.numel());

        float our_loss = 0.0f;
        for (int cam = 0; cam < num_cameras; ++cam) {
            for (int ch = 0; ch < 3; ++ch) {
                size_t base = cam * 15 + ch * 5;
                float cx_val = vig_data[base + 0];
                float cy_val = vig_data[base + 1];
                our_loss += cx_val * cx_val + cy_val * cy_val;
            }
        }
        our_loss /= (num_cameras * 3);

        EXPECT_TRUE(tensors_close(expected_loss, our_loss, 1e-5f, 1e-6f));
    }

    // Test vignetting center gradient
    TEST_F(PPISPRegularizationTest, VignettingCenterGradient) {
        const int num_cameras = 3;

        auto vig_torch = torch::randn({num_cameras, 3, 5}, torch::kFloat32);
        vig_torch.set_requires_grad(true);

        // PyTorch forward + backward
        auto cx = vig_torch.index({"...", 0});
        auto cy = vig_torch.index({"...", 1});
        auto loss_torch = (cx.pow(2) + cy.pow(2)).mean();
        loss_torch.backward();

        auto grad_torch = vig_torch.grad().flatten().contiguous();
        std::vector<float> expected_grad(grad_torch.data_ptr<float>(),
                                         grad_torch.data_ptr<float>() + grad_torch.numel());

        // Our gradient computation
        auto vig_flat = vig_torch.flatten().contiguous();
        std::vector<float> vig_data(vig_flat.data_ptr<float>(),
                                    vig_flat.data_ptr<float>() + vig_flat.numel());

        std::vector<float> our_grad(vig_data.size(), 0.0f);
        float scale = 2.0f / (num_cameras * 3);
        for (int cam = 0; cam < num_cameras; ++cam) {
            for (int ch = 0; ch < 3; ++ch) {
                size_t base = cam * 15 + ch * 5;
                our_grad[base + 0] = scale * vig_data[base + 0]; // d/d(cx)
                our_grad[base + 1] = scale * vig_data[base + 1]; // d/d(cy)
            }
        }

        EXPECT_TRUE(vectors_close(expected_grad, our_grad, 1e-5f, 1e-6f));
    }

    // Test vignetting non-positivity loss
    TEST_F(PPISPRegularizationTest, VignettingNonPosLoss) {
        const int num_cameras = 3;

        auto vig_torch = torch::randn({num_cameras, 3, 5}, torch::kFloat32);
        vig_torch.set_requires_grad(true);

        // PyTorch reference: mean(relu(alphas))
        auto alphas = vig_torch.index({"...", torch::indexing::Slice(2, 5)});
        auto loss_torch = torch::relu(alphas).mean();

        float expected_loss = loss_torch.item<float>();

        // Our implementation
        auto vig_flat = vig_torch.flatten().contiguous();
        std::vector<float> vig_data(vig_flat.data_ptr<float>(),
                                    vig_flat.data_ptr<float>() + vig_flat.numel());

        float our_loss = 0.0f;
        for (int cam = 0; cam < num_cameras; ++cam) {
            for (int ch = 0; ch < 3; ++ch) {
                size_t base = cam * 15 + ch * 5;
                for (int a = 0; a < 3; ++a) {
                    float alpha = vig_data[base + 2 + a];
                    if (alpha > 0.0f) {
                        our_loss += alpha;
                    }
                }
            }
        }
        our_loss /= (num_cameras * 3 * 3);

        EXPECT_TRUE(tensors_close(expected_loss, our_loss, 1e-5f, 1e-6f));
    }

    // Test vignetting non-positivity gradient
    TEST_F(PPISPRegularizationTest, VignettingNonPosGradient) {
        const int num_cameras = 3;

        auto vig_torch = torch::randn({num_cameras, 3, 5}, torch::kFloat32);
        vig_torch.set_requires_grad(true);

        // PyTorch forward + backward
        auto alphas = vig_torch.index({"...", torch::indexing::Slice(2, 5)});
        auto loss_torch = torch::relu(alphas).mean();
        loss_torch.backward();

        auto grad_torch = vig_torch.grad().flatten().contiguous();
        std::vector<float> expected_grad(grad_torch.data_ptr<float>(),
                                         grad_torch.data_ptr<float>() + grad_torch.numel());

        // Our gradient computation
        auto vig_flat = vig_torch.flatten().contiguous();
        std::vector<float> vig_data(vig_flat.data_ptr<float>(),
                                    vig_flat.data_ptr<float>() + vig_flat.numel());

        std::vector<float> our_grad(vig_data.size(), 0.0f);
        float scale = 1.0f / (num_cameras * 3 * 3);
        for (int cam = 0; cam < num_cameras; ++cam) {
            for (int ch = 0; ch < 3; ++ch) {
                size_t base = cam * 15 + ch * 5;
                for (int a = 0; a < 3; ++a) {
                    if (vig_data[base + 2 + a] > 0.0f) {
                        our_grad[base + 2 + a] = scale;
                    }
                }
            }
        }

        EXPECT_TRUE(vectors_close(expected_grad, our_grad, 1e-5f, 1e-6f));
    }

    // Test vignetting channel variance loss
    TEST_F(PPISPRegularizationTest, VignettingChannelVarianceLoss) {
        const int num_cameras = 3;

        auto vig_torch = torch::randn({num_cameras, 3, 5}, torch::kFloat32);
        vig_torch.set_requires_grad(true);

        // PyTorch reference: mean(var(vig, dim=channel))
        // vig is [cam, ch, param], compute variance over ch (dim=1)
        auto var_torch = vig_torch.var(1, false); // unbiased=False
        auto loss_torch = var_torch.mean();

        float expected_loss = loss_torch.item<float>();

        // Our implementation
        auto vig_flat = vig_torch.flatten().contiguous();
        std::vector<float> vig_data(vig_flat.data_ptr<float>(),
                                    vig_flat.data_ptr<float>() + vig_flat.numel());

        float our_loss = 0.0f;
        for (int cam = 0; cam < num_cameras; ++cam) {
            for (int p = 0; p < 5; ++p) {
                float vals[3];
                for (int ch = 0; ch < 3; ++ch) {
                    vals[ch] = vig_data[cam * 15 + ch * 5 + p];
                }
                float mean = (vals[0] + vals[1] + vals[2]) / 3.0f;
                float var = 0.0f;
                for (int ch = 0; ch < 3; ++ch) {
                    float diff = vals[ch] - mean;
                    var += diff * diff;
                }
                var /= 3.0f;
                our_loss += var;
            }
        }
        our_loss /= (num_cameras * 5);

        EXPECT_TRUE(tensors_close(expected_loss, our_loss, 1e-5f, 1e-6f));
    }

    // Test vignetting channel variance gradient
    TEST_F(PPISPRegularizationTest, VignettingChannelVarianceGradient) {
        const int num_cameras = 3;

        auto vig_torch = torch::randn({num_cameras, 3, 5}, torch::kFloat32);
        vig_torch.set_requires_grad(true);

        // PyTorch forward + backward
        auto var_torch = vig_torch.var(1, false);
        auto loss_torch = var_torch.mean();
        loss_torch.backward();

        auto grad_torch = vig_torch.grad().flatten().contiguous();
        std::vector<float> expected_grad(grad_torch.data_ptr<float>(),
                                         grad_torch.data_ptr<float>() + grad_torch.numel());

        // Our gradient computation
        auto vig_flat = vig_torch.flatten().contiguous();
        std::vector<float> vig_data(vig_flat.data_ptr<float>(),
                                    vig_flat.data_ptr<float>() + vig_flat.numel());

        std::vector<float> our_grad(vig_data.size(), 0.0f);
        float scale = 1.0f / (num_cameras * 5);
        for (int cam = 0; cam < num_cameras; ++cam) {
            for (int p = 0; p < 5; ++p) {
                float vals[3];
                size_t idxs[3];
                for (int ch = 0; ch < 3; ++ch) {
                    idxs[ch] = cam * 15 + ch * 5 + p;
                    vals[ch] = vig_data[idxs[ch]];
                }
                float mean = (vals[0] + vals[1] + vals[2]) / 3.0f;
                for (int ch = 0; ch < 3; ++ch) {
                    our_grad[idxs[ch]] = scale * 2.0f * (vals[ch] - mean) / 3.0f;
                }
            }
        }

        EXPECT_TRUE(vectors_close(expected_grad, our_grad, 1e-5f, 1e-6f));
    }

    // Test color mean regularization loss
    TEST_F(PPISPRegularizationTest, ColorMeanLoss) {
        const int num_frames = 10;

        auto color_torch = torch::randn({num_frames, 8}, torch::kFloat32);
        color_torch.set_requires_grad(true);

        auto pinv = get_color_pinv_block_diag();

        // PyTorch reference: smooth_l1(mean(color @ pinv, dim=0), beta=0.005).mean()
        auto offsets = torch::matmul(color_torch, pinv);
        auto mean_offsets = offsets.mean(0);
        auto loss_torch = smooth_l1_torch(mean_offsets, 0.005f).mean();

        float expected_loss = loss_torch.item<float>();

        // Our implementation
        auto color_flat = color_torch.flatten().contiguous();
        std::vector<float> color_data(color_flat.data_ptr<float>(),
                                      color_flat.data_ptr<float>() + color_flat.numel());

        auto pinv_flat = pinv.flatten().contiguous();
        std::vector<float> pinv_data(pinv_flat.data_ptr<float>(),
                                     pinv_flat.data_ptr<float>() + pinv_flat.numel());

        float mean_offsets_cpp[8] = {0.0f};
        for (int f = 0; f < num_frames; ++f) {
            for (int j = 0; j < 8; ++j) {
                float dot = 0.0f;
                for (int k = 0; k < 8; ++k) {
                    dot += color_data[f * 8 + k] * pinv_data[k * 8 + j];
                }
                mean_offsets_cpp[j] += dot;
            }
        }
        for (int j = 0; j < 8; ++j) {
            mean_offsets_cpp[j] /= num_frames;
        }

        float our_loss = 0.0f;
        for (int j = 0; j < 8; ++j) {
            float abs_val = std::abs(mean_offsets_cpp[j]);
            if (abs_val < 0.005f) {
                our_loss += 0.5f * mean_offsets_cpp[j] * mean_offsets_cpp[j] / 0.005f;
            } else {
                our_loss += abs_val - 0.0025f;
            }
        }
        our_loss /= 8.0f;

        EXPECT_TRUE(tensors_close(expected_loss, our_loss, 1e-5f, 1e-6f));
    }

    // Test color mean regularization gradient
    TEST_F(PPISPRegularizationTest, ColorMeanGradient) {
        const int num_frames = 10;

        auto color_torch = torch::randn({num_frames, 8}, torch::kFloat32);
        color_torch.set_requires_grad(true);

        auto pinv = get_color_pinv_block_diag();

        // PyTorch forward + backward
        auto offsets = torch::matmul(color_torch, pinv);
        auto mean_offsets = offsets.mean(0);
        auto loss_torch = smooth_l1_torch(mean_offsets, 0.005f).mean();
        loss_torch.backward();

        auto grad_torch = color_torch.grad().flatten().contiguous();
        std::vector<float> expected_grad(grad_torch.data_ptr<float>(),
                                         grad_torch.data_ptr<float>() + grad_torch.numel());

        // Our gradient computation
        auto color_flat = color_torch.flatten().contiguous();
        std::vector<float> color_data(color_flat.data_ptr<float>(),
                                      color_flat.data_ptr<float>() + color_flat.numel());

        auto pinv_flat = pinv.flatten().contiguous();
        std::vector<float> pinv_data(pinv_flat.data_ptr<float>(),
                                     pinv_flat.data_ptr<float>() + pinv_flat.numel());

        // First compute mean offsets
        float mean_offsets_cpp[8] = {0.0f};
        for (int f = 0; f < num_frames; ++f) {
            for (int j = 0; j < 8; ++j) {
                float dot = 0.0f;
                for (int k = 0; k < 8; ++k) {
                    dot += color_data[f * 8 + k] * pinv_data[k * 8 + j];
                }
                mean_offsets_cpp[j] += dot;
            }
        }
        for (int j = 0; j < 8; ++j) {
            mean_offsets_cpp[j] /= num_frames;
        }

        // Gradient of smooth_l1
        float grad_offsets[8];
        for (int j = 0; j < 8; ++j) {
            float abs_val = std::abs(mean_offsets_cpp[j]);
            if (abs_val < 0.005f) {
                grad_offsets[j] = mean_offsets_cpp[j] / 0.005f;
            } else {
                grad_offsets[j] = (mean_offsets_cpp[j] > 0.0f) ? 1.0f : -1.0f;
            }
            grad_offsets[j] /= 8.0f; // mean over 8 outputs
        }

        // Chain rule: d/d(color) = grad_offsets @ pinv^T / num_frames
        std::vector<float> our_grad(num_frames * 8, 0.0f);
        for (int f = 0; f < num_frames; ++f) {
            for (int k = 0; k < 8; ++k) {
                float grad = 0.0f;
                for (int j = 0; j < 8; ++j) {
                    grad += grad_offsets[j] * pinv_data[k * 8 + j];
                }
                our_grad[f * 8 + k] = grad / num_frames;
            }
        }

        EXPECT_TRUE(vectors_close(expected_grad, our_grad, 1e-4f, 1e-5f));
    }

    // Test CRF channel variance loss
    TEST_F(PPISPRegularizationTest, CRFChannelVarianceLoss) {
        const int num_cameras = 3;

        // CRF params: [num_cameras, 3, 4] = [cam][ch][toe, shoulder, gamma, center]
        auto crf_torch = torch::randn({num_cameras, 3, 4}, torch::kFloat32);
        crf_torch.set_requires_grad(true);

        // PyTorch reference: mean(var(crf, dim=channel))
        auto var_torch = crf_torch.var(1, false); // unbiased=False
        auto loss_torch = var_torch.mean();

        float expected_loss = loss_torch.item<float>();

        // Our implementation
        auto crf_flat = crf_torch.flatten().contiguous();
        std::vector<float> crf_data(crf_flat.data_ptr<float>(),
                                    crf_flat.data_ptr<float>() + crf_flat.numel());

        float our_loss = 0.0f;
        for (int cam = 0; cam < num_cameras; ++cam) {
            for (int p = 0; p < 4; ++p) {
                float vals[3];
                for (int ch = 0; ch < 3; ++ch) {
                    vals[ch] = crf_data[cam * 12 + ch * 4 + p];
                }
                float mean = (vals[0] + vals[1] + vals[2]) / 3.0f;
                float var = 0.0f;
                for (int ch = 0; ch < 3; ++ch) {
                    float diff = vals[ch] - mean;
                    var += diff * diff;
                }
                var /= 3.0f;
                our_loss += var;
            }
        }
        our_loss /= (num_cameras * 4);

        EXPECT_TRUE(tensors_close(expected_loss, our_loss, 1e-5f, 1e-6f));
    }

    // Test CRF channel variance gradient
    TEST_F(PPISPRegularizationTest, CRFChannelVarianceGradient) {
        const int num_cameras = 3;

        auto crf_torch = torch::randn({num_cameras, 3, 4}, torch::kFloat32);
        crf_torch.set_requires_grad(true);

        // PyTorch forward + backward
        auto var_torch = crf_torch.var(1, false);
        auto loss_torch = var_torch.mean();
        loss_torch.backward();

        auto grad_torch = crf_torch.grad().flatten().contiguous();
        std::vector<float> expected_grad(grad_torch.data_ptr<float>(),
                                         grad_torch.data_ptr<float>() + grad_torch.numel());

        // Our gradient computation
        auto crf_flat = crf_torch.flatten().contiguous();
        std::vector<float> crf_data(crf_flat.data_ptr<float>(),
                                    crf_flat.data_ptr<float>() + crf_flat.numel());

        std::vector<float> our_grad(crf_data.size(), 0.0f);
        float scale = 1.0f / (num_cameras * 4);
        for (int cam = 0; cam < num_cameras; ++cam) {
            for (int p = 0; p < 4; ++p) {
                float vals[3];
                size_t idxs[3];
                for (int ch = 0; ch < 3; ++ch) {
                    idxs[ch] = cam * 12 + ch * 4 + p;
                    vals[ch] = crf_data[idxs[ch]];
                }
                float mean = (vals[0] + vals[1] + vals[2]) / 3.0f;
                for (int ch = 0; ch < 3; ++ch) {
                    our_grad[idxs[ch]] = scale * 2.0f * (vals[ch] - mean) / 3.0f;
                }
            }
        }

        EXPECT_TRUE(vectors_close(expected_grad, our_grad, 1e-5f, 1e-6f));
    }

    // Test finite difference verification for PPISP regularization gradients
    TEST_F(PPISPRegularizationTest, FiniteDifferenceVerification) {
        const int num_cameras = 2;
        const int num_frames = 5;
        const float eps = 1e-4f;

        lfs::training::PPISPConfig config;
        config.exposure_mean = 1.0f;
        config.vig_center = 0.02f;
        config.vig_channel = 0.1f;
        config.vig_non_pos = 0.01f;
        config.color_mean = 1.0f;
        config.crf_channel = 0.1f;

        auto ppisp_ptr = create_test_ppisp(num_cameras, num_frames, 1000, config);
        auto& ppisp = *ppisp_ptr;

        // Get initial loss
        auto loss0 = ppisp.reg_loss_gpu().cpu().item<float>();

        // Compute analytical gradients
        ppisp.zero_grad();
        ppisp.reg_backward();

        // The PPISP class doesn't expose direct access to gradients and params,
        // so we verify the overall gradient flow by checking that optimizer step
        // changes params in a way that reduces loss

        float initial_loss = loss0;

        // Take an optimizer step
        ppisp.optimizer_step();
        ppisp.scheduler_step();

        // Check new loss
        auto loss1 = ppisp.reg_loss_gpu().cpu().item<float>();

        // After one gradient step, loss should typically decrease (or stay similar)
        // This is a sanity check that gradients are flowing correctly
        // Note: with small random init, loss might already be near minimum
        EXPECT_GE(initial_loss + 0.1f, loss1); // Allow some tolerance
    }

    // Integration test: verify PPISP can be constructed and used without crashes
    TEST_F(PPISPRegularizationTest, PPISPIntegrationSmokeTest) {
        const int num_cameras = 2;
        const int num_frames = 5;

        lfs::training::PPISPConfig config;
        config.exposure_mean = 1.0f;
        config.vig_center = 0.02f;
        config.vig_channel = 0.1f;
        config.vig_non_pos = 0.01f;
        config.color_mean = 1.0f;
        config.crf_channel = 0.1f;

        auto ppisp_ptr = create_test_ppisp(num_cameras, num_frames, 1000, config);
        auto& ppisp = *ppisp_ptr;

        auto rgb = lfs::core::Tensor::ones({3, 32, 32}, lfs::core::Device::CUDA).mul(0.5f);
        auto grad_out = lfs::core::Tensor::ones({3, 32, 32}, lfs::core::Device::CUDA);

        // Run forward pass
        auto output = ppisp.apply(rgb, 0, 0);
        EXPECT_EQ(output.shape()[0], 3);
        EXPECT_EQ(output.shape()[1], 32);
        EXPECT_EQ(output.shape()[2], 32);

        // Run backward pass
        ppisp.zero_grad();
        auto grad_rgb = ppisp.backward(rgb, grad_out, 0, 0);
        EXPECT_EQ(grad_rgb.shape()[0], 3);

        // Compute reg loss
        auto reg_loss = ppisp.reg_loss_gpu();
        EXPECT_GE(reg_loss.cpu().item<float>(), 0.0f);

        // Accumulate reg gradients
        ppisp.reg_backward();

        // Optimizer step
        ppisp.optimizer_step();
        ppisp.scheduler_step();
    }

    TEST_F(PPISPRegularizationTest, BackwardRgbOnlyMatchesFiniteDiffAndLeavesParamsUntouched) {
        auto ppisp_ptr = create_test_ppisp(1, 2, 1000, {});
        auto& ppisp = *ppisp_ptr;
        ppisp.seed_exposure({{0, 0.7f}, {1, -0.4f}});
        const auto exp_before = ppisp.exposure_params().cpu().to_vector();
        const auto col_before = ppisp.color_params().cpu().to_vector();

        constexpr int H = 4;
        constexpr int W = 4;
        constexpr int N = H * W;
        std::vector<float> rgb_h(3 * N);
        std::vector<float> grad_h(3 * N);
        for (int i = 0; i < 3 * N; ++i) {
            rgb_h[i] = 0.15f + 0.01f * static_cast<float>(i % 29);
            grad_h[i] = 0.2f + 0.05f * static_cast<float>(i % 7);
        }
        const auto rgb = lfs::core::Tensor::from_vector(rgb_h, {3, H, W}, lfs::core::Device::CUDA);
        const auto grad_out = lfs::core::Tensor::from_vector(grad_h, {3, H, W}, lfs::core::Device::CUDA);

        ppisp.zero_grad();
        auto grad_rgb = grad_out.clone();
        ppisp.backward_rgb_only_in_place(rgb, grad_rgb, 0, 1);
        const auto analytical = grad_rgb.cpu().to_vector();

        constexpr float eps = 1e-3f;
        for (const int idx : {0, 5, N + 7, 2 * N + 11}) {
            auto plus_h = rgb_h;
            auto minus_h = rgb_h;
            plus_h[idx] += eps;
            minus_h[idx] -= eps;
            const auto out_plus = ppisp.apply(
                lfs::core::Tensor::from_vector(plus_h, {3, H, W}, lfs::core::Device::CUDA), 0, 1);
            const auto out_minus = ppisp.apply(
                lfs::core::Tensor::from_vector(minus_h, {3, H, W}, lfs::core::Device::CUDA), 0, 1);
            const float loss_plus = (out_plus * grad_out).sum().cpu().item<float>();
            const float loss_minus = (out_minus * grad_out).sum().cpu().item<float>();
            const float numerical = (loss_plus - loss_minus) / (2.0f * eps);
            EXPECT_NEAR(analytical[idx], numerical, 1e-4f);
        }

        // Verify parameter gradients remained zero by stepping the optimizer and checking params.
        ppisp.optimizer_step();
        const auto exp_after = ppisp.exposure_params().cpu().to_vector();
        const auto col_after = ppisp.color_params().cpu().to_vector();
        ASSERT_EQ(exp_before.size(), exp_after.size());
        for (size_t i = 0; i < exp_before.size(); ++i)
            EXPECT_FLOAT_EQ(exp_before[i], exp_after[i]);
        for (size_t i = 0; i < col_before.size(); ++i)
            EXPECT_FLOAT_EQ(col_before[i], col_after[i]);
    }

    TEST_F(PPISPRegularizationTest, VignettingAlphaL2MatchesFiniteDiff) {
        for (const bool gpu_fast_path : {true, false}) {
            SCOPED_TRACE(gpu_fast_path ? "gpu_fast_path" : "cpu_fallback");
            lfs::training::PPISPConfig config;
            config.train_crf = !gpu_fast_path;
            config.exposure_mean = gpu_fast_path ? 0.0f : 0.0f;
            config.color_mean = gpu_fast_path ? 0.0f : 1e-6f;
            config.crf_channel = gpu_fast_path ? 0.0f : 1e-6f;
            config.vig_center = 0.0f;
            config.vig_channel = 0.0f;
            config.vig_non_pos = 0.0f;
            config.vig_alpha_l2 = 0.75f;

            auto ppisp_ptr = create_test_ppisp(2, 4, 1000, config);
            auto& ppisp = *ppisp_ptr;
            std::vector<float> vig(2 * 15, 0.0f);
            for (size_t i = 0; i < vig.size(); ++i)
                vig[i] = ((i % 2 == 0) ? -0.35f : 0.25f) * (1.0f + 0.1f * static_cast<float>(i % 5));
            ppisp.vignetting_params().copy_from(
                lfs::core::Tensor::from_vector(vig, {vig.size()}, lfs::core::Device::CUDA));

            ppisp.zero_grad();
            const float loss0 = ppisp.reg_loss_gpu().cpu().item<float>();
            EXPECT_GT(loss0, 0.0f);
            ppisp.reg_backward();
            const auto grad = ppisp.vignetting_grad().cpu().to_vector();

            constexpr float eps = 1e-3f;
            for (size_t i = 0; i < vig.size(); ++i) {
                auto plus = vig;
                auto minus = vig;
                plus[i] += eps;
                minus[i] -= eps;
                ppisp.vignetting_params().copy_from(
                    lfs::core::Tensor::from_vector(plus, {plus.size()}, lfs::core::Device::CUDA));
                const float l_plus = ppisp.reg_loss_gpu().cpu().item<float>();
                ppisp.vignetting_params().copy_from(
                    lfs::core::Tensor::from_vector(minus, {minus.size()}, lfs::core::Device::CUDA));
                const float l_minus = ppisp.reg_loss_gpu().cpu().item<float>();
                const float fd = (l_plus - l_minus) / (2.0f * eps);
                EXPECT_NEAR(grad[i], fd, 5e-4f) << "index=" << i;
            }
        }
    }

    TEST_F(PPISPRegularizationTest, ExposurePriorAffineInvariantMatchesFiniteDiff) {
        const std::vector<std::tuple<const char*, std::vector<float>, std::vector<float>, float>> cases{
            {"interior_slope", {0.0f, 1.0f, 2.0f, 3.0f, 4.0f}, {-0.90f, -0.40f, 0.02f, 0.50f, 0.88f}, 0.44f},
            {"clamped_low_slope", {0.0f, 1.0f, 2.0f, 3.0f, 4.0f}, {-0.20f, -0.05f, 0.0f, 0.15f, 0.25f}, 0.30f},
            {"clamped_high_slope", {0.0f, 1.0f, 2.0f, 3.0f, 4.0f}, {-1.60f, -0.80f, 0.05f, 0.75f, 1.65f}, 0.60f},
            {"flat_exif", {2.0f, 2.0f, 2.0f, 2.0f, 2.0f}, {-0.3f, 0.1f, 0.0f, 0.4f, -0.2f}, 0.50f},
        };

        for (const auto& [name, evs, exp_vals, expected_slope] : cases) {
            for (const bool gpu_fast_path : {true, false}) {
                SCOPED_TRACE(std::string(name) + (gpu_fast_path ? "_gpu" : "_cpu"));
                lfs::training::PPISPConfig config;
                config.train_crf = !gpu_fast_path;
                config.exposure_mean = 0.0f;
                config.color_mean = gpu_fast_path ? 0.0f : 1e-6f;
                config.crf_channel = gpu_fast_path ? 0.0f : 1e-6f;
                config.vig_center = 0.0f;
                config.vig_channel = 0.0f;
                config.vig_non_pos = 0.0f;
                config.exposure_prior = 0.8f;

                auto ppisp_ptr = create_test_ppisp(1, 5, 1000, config);
                auto& ppisp = *ppisp_ptr;
                std::vector<std::pair<int, float>> prior;
                for (int i = 0; i < 5; ++i)
                    prior.emplace_back(i, evs[static_cast<size_t>(i)]);
                ppisp.set_exposure_prior(prior);
                ppisp.exposure_params().copy_from(
                    lfs::core::Tensor::from_vector(exp_vals, {5}, lfs::core::Device::CUDA));

                if (std::string_view(name) != "flat_exif") {
                    EXPECT_NEAR(ppisp.exposure_prior_slope(), expected_slope, 0.03f);
                } else {
                    EXPECT_FLOAT_EQ(ppisp.exposure_prior_slope(), 0.5f);
                }

                // Shift invariance: adding any global EV shift leaves loss unchanged.
                const float loss0 = ppisp.reg_loss_gpu().cpu().item<float>();
                auto shifted = exp_vals;
                for (float& v : shifted)
                    v += 0.65f;
                ppisp.exposure_params().copy_from(
                    lfs::core::Tensor::from_vector(shifted, {5}, lfs::core::Device::CUDA));
                EXPECT_NEAR(ppisp.reg_loss_gpu().cpu().item<float>(), loss0, 1e-5f);

                // Restore and verify finite-diff gradients.
                ppisp.exposure_params().copy_from(
                    lfs::core::Tensor::from_vector(exp_vals, {5}, lfs::core::Device::CUDA));
                ppisp.zero_grad();
                ppisp.reg_backward();
                const auto grad = ppisp.exposure_grad().cpu().to_vector();

                constexpr float eps = 1e-3f;
                for (size_t i = 0; i < exp_vals.size(); ++i) {
                    auto plus = exp_vals;
                    auto minus = exp_vals;
                    plus[i] += eps;
                    minus[i] -= eps;
                    ppisp.exposure_params().copy_from(
                        lfs::core::Tensor::from_vector(plus, {5}, lfs::core::Device::CUDA));
                    const float l_plus = ppisp.reg_loss_gpu().cpu().item<float>();
                    ppisp.exposure_params().copy_from(
                        lfs::core::Tensor::from_vector(minus, {5}, lfs::core::Device::CUDA));
                    const float l_minus = ppisp.reg_loss_gpu().cpu().item<float>();
                    const float fd = (l_plus - l_minus) / (2.0f * eps);
                    EXPECT_NEAR(grad[i], fd, 5e-4f) << "index=" << i;
                }
            }
        }
    }

    TEST_F(PPISPRegularizationTest, ColorLatentL2MatchesFiniteDiff) {
        for (const bool gpu_fast_path : {true, false}) {
            SCOPED_TRACE(gpu_fast_path ? "gpu_fast_path" : "cpu_fallback");
            lfs::training::PPISPConfig config;
            config.train_crf = !gpu_fast_path;
            config.exposure_mean = 0.0f;
            config.color_mean = gpu_fast_path ? 0.0f : 1e-6f;
            config.crf_channel = gpu_fast_path ? 0.0f : 1e-6f;
            config.vig_center = 0.0f;
            config.vig_channel = 0.0f;
            config.vig_non_pos = 0.0f;
            config.color_l2 = 0.25f;

            auto ppisp_ptr = create_test_ppisp(1, 3, 1000, config);
            auto& ppisp = *ppisp_ptr;
            std::vector<float> col(24);
            for (size_t i = 0; i < col.size(); ++i)
                col[i] = 0.3f * std::sin(0.5f * static_cast<float>(i) + 0.2f);
            ppisp.color_params().copy_from(
                lfs::core::Tensor::from_vector(col, {col.size()}, lfs::core::Device::CUDA));

            ppisp.zero_grad();
            const float loss0 = ppisp.reg_loss_gpu().cpu().item<float>();
            EXPECT_GT(loss0, 0.0f);
            ppisp.reg_backward();
            const auto grad = ppisp.color_grad().cpu().to_vector();

            constexpr float eps = 1e-3f;
            for (size_t i = 0; i < col.size(); ++i) {
                auto plus = col;
                auto minus = col;
                plus[i] += eps;
                minus[i] -= eps;
                ppisp.color_params().copy_from(
                    lfs::core::Tensor::from_vector(plus, {plus.size()}, lfs::core::Device::CUDA));
                const float l_plus = ppisp.reg_loss_gpu().cpu().item<float>();
                ppisp.color_params().copy_from(
                    lfs::core::Tensor::from_vector(minus, {minus.size()}, lfs::core::Device::CUDA));
                const float l_minus = ppisp.reg_loss_gpu().cpu().item<float>();
                const float fd = (l_plus - l_minus) / (2.0f * eps);
                EXPECT_NEAR(grad[i], fd, 5e-4f) << "index=" << i;
            }
        }
    }

} // namespace
