// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/abi.hpp"
#include "core/argument_parser.hpp"
#include "core/crash_handler.hpp"
#include "core/cuda_error.hpp"
#include "core/logger.hpp"
#include "git_version.h"
#include "lfs_core_abi_stamp.h"
#include "runtime.hpp"
#include <cstdlib>
#include <cuda_runtime.h>
#include <exception>
#include <print>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (!lfs_core_abi_matches(LFS_CORE_ABI_STAMP)) {
        std::println(stderr, "lfs-train: lfs_core ABI mismatch; rebuild in a clean train directory");
        return 2;
    }
    try {
        lfs::core::Logger::get().init();
        // Reuse the upstream training parser, with headless semantics by default.
        std::vector<const char*> args(argv, argv + argc);
        args.push_back("--headless");
        auto parsed = lfs::core::args::parse_args(static_cast<int>(args.size()), args.data());
        if (!parsed) {
            std::println(stderr, "lfs-train: {}", parsed.error());
            return 2;
        }
        if (std::holds_alternative<lfs::core::args::HelpMode>(*parsed))
            return 0;
        if (std::holds_alternative<lfs::core::args::VersionMode>(*parsed)) {
            std::println("lfs-train {}", GIT_COMMIT_HASH_SHORT);
            return 0;
        }
        auto* training = std::get_if<lfs::core::args::TrainingMode>(&*parsed);
        if (!training) {
            std::println(stderr, "lfs-train supports training only; conversion, preprocessing and plugins are unavailable");
            return 2;
        }
        lfs::app::validate_train_parameters(*training->params);
#ifdef _WIN32
        _putenv_s("CUDA_MODULE_LOADING", "LAZY");
#else
        setenv("CUDA_MODULE_LOADING", "LAZY", 0);
#endif
        struct CudaCleanup {
            ~CudaCleanup() { lfs::core::teardown_gpu_before_exit(); }
        } cuda_cleanup;
        lfs::core::install_crash_handlers();
        lfs::core::initialize_cuda_diagnostics();
        cudaDeviceProp device{};
        auto status = cudaGetDeviceProperties(&device, 0);
        if (status != cudaSuccess) {
            std::println(stderr, "lfs-train: CUDA initialization failed: {}", cudaGetErrorString(status));
            return 1;
        }
        if (device.major * 10 + device.minor < LFS_MIN_SM) {
            std::println(stderr, "lfs-train requires SM {} or newer", LFS_MIN_SM);
            return 1;
        }
        return lfs::app::run_train(std::move(*training->params));
    } catch (const std::exception& error) {
        std::println(stderr, "lfs-train: {}", error.what());
        return 1;
    } catch (...) {
        std::println(stderr, "lfs-train: unknown error");
        return 1;
    }
}
