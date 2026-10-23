/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// lfs_train: headless training entry point of the LFS_TRAINER_ONLY build.
// Trains a dataset folder the same way `LichtFeld-Studio --headless` does,
// without the visualizer, Python, video, MCP/TCP or MoGe preprocessing.

#include "app/headless_run_coordinator.hpp"
#include "control/command_api.hpp"
#include "core/abi.hpp"
#include "core/argument_parser.hpp"
#include "core/crash_handler.hpp"
#include "core/cuda_error.hpp"
#include "core/cuda_version.hpp"
#include "core/environment.hpp"
#include "core/event_bridge/command_center_bridge.hpp"
#include "core/image_loader.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/scene.hpp"
#include "core/session_breadcrumb.hpp"
#include "core/tensor.hpp"
#include "git_version.h"
#include "io/cache_image_loader.hpp"
#include "lfs_core_abi_stamp.h"
#include "training/trainer.hpp"
#include "training/training_setup.hpp"

#include <cstdlib>
#include <cuda_runtime.h>
#include <filesystem>
#include <memory>
#include <optional>
#include <print>
#include <string>
#include <variant>
#include <vector>

#ifdef __GLIBC__
#include <malloc.h>
#endif

#ifndef LFS_MIN_SM
#error "LFS_MIN_SM must be defined by the build (src/train_cli/CMakeLists.txt)"
#endif

namespace {

    // Must precede any CUDA call; see applyCudaContextTuning in src/app/main.cpp.
    void applyCudaContextTuning() {
#ifdef _WIN32
        _putenv_s("CUDA_MODULE_LOADING", "LAZY");
#else
        setenv("CUDA_MODULE_LOADING", "LAZY", /*overwrite=*/0);
#endif
    }

    bool preflightGpu() {
        const auto info = lfs::core::check_cuda_version();
        if (info.query_failed) {
            LOG_WARN("Failed to query CUDA driver version");
        } else {
            LOG_INFO("CUDA driver version: {}.{}", info.major, info.minor);
            if (!info.supported) {
                LOG_ERROR("CUDA {}.{} is too old; lfs_train requires CUDA 12.8 or newer "
                          "(NVIDIA driver 570 or newer)",
                          info.major, info.minor);
                return false;
            }
        }

        cudaDeviceProp prop{};
        if (const cudaError_t err = cudaGetDeviceProperties(&prop, 0); err != cudaSuccess) {
            LOG_ERROR("No usable NVIDIA GPU found ({})", cudaGetErrorString(err));
            return false;
        }
        LOG_INFO("GPU: {} (SM {}.{}, {} MB)", prop.name, prop.major, prop.minor,
                 prop.totalGlobalMem / (1024 * 1024));
        if (const int device_sm = prop.major * 10 + prop.minor; device_sm < LFS_MIN_SM) {
            LOG_ERROR("GPU {} has compute capability {}.{}, below the {}.{} this build requires",
                      prop.name, prop.major, prop.minor, LFS_MIN_SM / 10, LFS_MIN_SM % 10);
            return false;
        }
        return true;
    }

    void installImageLoader(const bool use_cpu_memory) {
        lfs::io::CacheLoader::getInstance(use_cpu_memory);
        lfs::core::set_image_loader([](const lfs::core::ImageLoadParams& p) {
            return lfs::io::CacheLoader::getInstance().load_cached_image(
                p.path,
                {.resize_factor = p.resize_factor,
                 .max_width = p.max_width,
                 .cuda_stream = p.stream,
                 .output_uint8 = p.output_uint8,
                 .skip_blob_cache = p.skip_blob_cache,
                 .mask_grayscale = p.mask_grayscale});
        });
    }

    std::optional<std::filesystem::path> lpipsWeightsFromEnvironment() {
        if (const auto path = lfs::core::environment::value("LFS_LPIPS_WEIGHTS"))
            return lfs::core::utf8_to_path(*path);
        LOG_WARN("LPIPS unavailable: set LFS_LPIPS_WEIGHTS to the weights file "
                 "(lfs_train does not download it)");
        return std::nullopt;
    }

    // Options that need a component the trainer-only build leaves out.
    std::optional<std::string> unsupportedOption(const lfs::core::param::TrainingParameters& params) {
        if (params.render_path)
            return "--render-camera-path (video rendering)";
        if (params.evaluate_only)
            return "the eval subcommand";
        if (params.dataset_project)
            return "a .licht project as --data-path";
        if (params.resume_project || params.resume_checkpoint)
            return "--resume";
        if (params.server.tcp_connection)
            return "--tcp-connection";
        if (!params.python_scripts.empty())
            return "--python-script";
        if (params.optimization.debug_python)
            return "--debug-python";
        return std::nullopt;
    }

    int train(std::unique_ptr<lfs::core::param::TrainingParameters> params) {
        if (const auto option = unsupportedOption(*params)) {
            LOG_ERROR("{} is not supported by lfs_train; use the LichtFeld-Studio application", *option);
            return 1;
        }
        if (params->dataset.data_path.empty()) {
            LOG_ERROR("lfs_train requires --data-path");
            return 1;
        }
        params->optimization.headless = true;

        installImageLoader(params->dataset.loading_params.use_cpu_memory);
        lfs::event::CommandCenterBridge::instance().set(&lfs::training::CommandCenter::instance());
        lfs::app::HeadlessRunCoordinator coordinator;

        // Scene and trainer stay alive until flush_and_exit, like the headless application path.
        lfs::core::Scene scene;
        if (const auto result = lfs::training::loadTrainingDataIntoScene(*params, scene); !result) {
            LOG_ERROR("Failed to load training data: {}", result.error());
            return 1;
        }
        if (const auto result = lfs::training::initializeTrainingModel(*params, scene); !result) {
            LOG_ERROR("Failed to initialize model: {}", result.error());
            return 1;
        }

        auto trainer = std::make_unique<lfs::training::Trainer>(scene);
        if (const auto result = trainer->initialize(*params); !result) {
            LOG_ERROR("Failed to initialize trainer: {}", result.error());
            return 1;
        }
        if (params->optimization.enable_eval)
            trainer->set_lpips_weights_path(lpipsWeightsFromEnvironment());
        lfs::training::grant_headless_project_saves(*trainer, *params);

        lfs::core::Tensor::trim_memory_pool();

        if (const auto result = trainer->train(coordinator.stop_token()); !result) {
            LOG_ERROR("Training error: {}", lfs::format_for_developer(result.error()));
            return 1;
        }

        const auto exported = lfs::training::export_final_splats(*trainer, *params);
        trainer->shutdown();
        static_cast<void>(trainer.release());
        if (!exported) {
            LOG_ERROR("{}", lfs::format_for_developer(exported.error()));
            return 1;
        }

        lfs::core::teardown_gpu_before_exit();
        lfs::core::mark_clean_exit();
        lfs::core::flush_and_exit(coordinator.interrupted() ? coordinator.interrupted_exit_code() : 0);
    }

    int runMode(lfs::core::args::ParsedArgs args) {
        return std::visit([](auto&& mode) -> int {
            using T = std::decay_t<decltype(mode)>;

            if constexpr (std::is_same_v<T, lfs::core::args::HelpMode>) {
                return 0;
            } else if constexpr (std::is_same_v<T, lfs::core::args::VersionMode>) {
                std::println("lfs_train (LichtFeld Studio {} {})", GIT_TAGGED_VERSION, GIT_COMMIT_HASH_SHORT);
                return 0;
            } else if constexpr (std::is_same_v<T, lfs::core::args::TrainingMode>) {
#ifdef __GLIBC__
                // Training workers create many short-lived glibc arenas.
                if (std::getenv("MALLOC_ARENA_MAX") == nullptr)
                    (void)mallopt(M_ARENA_MAX, 4);
#endif
                LOG_INFO("lfs_train {} ({})", GIT_TAGGED_VERSION, GIT_COMMIT_HASH_SHORT);
                applyCudaContextTuning();
                if (!preflightGpu()) {
                    lfs::core::teardown_gpu_before_exit();
                    lfs::core::flush_and_exit(1);
                }
                return train(std::move(mode.params));
            } else {
                std::println(stderr, "Error: lfs_train only trains; this subcommand needs the "
                                     "LichtFeld-Studio application");
                return 1;
            }
        },
                          std::move(args));
    }

} // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t* wide_argv[]) {
    // The parser expects UTF-8; narrow CRT argv uses the ANSI code page.
    std::vector<std::string> utf8_args;
    utf8_args.reserve(argc);
    for (int i = 0; i < argc; ++i)
        utf8_args.push_back(lfs::core::wstring_to_utf8(wide_argv[i]));

    std::vector<const char*> utf8_argv;
    utf8_argv.reserve(argc + 1);
    for (const auto& arg : utf8_args)
        utf8_argv.push_back(arg.c_str());
    utf8_argv.push_back(nullptr);
    const auto* argv = utf8_argv.data();
#else
int main(int argc, char* argv[]) {
#endif
    const char* const loaded_core_stamp = lfs_core_abi_stamp();
    if (loaded_core_stamp == nullptr || !lfs_core_abi_matches(LFS_CORE_ABI_STAMP)) {
        std::println(stderr,
                     "Fatal: lfs_core ABI mismatch. lfs_train expects '{}' but loaded '{}'. "
                     "Remove stale binaries and rebuild.",
                     LFS_CORE_ABI_STAMP,
                     loaded_core_stamp != nullptr ? loaded_core_stamp : "<null>");
        return 2;
    }

    lfs::core::install_crash_handlers();
    lfs::core::record_session_start();
    lfs::core::initialize_cuda_diagnostics();

    auto result = lfs::core::args::parse_args(argc, argv);
    if (!result) {
        std::println(stderr, "Error: {}", result.error());
        return 1;
    }

    return lfs::core::run_with_exception_firewall(
        [&result] { return runMode(std::move(*result)); });
}
