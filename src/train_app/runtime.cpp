// SPDX-License-Identifier: GPL-3.0-or-later
#include "runtime.hpp"
#include "app/headless_recovery_document.hpp"
#include "app/headless_run_coordinator.hpp"
#include "core/checkpoint_format.hpp"
#include "core/event_bridge/command_center_bridge.hpp"
#include "core/image_loader.hpp"
#include "core/path_utils.hpp"
#include "core/scene.hpp"
#include "core/splat_data.hpp"
#include "io/cache_image_loader.hpp"
#include "io/embedded_dataset.hpp"
#include "io/project_document.hpp"
#include "io/project_recovery.hpp"
#include "training/control/command_api.hpp"
#include "training/trainer.hpp"
#include "training/training_setup.hpp"
#include <format>
#include <stdexcept>

namespace lfs::app {
    void validate_train_parameters(const core::param::TrainingParameters& p) {
        if (!p.python_scripts.empty() || p.optimization.debug_python)
            throw std::invalid_argument("Python scripts are unavailable in lfs-train");
        if (p.server.tcp_connection || p.mcp_port || p.render_path || !p.view_paths.empty() ||
            p.import_cameras_path || p.project_path || p.reset_preferences || p.reset_layout || p.reset_all_settings)
            throw std::invalid_argument("Editor, server and video options are unavailable in lfs-train");
        if (p.optimization.normal_auto_generate)
            throw std::invalid_argument("Automatic normals are unavailable; use --no-normal-auto-generate and precomputed maps");
        for (auto f : p.export_formats)
            if (f != core::param::OutputFormat::PLY)
                throw std::invalid_argument("lfs-train supports only --export ply");
    }
    namespace {
        void validate_scene(const core::param::TrainingParameters& p, const core::Scene& scene) {
            if (!scene.hasTrainingData())
                throw std::invalid_argument("No training cameras were loaded; provide a COLMAP or NeRF-transform dataset");
            for (const auto* n : scene.getNodes())
                if (n->type == core::NodeType::MESH || n->type == core::NodeType::PLY_SEQUENCE)
                    throw std::invalid_argument("lfs-train cannot train projects containing mesh or sequence assets");
            for (const auto& c : scene.getAllCameras()) {
                if (!c->has_image())
                    continue;
                if (p.optimization.mask_mode != core::param::MaskMode::None && !c->has_mask() &&
                    !(p.optimization.use_alpha_as_mask && c->has_alpha()))
                    throw std::invalid_argument("Required mask or image alpha missing: " + c->image_name());
                if (p.optimization.use_depth_loss && p.optimization.depth_loss_weight > 0 && !c->has_depth())
                    throw std::invalid_argument("Required depth map missing: " + c->image_name());
                if (p.optimization.use_normal_loss && p.optimization.normal_loss_weight > 0 && !c->has_normal())
                    throw std::invalid_argument("Required normal map missing: " + c->image_name());
            }
        }
        // Headless runs train into the project they were started from unless
        // -o redirects the result to a fresh project.licht.
        [[nodiscard]] std::filesystem::path headless_project_save_destination(
            const core::param::TrainingParameters& cli_params,
            const std::filesystem::path& source) {
            if (!cli_params.dataset.output_path_explicit)
                return source;

            const auto destination = cli_params.dataset.output_path / "project.licht";
            LOG_INFO("Headless project destination: {}",
                     core::path_to_utf8(destination));
            return destination;
        }

        // Empty for a plain dataset-folder run, which keeps the default
        // output_path/project.licht destination.
        [[nodiscard]] std::filesystem::path headless_dataset_project_destination(
            const core::param::TrainingParameters& params) {
            if (!params.dataset_project)
                return {};
            return headless_project_save_destination(params, *params.dataset_project);
        }

        [[nodiscard]] lfs::Error training_project_error(
            const lfs::ErrorCode code,
            std::string detail,
            const lfs::core::SourceSite source) {
            return lfs::make_error(lfs::ErrorInit{
                .code = code,
                .domain = lfs::ErrorDomain::App,
                .user_message =
                    "The training project could not be restored.",
                .detail = std::move(detail),
                .detection = source,
            });
        }

        // --data-path may name an untrained .licht. Resolve its dataset, the
        // external folder recorded in REFS or else the embedded copy extracted
        // to the per-user cache, and continue as a dataset-folder run that
        // trains into the project unless -o redirects the result.
        [[nodiscard]] lfs::Result<void> adoptDatasetProject(
            core::param::TrainingParameters& params) {
            const auto path = *params.dataset_project;
            auto document = io::project::ProjectDocument::open(
                path,
                io::project::ProjectDocumentOpenOptions{
                    .reader = {},
                    .geometry = {},
                    .defer_geometry_payloads = true,
                });
            if (!document) {
                return lfs::Status::failure(
                    std::move(document).error().with_context(
                        "open training project",
                        LFS_SOURCE_SITE_CURRENT()));
            }

            auto checkpoint_uuid = document->bound_checkpoint_uuid();
            if (!checkpoint_uuid) {
                return lfs::Status::failure(
                    std::move(checkpoint_uuid).error());
            }
            // Check whether there's a checkpoint, hence being trained before.
            if (*checkpoint_uuid) {
                return lfs::Status::failure(training_project_error(
                    lfs::ErrorCode::FailedPrecondition,
                    "The project already has a training checkpoint; pass it "
                    "to --resume to continue training",
                    LFS_SOURCE_SITE_CURRENT()));
            }

            auto snapshot = document->parameters().snapshot();
            if (!snapshot) {
                return lfs::Status::failure(
                    std::move(snapshot).error().with_context(
                        "read training project parameters",
                        LFS_SOURCE_SITE_CURRENT()));
            }

            // Prefer the original dataset folder recorded in REFS when it is
            // still reachable, so nothing has to be unpacked.
            std::optional<std::filesystem::path> dataset_root;
            std::string images_folder = snapshot->dataset.images;
            if (const auto dataset_ref =
                    document->project().dataset_reference();
                dataset_ref && *dataset_ref) {
                auto external = io::project::resolve_path_reference(
                    document->references(), path.parent_path(),
                    **dataset_ref);
                if (external && std::filesystem::is_directory(*external)) {
                    dataset_root = std::move(*external);
                }
            }
            // Otherwise unpack the embedded DSRC chunks into the per-user cache.
            // Files already present with a matching hash are reused.
            if (!dataset_root) {
                auto cache_dir =
                    io::project::embedded_dataset_cache_dir(*document);
                if (!cache_dir) {
                    return lfs::Status::failure(
                        std::move(cache_dir).error());
                }
                auto extracted = io::project::extract_embedded_dataset(
                    *document, *cache_dir);
                if (!extracted) {
                    return lfs::Status::failure(
                        std::move(extracted).error());
                }
                if (*extracted) {
                    dataset_root = std::move(**extracted);
                    // The manifest records which images folder was embedded.
                    if (const auto manifest =
                            document->parameters().embedded_dataset();
                        manifest && *manifest) {
                        images_folder = (*manifest)->images_folder;
                    }
                }
            }
            if (!dataset_root) {
                return lfs::Status::failure(training_project_error(
                    lfs::ErrorCode::NotFound,
                    "The project's dataset is neither reachable nor embedded",
                    LFS_SOURCE_SITE_CURRENT()));
            }

            // Without -o, exports land beside the project.
            if (!params.dataset.output_path_explicit)
                params.dataset.output_path = path.parent_path();
            io::project::adopt_project_training_parameters(
                params, std::move(*snapshot), std::move(*dataset_root),
                std::move(images_folder));
            // dataset_project stays set: it is the save destination unless
            // -o redirects the result, exactly like --resume.
            LOG_INFO(
                "Training from project {}; dataset resolved to {}",
                core::path_to_utf8(path),
                core::path_to_utf8(params.dataset.data_path));
            return {};
        }

        std::expected<core::param::TrainingParameters, std::string> loadCheckpointParams(const core::param::TrainingParameters& params, core::Scene& scene) {
            LOG_INFO("Resuming from checkpoint: {}", core::path_to_utf8(*params.resume_checkpoint));

            auto params_result = core::load_checkpoint_params(*params.resume_checkpoint);
            if (!params_result) {
                return std::unexpected(std::format("Failed to load checkpoint params: {}", params_result.error()));
            }
            auto checkpoint_params = std::move(*params_result);

            if (!params.dataset.data_path.empty())
                checkpoint_params.dataset.data_path = params.dataset.data_path;
            if (!params.dataset.output_path.empty())
                checkpoint_params.dataset.output_path = params.dataset.output_path;
            if (!params.dataset.output_name.empty())
                checkpoint_params.dataset.output_name = params.dataset.output_name;

            // Runtime-only CLI controls are not part of the serialized
            // training state. Preserve them when a resume is requested so
            // --perf-bench (and its warmup) still applies to the resumed run.
            checkpoint_params.optimization.perf_bench = params.optimization.perf_bench;
            checkpoint_params.optimization.perf_bench_warmup = params.optimization.perf_bench_warmup;
            checkpoint_params.cli_iterations_set = params.cli_iterations_set;
            checkpoint_params.cli_bg_color_set = params.cli_bg_color_set;
            if (params.cli_iterations_set)
                checkpoint_params.optimization.iterations = params.optimization.iterations;
            if (params.cli_bg_color_set)
                checkpoint_params.optimization.bg_color = params.optimization.bg_color;
            checkpoint_params.overrides = params.overrides;
            core::param::apply_explicit_training_overrides(
                checkpoint_params, checkpoint_params.overrides);

            validate_train_parameters(checkpoint_params);
            if (checkpoint_params.dataset.data_path.empty()) {
                return std::unexpected("Checkpoint has no dataset path and none provided via --data-path");
            }
            if (!std::filesystem::exists(checkpoint_params.dataset.data_path)) {
                return std::unexpected(std::format("Dataset path does not exist: {}", core::path_to_utf8(checkpoint_params.dataset.data_path)));
            }

            if (const auto result = training::validateDatasetPath(checkpoint_params); !result) {
                return std::unexpected(std::format("Dataset validation failed: {}", result.error()));
            }

            if (const auto result = training::loadTrainingDataIntoScene(checkpoint_params, scene); !result) {
                return std::unexpected(std::format("Failed to load training data: {}", result.error()));
            }

            for (const auto* node : scene.getNodes()) {
                if (node->type == core::NodeType::POINTCLOUD) {
                    scene.removeNode(node->name, false);
                    break;
                }
            }

            auto splat_result = core::load_checkpoint_splat_data(*params.resume_checkpoint);
            if (!splat_result) {
                return std::unexpected(std::format("Failed to load checkpoint splat data: {}", splat_result.error()));
            }

            auto splat_data = std::make_unique<core::SplatData>(std::move(*splat_result));
            const auto model_id = scene.addSplat("Model", std::move(splat_data), core::NULL_NODE);
            if (model_id == core::NULL_NODE) {
                return std::unexpected("Failed to add checkpoint training model to scene");
            }
            scene.setTrainingModelNode(model_id);

            checkpoint_params.resume_checkpoint = *params.resume_checkpoint;
            return checkpoint_params;
        }

        struct LoadedTrainingProject {
            detail::HeadlessRecoveryDocument document;
            core::param::TrainingParameters params;
            core::Uuid checkpoint_uuid;
            int iteration = 0;
        };

        lfs::Result<LoadedTrainingProject>
        loadTrainingProject(
            const core::param::TrainingParameters& cli_params,
            core::Scene& scene) {
            if (!cli_params.resume_project) {
                return training_project_error(
                    lfs::ErrorCode::InvalidArgument,
                    "No .licht resume project was provided",
                    LFS_SOURCE_SITE_CURRENT());
            }
            const auto& path =
                *cli_params.resume_project;
            std::filesystem::path open_path = path;
            std::optional<
                io::project::RecoverySession>
                recovery_session;
            LOG_INFO(
                "Opening training project: {}",
                core::path_to_utf8(path));

            auto recovery =
                io::project::inspect_autosave_recovery(
                    path);
            if (!recovery) {
                return std::move(recovery)
                    .error()
                    .with_context(
                        "inspect training project autosave",
                        LFS_SOURCE_SITE_CURRENT());
            }
            if (recovery->disposition ==
                    io::project::RecoveryDisposition::Offer &&
                recovery->selected_path) {
                auto session =
                    io::project::
                        begin_recovery_session(
                            path,
                            *recovery
                                 ->selected_path);
                if (!session) {
                    return std::move(session)
                        .error()
                        .with_context(
                            "hold headless recovery master lock",
                            LFS_SOURCE_SITE_CURRENT());
                }
                open_path = io::project::
                    recovery_session_temp_path(
                        path);
                auto materialized =
                    io::project::
                        materialize_recovered_project(
                            path,
                            *recovery
                                 ->selected_path,
                            open_path, *session);
                if (!materialized) {
                    return std::move(
                               materialized)
                        .error()
                        .with_context(
                            "materialize headless autosave recovery",
                            LFS_SOURCE_SITE_CURRENT());
                }
                recovery_session.emplace(
                    std::move(*session));
                LOG_INFO(
                    "Automatically recovering autosave sequence {} for headless project open",
                    recovery
                        ->autosave_sequence);
            }

            auto document =
                io::project::ProjectDocument::open(
                    open_path);
            if (!document) {
                if (recovery_session) {
                    static_cast<void>(
                        recovery_session
                            ->release());
                }
                return std::move(document)
                    .error()
                    .with_context(
                        "open training project",
                        LFS_SOURCE_SITE_CURRENT());
            }
            detail::HeadlessRecoveryDocument
                recovery_document(
                    std::move(*document),
                    std::move(recovery_session));
            auto checkpoint_uuid =
                recovery_document.document().bound_checkpoint_uuid();
            if (!checkpoint_uuid) {
                return std::move(checkpoint_uuid).error();
            }
            if (!*checkpoint_uuid) {
                return training_project_error(
                    lfs::ErrorCode::FailedPrecondition,
                    "Training project has no checkpoint to resume; pass an "
                    "untrained project to --data-path instead",
                    LFS_SOURCE_SITE_CURRENT());
            }
            const auto bound_checkpoint_uuid = **checkpoint_uuid;
            const auto* checkpoint =
                recovery_document.document()
                    .find_checkpoint(bound_checkpoint_uuid);
            if (!checkpoint) {
                return training_project_error(
                    lfs::ErrorCode::ContractViolation,
                    "Training project CKPT handle disappeared",
                    LFS_SOURCE_SITE_CURRENT());
            }

            std::optional<
                core::CheckpointParametersLoadResult>
                parsed_params;
            auto visited =
                checkpoint->visit_stream(
                    [&](std::istream& source,
                        const std::uint64_t bytes)
                        -> lfs::Result<void> {
                        parsed_params =
                            core::load_checkpoint_params(
                                source, bytes);
                        return {};
                    });
            if (!visited) {
                return std::move(visited)
                    .error()
                    .with_context(
                        "stream CKPT parameters",
                        LFS_SOURCE_SITE_CURRENT());
            }
            if (!parsed_params ||
                !*parsed_params) {
                return training_project_error(
                    lfs::ErrorCode::DataLoss,
                    parsed_params
                        ? std::format(
                              "Failed to parse CKPT parameters: {}",
                              parsed_params->error())
                        : "CKPT parameter visitor did not run",
                    LFS_SOURCE_SITE_CURRENT());
            }
            auto checkpoint_params =
                std::move(**parsed_params);

            if (!cli_params.dataset.data_path.empty()) {
                checkpoint_params.dataset.data_path =
                    cli_params.dataset.data_path;
            }
            if (!cli_params.dataset.output_path.empty()) {
                checkpoint_params.dataset.output_path =
                    cli_params.dataset.output_path;
            }
            if (!cli_params.dataset.output_name.empty()) {
                checkpoint_params.dataset.output_name =
                    cli_params.dataset.output_name;
            }
            if (cli_params.cli_iterations_set) {
                checkpoint_params.optimization.iterations =
                    cli_params.optimization.iterations;
            }
            checkpoint_params.optimization.headless =
                cli_params.optimization.headless;
            checkpoint_params.optimization.auto_train =
                cli_params.optimization.auto_train;
            checkpoint_params.optimization.no_splash =
                cli_params.optimization.no_splash;
            checkpoint_params.server =
                cli_params.server;
            if (!cli_params.python_scripts.empty())
                checkpoint_params.python_scripts = cli_params.python_scripts;
            checkpoint_params.resume_checkpoint.reset();
            checkpoint_params.resume_project = path;
            checkpoint_params.save_project_at_iteration =
                cli_params.save_project_at_iteration;
            checkpoint_params.save_project_path =
                cli_params.save_project_path;
            checkpoint_params.cli_iterations_set =
                cli_params.cli_iterations_set;
            checkpoint_params.cli_bg_color_set =
                cli_params.cli_bg_color_set;
            checkpoint_params.overrides = cli_params.overrides;
            core::param::apply_explicit_training_overrides(
                checkpoint_params, checkpoint_params.overrides);

            validate_train_parameters(checkpoint_params);
            auto hydration =
                recovery_document.document()
                    .hydrate(scene);
            if (!hydration) {
                return std::move(hydration)
                    .error()
                    .with_context(
                        "hydrate training project display state",
                        LFS_SOURCE_SITE_CURRENT());
            }
            if (!hydration->trainer_state_pending ||
                !hydration->checkpoint_uuid ||
                *hydration->checkpoint_uuid !=
                    bound_checkpoint_uuid ||
                !hydration->checkpoint_header) {
                return training_project_error(
                    lfs::ErrorCode::ContractViolation,
                    "Project hydration did not preserve the lazy CKPT "
                    "trainer-state barrier",
                    LFS_SOURCE_SITE_CURRENT());
            }
            if (hydration->checkpoint_header->iteration < 0) {
                return training_project_error(
                    lfs::ErrorCode::DataLoss,
                    "Project CKPT iteration is negative",
                    LFS_SOURCE_SITE_CURRENT());
            }

            return LoadedTrainingProject{
                .document =
                    std::move(recovery_document),
                .params =
                    std::move(checkpoint_params),
                .checkpoint_uuid =
                    bound_checkpoint_uuid,
                .iteration =
                    hydration
                        ->checkpoint_header
                        ->iteration,
            };
        }

    } // namespace
    int run_train(core::param::TrainingParameters params) {
        params.optimization.headless = true;
        validate_train_parameters(params);
        if (params.dataset_project) {
            auto result = adoptDatasetProject(params);
            if (!result)
                throw std::runtime_error(lfs::format_for_developer(result.error()));
            validate_train_parameters(params);
        }
        io::CacheLoader::getInstance(params.dataset.loading_params.use_cpu_memory);
        core::set_image_loader([](const core::ImageLoadParams& p) {
            return io::CacheLoader::getInstance().load_cached_image(p.path,
                                                                    {.resize_factor = p.resize_factor, .max_width = p.max_width, .cuda_stream = p.stream, .output_uint8 = p.output_uint8, .skip_blob_cache = p.skip_blob_cache});
        });
        struct LoaderGuard {
            ~LoaderGuard() {
                try {
                    io::CacheLoader::getInstance().reset_cache();
                } catch (const std::exception& e) { LOG_ERROR("Image cache cleanup failed: {}", e.what()); }
            }
        } loader_guard;
        event::CommandCenterBridge::instance().set(&training::CommandCenter::instance());
        HeadlessRunCoordinator coordinator;
        core::Scene scene;
        // The document and recovery lock must outlive the trainer's lazy checkpoint access.
        std::optional<LoadedTrainingProject> project;
        std::unique_ptr<training::Trainer> trainer;
        if (params.resume_project) {
            auto loaded = loadTrainingProject(params, scene);
            if (!loaded)
                throw std::runtime_error(lfs::format_for_developer(loaded.error()));
            project.emplace(std::move(*loaded));
            validate_scene(project->params, scene);
            std::optional<io::project::RecoverySession> recovery;
            if (const auto* r = project->document.recovery_session())
                recovery = *r;
            auto installed = training::installTrainerFromProjectCheckpoint(scene,
                                                                           project->document.document(), project->checkpoint_uuid, project->params,
                                                                           core::path_to_utf8(*params.resume_project), project->iteration, recovery);
            if (!installed)
                throw std::runtime_error(installed.error());
            trainer = std::move(installed->trainer);
            training::grant_headless_project_saves(*trainer, project->params,
                                                   headless_project_save_destination(params, *params.resume_project));
        } else {
            auto effective = params;
            if (params.resume_checkpoint) {
                auto loaded = loadCheckpointParams(params, scene);
                if (!loaded)
                    throw std::runtime_error(loaded.error());
                effective = std::move(*loaded);
            } else {
                auto loaded = training::loadTrainingDataIntoScene(effective, scene);
                if (!loaded)
                    throw std::runtime_error(loaded.error());
                auto initialized = training::initializeTrainingModel(effective, scene);
                if (!initialized)
                    throw std::runtime_error(initialized.error());
            }
            validate_train_parameters(effective);
            validate_scene(effective, scene);
            trainer = std::make_unique<training::Trainer>(scene);
            auto initialized = trainer->initialize(effective);
            if (!initialized)
                throw std::runtime_error(initialized.error());
            training::grant_headless_project_saves(*trainer, effective,
                                                   headless_dataset_project_destination(params));
            if (params.resume_checkpoint) {
                auto restored = trainer->load_checkpoint(*params.resume_checkpoint);
                if (!restored)
                    throw std::runtime_error(restored.error());
            }
            params.dataset.output_path = effective.dataset.output_path;
        }
        validate_train_parameters(trainer->getParams());
        validate_scene(trainer->getParams(), scene);
        params.dataset.output_path = trainer->get_output_path();
        core::Tensor::trim_memory_pool();
        auto trained = trainer->train(coordinator.stop_token());
        const bool exported = trained && training::export_final_splats(*trainer, params);
        // shutdown drains asynchronous saves before inspecting their error status.
        trainer->shutdown();
        const auto metrics = trainer->get_project_snapshot_metrics();
        if (!trained)
            throw std::runtime_error(lfs::format_for_developer(trained.error()));
        if (!metrics.last_writer_error.empty())
            throw std::runtime_error(metrics.last_writer_error);
        if (project) {
            auto rebound = project->document.rebind_after_durable_merge();
            if (!rebound)
                throw std::runtime_error(lfs::format_for_developer(rebound.error()));
        }
        if (!exported)
            return 1;
        return coordinator.interrupted_exit_code();
    }
} // namespace lfs::app
