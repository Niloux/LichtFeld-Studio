// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/argument_parser.hpp"
#include "core/parameters.hpp"
#include "core/evaluation_sampling.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
    void require(bool condition, const char* message) {
        if (!condition)
            throw std::runtime_error(message);
    }

    auto parse(const std::vector<std::string>& options) {
        std::vector<const char*> argv{"lfs-train", "--headless"};
        for (const auto& option : options)
            argv.push_back(option.c_str());
        return lfs::core::args::parse_args_and_params(static_cast<int>(argv.size()), argv.data());
    }
} // namespace

int main(int argc, char** argv) {
    const auto root = std::filesystem::temp_directory_path() /
                      ("lfs-config-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root);
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() { std::filesystem::remove_all(path); }
    } cleanup{root};
    try {
        const std::vector<int> stereo_ids{1, 2, 1, 2, 1, 2, 1, 2, 1, 2};
        require(lfs::core::sample_training_views(stereo_ids, 2) == std::vector<size_t>{0, 1, 4, 5, 8, 9},
                "interleaved cameras are sampled independently");
        require(lfs::core::sample_training_views({1, 1, 2}, 8) == std::vector<size_t>{0, 2},
                "short camera sequences each contribute a view");
        require(lfs::core::sample_training_views({}, 8).empty(), "empty sampling");
        require(lfs::core::sample_training_views({1, 2, 1}, 1) == std::vector<size_t>{0, 1, 2}, "stride one evaluates all views");
        auto legacy_dataset = lfs::core::param::DatasetConfig{}.to_json();
        legacy_dataset.erase("use_test_split");
        require(lfs::core::param::DatasetConfig::from_json(legacy_dataset).use_test_split, "old configs retain held-out evaluation");
        const auto config_path = root / "train.json";
        const auto init = root / "lidar.ply";
        std::ofstream(init) << "ply\n";
        nlohmann::json config{
            {"dataset", {{"data_path", root.string()}, {"output_path", (root / "output").string()}, {"resize_factor", 4}, {"max_width", 0}, {"test_every", 5}, {"loading_params", {{"use_cpu_memory", false}}}}},
            {"init_path", init.string()},
            {"export_formats", {"ply"}},
            {"optimization", {{"strategy", "mrnf"}, {"iterations", 30000}, {"gut", true}, {"sh_degree", 0}, {"use_exposure_correction", true}, {"sky_enabled", true}, {"sky_num_points", 100000}, {"enable_eval", true}, {"eval_steps", {3000, 7000, 30000}}}}};
        auto full_optimization = lfs::core::param::OptimizationParameters::mrnf_defaults().to_json();
        config["dataset"]["use_test_split"] = false;
        full_optimization.update(config.at("optimization"));
        config["optimization"] = full_optimization;
        auto save = [&]() { std::ofstream(config_path) << config.dump(2); };
        save();
        auto parsed = parse({"--config", config_path.string()});
        if (!parsed)
            throw std::runtime_error(parsed.error());
        const auto& p = **parsed;
        require(p.dataset.data_path == root && p.dataset.output_path == root / "output", "config paths");
        require(p.dataset.output_path_explicit && p.cli_iterations_set, "explicit config intent");
        require(p.dataset.resize_factor == 4 && p.dataset.max_width == 0 && p.dataset.test_every == 5, "config dataset");
        require(!p.dataset.loading_params.use_cpu_memory, "config loading parameters");
        require(!p.dataset.use_test_split, "training-view evaluation parsed");
        require(!lfs::core::param::DatasetConfig::from_json(p.dataset.to_json()).use_test_split, "training-view evaluation round trip");
        require(p.init_path == init.string() && p.export_formats == std::vector{lfs::core::param::OutputFormat::PLY}, "config init/export");
        require(p.optimization.gut && p.optimization.sky_enabled && p.optimization.sky_num_points == 100000, "config sky");
        require(p.optimization.iterations == 30000 && p.optimization.max_cap == 5000000, "strategy defaults");
        require(p.optimization.eval_steps == std::vector<size_t>{3000, 7000, 30000}, "config eval steps");

        parsed = parse({"--config", config_path.string(), "--iter", "3000", "-r", "2", "--max-width", "1600",
                        "-d", (root / "other-data").string(), "-o", (root / "override").string(), "--test-every", "8"});
        if (!parsed)
            throw std::runtime_error(parsed.error());
        auto& override = **parsed;
        require(override.optimization.iterations == 3000 && override.dataset.resize_factor == 2 && override.dataset.max_width == 1600, "CLI overrides config");
        lfs::core::param::apply_explicit_training_overrides(override, override.overrides);
        require(!override.dataset.use_test_split, "resume preserves explicit evaluation mode");
        require(override.dataset.data_path == root / "other-data" && override.dataset.output_path == root / "override" && override.dataset.test_every == 8, "resume overlays preserve CLI paths");
        require(!parse({"--config", config_path.string(), "--strategy", "mcmc"}), "strategy conflict rejected");

        config["init_path"] = (root / "missing.ply").string();
        config["export_formats"] = {"invalid"};
        save();
        require(!parse({"--config", config_path.string()}), "invalid launch fields rejected");
        require(parse({"--config", config_path.string(), "--init", init.string(), "--export", "ply"}).has_value(), "CLI init/export replacement");

        config = lfs::core::param::OptimizationParameters::mrnf_defaults().to_json();
        config["iterations"] = 12000;
        save();
        parsed = parse({"--config", config_path.string(), "-d", root.string(), "-o", (root / "legacy").string()});
        require(parsed && (*parsed)->optimization.iterations == 12000 && (*parsed)->dataset.resize_factor == 1, "legacy flat config");
        config.erase("means_lr");
        save();
        require(!parse({"--config", config_path.string(), "-d", root.string(), "-o", (root / "incomplete").string()}), "incomplete optimization snapshot rejected");
        parsed = parse({"-d", root.string(), "-o", (root / "defaults").string()});
        require(parsed && (*parsed)->dataset.resize_factor == 1 && !(*parsed)->cli_iterations_set, "CLI-only defaults");
        config = {{"dataset", false}};
        save();
        require(!parse({"--config", config_path.string()}), "invalid dataset type rejected");
        std::ofstream(config_path) << "{broken";
        require(!parse({"--config", config_path.string()}), "malformed JSON rejected");
        require(!parse({"--config", (root / "missing.json").string()}), "missing config rejected");
        if (argc > 1) {
            std::ifstream preset(argv[1]);
            config = nlohmann::json::parse(preset);
            config["dataset"]["data_path"] = root.string();
            config["dataset"]["output_folder"] = (root / "preset").string();
            config["init_path"] = init.string();
            save();
            parsed = parse({"--config", config_path.string()});
            if (!parsed)
                throw std::runtime_error(parsed.error());
            require((*parsed)->optimization.sky_enabled && (*parsed)->optimization.use_exposure_correction &&
                        (*parsed)->optimization.sh_degree == 0 && (*parsed)->dataset.resize_factor == 4,
                    "shipped contextcapture preset");
        }
        std::cout << "Training config tests passed (CPU only)\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
