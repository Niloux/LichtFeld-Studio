// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "core/parameters.hpp"
namespace lfs::app {
    void validate_train_parameters(const core::param::TrainingParameters& params);
    int run_train(core::param::TrainingParameters params);
} // namespace lfs::app
