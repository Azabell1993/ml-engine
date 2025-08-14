#pragma once
#include "ml/model_base.hpp"
#include <torch/torch.h>
#include <utility>

namespace ml {

    struct TrainConfig;

struct DatasetFactory {
    using Mnist = torch::data::datasets::MNIST;

    static std::pair<Mnist, Mnist>  make_mnist(const TrainConfig& cfg);
};
} // namespace ml