#include "ml/dataset.hpp"
#include <torch/torch.h>
#include "ml/model_base.hpp"     // TrainConfig 정의 포함
#include <string>

namespace ml {

std::pair<DatasetFactory::Mnist, DatasetFactory::Mnist>
DatasetFactory::make_mnist(const TrainConfig& cfg) {
  std::string root = cfg.dataset_root.empty() ? "./data/mnist" : cfg.dataset_root;
  DatasetFactory::Mnist train(root, DatasetFactory::Mnist::Mode::kTrain);
  DatasetFactory::Mnist test (root, DatasetFactory::Mnist::Mode::kTest);
  return { std::move(train), std::move(test) };
}

} // namespace ml