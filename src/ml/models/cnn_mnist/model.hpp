#pragma once
#include "ml/model_base.hpp"

namespace ml {
struct CNNMNISTImpl : public IModel {
  torch::nn::Sequential net;
  CNNMNISTImpl();
  torch::Tensor forward(torch::Tensor x) override;
};
TORCH_MODULE(CNNMNIST);
} // namespace ml