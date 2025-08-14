#include "ml/registry.hpp"
#include "model.hpp"

namespace ml {
CNNMNISTImpl::CNNMNISTImpl() {
  net = torch::nn::Sequential(
    torch::nn::Conv2d(torch::nn::Conv2dOptions(1,32,3).stride(1).padding(1)),
    torch::nn::ReLU(),
    torch::nn::MaxPool2d(2),
    torch::nn::Conv2d(torch::nn::Conv2dOptions(32,64,3).stride(1).padding(1)),
    torch::nn::ReLU(),
    torch::nn::AdaptiveAvgPool2d(1),
    torch::nn::Flatten(),
    torch::nn::Linear(64,10)
  );
  register_module("net", net);
}
torch::Tensor CNNMNISTImpl::forward(torch::Tensor x) {
  return net->forward(x);
}

static ml::IModel::Ptr make_cnn(const ml::TrainConfig&) {
  return std::make_shared<CNNMNISTImpl>();
}
REGISTER_MODEL(cnn_mnist, make_cnn);

} // namespace ml