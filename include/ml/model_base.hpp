#pragma once
#include <torch/torch.h>
#include <string>
#include <memory>

namespace ml {

struct TrainConfig {
  int64_t epochs{3};
  int64_t batch_size{64};
  double  lr{1e-3};
  int64_t num_workers{2};
  bool    amp{false};
  std::string device{"cpu"};      // "cuda"|"cpu"
  std::string ckpt_dir{"./runs"}; // runs/<model>
  std::string dataset_root{"./data/mnist"};
  int64_t seed{42};
};

class IModel : public torch::nn::Module {
public:
  using Ptr = std::shared_ptr<IModel>;
  virtual ~IModel() = default;
  virtual torch::Tensor forward(torch::Tensor x) = 0;
};

} // namespace ml