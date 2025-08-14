#pragma once
#include <torch/torch.h>
#include <torch/data/transforms/base.h>
#include <torch/data/example.h>

namespace ml {
namespace transforms {

struct EnsureChannel
  : torch::data::transforms::Transform<
        torch::data::Example<>, torch::data::Example<>> {

  torch::data::Example<> apply(torch::data::Example<> ex) override {
    // 입력 텐서가 [H,W]면 [1,H,W]로 보정 (MNIST 등 1채널 가정)
    if (ex.data.dim() == 2) {
      ex.data = ex.data.unsqueeze(0);
    }
    return ex;
  }
};

} // namespace transforms
} // namespace ml