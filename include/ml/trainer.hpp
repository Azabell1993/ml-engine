#pragma once
#include "ml/model_base.hpp"
#include <torch/torch.h>
#include <filesystem>
#include "log.h"

namespace ml {

class Trainer {
public:
  template <typename TrainDataset, typename ValidDataset>
  static void fit(IModel::Ptr model, const TrainConfig& cfg,
                  TrainDataset& train, ValidDataset& valid) {
    torch::Device dev((cfg.device == "cuda" && torch::cuda::is_available()) ? torch::kCUDA : torch::kCPU);
    torch::manual_seed(cfg.seed);
    model->to(dev);

    auto train_loader = torch::data::make_data_loader(
        train, torch::data::DataLoaderOptions()
                   .batch_size(cfg.batch_size)
                   .workers(cfg.num_workers)
                   .drop_last(true));

    auto val_loader = torch::data::make_data_loader(
        valid, torch::data::DataLoaderOptions()
                   .batch_size(cfg.batch_size)
                   .workers(cfg.num_workers));

    torch::optim::AdamW opt(model->parameters(), torch::optim::AdamWOptions(cfg.lr));
    std::filesystem::create_directories(cfg.ckpt_dir);

    for (int64_t epoch = 1; epoch <= cfg.epochs; ++epoch) {
      model->train();
      double loss_sum = 0.0;
      size_t steps = 0;

      for (auto& batch : *train_loader) {
        // --- 입력 전처리 ---
        auto data   = batch.data.to(dev, /*dtype=*/torch::kFloat32).div_(255.0);
        if (data.dim() == 3) data = data.unsqueeze(0);       // {1,28,28} -> {1,1,28,28}
        if (data.dim() == 3) data = data.unsqueeze(1);       // 안전망 (개별 샘플 처리 시)
        if (data.dim() == 4 && data.size(1) != 1) {
          // 혹시 채널이 없거나 다른 경우를 강제 1채널로 맞춤 (필요 시 주석 처리)
          data = data.narrow(/*dim=*/1, /*start=*/0, /*length=*/1);
        }
        auto target = batch.target.to(dev);

        // --- 학습 스텝 ---
        opt.zero_grad(true);
        auto logits = model->forward(data);
        auto loss   = torch::nn::functional::cross_entropy(logits, target);
        loss.backward();
        opt.step();

        // 의존 템플릿 호출
        loss_sum += loss.template item<double>();
        ++steps;
      }

      // --- 검증 ---
      model->eval();
      torch::NoGradGuard ng;
      size_t correct = 0, total = 0;

      for (auto& batch : *val_loader) {
        auto data   = batch.data.to(dev, torch::kFloat32).div_(255.0);
        if (data.dim() == 3) data = data.unsqueeze(0);
        if (data.dim() == 3) data = data.unsqueeze(1);
        if (data.dim() == 4 && data.size(1) != 1) {
          data = data.narrow(1, 0, 1);
        }
        auto target = batch.target.to(dev);

        auto logits = model->forward(data);
        auto pred   = logits.argmax(1);
        correct += (pred == target).sum().template item<int64_t>();
        total   += target.size(0);
      }

      const double mean_loss = loss_sum / std::max<size_t>(1, steps);
      const double acc = total ? static_cast<double>(correct) / total : 0.0;
      SQLOG_I("[epoch %lld] loss=%.5f val_acc=%.4f", (long long)epoch, mean_loss, acc);

      const auto path = cfg.ckpt_dir + "/epoch_" + std::to_string(epoch) + ".pt";
      torch::save(model, path);
    }
  }
};

} // namespace ml