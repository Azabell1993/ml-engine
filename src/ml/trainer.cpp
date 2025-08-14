#include "ml/trainer.hpp"
#include "log.h"
#include <filesystem>

namespace ml {
void Trainer::fit(IModel::Ptr model, const TrainConfig& cfg,
                  torch::data::datasets::Dataset<>& train,
                  torch::data::datasets::Dataset<>& valid) {
  torch::Device dev((cfg.device=="cuda" && torch::cuda::is_available()) ? torch::kCUDA : torch::kCPU);
  torch::manual_seed(cfg.seed);
  model->to(dev);

  auto train_loader = torch::data::make_data_loader(
    train, torch::data::DataLoaderOptions().batch_size(cfg.batch_size).workers(cfg.num_workers).drop_last(true));
  auto val_loader = torch::data::make_data_loader(
    valid, torch::data::DataLoaderOptions().batch_size(cfg.batch_size).workers(cfg.num_workers));

  torch::optim::AdamW opt(model->parameters(), torch::optim::AdamWOptions(cfg.lr));
  std::filesystem::create_directories(cfg.ckpt_dir);

  for (int64_t epoch=1; epoch<=cfg.epochs; ++epoch) {
    model->train();
    double loss_sum=0.0; size_t steps=0;
    for (auto& batch : *train_loader) {
      auto data = batch.data.to(dev);
      auto target = batch.target.to(dev);
      auto logits = model->forward(data);
      auto loss = torch::nn::functional::cross_entropy(logits, target);
      loss.backward();
      opt.step();
      opt.zero_grad();
      loss_sum += loss.item<double>(); ++steps;
    }

    // eval
    model->eval();
    torch::NoGradGuard ng;
    size_t correct=0, total=0;
    for (auto& batch: *val_loader) {
      auto data = batch.data.to(dev);
      auto target = batch.target.to(dev);
      auto logits = model->forward(data);
      auto pred = logits.argmax(1);
      correct += (pred==target).sum().item<int64_t>();
      total   += target.size(0);
    }
    double acc = total ? (double)correct/total : 0.0;
    SQLOG_I("[epoch %lld] loss=%.5f val_acc=%.4f", (long long)epoch, loss_sum/std::max<size_t>(1,steps), acc);

    // save
    const auto path = cfg.ckpt_dir + "/epoch_" + std::to_string(epoch) + ".pt";
    torch::save(model, path);
  }
}
} // namespace ml