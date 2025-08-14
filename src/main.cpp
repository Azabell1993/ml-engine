#include "engine/engine.hpp"
#include "ml/registry.hpp"
#include "ml/trainer.hpp"
#include "log.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <iostream>
#include <torch/torch.h> // MNIST 직접 생성용

int main(int argc, char** argv) {
#ifdef _DEBUG
  start_log_thread();
  atexit(stop_log_thread);
#endif

  // 모드: 1) 서버  2) CLI 학습
  if (argc>1 && std::string(argv[1])=="train-cli") {
    std::string name = (argc>2)? argv[2] : "cnn_mnist";
    ml::TrainConfig cfg; cfg.ckpt_dir = "./runs/" + name;
    try {
      auto model = ml::Registry::get().create(name, cfg);

      // MNIST 데이터셋 직접 생성
      auto train = torch::data::datasets::MNIST(cfg.dataset_root, torch::data::datasets::MNIST::Mode::kTrain)
                      .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                      .map(torch::data::transforms::Stack<>());
      auto valid = torch::data::datasets::MNIST(cfg.dataset_root, torch::data::datasets::MNIST::Mode::kTest)
                      .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                      .map(torch::data::transforms::Stack<>());

      ml::Trainer::fit(model, cfg, train, valid);
      SQLOG_I("CLI training finished: %s", name.c_str());
    } catch (const std::exception& e) {
      SQLOG_E("CLI training failed: %s", e.what());
      return 1;
    }
    return 0;
  }

  // 서버 모드
  auto eng = engine::Engine::create();
  if (eng->loadConfig("./config/engine-config.json") != engine::EngineState::Success) return 2;
  if (eng->init() != engine::EngineState::Success) return 3;
  auto ret = eng->run();
  eng->release();
  return ret==engine::EngineState::Success ? 0 : 4;
}