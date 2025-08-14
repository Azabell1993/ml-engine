#include "api/api_server.hpp"
#include "api/handler/handler_base.hpp"
#include "ml/registry.hpp"
#include "ml/trainer.hpp"
#include "log.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <torch/torch.h> // MNIST 직접 생성용
#include "llm/llm_engine.hpp"
#include "ml/transforms/ensure_channel.hpp"

using nlohmann::json;

namespace engine {

ApiServer::ApiServer(const std::string& ip, int port) : ip_(ip), port_(port) {}

void ApiServer::init() {
  registerRoutes();
  SQLOG_I("Crow API initialized at %s:%d", ip_.c_str(), port_);
}

void ApiServer::start() {
  app_.signal_clear();
  app_.port(port_).multithreaded().run();
}

void ApiServer::stop() { app_.stop(); }

void ApiServer::addRoute(const std::string& path, crow::HTTPMethod method, Handler h) {
  app_.route_dynamic(path).methods(method)([h](const crow::request& req){ return h(req); });
  SQLOG_I("Route: [%d] %s", (int)method, path.c_str());
}

void ApiServer::registerRoutes() {
  addRoute("/health", crow::HTTPMethod::Get, [this](auto& r){ return health(r); });

  // ML
  addRoute("/ml/models", crow::HTTPMethod::Get, [this](auto& r){ return listModels(r); });
  addRoute("/ml/train-all", crow::HTTPMethod::Post, [this](auto& r){ return trainAll(r); });
  addRoute("/ml/train", crow::HTTPMethod::Post, [this](auto& r){ return trainOne(r); });

  // LLM
  addRoute("/llm/generate", crow::HTTPMethod::Post, [this](auto& r){ return llmGenerate(r); });
}

crow::response ApiServer::llmGenerate(const crow::request& req) {
  auto body = json::parse(req.body, nullptr, false);
  if (body.is_discarded()) return handler::msg(handler::StatusCode::_400, "invalid json");

  llm::Params p;
  p.n_threads       = body.value("n_threads", 4);
  p.n_ctx           = body.value("n_ctx", 512);
  p.temperature     = body.value("temperature", 0.8);
  p.top_k           = body.value("top_k", 40);
  p.top_p           = body.value("top_p", 0.95);
  p.backend         = body.value("backend", "mock");       // "mock" or "llama"
  p.llama_exec_path = body.value("llama_exec_path", "");
  if (body.contains("extra_args") && body["extra_args"].is_array()) {
    for (auto& a : body["extra_args"]) p.extra_args.push_back(a.get<std::string>());
  }

  std::string prompt = body.value("prompt", "");
  if (prompt.empty()) return handler::msg(handler::StatusCode::_400, "missing 'prompt'");

  auto& eng = llm::Engine::instance();
  std::string out = eng.generate(prompt, p);

  return handler::jsonResp(handler::StatusCode::_200, {
    {"prompt", prompt},
    {"output", out},
    {"engine", p.backend},
    {"params", {
      {"n_threads", p.n_threads},
      {"n_ctx",     p.n_ctx},
      {"temperature", p.temperature},
      {"top_k", p.top_k},
      {"top_p", p.top_p},
      {"extra_args", p.extra_args}
    }}
  });
}

crow::response ApiServer::health(const crow::request&) {
  return handler::jsonResp(handler::StatusCode::_200, {{"status","ok"},{"service","ml-engine"}});
}

crow::response ApiServer::listModels(const crow::request&) {
  json j;
  j["models"] = ml::Registry::get().names();
  return handler::jsonResp(handler::StatusCode::_200, j);
}

static ml::TrainConfig readTrainCfg(const json& j) {
  ml::TrainConfig c;
  if (j.contains("epochs"))       c.epochs       = j["epochs"];
  if (j.contains("batch_size"))   c.batch_size   = j["batch_size"];
  if (j.contains("lr"))           c.lr           = j["lr"];
  if (j.contains("device"))       c.device       = j["device"];
  if (j.contains("dataset_root")) c.dataset_root = j["dataset_root"];
  if (j.contains("ckpt_dir"))     c.ckpt_dir     = j["ckpt_dir"];
  if (j.contains("seed"))         c.seed         = j["seed"];
  return c;
}

crow::response ApiServer::trainAll(const crow::request& req) {
  json body = json::parse(req.body, nullptr, false);
  if (body.is_discarded()) return handler::msg(handler::StatusCode::_400, "invalid json");

  auto names = ml::Registry::get().names();
  json result = json::array();

  for (auto& name : names) {
    auto cfg = readTrainCfg(body);
    if (cfg.dataset_root.empty()) cfg.dataset_root = "./data/mnist";
    cfg.ckpt_dir = "./runs/" + name;
    try {
      auto model = ml::Registry::get().create(name, cfg);

      auto train = torch::data::datasets::MNIST(cfg.dataset_root, torch::data::datasets::MNIST::Mode::kTrain)
                      .map(ml::transforms::EnsureChannel{})                                          // (1,28,28) 보장
                      .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))    // 정규화
                      .map(torch::data::transforms::Stack<>());                      // 배치 스택

      auto valid = torch::data::datasets::MNIST(cfg.dataset_root, torch::data::datasets::MNIST::Mode::kTest)
                      .map(ml::transforms::EnsureChannel{})
                      .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                      .map(torch::data::transforms::Stack<>());

      ml::Trainer::fit(model, cfg, train, valid);
      result.push_back({{"model",name},{"status","ok"}});
    } catch (const std::exception& e) {
      result.push_back({{"model",name},{"status","fail"},{"error",e.what()}});
    }
  }
  return handler::jsonResp(handler::StatusCode::_200, {{"results", result}});
}

crow::response ApiServer::trainOne(const crow::request& req) {
  json body = json::parse(req.body, nullptr, false);
  if (body.is_discarded()) return handler::msg(handler::StatusCode::_400, "invalid json");

  std::string name = body.value("model", "");
  if (name.empty()) return handler::msg(handler::StatusCode::_400, "missing 'model'");

  auto cfg = readTrainCfg(body);
  if (cfg.dataset_root.empty()) cfg.dataset_root = "./data/mnist";
  cfg.ckpt_dir = "./runs/" + name;

  try {
    auto model = ml::Registry::get().create(name, cfg);

    auto train = torch::data::datasets::MNIST(cfg.dataset_root, torch::data::datasets::MNIST::Mode::kTrain)
                   .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                   .map(torch::data::transforms::Stack<>());
    auto valid = torch::data::datasets::MNIST(cfg.dataset_root, torch::data::datasets::MNIST::Mode::kTest)
                   .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                   .map(torch::data::transforms::Stack<>());

    ml::Trainer::fit(model, cfg, train, valid);
    return handler::jsonResp(handler::StatusCode::_200, {{"model",name},{"status","ok"}});
  } catch (const std::exception& e) {
    return handler::jsonResp(handler::StatusCode::_500, {{"model",name},{"status","fail"},{"error",e.what()}});
  }
}

} // namespace ml