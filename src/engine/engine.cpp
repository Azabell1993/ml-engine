#include "engine/engine.hpp"
#include "api/api_server.hpp"
#include "log.h"
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>

static std::atomic<bool> g_exit{false};
static void onSignal(int s){ SQLOG_W("signal %d", s); g_exit=true; }

namespace engine {

std::unique_ptr<Engine> Engine::create(){ return std::make_unique<Engine>(); }
Engine::~Engine() = default;

EngineState Engine::loadConfig(const std::string& path) {
  // 최소 동작: JSON 읽어 api_port 설정 (실사용은 utils::loadEngineConfig 대체)
  // 여기서는 디폴트 유지
  (void)path;
  return EngineState::Success;
}

EngineState Engine::init() {
  api_ = std::make_unique<ApiServer>("0.0.0.0", api_port_);
  api_->init();
  return EngineState::Success;
}

EngineState Engine::run() const {
  std::signal(SIGINT, onSignal);
  std::thread t([this](){ api_->start(); });

  while(!g_exit) std::this_thread::sleep_for(std::chrono::milliseconds(200));

  if (api_) api_->stop();
  if (t.joinable()) t.join();
  return EngineState::Success;
}

EngineState Engine::release() {
  api_.reset();
  return EngineState::Success;
}

} // namespace engine