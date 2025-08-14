#pragma once
#include "engine/engine_state.hpp"
#include <memory>
#include <string>

namespace engine {
class ApiServer;

class Engine {
public:
  static std::unique_ptr<Engine> create();
  ~Engine();
  EngineState loadConfig(const std::string& path);
  EngineState init();
  EngineState run() const;
  EngineState release();

private:
  std::unique_ptr<ApiServer> api_;
  int api_port_{18080};
};
} // namespace engine