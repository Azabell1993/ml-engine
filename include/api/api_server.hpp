#pragma once
#include <crow.h>
#include <functional>
#include <memory>
#include <string>

namespace engine {

class ApiServer {
public:
  using Handler = std::function<crow::response(const crow::request&)>;

  ApiServer(const std::string& ip, int port);
  void init();
  void start();
  void stop();

private:
  void registerRoutes();
  void addRoute(const std::string& path, crow::HTTPMethod method, Handler h);

  // --- 라우트 핸들러 ---
  crow::response health(const crow::request&);
  crow::response listModels(const crow::request&);
  crow::response trainAll(const crow::request&);
  crow::response trainOne(const crow::request&);
  crow::response llmGenerate(const crow::request&);

private:
  crow::SimpleApp app_;
  std::string ip_;
  int port_;
};

} // namespace engine