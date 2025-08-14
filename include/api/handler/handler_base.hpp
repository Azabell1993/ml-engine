#pragma once
#include <crow.h>
#include <string>
#include <nlohmann/json.hpp>

namespace handler {
enum class StatusCode { _200=200, _201=201, _400=400, _403=403, _404=404, _500=500 };

inline crow::response jsonResp(StatusCode code, const nlohmann::json& j) {
  crow::response r{(int)code};
  r.set_header("Content-Type", "application/json");
  r.body = j.dump(2);
  return r;
}
inline crow::response msg(StatusCode code, const std::string& m) {
  return jsonResp(code, {{"message", m}});
}
} // namespace handler