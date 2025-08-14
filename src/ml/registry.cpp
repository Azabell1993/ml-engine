#include "ml/registry.hpp"
#include <stdexcept>

namespace ml {
IModel::Ptr Registry::create(const std::string& name, const TrainConfig& cfg) const {
  auto it = map_.find(name);
  if (it==map_.end()) throw std::runtime_error("Model not found: " + name);
  return (it->second)(cfg);
}
std::vector<std::string> Registry::names() const {
  std::vector<std::string> v; v.reserve(map_.size());
  for (auto& kv: map_) v.push_back(kv.first);
  return v;
}
} // namespace ml