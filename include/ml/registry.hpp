#pragma once
#include "ml/model_base.hpp"
#include <functional>
#include <unordered_map>
#include <vector>

namespace ml {

class Registry {
public:
  using Factory = std::function<IModel::Ptr(const TrainConfig&)>;
  static Registry& get() { static Registry r; return r; }
  void add(const std::string& name, Factory f) { map_[name]=std::move(f); }
  IModel::Ptr create(const std::string& name, const TrainConfig& cfg) const;
  std::vector<std::string> names() const;
private:
  std::unordered_map<std::string, Factory> map_;
};

#define REGISTER_MODEL(NAME, FACTORY) \
  static bool _reg_##NAME = [](){ \
    ::ml::Registry::get().add(#NAME, FACTORY); \
    return true; \
  }()

} // namespace ml