#pragma once
namespace engine {
enum class EngineState {
  Success,
  EngineConfigLoadFailed,
  EngineInitFailed,
  EngineLicenseCheckFailed
};
}