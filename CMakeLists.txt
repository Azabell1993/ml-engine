cmake_minimum_required(VERSION 3.22)
project(ml_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OSX_ARCHITECTURES arm64)
set(CMAKE_BUILD_TYPE Release)

# LibTorch
set(Torch_DIR "${CMAKE_SOURCE_DIR}/third_party/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

# Threads (Crow/기본 런타임에 필요)
find_package(Threads REQUIRED)

# (선택) OpenMP
find_package(OpenMP QUIET)

# 소스들
set(SRCS
  src/main.cpp
  src/api/api_server.cpp
  src/api/handler/handler_base.cpp
  src/engine/engine.cpp
  src/ml/registry.cpp
#   src/ml/trainer.cpp
  src/ml/dataset.cpp
  src/ml/models/cnn_mnist/model.cpp
  src/llm/llm_engine.cpp
)

# 타깃 생성 — <<< 이 줄이 먼저 와야 합니다
add_executable(ml_engine ${SRCS})

# 포함 경로
target_include_directories(ml_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/third_party/libtorch/include
  ${CMAKE_SOURCE_DIR}/third_party/libtorch/include/torch/csrc/api/include
)

# 링크
target_link_libraries(ml_engine PRIVATE
  ${TORCH_LIBRARIES}
  Threads::Threads
)

# if (OpenMP_CXX_FOUND)
#   target_link_libraries(ml_engine PRIVATE OpenMP::OpenMP_CXX)
# endif()

# 최적화
if (CMAKE_BUILD_TYPE MATCHES "Release")
  target_compile_options(ml_engine PRIVATE -O3)
endif()

# macOS rpath (실행파일 기준: ../third_party/libtorch/lib)
set_target_properties(ml_engine PROPERTIES
  MACOSX_RPATH ON
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH "@loader_path/../third_party/libtorch/lib"
  INSTALL_RPATH "@loader_path/../third_party/libtorch/lib"
)