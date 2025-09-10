#!/bin/bash
if [ -d build ]; then
  rm -rf build
fi
mkdir build && cd build

# 가장 안전: CMAKE_PREFIX_PATH도 같이 넘기기
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=../third_party/libtorch \
      -DTorch_DIR=../third_party/libtorch/share/cmake/Torch \
      -DCMAKE_OSX_ARCHITECTURES=x86_64 \
      ..

cmake --build . -j
