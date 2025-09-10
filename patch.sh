set -euo pipefail

# 1) 기존 libtorch 정리 후 재다운로드 (Intel macOS, CPU)
rm -rf third_party/libtorch*                       # 깨끗이
mkdir -p third_party && cd third_party

LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-macos-2.1.2.zip"
echo "[INFO] Downloading LibTorch from $LIBTORCH_URL"
curl -L -C - -o libtorch.zip "$LIBTORCH_URL"

unzip -q libtorch.zip
mv libtorch libtorch-macos-2.1.2
ln -sfn libtorch-macos-2.1.2 libtorch

# Gatekeeper 격리 속성 제거(간혹 필요)
xattr -dr com.apple.quarantine libtorch-macos-2.1.2 || true

# 필수 파일 존재 확인
test -f libtorch/share/cmake/Torch/TorchConfig.cmake || { echo "[ERR] TorchConfig.cmake not found"; exit 1; }
test -f libtorch/lib/libc10.dylib || { echo "[ERR] libc10.dylib not found"; exit 1; }

# 아키텍처 확인(반드시 x86_64)
file libtorch/lib/libc10.dylib | grep -q "x86_64" || { echo "[ERR] Not x86_64 LibTorch"; exit 1; }

cd ..

# 2) 환경변수(한 세션용). 반복사용이면 ~/.zshrc에 넣으세요.
export CMAKE_PREFIX_PATH="$(pwd)/third_party/libtorch:${CMAKE_PREFIX_PATH:-}"
if command -v brew >/dev/null 2>&1; then
  export DYLD_LIBRARY_PATH="$(brew --prefix libomp)/lib:$(pwd)/third_party/libtorch/lib:${DYLD_LIBRARY_PATH:-}"
else
  export DYLD_LIBRARY_PATH="$(pwd)/third_party/libtorch/lib:${DYLD_LIBRARY_PATH:-}"
fi

# 3) 클린 빌드 (인텔 맥이므로 x86_64로 고정)
rm -rf build
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=x86_64 ..
cmake --build . -j"$(sysctl -n hw.ncpu)"

# 4) 링크 확인 (참고)
otool -L ./ml_engine | egrep "libomp|libtorch|libc10" || true
