# 프로젝트 루트에서
rm -rf third_party/libtorch*                 # 깨끗이 정리
mkdir -p third_party && cd third_party

# 인텔 맥용 LibTorch (CPU) 2.1.2
LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-macos-2.1.2.zip"

# 이어받기 가능하게 다운로드 (curl)
curl -L -C - -o libtorch.zip "$LIBTORCH_URL"
unzip -q libtorch.zip

# 버전 폴더명 정리 + 심볼릭 링크 통일
mv libtorch libtorch-macos-2.1.2
ln -sfn libtorch-macos-2.1.2 libtorch

# (권장) Gatekeeper 격리 속성 제거
xattr -dr com.apple.quarantine libtorch-macos-2.1.2

# 확인: 필수 dylib가 보여야 합니다
ls -l libtorch/lib | egrep "libc10|libtorch|libcaffe2"
cd ..

