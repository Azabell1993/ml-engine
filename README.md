# ml-engine
### C++/LibTorch Training & llama.cpp Inference for macOS M2


[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS%20M2%20%7C%20Linux%20%7C%20CPU%2FGPU-blue.svg)]()
[![Language](https://img.shields.io/badge/language-C%2B%2B%20%7C%20LibTorch%20%7C%20llama.cpp-orange.svg)]()

<p align="right">
<a href="#13-부록-빠른-시작" style="font-weight:bold;background:#e0f7fa;border-radius:8px;padding:6px 16px;text-decoration:none;">🚀 Quick Start 바로가기</a>
</p>

---

## 개요

**ml-engine**은 C++/LibTorch 기반의 모델(Module)을 직접 빌드해 데이터셋의 loss를 최소화하는 학습 엔진이며, 동시에 외부 실행(예: llama.cpp) 을 통해 LLM 추론을 REST로 오케스트레이션하는 서버/CLI입니다.

- 학습 대상: `src/ml/models/...` 아래 LibTorch C++ Module 클래스만
- 검증 데이터셋: MNIST (28×28, 흑백, 0–9)
- 추론: llama.cpp 바이너리를 서브프로세스로 호출하여 GGUF 모델 실행
- macOS(Apple Silicon): LibTorch Metal 학습 미지원 → CPU 학습, llama.cpp는 Metal 가속(-ngl 99) 가능
- LLM 파인튜닝은 PyTorch/Hugging Face + (Q)LoRA 또는 llama.cpp의 LoRA/QLoRA 경로를 사용
- thirdparty의 libtorch는 본 프로그램 데모의 경우 `libtorch-macos-arm64-2.8.0.zip`를 사용
- license 
---


## 목차

1. [특징](#1-특징)
2. [환경 및 기술 개요](#2-환경-및-기술-개요)
3. [프로젝트 구조](#3-프로젝트-구조)
4. [빌드 & 실행](#4-빌드--실행)
5. [REST API](#5-rest-api)
6. [학습/추론 예시](#6-학습추론-예시)
7. [모델 추천 & 양자화 가이드](#7-모델-추천--양자화-가이드)
8. [운영 체크리스트](#8-운영-체크리스트)
9. [로드맵](#9-로드맵)
10. [자주 발생하는 이슈](#10-자주-발생하는-이슈)
11. [용어 사전](#11-용어-사전)
12. [라이선스](#12-라이선스)
13. [부록: 빠른 시작](#13-부록-빠른-시작)
14. [DeepSeek-Coder-V2-Lite Instruct 선택 및 실전 로그](#14-deepseek-coder-v2-lite-instruct-선택-및-실전-로그)
15. [부록: DeepSeek-Coder-V2-Lite Instruct GGUF 메타데이터 및 실행 로그 해설](#15-부록-deepseek-coder-v2-lite-instruct-gguf-메타데이터-및-실행-로그-해설)
16. [부록: 서버 기동 및 라이선스 체크 동작](#16-부록-서버-기동-및-라이선스-체크-동작)

---

## 1. 특징

- C++/LibTorch 학습 엔진
- `src/ml/models/...`의 Module만 학습 가능 (예: cnn_mnist)
- seed 고정, DataLoader shuffle 고정, cuDNN deterministic 제어(해당 환경)
- stdout + 파일 로그 기본, JSONL/TensorBoard/wandb C++로 확장 가능
- llama.cpp 연동 LLM 추론 (서브프로세스, GGUF 모델)
- macOS Metal 가속(빌드 시 -DGGML_METAL=ON, 실행 시 -ngl 99)
- 플러그인화(선택): .so/.dll로 빌드 후 런타임 교체/추가(초기엔 정적 링크로도 충분)
- 분산/멀티GPU(확장): torch::distributed + DDP로 확장 가능

---

## 2. 환경 및 기술 개요

- CUDA(NVIDIA): LibTorch CUDA 빌드 설치 시 GPU 학습 가능
- ROCm(AMD): PyTorch 일부 버전 지원
- CPU: GPU 없거나 macOS Metal 학습 미지원 시 CPU 학습/추론
- macOS(Apple Silicon): LibTorch Metal 학습 미지원, llama.cpp는 Metal 가속(-ngl 99) 가능
- AMP: torch::autocast 사용, 메모리 절약/속도 향상
- 데이터셋: MNIST, ImageFolderDataset, CSV/Parquet 커스텀 Dataset
- 로깅: 콘솔/파일 기본, JSONL/TensorBoard/wandb 확장 가능

---

## 3. 프로젝트 구조

```text
ml-engine/
├─ CMakeLists.txt
├─ third_party/
│  └─ libtorch/
├─ include/
│  ├─ log.h
│  ├─ safe_arithmetic_ops.h
│  ├─ api/
│  │  ├─ api_server.hpp
│  │  └─ handler/handler_base.hpp
│  ├─ engine/
│  │  ├─ engine.hpp
│  │  └─ engine_state.hpp
│  └─ ml/
│     ├─ model_base.hpp
│     ├─ registry.hpp
│     ├─ trainer.hpp
│     └─ dataset.hpp
├─ src/
│  ├─ main.cpp
│  ├─ api/
│  │  ├─ api_server.cpp
│  │  └─ handler/handler_base.cpp
│  ├─ engine/
│  │  └─ engine.cpp
│  └─ ml/
│     ├─ registry.cpp
│     ├─ trainer.cpp
│     ├─ dataset.cpp
│     └─ models/
│        └─ cnn_mnist/
│           ├─ model.hpp
│           └─ model.cpp
├─ config/
│  ├─ engine-config.json
├─ php/
│  ├─ index.php
│  └─ .env.php.dist
└─ README.md
```

---

## 4. 빌드 & 실행

### 의존성
- CMake 3.20+
- Clang/GCC (macOS AppleClang OK)
- LibTorch (CPU or CUDA 빌드와 일치)
- (macOS) brew install libomp

### 빌드
```sh
bash ./build.sh
# [100%] Built target ml_engine
```

### 런타임 환경
```sh
export DYLD_LIBRARY_PATH="$(brew --prefix libomp)/lib:third_party/libtorch/lib:${DYLD_LIBRARY_PATH:-}"
```

### CLI 학습 (MNIST)
```sh
build/ml_engine train-cli cnn_mnist
# [epoch 1] loss=1.37270 val_acc=0.7693
# [epoch 2] loss=0.65143 val_acc=0.8713
# [epoch 3] loss=0.44448 val_acc=0.9037
# CLI training finished: cnn_mnist
```

### 체크포인트 확인
```sh
ls -lh runs/cnn_mnist
# epoch_1.pt  epoch_2.pt  epoch_3.pt  (각 ~85KB)
```

---

## 5. REST API

### 5.1 서버 기동

```bash
# 서버 모드 (인자 없음)
./build/ml_engine

# 기본 설정 파일 위치
# ./config/engine-config.json
#   - port: HTTP 포트 (기본: 18080)
#   - threads: 워커 스레드 수
#   - routes: 활성화할 엔드포인트 목록
#   - llm_backend: LLM 실행 경로 및 옵션
```

### 서버 기동 시 출력 모습
```
[INFO] [2025-08-14 17:31:06] - Route: [1] /health
[INFO] [2025-08-14 17:31:06] - Route: [1] /ml/models
[INFO] [2025-08-14 17:31:06] - Route: [3] /ml/train-all
[INFO] [2025-08-14 17:31:06] - Route: [3] /ml/train
[INFO] [2025-08-14 17:31:06] - Route: [3] /llm/generate
Crow API initialized at 0.0.0.0:18080
Crow/1.2.1 server is running ... using 8 threads
```

---

### 5.2 엔드포인트 목록

| 엔드포인트         | Method | 설명 |
|--------------------|--------|------|
| **`/health`**      | GET    | 서버 상태 확인 (`200 OK` 시 정상) |
| **`/ml/models`**   | GET    | 현재 등록된 학습 가능한 ML 모델 목록 반환 |
| **`/ml/train`**    | POST   | 지정 모델 1개 학습 시작 (JSON 요청 바디 필요) |
| **`/ml/train-all`**| POST   | 등록된 모든 모델 학습 시작 |
| **`/llm/generate`**| POST   | LLM(예: llama.cpp) 텍스트 생성 요청 |

---

### 5.3 요청/응답 예시

#### 1) 체크
```bash
curl -s http://localhost:18080/health
# 응답
{"service":"ml-engine","status":"ok"}
```

#### 2) 등록 모델 조회
```bash
curl -s http://localhost:18080/ml/models
# 응답
{"models":["cnn_mnist"]}
```

#### 3) 단일 모델 학습
```bash
curl -s -X POST http://localhost:18080/ml/train \
  -H "Content-Type: application/json" \
  -d '{
        "model": "cnn_mnist",
        "epochs": 1,
        "batch_size": 64,
        "dataset_root": "./data/mnist",
        "ckpt_dir": "./runs/cnn_mnist_from_api"
      }'

# 응답
{
  "model": "cnn_mnist",
  "status": "ok"
}
```

#### 4) 전체 모델 학습
```bash
curl -s -X POST http://localhost:18080/ml/train-all \
  -H "Content-Type: application/json" \
  -d '{
        "epochs": 1,
        "batch_size": 64,
        "dataset_root": "./data/mnist"
      }'

# 응답
{
  "results": [
    {
      "model": "cnn_mnist",
      "status": "ok"
    }
  ]
}
```

#### 5) LLM 텍스트 생성
```bash
curl -s -X POST http://localhost:18080/llm/generate \
  -H "Content-Type: application/json" \
  -d '{
        "backend": "llama",
        "prompt": "Hello from ml-engine",
        "extra_args": ["-m", "./models/model.gguf", "-n", "64"]
      }'

# 응답 예시 (LLM 백엔드 설정에 따라 다름)
{"status":"ok","output":"Hello from ml-engine..."}
```

---

### 5.4 실행 모드와 유효 인자

| 명령 예시 | 동작 |
|-----------|------|
| `./ml_engine` | 서버 모드 (REST API 활성) |
| `./ml_engine --help=server` | 서버 모드 가이드 출력 |
| `./ml_engine train-cli` | 기본 모델(`cnn_mnist`) CLI 학습 |
| `./ml_engine train-cli cnn_mnist` | 지정 모델 CLI 학습 |
| `./ml_engine f` | 벤치마크 모드 실행 |
| `./ml_engine tr` | ❌ 에러: 잘못된 인자 → 서버 미기동 |

---

### 5.5 트러블슈팅

- **404 에러** → 해당 엔드포인트가 빌드에 포함되지 않았거나 구현되지 않음  
- **포트 충돌** → `config/engine-config.json`에서 포트 변경  
- **LLM 응답이 200인데 내용 없음** → 백엔드 경로와 `.gguf` 모델 파일 존재 여부 확인  
- **`train-cli` 실행 시 데이터셋 오류** → `dataset_root` 경로에 MNIST 데이터 존재 여부 확인  

  
---

## 6. 학습/추론 예시

### CLI 학습
```sh
build/ml_engine train-cli cnn_mnist
```

### llama.cpp 바이너리 직접 테스트
```sh
llama-cli -p "테스트\n\n" -t 8 -c 2048 --temp 0.7 --top-k 40 --top-p 0.95 \
  -m deepseek-coder-v2-lite-instruct-q4_k_m.gguf -ngl 0 --simple-io -n 64 -r "User:"
```

### REST 추론
```sh
curl -sS -X POST http://localhost:18080/llm/generate \
  -H "Content-Type: application/json" \
  -d '{ ... }'
```
서버 로그에는 load time / prompt eval time / eval time / TPS 등 성능 지표가 출력됩니다.

---

## 7. 모델 추천 & 양자화 가이드

### 용도별 추천 (Apple Silicon, 로컬 추론 위주)

#### 범용 비서 + 기본 코드 보조
- Llama 3.1 Instruct 8B: Q4_K_M(기본), Q5_K_M(여유) ~4.5GB
- Mistral Instruct 7B: Q4_K_M ~4.1GB
- Phi-3.5-mini Instruct 3.8B: Q4_K_M ~2.2GB

#### 코드 생성 중심
- Qwen2.5-Coder Instruct 7B: Q4_K_M(기본), Q5_K_M ~4.6GB
- CodeLlama Instruct 7B: Q4_K_M ~4.3GB
- DeepSeek-Coder-V2-Lite Instruct 16B: Q4_K_M ~8–9GB (16GB RAM↑ 권장)

#### 요약
- RAM 16GB: Qwen2.5-Coder-7B-Instruct Q4_K_M
- RAM 32GB↑: Llama 3.1-8B-Instruct Q5_K_M 또는 DeepSeek-Coder-V2-Lite Q4_K_M
- 매우 가볍게: Mistral-7B-Instruct Q4_K_M

### 양자화 가이드
- Q4_K_M: 기본 추천 (메모리↓, 품질 손실 적음)
- Q5_K_M: 메모리 여유 시 품질↑
- Q8_0: FP16 근접, 메모리/속도 비용 큼
- 런타임 메모리: GGUF 파일 크기 × 1.2~1.4

### 파일 배치 & 호출 예시(JSON 페이로드)
#### Request
```json
curl -sS -X POST http://localhost:18080/llm/generate \
  -H "Content-Type: application/json" \
  -d '{
    "backend": "llama",
    "prompt": "C++로 멀티스레드 예제 작성해줘. 코드 블록으로만 답해줘.",
    "llama_exec_path": "/Users/mac/Desktop/workspace/miniGPT/ml-engine/third_party/llama.cpp/build/bin/llama-cli",
    "n_threads": 8,
    "n_ctx": 1024,
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.95,
    "extra_args": [
      "-m","/Users/mac/Desktop/workspace/miniGPT/ml-engine/models/deepseek-coder-v2-lite-instruct-q4_k_m.gguf",
      "-ngl","24",
      "-c","1024",
      "-b","1024",
      "--simple-io",
      "-no-cnv",
      "-n","128"
    ]
  }' | jq .
```

#### Response

```json
{
  "engine": "llama",
  "output": "C++로 멀티스레드 예제 작성해줘. 코드 블록으로만 답해줘.\n\n```cpp\n#include <iostream>\n#include <thread>\n#include <mutex>\n\nstd::mutex mtx;\n\nvoid print_thread_id(int id) {\n    mtx.lock();\n    std::cout << \"Thread \" << id << \" is running\\n\";\n    mtx.unlock();\n}\n\nint main() {\n    std::thread threads[10];\n    for (int i = 0; i < 10; ++i) {\n        threads[i] = std::thread(print_thread_\n\n",
  "params": {
    "extra_args": [
      "-m",
      "/Users/mac/Desktop/workspace/miniGPT/ml-engine/models/deepseek-coder-v2-lite-instruct-q4_k_m.gguf",
      "-ngl",
      "24",
      "-c",
      "1024",
      "-b",
      "1024",
      "--simple-io",
      "-no-cnv",
      "-n",
      "128"
    ],
    "n_ctx": 1024,
    "n_threads": 8,
    "temperature": 0.699999988079071,
    "top_k": 40,
    "top_p": 0.949999988079071
  },
  "prompt": "C++로 멀티스레드 예제 작성해줘. 코드 블록으로만 답해줘."
}
```

---

## 8. 운영 체크리스트

- 실행 권한: llama_exec_path → chmod +x
- DYLD_LIBRARY_PATH: libomp, third_party/libtorch/lib 추가
- 길이 제어: n_ctx(컨텍스트), -n(생성 길이)
- 라이선스 준수: 모델/데이터/코드 각각 확인
- 텐서 차원 안전망: {N,28,28} → {N,1,28,28} unsqueeze (MNIST 1채널 전제 시)
- macOS DataLoader: num_workers>0 이슈 시 0으로

---

## 9. 로드맵

| 단계 | 목표 | 왜/무엇 |
|------|------|---------|
| 1 | 아티팩트 확인 | runs/cnn_mnist/epoch_*.pt 생성 확인 |
| 2 | 추론 루틴 추가 | 체크포인트 로딩 → 단일/배치 예측 |
| 3 | REST 예측 API | POST /ml/predict (Crow) |
| 4 | 베스트만 저장 | 최고 검증점만 저장(early-stop/최고점 관리) |
| 5 | 하이퍼파라미터 제어 | CLI/REST로 epochs/batch/lr/workers 노출 |
| 6 | 영구 러ntime 경로 | rpath 또는 쉘 초기화로 DYLD 영구화 |
| 7 | PHP UI 연동 | Train/Predict 버튼(공유호스팅) |

---

## 10. 자주 발생하는 이슈

- CUDA/cuDNN/LibTorch 불일치 → 로딩/링킹 오류. 버전 정합성 필수
- macOS 포크 제약 → DataLoader(num_workers>0) 문제 시 0으로
- llama.cpp Makefile 경고 → CMake 빌드 사용
- 출력이 중간에 끊김 → -n(생성 길이) 및 n_ctx(컨텍스트) 확대
- CPU만 느림 → llama.cpp Metal 가속 (-DGGML_METAL=ON, 실행 -ngl 99)

---

## 11. 용어 사전

- n_ctx: 컨텍스트 길이(토큰)
- -n / n_predict: 생성할 최대 토큰 수
- n_threads: CPU 스레드 수
- -ngl: GPU(Metal) 오프로딩 레이어 수 (0=CPU, 99=최대)
- -c/-b: 컨텍스트/배치 크기
- --simple-io: 단순 STDIO I/O
- -no-cnv: chat template 비활성화
- temperature/top_k/top_p: 샘플링 제어
- repeat_penalty: 반복 억제
- BOS/EOS/EOG/PAD: 시작/끝/생성종료/패딩 토큰
- FIM: Fill-In-the-Middle 토큰 (PRE/SUF/MID)
- rope(yarn): RoPE 위치 인코딩(+긴 컨텍스트 스케일링)
- kv cache: K/V 캐시(메모리↔속도 트레이드오프)
- prompt eval / eval time: 프롬프트 처리/생성 토큰당 소요
- TPS: 초당 생성 토큰 수

---


## 12. 라이선스

<div align="left">
<details open>
<summary><strong>MIT License 전문</strong></summary>

<blockquote style="background:#f5f5f5;border-radius:12px;padding:16px 24px;box-shadow:0 2px 8px #eee;">

MIT License<br>
Copyright (c) 2025 …<br><br>
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:<br><br>
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.<br><br>
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.<br><br>

</blockquote>
</details>
</div>

---

## 13. 부록: 빠른 시작

```sh
# 1) 빌드
bash ./build.sh

# 2) 런타임 경로
export DYLD_LIBRARY_PATH="$(brew --prefix libomp)/lib:third_party/libtorch/lib:${DYLD_LIBRARY_PATH:-}"

# 3) 학습 (MNIST)
build/ml_engine train-cli cnn_mnist
ls runs/cnn_mnist   # epoch_*.pt 확인

# 4) 서버 기동
build/ml_engine
curl -s http://localhost:18080/health
curl -s http://localhost:18080/ml/models

# 5) LLM 추론 (llama.cpp 연동)
curl -sS -X POST http://localhost:18080/llm/generate \
  -H "Content-Type: application/json" \
  -d '{ ... }'
```

---

## 14. DeepSeek-Coder-V2-Lite Instruct 선택 및 실전 로그

<details>
<summary>실전 로그 및 상세 설명 (클릭하여 펼치기)</summary>

**DeepSeek-Coder-V2-Lite Instruct로 선택**  
https://huggingface.co/models

맨 아래쪽 `sugatoxay/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M-GGUF` 를 추천한 건,
지금 상황이 “자체 LLM 패널에서 추론 테스트” 목적이라서 가장 가볍고 실행이 빠른 `GGUF` 변환본이 필요하기 때문입니다.

https://huggingface.co/sugatoray/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M-GGUF/blob/main/deepseek-coder-v2-lite-instruct-q4_k_m.gguf

• 데이터 차원 보정은 모델 구현(예: cnn_mnist::Model)이 NCHW(1채널) 전제를 가질 때만 필요합니다. 이미 MNIST DataSet이 {N,1,28,28}로 배치한다면 unsqueeze 블록은 건너뛰지만, 위처럼 조건부로 안전망을 두면 환경 차이(C++ API 버전/커스텀 Dataset)에서도 안전합니다.
• macOS CPU에서 num_workers>0일 때 fork 제약으로 드물게 문제가 생기면 cfg.num_workers=0로 낮춰보세요.

---

### [build 완료]
```sh
mac@azabell-mac ml-engine % bash ./build.sh
-- The CXX compiler identification is AppleClang 16.0.0.16000026
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Found Torch: /Users/mac/Desktop/workspace/miniGPT/ml-engine/third_party/libtorch/lib/libtorch.dylib
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Success
-- Found Threads: TRUE
-- Configuring done (3.5s)
-- Generating done (0.0s)
-- Build files have been written to: /Users/mac/Desktop/workspace/miniGPT/ml-engine/build
[ 11%] Building CXX object CMakeFiles/ml_engine.dir/src/api/api_server.cpp.o
[ 22%] Building CXX object CMakeFiles/ml_engine.dir/src/ml/registry.cpp.o
[ 33%] Building CXX object CMakeFiles/ml_engine.dir/src/engine/engine.cpp.o
[ 44%] Building CXX object CMakeFiles/ml_engine.dir/src/api/handler/handler_base.cpp.o
[ 55%] Building CXX object CMakeFiles/ml_engine.dir/src/llm/llm_engine.cpp.o
[ 66%] Building CXX object CMakeFiles/ml_engine.dir/src/ml/models/cnn_mnist/model.cpp.o
[ 77%] Building CXX object CMakeFiles/ml_engine.dir/src/main.cpp.o
[ 88%] Building CXX object CMakeFiles/ml_engine.dir/src/ml/dataset.cpp.o
[100%] Linking CXX executable ml_engine
[100%] Built target ml_engine
```

---

```sh
$ export DYLD_LIBRARY_PATH="$(brew --prefix libomp)/lib:../third_party/libtorch/lib:${DYLD_LIBRARY_PATH:-}"
```

---

### [최종 테스트]
#### Request
```sh
$ mac@azabell-mac ml-engine % curl -s http://localhost:18080/health
curl -s http://localhost:18080/ml/models
curl -s -X POST http://localhost:18080/llm/generate \
  -H "Content-Type: application/json" \
  -d '{"backend":"mock","prompt":"Hello"}'
```

#### Response
```json
{
  "service": "ml-engine",
  "status": "ok"
}
{
  "models": [
    "cnn_mnist"
  ]
}
{
  "engine": "mock",
  "output": "[mock] codegen for: Hello",
  "params": {
    "extra_args": [],
    "n_ctx": 512,
    "n_threads": 4,
    "temperature": 0.800000011920929,
    "top_k": 40,
    "top_p": 0.949999988079071
  },
  "prompt": "Hello"
}
```

---

### [llama.cpp 빌드 및 테스트]
```sh
cd /Users/mac/Desktop/workspace/miniGPT/ml-engine/third_party/llama.cpp
git pull
make clean
cmake -B build -DGGML_METAL=ON
cmake --build build -j
```

빌드 로그 및 경고, Metal/CPU 백엔드 감지, OpenMP 미탐지 경고 등은 실제 환경에 따라 다를 수 있습니다.

---

### [LLM 데모 실행 예시]
```sh
/Users/mac/Desktop/workspace/miniGPT/ml-engine/third_party/llama.cpp/build/bin/llama-cli \
  -p "테스트\n\n" \
  -t 8 -c 2048 --temp 0.7 --top-k 40 --top-p 0.95 \
  -m /Users/mac/Desktop/workspace/miniGPT/ml-engine/models/deepseek-coder-v2-lite-instruct-q4_k_m.gguf \
  -ngl 0 --simple-io -n 64 -r "User:"
```

실행 로그, 모델 메타데이터, 토큰 정보, Metal/CPU 메모리 할당, 성능 지표(TPS, eval time 등)는 실제 출력 예시를 참고하세요.

---

### [Crow Server REST API 요청/응답 예시]
#### Request
```sh
curl -sS -X POST http://localhost:18080/llm/generate \
  -H "Content-Type: application/json" \
  -d '{
    "backend": "llama",
    "prompt": "C++로 멀티스레드 예제 작성해줘. 코드 블록으로만 답해줘.",
    "llama_exec_path": "/Users/mac/Desktop/workspace/miniGPT/ml-engine/third_party/llama.cpp/build/bin/llama-cli",
    "n_threads": 8,
    "n_ctx": 1024,
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.95,
    "extra_args": [
      "-m","/Users/mac/Desktop/workspace/miniGPT/ml-engine/models/deepseek-coder-v2-lite-instruct-q4_k_m.gguf",
      "-ngl","24",
      "-c","1024",
      "-b","1024",
      "--simple-io",
      "-no-cnv",
      "-n","128"
    ]
  }' | jq .
```

#### Response
```json
{
  "engine": "llama",
  "output": "C++로 멀티스레드 예제 작성해줘. 코드 블록으로만 답해줘.\n\n```cpp\n#include <iostream>\n#include <thread>\n#include <mutex>\n\nstd::mutex mtx;\n\nvoid print_thread_id(int id) {\n    mtx.lock();\n    std::cout << \"Thread \" << id << \" is running\\n\";\n    mtx.unlock();\n}\n\nint main() {\n    std::thread threads[10];\n    for (int i = 0; i < 10; ++i) {\n        threads[i] = std::thread(print_thread_\n\n",
  "params": {
    "extra_args": [
      "-m",
      "/Users/mac/Desktop/workspace/miniGPT/ml-engine/models/deepseek-coder-v2-lite-instruct-q4_k_m.gguf",
      "-ngl",
      "24",
      "-c",
      "1024",
      "-b",
      "1024",
      "--simple-io",
      "-no-cnv",
      "-n",
      "128"
    ],
    "n_ctx": 1024,
    "n_threads": 8,
    "temperature": 0.699999988079071,
    "top_k": 40,
    "top_p": 0.949999988079071
  },
  "prompt": "C++로 멀티스레드 예제 작성해줘. 코드 블록으로만 답해줘."
}
```

서버 성능 로그(TPS, load/prompt/eval time 등)는 실제 환경에 따라 다를 수 있습니다.

---

### [추가 팁]

- 데이터 차원 보정은 모델 구현(예: cnn_mnist::Model)이 NCHW(1채널) 전제를 가질 때만 필요합니다. 이미 MNIST DataSet이 {N,1,28,28}로 배치한다면 unsqueeze 블록은 건너뛰지만, 위처럼 조건부로 안전망을 두면 환경 차이(C++ API 버전/커스텀 Dataset)에서도 안전합니다.
- macOS CPU에서 num_workers>0일 때 fork 제약으로 드물게 문제가 생기면 cfg.num_workers=0로 낮춰보세요.

</details>

---

## 15. 부록: DeepSeek-Coder-V2-Lite Instruct GGUF 메타데이터 및 실행 로그 해설

<details>
<summary>실행 로그 및 메타데이터 상세 해설 (클릭하여 펼치기)</summary>

아래는 llama.cpp가 DeepSeek-Coder-V2-Lite-Instruct GGUF 모델을 로드할 때 출력하는 주요 로그와 각 항목의 설명입니다.

### 모델 로딩 및 메타데이터

- **llama_model_load_from_file_impl: using device Metal (Apple M2) - 10922 MiB free**
  - Metal(Apple GPU)에서 모델을 로드하며, 사용 가능한 VRAM(메모리) 용량을 표시합니다.

- **llama_model_loader: loaded meta data with 38 key-value pairs and 377 tensors ...**
  - 모델 파일 내에 포함된 메타데이터(키-값 쌍)와 텐서(가중치 등) 개수를 보여줍니다.

#### 주요 메타데이터 항목 설명

| 키 | 값 | 설명 |
|----|-----|------|
| general.architecture | deepseek2 | 모델 아키텍처 이름 (DeepSeek 2) |
| general.name | DeepSeek-Coder-V2-Lite-Instruct | 모델 이름 |
| deepseek2.block_count | 27 | Transformer 블록(레이어) 개수 |
| deepseek2.context_length | 163840 | 최대 컨텍스트 길이(토큰) |
| deepseek2.embedding_length | 2048 | 임베딩 차원 수 |
| deepseek2.feed_forward_length | 10944 | FFN(Feed Forward Network) 차원 |
| deepseek2.attention.head_count | 16 | 어텐션 헤드 개수 |
| deepseek2.attention.head_count_kv | 16 | KV 어텐션 헤드 개수 |
| deepseek2.rope.freq_base | 10000.0 | RoPE(위치 인코딩) 기본 주파수 |
| deepseek2.attention.layer_norm_rms_epsilon | 0.000001 | RMSNorm epsilon 값 |
| deepseek2.expert_used_count | 6 | 사용되는 expert 개수(MoE 구조) |
| general.file_type | 15 | 파일 타입(내부용) |
| deepseek2.leading_dense_block_count | 1 | 선행 Dense 블록 개수 |
| deepseek2.vocab_size | 102400 | 토크나이저의 vocab 크기 |
| deepseek2.attention.kv_lora_rank | 512 | KV LoRA 랭크(파라미터 효율화) |
| deepseek2.attention.key_length | 192 | 어텐션 key 벡터 차원 |
| deepseek2.attention.value_length | 128 | 어텐션 value 벡터 차원 |
| deepseek2.expert_feed_forward_length | 1408 | expert FFN 차원 |
| deepseek2.expert_count | 64 | expert 전체 개수 |
| deepseek2.expert_shared_count | 2 | expert 공유 개수 |
| deepseek2.expert_weights_scale | 1.0 | expert 가중치 스케일 |
| deepseek2.rope.dimension_count | 64 | RoPE 차원 수 |
| deepseek2.rope.scaling.type | yarn | RoPE 스케일링 방식(yarn: 긴 컨텍스트 지원) |
| deepseek2.rope.scaling.factor | 40.0 | RoPE 스케일링 팩터 |
| deepseek2.rope.scaling.original_context_length | 4096 | 원래 컨텍스트 길이 |
| deepseek2.rope.scaling.yarn_log_multiplier | 0.0707 | yarn 스케일링 로그 멀티플라이어 |
| tokenizer.ggml.model | gpt2 | 토크나이저 모델(gpt2 기반) |
| tokenizer.ggml.pre | deepseek-llm | 토크나이저 prefix |
| tokenizer.ggml.tokens | arr[str,102400] | 토큰 리스트(102400개) |
| tokenizer.ggml.token_type | arr[i32,102400] | 토큰 타입 리스트 |
| tokenizer.ggml.merges | arr[str,99757] | BPE 병합 규칙 리스트 |
| tokenizer.ggml.bos_token_id | 100000 | BOS(문장 시작) 토큰 ID |
| tokenizer.ggml.eos_token_id | 100001 | EOS(문장 끝) 토큰 ID |
| tokenizer.ggml.padding_token_id | 100001 | PAD(패딩) 토큰 ID |
| tokenizer.ggml.add_bos_token | true | BOS 토큰 자동 추가 여부 |
| tokenizer.ggml.add_eos_token | false | EOS 토큰 자동 추가 여부 |
| tokenizer.chat_template | ... | 대화 템플릿(프롬프트 포맷) |
| general.quantization_version | 2 | 양자화 버전 |

- **type f32/q4_K/q5_0/q8_0/q6_K: ... tensors**
  - 각 양자화 타입별 텐서 개수(f32: float, q4_K: 4비트 양자화 등)

- **print_info: file format = GGUF V3 (latest)**
  - GGUF 파일 포맷 버전(최신)
- **print_info: file type   = Q4_K - Medium**
  - 양자화 타입(Q4_K: 4비트, Medium)
- **print_info: file size   = 9.65 GiB (5.28 BPW)**
  - 모델 파일 크기 및 BPW(bits per weight)

### 토크나이저 및 특수 토큰

- **BOS token        = 100000 '<｜begin▁of▁sentence｜>'**
- **EOS token        = 100001 '<｜end▁of▁sentence｜>'**
- **PAD token        = 100001 '<｜end▁of▁sentence｜>'**
- **FIM PRE/SUF/MID token = 100003/100002/100004**
  - FIM(Fill-In-the-Middle) 프롬프트용 특수 토큰
- **EOG token        = 100001 '<｜end▁of▁sentence｜>'**
- **max token length = 256**
  - 한 토큰의 최대 길이

### 모델 로딩 및 실행 환경

- **load_tensors: loading model tensors, this can take a while... (mmap = true)**
  - 텐서(가중치) 로딩 중, mmap(메모리 매핑) 사용
- **load_tensors: offloading 0 repeating layers to GPU**
  - 반복 레이어를 GPU로 오프로딩(여기선 0)
- **CPU_Mapped model buffer size = 9880.47 MiB**
  - CPU에 매핑된 모델 버퍼 크기

### llama_context 및 실행 파라미터

- **llama_context: constructing llama_context**
  - 추론 컨텍스트 생성
- **n_seq_max     = 1**
  - 최대 시퀀스 개수(대화 세션)
- **n_ctx         = 2048**
  - 컨텍스트 길이(토큰)
- **n_batch       = 2048**
  - 배치 크기
- **causal_attn   = 1**
  - 인과적 어텐션 사용
- **kv_unified    = false**
  - KV 캐시 통합 여부
- **freq_base     = 10000.0**
  - RoPE 주파수
- **n_ctx_per_seq (2048) < n_ctx_train (163840) -- the full capacity of the model will not be utilized**
  - 실제 추론 컨텍스트가 학습 시 최대값보다 작아 전체 용량을 다 쓰지 않음

### Metal(GPU) 환경 정보

- **ggml_metal_init: found device: Apple M2**
  - Metal 백엔드에서 Apple M2 GPU 감지
- **ggml_metal_init: recommendedMaxWorkingSetSize  = 11453.25 MB**
  - Metal에서 추천하는 최대 워킹셋(메모리) 크기
- **ggml_metal_init: skipping kernel_xxx (not supported)**
  - 지원하지 않는 커널(연산)은 스킵

### 기타 실행 파라미터 및 샘플링

- **main: llama threadpool init, n_threads = 8**
  - 추론에 사용할 스레드 수
- **main: chat template is available, enabling conversation mode (disable it with -no-cnv)**
  - 대화 템플릿 활성화(기본), -no-cnv로 비활성화 가능
- **sampler params: ...**
  - 샘플링 파라미터(반복 억제, top_k, top_p, temperature 등)
- **generate: n_ctx = 2048, n_batch = 2048, n_predict = 64, n_keep = 1**
  - 추론 시 컨텍스트, 배치, 생성 토큰 수, 유지 토큰 수

### 대화 예시 및 프롬프트

- **main: chat template example:**
  - 프롬프트 예시: User/Assistant 역할

---

이 로그들은 GGUF 모델의 구조, 토크나이저, 실행 환경, 샘플링 파라미터 등 LLM 추론의 모든 핵심 정보를 담고 있습니다. 각 항목은 실제 추론 품질, 속도, 메모리 사용량에 직접적인 영향을 미치므로, 환경에 맞게 조정하며 활용하세요.

</details>

---

## 16 부록: 서버 기동 및 라이선스 체크 동작
<details>
<summary>라이선스 상세 설명 (클릭하여 펼치기)</summary>

### 1. config/engine-config.json 구조

```json
{
  "common": {
    "api_port": 18080,
    "license": "./license.json",
    "public_key_path": "./public_key.pem"
  }
}
```
- `license`: 실제 라이선스 JSON 파일 경로
- `public_key_path`: 서명 검증용 공개키 경로

### 2. config/license.json 샘플

```json
{
  "license_id": "trial-2025-08-14",
  "issued_to": "user@example.com",
  "issued_at": "2025-08-14T00:00:00Z",
  "expires_at": "2025-09-14T00:00:00Z",
  "public_key_path": "public_key.pem",
  "signature": "<base64-signature>",
  "features": ["api", "train", "predict"],
  "notes": "Demo license for engine activation."
}
```

### 3. config/public_key.pem 샘플

```
-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEArandompublickeydemo
-----END PUBLIC KEY-----
```

### 4. utils::loadEngineConfig (engine-config.json 파싱)
- `license`, `public_key_path` 필드를 읽어 실제 경로로 변환
- 구조체에 채워넣어 이후 엔진 초기화에 사용

### 5. secure::loadLicenseFile (license.json 로딩)
- 라이선스 파일을 통째로 읽어 문자열로 반환
- 실제 환경에서는 암호화/서명 검증 필요

### 6. engine.cpp의 라이선스 연동 흐름

- config에서 license/public_key_path 경로를 읽어 실제 파일 경로로 보정
- 라이선스 파일을 읽어 JSON 파싱
- 라이선스 내 public_key_path가 있으면 우선 적용
- secure::AntiPiracy, SignatureVerifier 등으로 무결성/서명 검증
- features, expires_at 등 라이선스 필드 활용 가능

#### 주요 코드 흐름
```cpp
// 1) 경로 보정 및 로그
std::string license_json_path = m_config.common.license;
std::string public_key_path = m_config.common.public_key_path;
// 상대경로 → 절대경로 보정
std::filesystem::path config_dir = std::filesystem::path(m_config_filepath).parent_path();
std::filesystem::path abs_public_key_path = config_dir / public_key_path;
public_key_path = abs_public_key_path.string();

// 2) 라이선스 파일 로딩
std::string license_json;
secure::loadLicenseFile(license_json_path, license_json);

// 3) 라이선스 JSON 파싱 및 public_key_path 우선 적용
auto root = nlohmann::json::parse(license_json);
if (root.contains("public_key_path")) {
    std::filesystem::path license_dir = std::filesystem::path(license_json_path).parent_path();
    std::filesystem::path abs_license_public_key_path = license_dir / root["public_key_path"].get<std::string>();
    public_key_path = abs_license_public_key_path.string();
}

// 4) 무결성/서명 검증
secure::AntiPiracy::verifyProgramIntegrity();
secure::AntiPiracy::activateOnlineFromJson(license_json);
secure::SignatureVerifier::verifySignatureFromJson(license_json, license_json_path);
```

### 7. 실제 활용 예시
- 라이선스 만료, feature 제한, 서명 검증 등 엔진 동작 제어 가능
- 예: `features`에 "train"이 없으면 학습 API 비활성화 등

---
