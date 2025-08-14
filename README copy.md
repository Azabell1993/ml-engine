ml-engine — C++/LibTorch Training + llama.cpp Inference (macOS M2)


⸻

0) 개요
	•	목적: ml-engine은 C++/LibTorch 기반의 모델(Module)을 직접 빌드해 데이터셋의 loss를 최소화하는 학습 엔진이며, 동시에 외부 실행(예: llama.cpp) 을 통해 LLM 추론을 REST로 오케스트레이션합니다.
	•	핵심 포인트
	•	학습 가능한 대상은 src/ml/models/... 아래에 있는 LibTorch C++ Module 들뿐.
	•	MNIST(28×28, 흑백 0–9) 로 도구 체인/학습 루프 검증.
	•	miniGPT/models/ImSQL, LISA_SOURCE_APPS, mini-gpt, QT_Kernel_OS, WidevineKey 등은 딥러닝 모델이 아니라 일반 프로젝트 소스이므로 학습 대상이 아님.
	•	코드 자체를 “학습”하려면(=LLM 파인튜닝) PyTorch/Hugging Face + (Q)LoRA 파이프라인을 활용하거나 llama.cpp의 LoRA/QLoRA 경로를 사용. C++ 순정 LibTorch로 거대한 LLM을 macOS CPU에서 학습하는 것은 현실적으로 불가.
ml-engine은 C++/LibTorch로 구현된 미니멀 학습 엔진과, 외부 실행(예: llama.cpp) 연동 LLM 추론을 REST로 오케스트레이션하는 서버/CLI입니다.
	•	학습 대상: src/ml/models/...의 LibTorch Module(C++ 클래스)
	•	검증 데이터셋: MNIST (28×28, 흑백, 0–9)
	•	추론: llama.cpp 바이너리를 서브프로세스로 호출하여 GGUF 모델 실행
	•	macOS(Apple Silicon): LibTorch Metal 학습 미지원 → CPU 학습, llama.cpp는 Metal 가속(-ngl 99) 가능
ml-engine: C++/LibTorch 학습 + llama.cpp 연동 추론 (macOS M2 기준 종합 가이드)

⸻

1) 환경 및 주요 기술 상세 설명

1-1) CUDA / ROCm / CPU
	•	CUDA (NVIDIA): NVIDIA GPU에서 병렬 연산 가속. LibTorch CUDA 빌드를 설치하면 GPU 학습 가능.
	•	ROCm (AMD): AMD GPU용 오픈소스 가속 플랫폼. PyTorch 일부 버전만 지원.
	•	CPU: GPU가 없거나 macOS(Apple Silicon) 처럼 LibTorch의 Metal 학습 미지원 시 CPU-only 학습/추론.

1-2) macOS (Apple Silicon)
	•	LibTorch Metal 학습 미지원: 공식적으로 Metal(GPU) 학습 미지원; 추론은 CoreML로 일부 가능.
	•	실제 학습: CPU 모드만 가능, 속도 느림 → 대규모 학습은 Linux/NVIDIA 권장.

1-3) Linux / NVIDIA
	•	CUDA + cuDNN: GPU 학습 필수 조합. LibTorch를 CUDA 버전에 맞게 빌드·배포 필요.
	•	버전 일치: CUDA / cuDNN / LibTorch 버전이 정확히 일치해야 오류 없이 동작.

1-4) AMP (Automatic Mixed Precision)
	•	C++ 네임스페이스: PyTorch Python의 autocast에 대응하는 torch::autocast 사용.
	•	효과: VRAM/메모리 절약, 속도 향상. 단, 지원되는 연산/모델에서만 안정적.

1-5) 데이터셋 일반화
	•	MNIST: 28×28 흑백 손글씨(0–9), 기본 예제.
	•	ImageFolderDataset: 폴더 구조로 이미지 분류 데이터 로딩.
	•	CSV/Parquet 커스텀 Dataset: 표/테이블 데이터에 대해 Dataset 구현으로 학습 확장 가능.

1-6) 재현성
	•	seed 고정: torch::manual_seed(...).
	•	cuDNN 결정론: PyTorch C++에서는 환경변수/플래그로 제어.
	•	DataLoader shuffle 고정: 섞기 순서도 seed 고정.

1-7) 로깅
	•	stdout + 파일 로그: 기본 로그를 콘솔/파일에 남김.
	•	확장: JSONL, TensorBoard 프로토콜, wandb C++ SDK 등으로 시각화/분석.

1-8) 분산/멀티 GPU
	•	초기: 단일 GPU만 지원.
	•	확장: torch::distributed + DDP로 다중 GPU/다중 노드 병렬 학습 가능.

1-9) 플러그인화(선택)
	•	.so/.dll 빌드: 모델을 동적 라이브러리로 빌드 후 런타임 교체/추가.
	•	dlopen: C++에서 동적 로딩.
	•	초기 버전: 정적 링크만으로도 충분히 운영 가능.

⸻

2) 실제 환경 (macOS M2) 및 프로젝트 구조

2-1) 학습 범위와 제한
	•	학습 가능한 모델: src/ml/models/... 아래 LibTorch Module 클래스만.
	•	데이터셋: MNIST 등을 직접 생성/로딩.
	•	학습 대상 아님 (중요): miniGPT/models/ImSQL, LISA_SOURCE_APPS, mini-gpt, QT_Kernel_OS, WidevineKey 등 일반 소스 폴더.
	•	대안
	•	A: 위 폴더 중 딥러닝 네트워크로 정의 가능한 부분을 LibTorch Module로 신규 작성.
	•	B: 코드 자체를 학습(LLM 파인튜닝)하려면 PyTorch/HF + (Q)LoRA 또는 llama.cpp의 (Q)LoRA 파이프라인 사용.

2-2) 디렉터리 트리

ml-engine/
├─ CMakeLists.txt
├─ third_party/
│  └─ libtorch/                    # (다운로드한 LibTorch 위치; CUDA/CPU 빌드와 일치 필수)
├─ include/
│  ├─ log.h                        # (요청 주신 버전 유지 + 소소 개선)
│  ├─ safe_arithmetic_ops.h        # (요청 주신 헤더 유지)
│  ├─ api/
│  │  ├─ api_server.hpp
│  │  └─ handler/handler_base.hpp  # 공통 응답 유틸
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
│  ├─ engine-config.json           # 공용 설정
│  └─ license.json                 # (옵션) 라이선스 예시
├─ php/                            # 공유호스팅용 프록시/자동화
│  ├─ index.php
│  └─ .env.php.dist
└─ README.md

재강조: ml-engine은 LibTorch Module 학습 엔진입니다. 따라서 학습 대상은 src/ml/models/... 뿐입니다. (중복 기재된 설명까지 반영)

⸻

3) 빌드 및 실행 (macOS M2)

3-1) 빌드 로그 (발췌)

$ bash ./build.sh
-- Found Torch: .../third_party/libtorch/lib/libtorch.dylib
[100%] Built target ml_engine

3-2) 런타임 준비

export DYLD_LIBRARY_PATH="$(brew --prefix libomp)/lib:third_party/libtorch/lib:${DYLD_LIBRARY_PATH:-}"

3-3) CLI 학습 예시 (MNIST)

build/ml_engine train-cli cnn_mnist
# 예시 로그
[INFO] ... [epoch 1] loss=1.37270 val_acc=0.7693
[INFO] ... [epoch 2] loss=0.65143 val_acc=0.8713
[INFO] ... [epoch 3] loss=0.44448 val_acc=0.9037
[INFO] ... CLI training finished: cnn_mnist

3-4) 체크포인트 생성 확인

$ ls -lh runs/cnn_mnist
epoch_1.pt  epoch_2.pt  epoch_3.pt   # 각 ~85KB


⸻

4) REST API 서버

4-1) 기동

build/ml_engine
# 예시:
[INFO] Route: [1] /health
[INFO] Route: [1] /ml/models
[INFO] Route: [3] /ml/train-all
[INFO] Route: [3] /ml/train
[INFO] Route: [3] /llm/generate
[INFO] Crow API initialized at 0.0.0.0:18080

4-2) 엔드포인트 목록

엔드포인트	Method	설명
/health	GET	서버 상태 확인 (200 OK)
/ml/models	GET	등록된 학습 모델 목록
/ml/train-all	POST	모든 모델 학습 시작
/ml/train	POST	단일 모델 학습 시작 (JSON 필요)
/llm/generate	POST	LLM 텍스트 생성 (외부 실행 연동)

4-3) 헬스/모델 확인

curl -s http://localhost:18080/health
# {"service":"ml-engine","status":"ok"}

curl -s http://localhost:18080/ml/models
# {"models":["cnn_mnist"]}

4-4) 학습 시작 (REST)

curl -s -X POST http://localhost:18080/ml/train \
  -H "Content-Type: application/json" \
  -d '{"model":"cnn_mnist","epochs":3,"device":"cpu"}'
# {"model":"cnn_mnist","status":"ok"}

curl -s -X POST http://localhost:18080/ml/train-all
# {"message":"invalid json"}  ← body 미포함 시 서버 검증에 의해 실패


⸻

5) LLM 추론 (외부 실행: llama.cpp)

5-1) 서버 API 사용 예시

curl -s -X POST http://localhost:18080/llm/generate \
  -H "Content-Type: application/json" \
  -d '{
    "backend": "llama",
    "prompt": "C++로 멀티스레드 예제 작성해줘",
    "llama_exec_path": "/Users/mac/Desktop/workspace/miniGPT/ml-engine/third_party/llama.cpp/build/bin/llama-cli",
    "n_threads": 8,
    "n_ctx": 2048,
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.95,
    "extra_args": [
      "-m","/Users/mac/Desktop/workspace/miniGPT/ml-engine/models/deepseek-coder-v2-lite-instruct-q4_k_m.gguf",
      "-ngl","0",
      "-n","256"
    ]
  }'

응답 예시(요약): engine":"llama", output:"... (모델 생성물) ..."
서버 로그에는 load time / prompt eval time / eval time / tokens per second 등의 성능 지표가 함께 출력됩니다.

5-2) llama.cpp 바이너리 직접 테스트 (성공 케이스)

/Users/.../llama-cli \
  -p "테스트\n\n" \
  -t 8 -c 2048 --temp 0.7 --top-k 40 --top-p 0.95 \
  -m /Users/.../deepseek-coder-v2-lite-instruct-q4_k_m.gguf \
  -ngl 0 --simple-io -n 64 -r "User:"
# Assistant: 여기에 응답이 나오면 성공.

5-3) llama.cpp 빌드 메모
	•	third_party/llama.cpp가 git 리포지토리 아님으로 감지될 수 있음(로그: not a git repository). CMake 빌드 사용 권장(메이크파일 빌드는 deprecated 경고).
	•	Metal 가속: -DGGML_METAL=ON 또는 LLAMA_METAL=1로 빌드 후 **-ngl 99**로 대부분의 레이어를 GPU로 오프로딩하여 M시리즈에서 속도 개선.
	•	일부 빌드 로그에 OpenMP 미탐지 경고가 나타날 수 있으나, Accelerate/Metal 백엔드로 진행 가능.

5-4) 서버 측 성능 로그 예시(발췌)

sampling time = 14.28 ms / 189 runs (0.08 ms/token, 13k tok/s)
prompt eval time = 3262 ms / 61 tokens (53.48 ms/token)
eval time = 14670 ms / 127 runs (115.52 ms/token)

	•	의미: 로드 시간(load), 프롬프트 평가 시간(prompt eval), 토큰당 추론 시간(eval/ts), TPS(tokens/s) 등으로 현재 하드웨어/설정에서의 처리량을 가늠.

⸻

6) 모델 추천 및 양자화 옵션 (Apple Silicon, 로컬 추론 위주)

6-1) 용도별 추천

(A) 범용 비서 + 기본 코드 보조

모델	파라미터	강점	권장 양자화	대략 용량	비고
Llama 3.1 Instruct	8B	한/영 균형, 대화 품질 안정	Q4_K_M(기본), Q5_K_M(여유)	~4.5GB	메타 라이선스 확인 필요
Mistral Instruct	7B	가벼움/속도, 합리적 품질	Q4_K_M	~4.1GB	메모리 적음
Phi-3.5-mini Instruct	3.8B	저메모리·요약·짧은 코드	Q4_K_M	~2.2GB	소형, 품질은 7–8B <

(B) 코드 생성 중심

모델	파라미터	강점	권장 양자화	대략 용량	비고
Qwen2.5-Coder Instruct	7B	최신 코드 지식, 포맷 안정	Q4_K_M(기본), Q5_K_M	~4.6GB	코드 생성/수정 강점
CodeLlama Instruct	7B	전통적 코드 LLM, 안정	Q4_K_M	~4.3GB	가벼움
DeepSeek-Coder-V2-Lite Instruct	16B	고품질 코드/리팩토링	Q4_K_M	~8–9GB	16GB RAM↑ 권장, 느릴 수 있음

요약
	•	RAM 16GB: Qwen2.5-Coder-7B-Instruct Q4_K_M
	•	RAM 32GB↑: Llama 3.1-8B-Instruct Q5_K_M(범용) 또는 DeepSeek-Coder-V2-Lite Q4_K_M(코드 품질)
	•	매우 가볍게: Mistral-7B-Instruct Q4_K_M

6-2) 양자화 가이드
	•	Q4_K_M: 기본 추천(메모리↓, 품질 손실 적음)
	•	Q5_K_M: 메모리 여유 시 품질↑
	•	Q8_0: FP16 근접 품질, 메모리/속도 비용 큼
	•	런타임 상주 메모리(대략): GGUF 파일 크기 × 1.2~1.4

6-3) 파일 배치 & 호출 예시

{
  "backend": "llama",
  "prompt": "REST API Hello world 코드를 생성해줘",
  "llama_exec_path": "/Users/mac/src/llama.cpp/llama-cli",
  "n_threads": 8,
  "n_ctx": 2048,
  "temperature": 0.7,
  "top_k": 40,
  "top_p": 0.95,
  "extra_args": [
    "-m","/Users/mac/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf",
    "-ngl","0",      // CPU 전용. Metal 사용시 99
    "-n","512"
  ]
}

6-4) 체크리스트(LLM 연동)
	•	llama_exec_path 실행 권한(chmod +x)
	•	DYLD_LIBRARY_PATH: libomp, libtorch 필요 시 설정
	•	긴 프롬프트/출력: n_ctx, -n 조절
	•	라이선스/사용조건(특히 Llama 계열) 준수

6-5) DeepSeek-Coder-V2-Lite Instruct 선택 이유 (링크 맥락)
	•	목적: 자체 LLM 패널에서 추론 테스트 → 가볍고 빠른 GGUF 변환본이 적합
	•	예시 링크(사용자 제공):
	•	모델 목록: https://huggingface.co/models
	•	GGUF 예: https://huggingface.co/sugatoray/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M-GGUF/blob/main/deepseek-coder-v2-lite-instruct-q4_k_m.gguf
주: 텍스트 중 sugatoxay와 sugatoray 표기가 혼재. 실제 파일 경로는 **sugatoray**로 기재되어 있음.

⸻

7) 추론/학습 운영 체크리스트
	•	실행 권한: llama_exec_path에 chmod +x
	•	DYLD_LIBRARY_PATH: libomp, third_party/libtorch/lib
	•	길이 제어: n_ctx(컨텍스트), -n(생성 길이)
	•	라이선스: 모델별 약관 준수 (Llama 등)
	•	데이터 차원 안전망: 모델이 NCHW(1채널) 가정이면, {N, 28, 28} → {N, 1, 28, 28} unsqueeze 방어 로직 유용
	•	macOS DataLoader: num_workers > 0에서 fork 제약 발생 시 num_workers=0 시도

⸻

8) 엔드투엔드 예시 세트 (요약)

8-1) 학습(CLI)

cd ~/Desktop/workspace/miniGPT/ml-engine
export DYLD_LIBRARY_PATH="$(brew --prefix libomp)/lib:third_party/libtorch/lib:${DYLD_LIBRARY_PATH:-}"
build/ml_engine train-cli cnn_mnist
# Epoch 1~3 진행, val_acc ~0.90, runs/cnn_mnist/epoch_*.pt 생성

8-2) 서버 기동 & 점검

build/ml_engine
curl -s http://localhost:18080/health
curl -s http://localhost:18080/ml/models

8-3) REST 학습

curl -s -X POST http://localhost:18080/ml/train \
  -H "Content-Type: application/json" \
  -d '{"model":"cnn_mnist","epochs":3,"device":"cpu"}'

8-4) REST 추론 (llama.cpp)

curl -sS -X POST http://localhost:18080/llm/generate \
  -H "Content-Type: application/json" \
  -d '{
    "backend":"llama",
    "prompt":"C++로 멀티스레드 예제 작성해줘. 코드 블록으로만 답해줘.",
    "llama_exec_path":"/Users/.../llama-cli",
    "n_threads":8,
    "n_ctx":1024,
    "temperature":0.7,
    "top_k":40,
    "top_p":0.95,
    "extra_args":[
      "-m","/Users/.../deepseek-coder-v2-lite-instruct-q4_k_m.gguf",
      "-ngl","24", "-c","1024", "-b","1024", "--simple-io", "-no-cnv", "-n","128"
    ]
  }'
# 서버 로그에 load/prompt/eval 시간과 TPS가 출력됨


⸻

9) 로드맵/작업 단계 (체크리스트 확장)

단계	목표	무엇을/왜
1	아티팩트 확인	runs/cnn_mnist/epoch_*.pt 생성 확인
2	추론 루틴 추가	체크포인트 로딩 → 단일/배치 예측
3	REST 예측 API	POST /ml/predict (Crow)
4	베스트만 저장	검증 최고 정확도만 저장 (early-stop/최고점 관리)
5	하이퍼파라미터 제어	CLI/REST로 epochs/batch/lr/workers 노출
6	영구 런타임 경로	rpath 또는 쉘 초기화로 DYLD 설정 영구화
7	PHP UI 연동	버튼으로 Train/Predict 호출 (공유호스팅용)


⸻

10) 파라미터/로그 용어 사전 (llama.cpp / 엔진 공통)
	•	n_ctx: 모델 컨텍스트 길이(토큰). 길수록 긴 프롬프트/출력 처리, 메모리↑.
	•	n_predict / -n: 생성할 최대 토큰 수.
	•	n_threads: CPU 스레드 수.
	•	-ngl: GPU(Metal)로 오프로딩할 레이어 수. 0은 CPU-only, 99는 대부분 레이어를 GPU로.
	•	-c: 컨텍스트(n_ctx) 지정 (llama.cpp).
	•	-b: 배치 크기(batch). 큰 값은 속도↑(메모리 여유 필요).
	•	--simple-io: 프롬프트/응답을 단순 STDIO로 입출력.
	•	-no-cnv: chat template 비활성(대화 모드 끄기).
	•	temperature: 확률 분포 평탄화(↑=창의↑, 안정↓).
	•	top_k: 상위 k개 토큰만 샘플링.
	•	top_p: 누적 확률 p까지 포함하는 Nucleus Sampling.
	•	repeat_penalty: 반복 억제 가중.
	•	mirostat: 동적 엔트로피 타깃팅(일부 빌드).
	•	logits / logit bias: 토큰 로짓(미정규화 점수) / 특정 토큰 선택 편향값.
	•	BOS/EOS/EOG/PAD: 문장 시작/끝/생성 종료/패딩 토큰.
	•	FIM 토큰: Fill-In-the-Middle 프롬프트용 특수 토큰 (PRE/MID/SUF).
	•	rope: RoPE 위치 인코딩; yarn 스케일링은 긴 컨텍스트 확장 기법.
	•	kv cache: Key/Value 캐시. 길수록 메모리↑, 재사용으로 속도↑.
	•	prompt eval time / eval time: 프롬프트 소화 시간 / 토큰 생성 시간(토큰당 ms).
	•	tokens per second (TPS): 초당 생성 토큰 수.
	•	load time: 모델/가중치 로드 및 초기화 시간.

⸻

11) 체크리스트 (요약 재수록)
	•	llama_exec_path 실행권한
	•	DYLD_LIBRARY_PATH 설정(libomp, libtorch)
	•	프롬프트/출력 길이 제어: n_ctx, -n
	•	라이선스 검토(특히 Llama 계열)

위 내용을 통해 macOS M2 환경에서 ml-engine을 활용한 학습/추론/운영의 모든 주요 포인트를 상세히 이해하고 재현할 수 있습니다.

⸻

12) 참고: “일반 폴더 전체를 ‘학습’”에 대한 정리
	•	miniGPT/models의 일반 폴더는 ✔️ 컴파일/테스트 대상이지 ❌ 학습 대상이 아님.
	•	대안 A: 네트워크로 정의 가능한 부분을 LibTorch Module로 다시 작성.
	•	대안 B: LLM 파인튜닝은 PyTorch/HF + (Q)LoRA 또는 llama.cpp (Q)LoRA 경로.

⸻

13) 자주 발생하는 이슈와 팁
	•	macOS DataLoader: num_workers > 0에서 fork 제약 이슈가 드물게 발생 → num_workers=0 권장.
	•	CUDA/cuDNN/LibTorch 버전 불일치: 로딩/링킹 에러 원인 1순위. 정확한 버전 매칭 필수.
	•	MNIST 텐서 차원: {N, 28, 28} 입력이 오면 **unsqueeze(1)**로 {N, 1, 28, 28} 보정.
	•	llama.cpp 빌드 경고: Makefile deprecated → CMake 빌드 사용.
	•	성능 측정: TPS, eval time/token으로 설정 변화 영향 확인(예: -ngl 0 ↔ 99, -b, n_ctx).

⸻

부록) 샘플 호출(요청/응답) 요약
	•	요청: C++ 멀티스레드 예제 생성
	•	extra_args: -m <GGUF> -ngl 24 -c 1024 -b 1024 --simple-io -no-cnv -n 128
	•	응답: 코드 블록을 포함한 생성물(일부 잘릴 수 있음. -n/n_ctx 조절)
	•	서버 성능 로그:
	•	load time ≈ 10~12s, prompt eval ≈ 2.6~3.2s, eval ≈ 12~15s/127 tokens, TPS ≈ 8.6~10 tok/s (예시)

⸻

끝. 필요하시면 이 문서를 기반으로 README.md에 바로 붙여넣기 가능한 포맷(머리말/배지/라이선스 섹션 포함)으로도 정리해 드리겠습니다.
⸻

목차
	•	특징
	•	환경 및 기술 개요
	•	프로젝트 구조
	•	빌드 & 실행
	•	REST API
	•	학습/추론 예시
	•	모델 추천 & 양자화 가이드
	•	운영 체크리스트
	•	로드맵
	•	자주 발생하는 이슈
	•	용어 사전
	•	라이선스

⸻

특징
	•	C++/LibTorch 학습 엔진
	•	src/ml/models/...의 Module만 학습 가능 (예: cnn_mnist)
	•	seed 고정, DataLoader shuffle 고정, cuDNN deterministic 제어(해당 환경)
	•	stdout + 파일 로그 기본, **JSONL/TensorBoard/wandb C++**로 확장 가능
	•	llama.cpp 연동 LLM 추론
	•	/llm/generate → 외부 실행(서브프로세스) 로 GGUF 모델 실행
	•	macOS Metal 가속(빌드 시 -DGGML_METAL=ON, 실행 시 -ngl 99)
	•	플러그인화(선택)
	•	모델을 .so/.dll로 빌드 후 런타임 교체/추가(초기엔 정적 링크로도 충분)
	•	분산/멀티GPU(확장)
	•	초기 단일 GPU(또는 CPU) → torch::distributed + DDP로 확장 가능

중요: miniGPT/models/ImSQL, LISA_SOURCE_APPS, mini-gpt, QT_Kernel_OS, WidevineKey 등은 일반 소스이며, 그 상태로는 학습 대상이 아닙니다.
	•	대안 A: 해당 소스를 LibTorch Module로 재정의
	•	대안 B: LLM 파인튜닝은 PyTorch/Hugging Face + (Q)LoRA 또는 llama.cpp (Q)LoRA 사용

⸻

환경 및 기술 개요

CUDA / ROCm / CPU
	•	CUDA(NVIDIA): LibTorch CUDA 빌드 설치 시 GPU 학습 가능
	•	ROCm(AMD): PyTorch 일부 버전 지원
	•	CPU: GPU 없거나 macOS Metal 학습 미지원 시 CPU 학습/추론

macOS (Apple Silicon)
	•	LibTorch Metal 학습 미지원 (CoreML 추론 일부 가능)
	•	대규모 학습은 Linux/NVIDIA 권장
	•	llama.cpp는 Metal 가속으로 빠른 추론 가능

Linux / NVIDIA
	•	CUDA + cuDNN 필수
	•	CUDA / cuDNN / LibTorch 버전 정합성 중요

AMP (Automatic Mixed Precision)
	•	C++: torch::autocast 사용
	•	효과: 메모리 절약, 속도 향상(지원 연산/모델에 한함)

데이터셋
	•	MNIST, ImageFolderDataset, CSV/Parquet 커스텀 Dataset(직접 구현)

로깅
	•	콘솔/파일 기본
	•	JSONL/TensorBoard/wandb 확장 가능

⸻

프로젝트 구조

ml-engine/
├─ CMakeLists.txt
├─ third_party/
│  └─ libtorch/                    # LibTorch (CUDA/CPU 빌드와 일치)
├─ include/
│  ├─ log.h                        # 경량 로깅
│  ├─ safe_arithmetic_ops.h        # 안전 산술 유틸
│  ├─ api/
│  │  ├─ api_server.hpp
│  │  └─ handler/handler_base.hpp  # 공통 응답 유틸
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
│  ├─ engine-config.json           # 공용 설정
│  └─ license.json                 # (옵션) 라이선스 예시
├─ php/                            # 공유호스팅용 프록시/자동화
│  ├─ index.php
│  └─ .env.php.dist
└─ README.md


⸻

빌드 & 실행

의존성
	•	CMake 3.20+
	•	Clang/GCC (macOS AppleClang OK)
	•	LibTorch (CPU or CUDA 빌드와 일치)
	•	(macOS) brew install libomp

빌드

bash ./build.sh
# ...
# [100%] Built target ml_engine

런타임 환경

export DYLD_LIBRARY_PATH="$(brew --prefix libomp)/lib:third_party/libtorch/lib:${DYLD_LIBRARY_PATH:-}"

CLI 학습 (MNIST)

build/ml_engine train-cli cnn_mnist

# 예시 로그
# [epoch 1] loss=1.37270 val_acc=0.7693
# [epoch 2] loss=0.65143 val_acc=0.8713
# [epoch 3] loss=0.44448 val_acc=0.9037
# CLI training finished: cnn_mnist

체크포인트 확인

ls -lh runs/cnn_mnist
# epoch_1.pt  epoch_2.pt  epoch_3.pt  (각 ~85KB)


⸻

REST API

서버 기동:

build/ml_engine
# Crow server on 0.0.0.0:18080

엔드포인트:

엔드포인트	Method	설명
/health	GET	서버 상태 확인 (200 OK)
/ml/models	GET	등록된 학습 모델 목록
/ml/train-all	POST	모든 모델 학습 시작
/ml/train	POST	단일 모델 학습 시작 (JSON 바디 필요)
/llm/generate	POST	LLM 텍스트 생성 (llama.cpp 연동)

예시:

curl -s http://localhost:18080/health
# {"service":"ml-engine","status":"ok"}

curl -s http://localhost:18080/ml/models
# {"models":["cnn_mnist"]}

curl -s -X POST http://localhost:18080/ml/train \
  -H "Content-Type: application/json" \
  -d '{"model":"cnn_mnist","epochs":3,"device":"cpu"}'
# {"model":"cnn_mnist","status":"ok"}

curl -s -X POST http://localhost:18080/ml/train-all
# {"message":"invalid json"}  # 바디 미포함시 검증 실패 예시


⸻

학습/추론 예시

1) 학습 (CLI)

cd ~/Desktop/workspace/miniGPT/ml-engine
export DYLD_LIBRARY_PATH="$(brew --prefix libomp)/lib:third_party/libtorch/lib:${DYLD_LIBRARY_PATH:-}"
build/ml_engine train-cli cnn_mnist

2) llama.cpp 바이너리 직접 테스트 (성공 케이스)

/Users/.../llama.cpp/build/bin/llama-cli \
  -p "테스트\n\n" \
  -t 8 -c 2048 --temp 0.7 --top-k 40 --top-p 0.95 \
  -m /Users/.../deepseek-coder-v2-lite-instruct-q4_k_m.gguf \
  -ngl 0 --simple-io -n 64 -r "User:"
# Assistant: 여기에 응답이 나오면 성공.

3) REST 추론

curl -sS -X POST http://localhost:18080/llm/generate \
  -H "Content-Type: application/json" \
  -d '{
    "backend": "llama",
    "prompt": "C++로 멀티스레드 예제 작성해줘. 코드 블록으로만 답해줘.",
    "llama_exec_path": "/Users/.../llama.cpp/build/bin/llama-cli",
    "n_threads": 8,
    "n_ctx": 1024,
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.95,
    "extra_args": [
      "-m","/Users/.../deepseek-coder-v2-lite-instruct-q4_k_m.gguf",
      "-ngl","24",
      "-c","1024",
      "-b","1024",
      "--simple-io",
      "-no-cnv",
      "-n","128"
    ]
  }'

서버 로그에는 load time / prompt eval time / eval time / tokens per second(TPS) 등 성능 지표가 출력됩니다.

⸻

모델 추천 & 양자화 가이드

용도별 추천 (Apple Silicon, 로컬 추론 위주)

A. 범용 비서 + 기본 코드 보조

모델	파라미터	강점	권장 양자화	용량(대략)	비고
Llama 3.1 Instruct	8B	한/영 균형, 대화 품질 안정	Q4_K_M(기본), Q5_K_M	~4.5GB	메타 라이선스 확인
Mistral Instruct	7B	가벼움/속도, 합리적 품질	Q4_K_M	~4.1GB	저메모리
Phi-3.5-mini Instruct	3.8B	저메모리·요약·짧은 코드	Q4_K_M	~2.2GB	품질은 7–8B <

B. 코드 생성 중심

모델	파라미터	강점	권장 양자화	용량(대략)	비고
Qwen2.5-Coder Instruct	7B	최신 코드지식, 포맷 안정	Q4_K_M(기본), Q5_K_M	~4.6GB	코드 생성/수정 강점
CodeLlama Instruct	7B	전통적 코드 LLM, 안정	Q4_K_M	~4.3GB	가벼움
DeepSeek-Coder-V2-Lite Instruct	16B	고품질 코드/리팩토링	Q4_K_M	~8–9GB	16GB RAM↑ 권장

요약
	•	RAM 16GB: Qwen2.5-Coder-7B-Instruct Q4_K_M
	•	RAM 32GB↑: Llama 3.1-8B-Instruct Q5_K_M(범용) 또는 DeepSeek-Coder-V2-Lite Q4_K_M(코드품질)
	•	매우 가볍게: Mistral-7B-Instruct Q4_K_M

양자화 가이드
	•	Q4_K_M: 기본 추천 (메모리↓, 품질 손실 적음)
	•	Q5_K_M: 메모리 여유 시 품질↑
	•	Q8_0: FP16 근접, 메모리/속도 비용 큼
	•	런타임 메모리(경험칙): GGUF 파일 크기 × 1.2~1.4

파일 배치 & 호출 예시(JSON 페이로드)

{
  "backend": "llama",
  "prompt": "REST API Hello world 코드를 생성해줘",
  "llama_exec_path": "/Users/mac/src/llama.cpp/build/bin/llama-cli",
  "n_threads": 8,
  "n_ctx": 2048,
  "temperature": 0.7,
  "top_k": 40,
  "top_p": 0.95,
  "extra_args": [
    "-m","/Users/mac/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf",
    "-ngl","0",
    "-n","512"
  ]
}


⸻

운영 체크리스트
	•	실행 권한: llama_exec_path → chmod +x
	•	DYLD_LIBRARY_PATH: libomp, third_party/libtorch/lib 추가
	•	길이 제어: n_ctx(컨텍스트), -n(생성 길이)
	•	라이선스 준수: 모델/데이터/코드 각각 확인
	•	텐서 차원 안전망: {N,28,28} → {N,1,28,28} unsqueeze (MNIST 1채널 전제 시)
	•	macOS DataLoader: num_workers>0 이슈 시 0으로

⸻

로드맵

단계	목표	왜/무엇
1	아티팩트 확인	runs/cnn_mnist/epoch_*.pt 생성 확인
2	추론 루틴 추가	체크포인트 로딩 → 단일/배치 예측
3	REST 예측 API	POST /ml/predict (Crow)
4	베스트만 저장	최고 검증점만 저장(early-stop/최고점 관리)
5	하이퍼파라미터 제어	CLI/REST로 epochs/batch/lr/workers 노출
6	영구 러ntime 경로	rpath 또는 쉘 초기화로 DYLD 영구화
7	PHP UI 연동	Train/Predict 버튼(공유호스팅)


⸻

자주 발생하는 이슈
	•	CUDA/cuDNN/LibTorch 불일치 → 로딩/링킹 오류. 버전 정합성 필수
	•	macOS 포크 제약 → DataLoader(num_workers>0) 문제 시 0으로
	•	llama.cpp Makefile 경고 → CMake 빌드 사용
	•	출력이 중간에 끊김 → -n(생성 길이) 및 n_ctx(컨텍스트) 확대
	•	CPU만 느림 → llama.cpp Metal 가속 (-DGGML_METAL=ON, 실행 -ngl 99)

⸻

용어 사전
	•	n_ctx: 컨텍스트 길이(토큰)
	•	-n / n_predict: 생성할 최대 토큰 수
	•	n_threads: CPU 스레드 수
	•	-ngl: GPU(Metal) 오프로딩 레이어 수 (0=CPU, 99=최대)
	•	-c/-b: 컨텍스트/배치 크기
	•	--simple-io: 단순 STDIO I/O
	•	-no-cnv: chat template 비활성화
	•	temperature/top_k/top_p: 샘플링 제어
	•	repeat_penalty: 반복 억제
	•	BOS/EOS/EOG/PAD: 시작/끝/생성종료/패딩 토큰
	•	FIM: Fill-In-the-Middle 토큰 (PRE/SUF/MID)
	•	rope(yarn): RoPE 위치 인코딩(+긴 컨텍스트 스케일링)
	•	kv cache: K/V 캐시(메모리↔속도 트레이드오프)
	•	prompt eval / eval time: 프롬프트 처리/생성 토큰당 소요
	•	TPS: 초당 생성 토큰 수

13) 자주 발생하는 이슈와 팁
	•	macOS DataLoader: num_workers > 0에서 fork 제약 이슈가 드물게 발생 → num_workers=0 권장.
	•	CUDA/cuDNN/LibTorch 버전 불일치: 로딩/링킹 에러 원인 1순위. 정확한 버전 매칭 필수.
	•	MNIST 텐서 차원: {N, 28, 28} 입력이 오면 **unsqueeze(1)**로 {N, 1, 28, 28} 보정.
	•	llama.cpp 빌드 경고: Makefile deprecated → CMake 빌드 사용.
	•	성능 측정: TPS, eval time/token으로 설정 변화 영향 확인(예: -ngl 0 ↔ 99, -b, n_ctx).

⸻

⸻

License

MIT License

Copyright (c) 2025 …

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

상용 모델/데이터 사용 시 각자 모델/데이터 라이선스를 별도로 준수해야 합니다(Llama 등).

⸻

부록: 빠른 시작(복붙용)

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
  -d '{
    "backend":"llama",
    "prompt":"C++로 멀티스레드 예제 작성해줘. 코드 블록으로만 답해줘.",
    "llama_exec_path":"/Users/.../llama.cpp/build/bin/llama-cli",
    "n_threads":8,
    "n_ctx":1024,
    "temperature":0.7,
    "top_k":40,
    "top_p":0.95,
    "extra_args":[
      "-m","/Users/.../deepseek-coder-v2-lite-instruct-q4_k_m.gguf",
      "-ngl","24",
      "-c","1024",
      "-b","1024",
      "--simple-io",
      "-no-cnv",
      "-n","128"
    ]
  }'

10) 파라미터/로그 용어 사전 (llama.cpp / 엔진 공통)
	•	n_ctx: 모델 컨텍스트 길이(토큰). 길수록 긴 프롬프트/출력 처리, 메모리↑.
	•	n_predict / -n: 생성할 최대 토큰 수.
	•	n_threads: CPU 스레드 수.
	•	-ngl: GPU(Metal)로 오프로딩할 레이어 수. 0은 CPU-only, 99는 대부분 레이어를 GPU로.
	•	-c: 컨텍스트(n_ctx) 지정 (llama.cpp).
	•	-b: 배치 크기(batch). 큰 값은 속도↑(메모리 여유 필요).
	•	--simple-io: 프롬프트/응답을 단순 STDIO로 입출력.
	•	-no-cnv: chat template 비활성(대화 모드 끄기).
	•	temperature: 확률 분포 평탄화(↑=창의↑, 안정↓).
	•	top_k: 상위 k개 토큰만 샘플링.
	•	top_p: 누적 확률 p까지 포함하는 Nucleus Sampling.
	•	repeat_penalty: 반복 억제 가중.
	•	mirostat: 동적 엔트로피 타깃팅(일부 빌드).
	•	logits / logit bias: 토큰 로짓(미정규화 점수) / 특정 토큰 선택 편향값.
	•	BOS/EOS/EOG/PAD: 문장 시작/끝/생성 종료/패딩 토큰.
	•	FIM 토큰: Fill-In-the-Middle 프롬프트용 특수 토큰 (PRE/MID/SUF).
	•	rope: RoPE 위치 인코딩; yarn 스케일링은 긴 컨텍스트 확장 기법.
	•	kv cache: Key/Value 캐시. 길수록 메모리↑, 재사용으로 속도↑.
	•	prompt eval time / eval time: 프롬프트 소화 시간 / 토큰 생성 시간(토큰당 ms).
	•	tokens per second (TPS): 초당 생성 토큰 수.
	•	load time: 모델/가중치 로드 및 초기화 시간.

⸻

-----

DeepSeek-Coder-V2-Lite Instruct로 선택
https://huggingface.co/models

맨 아래쪽 sugatoxay/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M-GGUF 를 추천한 건,
지금 상황이 “자체 LLM 패널에서 추론 테스트” 목적이라서 가장 가볍고 실행이 빠른 GGUF 변환본이 필요하기 때문입니다.

https://huggingface.co/sugatoray/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M-GGUF/blob/main/deepseek-coder-v2-lite-instruct-q4_k_m.gguf

•	데이터 차원 보정은 모델 구현(예: cnn_mnist::Model)이 NCHW(1채널) 전제를 가질 때만 필요합니다. 이미 MNIST DataSet이 {N,1,28,28}로 배치한다면 unsqueeze 블록은 건너뛰지만, 위처럼 조건부로 안전망을 두면 환경 차이(C++ API 버전/커스텀 Dataset)에서도 안전합니다.
•	macOS CPU에서 num_workers>0일 때 fork 제약으로 드물게 문제가 생기면 cfg.num_workers=0로 낮춰보세요.


[build 완료]
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


$ export DYLD_LIBRARY_PATH="$(brew --prefix libomp)/lib:../third_party/libtorch/lib:${DYLD_LIBRARY_PATH:-}"

[최종]
$ mac@azabell-mac ml-engine % curl -s http://localhost:18080/health
curl -s http://localhost:18080/ml/models
curl -s -X POST http://localhost:18080/llm/generate \
  -H "Content-Type: application/json" \
  -d '{"backend":"mock","prompt":"Hello"}'
{
  "service": "ml-engine",
  "status": "ok"
}{
  "models": [
    "cnn_mnist"
  ]
}{
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
}%                                       


먼저 테스트 (llama.cpp)
mac@azabell-mac ml-engine % cd /Users/mac/Desktop/workspace/miniGPT/ml-engine/third_party/llama.cpp
git pull
make clean
cmake -B build -DGGML_METAL=ON
cmake --build build -j
fatal: not a git repository (or any of the parent directories): .git
Makefile:2: *** The Makefile build is deprecated. Use the CMake build instead. For more details, see https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md.  Stop.
fatal: not a git repository (or any of the parent directories): .git
fatal: not a git repository (or any of the parent directories): .git
-- Warning: ccache not found - consider installing it for faster compilation or disable this warning with GGML_CCACHE=OFF
-- CMAKE_SYSTEM_PROCESSOR: x86_64
-- GGML_SYSTEM_ARCH: x86
-- Including CPU backend
-- Accelerate framework found
-- Could NOT find OpenMP_C (missing: OpenMP_C_FLAGS OpenMP_C_LIB_NAMES) 
-- Could NOT find OpenMP_CXX (missing: OpenMP_CXX_FLAGS OpenMP_CXX_LIB_NAMES) 
-- Could NOT find OpenMP (missing: OpenMP_C_FOUND OpenMP_CXX_FOUND) 
CMake Warning at ggml/src/ggml-cpu/CMakeLists.txt:79 (message):
  OpenMP not found
Call Stack (most recent call first):
  ggml/src/CMakeLists.txt:372 (ggml_add_cpu_backend_variant_impl)


-- x86 detected
-- Adding CPU backend variant ggml-cpu: -march=native 
-- BLAS found, Libraries: /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework
-- BLAS found, Includes: 
-- Including BLAS backend
-- Metal framework found
-- Including METAL backend
-- ggml version: 0.0.0
-- ggml commit:  unknown
CMake Warning at common/CMakeLists.txt:32 (message):
  Git repository not found; to enable automatic generation of build info,
  make sure Git is installed and the project is a Git repository.


-- Configuring done (2.6s)
-- Generating done (0.5s)
-- Build files have been written to: /Users/mac/Desktop/workspace/miniGPT/ml-engine/third_party/llama.cpp/build
[  0%] Building CXX object common/CMakeFiles/build_info.dir/build-info.cpp.o
[  1%] Built target sha1
[  2%] Built target xxhash
[  2%] Built target sha256
[  3%] Built target llama-llava-cli
[  3%] Building CXX object ggml/src/CMakeFiles/ggml-base.dir/ggml-backend.cpp.o
[  3%] Building CXX object ggml/src/CMakeFiles/ggml-base.dir/ggml-opt.cpp.o
[  4%] Built target llama-qwen2vl-cli
[  4%] Building CXX object ggml/src/CMakeFiles/ggml-base.dir/ggml.cpp.o
[  4%] Building C object ggml/src/CMakeFiles/ggml-base.dir/ggml-quants.c.o
[  5%] Building CXX object ggml/src/CMakeFiles/ggml-base.dir/gguf.cpp.o
[  6%] Built target llama-minicpmv-cli
[  7%] Building CXX object ggml/src/CMakeFiles/ggml-base.dir/ggml-threading.cpp.o
[  8%] Building C object ggml/src/CMakeFiles/ggml-base.dir/ggml-alloc.c.o
[  9%] Built target llama-gemma3-cli
[ 10%] Building C object ggml/src/CMakeFiles/ggml-base.dir/ggml.c.o
[ 10%] Built target build_info
[ 10%] Linking CXX shared library ../../bin/libggml-base.dylib
[ 10%] Built target ggml-base
[ 11%] Linking C shared library ../../../bin/libggml-metal.dylib
[ 12%] Linking CXX shared library ../../../bin/libggml-blas.dylib
[ 12%] Linking CXX shared library ../../bin/libggml-cpu.dylib
[ 12%] Built target ggml-blas
[ 13%] Built target ggml-metal
[ 19%] Built target ggml-cpu
[ 19%] Linking CXX shared library ../../bin/libggml.dylib
[ 19%] Built target ggml
[ 21%] Linking CXX executable ../../bin/llama-gguf-hash
[ 21%] Linking CXX executable ../../bin/llama-gguf
[ 22%] Linking CXX shared library ../bin/libllama.dylib
[ 22%] Built target llama-gguf
[ 22%] Built target llama-gguf-hash
[ 32%] Built target llama
[ 33%] Linking C executable ../bin/test-c
[ 33%] Linking CXX executable ../../bin/llama-simple
[ 33%] Linking CXX executable ../../bin/llama-simple-chat
[ 33%] Linking CXX shared library ../../bin/libmtmd.dylib
[ 34%] Linking CXX static library libcommon.a
[ 34%] Built target test-c
[ 34%] Built target llama-simple
[ 35%] Built target llama-simple-chat
[ 37%] Built target mtmd
[ 42%] Built target common
[ 42%] Linking CXX executable ../bin/test-tokenizer-1-spm
[ 42%] Linking CXX executable ../bin/test-gbnf-validator
[ 42%] Linking CXX executable ../bin/test-grammar-integration
[ 43%] Linking CXX executable ../bin/test-model-load-cancel
[ 43%] Linking CXX executable ../bin/test-llama-grammar
[ 44%] Linking CXX executable ../../bin/llama-export-lora
[ 44%] Linking CXX executable ../bin/test-quantize-perf
[ 44%] Linking CXX executable ../../bin/llama-embedding
[ 45%] Linking CXX executable ../bin/test-log
[ 45%] Linking CXX executable ../../bin/llama-imatrix
[ 45%] Linking CXX executable ../../bin/llama-eval-callback
[ 45%] Linking CXX executable ../bin/test-autorelease
[ 45%] Linking CXX executable ../../bin/llama-bench
[ 46%] Linking CXX executable ../bin/test-json-partial
[ 45%] Linking CXX executable ../bin/test-chat-template
[ 47%] Linking CXX executable ../../bin/llama-convert-llama2c-to-ggml
[ 46%] Linking CXX executable ../../bin/llama-parallel
[ 47%] Linking CXX executable ../../bin/llama-batched-bench
[ 47%] Linking CXX executable ../../bin/llama-run
[ 49%] Linking CXX executable ../bin/test-chat
[ 49%] Linking CXX executable ../../bin/llama-lookup-stats
[ 47%] Linking CXX executable ../../bin/llama-speculative
[ 49%] Linking CXX executable ../../bin/llama-passkey
[ 49%] Linking CXX executable ../../bin/llama-diffusion-cli
[ 49%] Linking CXX executable ../../bin/llama-vdot
[ 49%] Linking CXX executable ../../bin/llama-tts
[ 49%] Linking CXX executable ../../bin/llama-mtmd-cli
[ 50%] Linking CXX executable ../../bin/llama-server
[ 50%] Linking CXX executable ../../bin/llama-finetune
[ 50%] Linking CXX executable ../bin/test-rope
[ 51%] Linking CXX executable ../bin/test-thread-safety
[ 52%] Linking CXX executable ../../bin/llama-perplexity
[ 52%] Linking CXX executable ../bin/test-arg-parser
[ 52%] Linking CXX executable ../../bin/llama-gguf-split
[ 53%] Linking CXX executable ../../bin/llama-q8dot
[ 54%] Linking CXX executable ../../bin/llama-lookahead
[ 54%] Linking CXX executable ../../bin/llama-quantize
[ 54%] Linking CXX executable ../../bin/llama-lookup-create
[ 55%] Linking CXX executable ../bin/test-mtmd-c-api
[ 55%] Linking CXX executable ../../bin/llama-batched
[ 55%] Linking CXX executable ../../bin/llama-gritlm
[ 55%] Linking CXX executable ../bin/test-quantize-stats
[ 56%] Linking CXX executable ../bin/test-grammar-parser
[ 56%] Linking CXX executable ../bin/test-json-schema-to-grammar
[ 56%] Linking CXX executable ../../bin/llama-gen-docs
[ 56%] Linking CXX executable ../../bin/llama-lookup
[ 57%] Linking CXX executable ../bin/test-tokenizer-0
[ 57%] Linking CXX executable ../bin/test-chat-parser
[ 57%] Linking CXX executable ../bin/test-quantize-fns
[ 59%] Linking CXX executable ../../bin/llama-speculative-simple
[ 59%] Linking CXX executable ../bin/test-sampling
[ 59%] Linking CXX executable ../../bin/llama-lookup-merge
[ 59%] Linking CXX executable ../../bin/llama-cvector-generator
[ 60%] Linking CXX executable ../../bin/llama-cli
[ 60%] Linking CXX executable ../../bin/llama-retrieval
[ 60%] Linking CXX executable ../bin/test-gguf
[ 61%] Linking CXX executable ../bin/test-backend-ops
[ 60%] Linking CXX executable ../../bin/llama-save-load-state
[ 61%] Linking CXX executable ../bin/test-regex-partial
[ 62%] Linking CXX executable ../bin/test-barrier
[ 62%] Linking CXX executable ../bin/test-tokenizer-1-bpe
[ 63%] Linking CXX executable ../../bin/llama-tokenize
[ 63%] Built target test-json-partial
[ 64%] Built target test-gbnf-validator
[ 65%] Built target test-llama-grammar
[ 66%] Built target test-log
[ 66%] Built target llama-eval-callback
[ 67%] Built target test-model-load-cancel
[ 67%] Built target llama-batched
[ 68%] Built target llama-bench
[ 69%] Built target test-gguf
[ 70%] Built target llama-embedding
[ 70%] Built target llama-export-lora
[ 71%] Built target test-quantize-perf
[ 72%] Built target test-tokenizer-1-spm
[ 73%] Built target test-grammar-integration
[ 73%] Built target llama-tts
[ 73%] Built target llama-quantize
[ 74%] Built target test-arg-parser
[ 75%] Built target llama-lookup-merge
[ 77%] Built target test-grammar-parser
[ 77%] Built target llama-convert-llama2c-to-ggml
[ 76%] Built target test-autorelease
[ 78%] Built target llama-passkey
[ 79%] Built target llama-batched-bench
[ 80%] Built target test-rope
[ 80%] Built target llama-gguf-split
[ 82%] Built target llama-speculative
[ 84%] Built target llama-vdot
[ 83%] Built target test-quantize-fns
[ 84%] Built target llama-imatrix
[ 85%] Built target test-sampling
[ 85%] Built target test-chat
[ 86%] Built target llama-save-load-state
[ 87%] Built target llama-run
[ 88%] Built target llama-parallel
[ 89%] Built target test-chat-template
[ 90%] Built target llama-lookup-create
[ 90%] Built target llama-tokenize
[ 90%] Built target llama-lookahead
[ 90%] Built target test-tokenizer-0
[ 90%] Built target llama-speculative-simple
[ 91%] Built target test-backend-ops
[ 92%] Built target test-chat-parser
[ 92%] Built target llama-perplexity
[ 93%] Built target llama-gritlm
[ 94%] Built target llama-diffusion-cli
[ 95%] Built target llama-finetune
[ 96%] Built target test-regex-partial
[ 97%] Built target test-quantize-stats
[ 97%] Built target llama-q8dot
[ 98%] Built target test-json-schema-to-grammar
[ 99%] Built target llama-gen-docs
[ 99%] Built target llama-lookup
[ 99%] Built target llama-server
[ 99%] Built target test-thread-safety
[ 99%] Built target llama-cli
[ 99%] Built target test-mtmd-c-api
[ 99%] Built target llama-lookup-stats
[100%] Built target llama-retrieval
[100%] Built target llama-cvector-generator
[100%] Built target llama-mtmd-cli
[100%] Built target test-tokenizer-1-bpe
[100%] Built target test-barrier
mac@azabell-mac llama.cpp % /Users/mac/Desktop/workspace/miniGPT/ml-engine/third_party/llama.cpp/build/bin/llama-cli \
  -p "테스트\n\n" \
  -t 8 -c 2048 --temp 0.7 --top-k 40 --top-p 0.95 \
  -m /Users/mac/Desktop/workspace/miniGPT/ml-engine/models/deepseek-coder-v2-lite-instruct-q4_k_m.gguf \
  -ngl 0 --simple-io -n 64 -r "User:"
build: 0 (unknown) with Apple clang version 16.0.0 (clang-1600.0.26.6) for x86_64-apple-darwin24.5.0
main: llama backend init
main: load the model and apply lora adapter, if any
llama_model_load_from_file_impl: using device Metal (Apple M2) - 10922 MiB free
llama_model_loader: loaded meta data with 38 key-value pairs and 377 tensors from /Users/mac/Desktop/workspace/miniGPT/ml-engine/models/deepseek-coder-v2-lite-instruct-q4_k_m.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.name str              = DeepSeek-Coder-V2-Lite-Instruct
llama_model_loader: - kv   2:                      deepseek2.block_count u32              = 27
llama_model_loader: - kv   3:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv   4:                 deepseek2.embedding_length u32              = 2048
llama_model_loader: - kv   5:              deepseek2.feed_forward_length u32              = 10944
llama_model_loader: - kv   6:             deepseek2.attention.head_count u32              = 16
llama_model_loader: - kv   7:          deepseek2.attention.head_count_kv u32              = 16
llama_model_loader: - kv   8:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv   9: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  10:                deepseek2.expert_used_count u32              = 6
llama_model_loader: - kv  11:                          general.file_type u32              = 15
llama_model_loader: - kv  12:        deepseek2.leading_dense_block_count u32              = 1
llama_model_loader: - kv  13:                       deepseek2.vocab_size u32              = 102400
llama_model_loader: - kv  14:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  15:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  16:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  17:       deepseek2.expert_feed_forward_length u32              = 1408
llama_model_loader: - kv  18:                     deepseek2.expert_count u32              = 64
llama_model_loader: - kv  19:              deepseek2.expert_shared_count u32              = 2
llama_model_loader: - kv  20:             deepseek2.expert_weights_scale f32              = 1.000000
llama_model_loader: - kv  21:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  22:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  23:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  24: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  25: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.070700
llama_model_loader: - kv  26:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  27:                         tokenizer.ggml.pre str              = deepseek-llm
llama_model_loader: - kv  28:                      tokenizer.ggml.tokens arr[str,102400]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  29:                  tokenizer.ggml.token_type arr[i32,102400]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  30:                      tokenizer.ggml.merges arr[str,99757]   = ["Ġ Ġ", "Ġ t", "Ġ a", "i n", "h e...
llama_model_loader: - kv  31:                tokenizer.ggml.bos_token_id u32              = 100000
llama_model_loader: - kv  32:                tokenizer.ggml.eos_token_id u32              = 100001
llama_model_loader: - kv  33:            tokenizer.ggml.padding_token_id u32              = 100001
llama_model_loader: - kv  34:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  35:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  36:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  37:               general.quantization_version u32              = 2
llama_model_loader: - type  f32:  108 tensors
llama_model_loader: - type q5_0:   14 tensors
llama_model_loader: - type q8_0:   13 tensors
llama_model_loader: - type q4_K:  229 tensors
llama_model_loader: - type q6_K:   13 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = Q4_K - Medium
print_info: file size   = 9.65 GiB (5.28 BPW) 
load: control-looking token: 100002 '<｜fim▁hole｜>' was not control-type; this is probably a bug in the model. its type will be overridden
load: control-looking token: 100004 '<｜fim▁end｜>' was not control-type; this is probably a bug in the model. its type will be overridden
load: control-looking token: 100003 '<｜fim▁begin｜>' was not control-type; this is probably a bug in the model. its type will be overridden
load: special_eos_id is not in special_eog_ids - the tokenizer config may be incorrect
load: printing all EOG tokens:
load:   - 100001 ('<｜end▁of▁sentence｜>')
load: special tokens cache size = 2400
load: token to piece cache size = 0.6661 MB
print_info: arch             = deepseek2
print_info: vocab_only       = 0
print_info: n_ctx_train      = 163840
print_info: n_embd           = 2048
print_info: n_layer          = 27
print_info: n_head           = 16
print_info: n_head_kv        = 16
print_info: n_rot            = 64
print_info: n_swa            = 0
print_info: is_swa_any       = 0
print_info: n_embd_head_k    = 192
print_info: n_embd_head_v    = 128
print_info: n_gqa            = 1
print_info: n_embd_k_gqa     = 3072
print_info: n_embd_v_gqa     = 2048
print_info: f_norm_eps       = 0.0e+00
print_info: f_norm_rms_eps   = 1.0e-06
print_info: f_clamp_kqv      = 0.0e+00
print_info: f_max_alibi_bias = 0.0e+00
print_info: f_logit_scale    = 0.0e+00
print_info: f_attn_scale     = 0.0e+00
print_info: n_ff             = 10944
print_info: n_expert         = 64
print_info: n_expert_used    = 6
print_info: causal attn      = 1
print_info: pooling type     = 0
print_info: rope type        = 0
print_info: rope scaling     = yarn
print_info: freq_base_train  = 10000.0
print_info: freq_scale_train = 0.025
print_info: n_ctx_orig_yarn  = 4096
print_info: rope_finetuned   = unknown
print_info: model type       = 16B
print_info: model params     = 15.71 B
print_info: general.name     = DeepSeek-Coder-V2-Lite-Instruct
print_info: n_layer_dense_lead   = 1
print_info: n_lora_q             = 0
print_info: n_lora_kv            = 512
print_info: n_embd_head_k_mla    = 0
print_info: n_embd_head_v_mla    = 0
print_info: n_ff_exp             = 1408
print_info: n_expert_shared      = 2
print_info: expert_weights_scale = 1.0
print_info: expert_weights_norm  = 0
print_info: expert_gating_func   = softmax
print_info: rope_yarn_log_mul    = 0.0707
print_info: vocab type       = BPE
print_info: n_vocab          = 102400
print_info: n_merges         = 99757
print_info: BOS token        = 100000 '<｜begin▁of▁sentence｜>'
print_info: EOS token        = 100001 '<｜end▁of▁sentence｜>'
print_info: EOT token        = 100001 '<｜end▁of▁sentence｜>'
print_info: PAD token        = 100001 '<｜end▁of▁sentence｜>'
print_info: LF token         = 185 'Ċ'
print_info: FIM PRE token    = 100003 '<｜fim▁begin｜>'
print_info: FIM SUF token    = 100002 '<｜fim▁hole｜>'
print_info: FIM MID token    = 100004 '<｜fim▁end｜>'
print_info: EOG token        = 100001 '<｜end▁of▁sentence｜>'
print_info: max token length = 256
load_tensors: loading model tensors, this can take a while... (mmap = true)
load_tensors: offloading 0 repeating layers to GPU
load_tensors: offloaded 0/28 layers to GPU
load_tensors:   CPU_Mapped model buffer size =  9880.47 MiB
.....................................................................................
llama_context: constructing llama_context
llama_context: n_seq_max     = 1
llama_context: n_ctx         = 2048
llama_context: n_ctx_per_seq = 2048
llama_context: n_batch       = 2048
llama_context: n_ubatch      = 512
llama_context: causal_attn   = 1
llama_context: flash_attn    = 0
llama_context: kv_unified    = false
llama_context: freq_base     = 10000.0
llama_context: freq_scale    = 0.025
llama_context: n_ctx_per_seq (2048) < n_ctx_train (163840) -- the full capacity of the model will not be utilized
ggml_metal_init: allocating
ggml_metal_init: found device: Apple M2
ggml_metal_init: picking default device: Apple M2
ggml_metal_load_library: using embedded metal library
ggml_metal_init: GPU name:   Apple M2
ggml_metal_init: GPU family: MTLGPUFamilyApple8  (1008)
ggml_metal_init: GPU family: MTLGPUFamilyCommon3 (3003)
ggml_metal_init: GPU family: MTLGPUFamilyMetal3  (5001)
ggml_metal_init: simdgroup reduction   = true
ggml_metal_init: simdgroup matrix mul. = true
ggml_metal_init: has residency sets    = false
ggml_metal_init: has bfloat            = true
ggml_metal_init: use bfloat            = false
ggml_metal_init: hasUnifiedMemory      = true
ggml_metal_init: recommendedMaxWorkingSetSize  = 11453.25 MB
ggml_metal_init: skipping kernel_get_rows_bf16                     (not supported)
ggml_metal_init: skipping kernel_set_rows_bf16                     (not supported)
ggml_metal_init: skipping kernel_mul_mv_bf16_f32                   (not supported)
ggml_metal_init: skipping kernel_mul_mv_bf16_f32_c4                (not supported)
ggml_metal_init: skipping kernel_mul_mv_bf16_f32_1row              (not supported)
ggml_metal_init: skipping kernel_mul_mv_bf16_f32_l4                (not supported)
ggml_metal_init: skipping kernel_mul_mv_bf16_bf16                  (not supported)
ggml_metal_init: skipping kernel_mul_mv_id_bf16_f32                (not supported)
ggml_metal_init: skipping kernel_mul_mm_bf16_f32                   (not supported)
ggml_metal_init: skipping kernel_mul_mm_id_bf16_f16                (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_bf16_h64           (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_bf16_h80           (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_bf16_h96           (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_bf16_h112          (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_bf16_h128          (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_bf16_h192          (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_bf16_hk192_hv128   (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_bf16_h256          (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_bf16_hk576_hv512   (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_vec_bf16_h64       (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_vec_bf16_h96       (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_vec_bf16_h128      (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_vec_bf16_h192      (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_vec_bf16_hk192_hv128 (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_vec_bf16_h256      (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_vec_bf16_hk576_hv512 (not supported)
ggml_metal_init: skipping kernel_cpy_f32_bf16                      (not supported)
ggml_metal_init: skipping kernel_cpy_bf16_f32                      (not supported)
ggml_metal_init: skipping kernel_cpy_bf16_bf16                     (not supported)
llama_context:        CPU  output buffer size =     0.39 MiB
llama_kv_cache_unified:        CPU KV buffer size =   540.00 MiB
llama_kv_cache_unified: size =  540.00 MiB (  2048 cells,  27 layers,  1/1 seqs), K (f16):  324.00 MiB, V (f16):  216.00 MiB
llama_context:        CPU compute buffer size =   236.25 MiB
llama_context: graph nodes  = 1844
llama_context: graph splits = 432 (with bs=512), 1 (with bs=1)
common_init_from_params: added <｜end▁of▁sentence｜> logit bias = -inf
common_init_from_params: setting dry_penalty_last_n to ctx_size = 2048
common_init_from_params: warming up the model with an empty run - please wait ... (--no-warmup to disable)
main: llama threadpool init, n_threads = 8
main: chat template is available, enabling conversation mode (disable it with -no-cnv)
*** User-specified prompt will pre-start conversation, did you mean to set --system-prompt (-sys) instead?
main: chat template example:
You are a helpful assistant

User: Hello

Assistant: Hi there<｜end▁of▁sentence｜>User: How are you?

Assistant:

system_info: n_threads = 8 (n_threads_batch = 8) / 8 | Metal : EMBED_LIBRARY = 1 | CPU : SSE3 = 1 | SSSE3 = 1 | LLAMAFILE = 1 | ACCELERATE = 1 | REPACK = 1 | 

main: interactive mode on.
Reverse prompt: 'User:'
sampler seed: 1217957002
sampler params: 
        repeat_last_n = 64, repeat_penalty = 1.000, frequency_penalty = 0.000, presence_penalty = 0.000
        dry_multiplier = 0.000, dry_base = 1.750, dry_allowed_length = 2, dry_penalty_last_n = 2048
        top_k = 40, top_p = 0.950, min_p = 0.050, xtc_probability = 0.000, xtc_threshold = 0.100, typical_p = 1.000, top_n_sigma = -1.000, temp = 0.700
        mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
sampler chain: logits -> logit-bias -> penalties -> dry -> top-n-sigma -> top-k -> typical -> top-p -> min-p -> xtc -> temp-ext -> dist 
generate: n_ctx = 2048, n_batch = 2048, n_predict = 64, n_keep = 1

== Running in interactive mode. ==
 - Press Ctrl+C to interject at any time.
 - Press Return to return control to the AI.
 - To return control without starting a new line, end your input with '/'.
 - If you want to submit another line, end your input with '\'.
 - Not using system message. To change it, set a different value via -sys PROMPT

User: 테스트



Assistant: 여기에 응답이 나오면 성공.



load_tensors:   CPU_Mapped model buffer size =  9880.47 MiB
.....................................................................................
llama_context:
이런거 용어 뜻 다해주기
-=---------


[Success]
 /Users/mac/Desktop/workspace/miniGPT/ml-engine/third_party/llama.cpp/build/bin/llama-cli \
  -p "테스트\n\n" \
  -t 8 -c 2048 --temp 0.7 --top-k 40 --top-p 0.95 \
  -m /Users/mac/Desktop/workspace/miniGPT/ml-engine/models/deepseek-coder-v2-lite-instruct-q4_k_m.gguf \
  -ngl 0 --simple-io -n 64 -r "User:"


[요청]
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


[response]
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


[서버]
llama_perf_sampler_print:    sampling time =      14.28 ms /   189 runs   (    0.08 ms per token, 13237.15 tokens per second)
llama_perf_context_print:        load time =   12103.93 ms
llama_perf_context_print: prompt eval time =    3262.02 ms /    61 tokens (   53.48 ms per token,    18.70 tokens per second)
llama_perf_context_print:        eval time =   14670.49 ms /   127 runs   (  115.52 ms per token,     8.66 tokens per second)
llama_perf_context_print:       total time =   18003.63 ms /   188 tokens
llama_perf_context_print:    graphs reused =        122
ggml_metal_free: deallocating
(2025-08-13 16:40:28) [INFO    ] Response: 0x10300f218 /llm/generate 200 0
(2025-08-13 16:40:34) [INFO    ] Request: 127.0.0.1:54225 0x103010618 HTTP/1.1 POST /llm/generate
build: 0 (unknown) with Apple clang version 16.0.0 (clang-1600.0.26.6) for x86_64-apple-darwin24.5.0
main: llama backend init
main: load the model and apply lora adapter, if any
llama_model_load_from_file_impl: using device Metal (Apple M2) - 10922 MiB free
llama_model_loader: loaded meta data with 38 key-value pairs and 377 tensors from /Users/mac/Desktop/workspace/miniGPT/ml-engine/models/deepseek-coder-v2-lite-instruct-q4_k_m.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.name str              = DeepSeek-Coder-V2-Lite-Instruct
llama_model_loader: - kv   2:                      deepseek2.block_count u32              = 27
llama_model_loader: - kv   3:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv   4:                 deepseek2.embedding_length u32              = 2048
llama_model_loader: - kv   5:              deepseek2.feed_forward_length u32              = 10944
llama_model_loader: - kv   6:             deepseek2.attention.head_count u32              = 16
llama_model_loader: - kv   7:          deepseek2.attention.head_count_kv u32              = 16
llama_model_loader: - kv   8:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv   9: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  10:                deepseek2.expert_used_count u32              = 6
llama_model_loader: - kv  11:                          general.file_type u32              = 15
llama_model_loader: - kv  12:        deepseek2.leading_dense_block_count u32              = 1
llama_model_loader: - kv  13:                       deepseek2.vocab_size u32              = 102400
llama_model_loader: - kv  14:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  15:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  16:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  17:       deepseek2.expert_feed_forward_length u32              = 1408
llama_model_loader: - kv  18:                     deepseek2.expert_count u32              = 64
llama_model_loader: - kv  19:              deepseek2.expert_shared_count u32              = 2
llama_model_loader: - kv  20:             deepseek2.expert_weights_scale f32              = 1.000000
llama_model_loader: - kv  21:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  22:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  23:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  24: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  25: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.070700
llama_model_loader: - kv  26:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  27:                         tokenizer.ggml.pre str              = deepseek-llm
llama_model_loader: - kv  28:                      tokenizer.ggml.tokens arr[str,102400]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  29:                  tokenizer.ggml.token_type arr[i32,102400]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  30:                      tokenizer.ggml.merges arr[str,99757]   = ["Ġ Ġ", "Ġ t", "Ġ a", "i n", "h e...
llama_model_loader: - kv  31:                tokenizer.ggml.bos_token_id u32              = 100000
llama_model_loader: - kv  32:                tokenizer.ggml.eos_token_id u32              = 100001
llama_model_loader: - kv  33:            tokenizer.ggml.padding_token_id u32              = 100001
llama_model_loader: - kv  34:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  35:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  36:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  37:               general.quantization_version u32              = 2
llama_model_loader: - type  f32:  108 tensors
llama_model_loader: - type q5_0:   14 tensors
llama_model_loader: - type q8_0:   13 tensors
llama_model_loader: - type q4_K:  229 tensors
llama_model_loader: - type q6_K:   13 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = Q4_K - Medium
print_info: file size   = 9.65 GiB (5.28 BPW) 
load: control-looking token: 100002 '<｜fim▁hole｜>' was not control-type; this is probably a bug in the model. its type will be overridden
load: control-looking token: 100004 '<｜fim▁end｜>' was not control-type; this is probably a bug in the model. its type will be overridden
load: control-looking token: 100003 '<｜fim▁begin｜>' was not control-type; this is probably a bug in the model. its type will be overridden
load: special_eos_id is not in special_eog_ids - the tokenizer config may be incorrect
load: printing all EOG tokens:
load:   - 100001 ('<｜end▁of▁sentence｜>')
load: special tokens cache size = 2400
load: token to piece cache size = 0.6661 MB
print_info: arch             = deepseek2
print_info: vocab_only       = 0
print_info: n_ctx_train      = 163840
print_info: n_embd           = 2048
print_info: n_layer          = 27
print_info: n_head           = 16
print_info: n_head_kv        = 16
print_info: n_rot            = 64
print_info: n_swa            = 0
print_info: is_swa_any       = 0
print_info: n_embd_head_k    = 192
print_info: n_embd_head_v    = 128
print_info: n_gqa            = 1
print_info: n_embd_k_gqa     = 3072
print_info: n_embd_v_gqa     = 2048
print_info: f_norm_eps       = 0.0e+00
print_info: f_norm_rms_eps   = 1.0e-06
print_info: f_clamp_kqv      = 0.0e+00
print_info: f_max_alibi_bias = 0.0e+00
print_info: f_logit_scale    = 0.0e+00
print_info: f_attn_scale     = 0.0e+00
print_info: n_ff             = 10944
print_info: n_expert         = 64
print_info: n_expert_used    = 6
print_info: causal attn      = 1
print_info: pooling type     = 0
print_info: rope type        = 0
print_info: rope scaling     = yarn
print_info: freq_base_train  = 10000.0
print_info: freq_scale_train = 0.025
print_info: n_ctx_orig_yarn  = 4096
print_info: rope_finetuned   = unknown
print_info: model type       = 16B
print_info: model params     = 15.71 B
print_info: general.name     = DeepSeek-Coder-V2-Lite-Instruct
print_info: n_layer_dense_lead   = 1
print_info: n_lora_q             = 0
print_info: n_lora_kv            = 512
print_info: n_embd_head_k_mla    = 0
print_info: n_embd_head_v_mla    = 0
print_info: n_ff_exp             = 1408
print_info: n_expert_shared      = 2
print_info: expert_weights_scale = 1.0
print_info: expert_weights_norm  = 0
print_info: expert_gating_func   = softmax
print_info: rope_yarn_log_mul    = 0.0707
print_info: vocab type       = BPE
print_info: n_vocab          = 102400
print_info: n_merges         = 99757
print_info: BOS token        = 100000 '<｜begin▁of▁sentence｜>'
print_info: EOS token        = 100001 '<｜end▁of▁sentence｜>'
print_info: EOT token        = 100001 '<｜end▁of▁sentence｜>'
print_info: PAD token        = 100001 '<｜end▁of▁sentence｜>'
print_info: LF token         = 185 'Ċ'
print_info: FIM PRE token    = 100003 '<｜fim▁begin｜>'
print_info: FIM SUF token    = 100002 '<｜fim▁hole｜>'
print_info: FIM MID token    = 100004 '<｜fim▁end｜>'
print_info: EOG token        = 100001 '<｜end▁of▁sentence｜>'
print_info: max token length = 256
load_tensors: loading model tensors, this can take a while... (mmap = true)

load_tensors: offloading 24 repeating layers to GPU
load_tensors: offloaded 24/28 layers to GPU
load_tensors: Metal_Mapped model buffer size =  8742.55 MiB
load_tensors:   CPU_Mapped model buffer size =  2567.24 MiB
...................................................................................
llama_context: constructing llama_context
llama_context: n_seq_max     = 1
llama_context: n_ctx         = 1024
llama_context: n_ctx_per_seq = 1024
llama_context: n_batch       = 1024
llama_context: n_ubatch      = 512
llama_context: causal_attn   = 1
llama_context: flash_attn    = 0
llama_context: kv_unified    = false
llama_context: freq_base     = 10000.0
llama_context: freq_scale    = 0.025
llama_context: n_ctx_per_seq (1024) < n_ctx_train (163840) -- the full capacity of the model will not be utilized
ggml_metal_init: allocating
ggml_metal_init: found device: Apple M2
ggml_metal_init: picking default device: Apple M2
ggml_metal_load_library: using embedded metal library
ggml_metal_init: GPU name:   Apple M2
ggml_metal_init: GPU family: MTLGPUFamilyApple8  (1008)
ggml_metal_init: GPU family: MTLGPUFamilyCommon3 (3003)
ggml_metal_init: GPU family: MTLGPUFamilyMetal3  (5001)
ggml_metal_init: simdgroup reduction   = true
ggml_metal_init: simdgroup matrix mul. = true
ggml_metal_init: has residency sets    = false
ggml_metal_init: has bfloat            = true
ggml_metal_init: use bfloat            = false
ggml_metal_init: hasUnifiedMemory      = true
ggml_metal_init: recommendedMaxWorkingSetSize  = 11453.25 MB
ggml_metal_init: skipping kernel_get_rows_bf16                     (not supported)
ggml_metal_init: skipping kernel_set_rows_bf16                     (not supported)
ggml_metal_init: skipping kernel_mul_mv_bf16_f32                   (not supported)
ggml_metal_init: skipping kernel_mul_mv_bf16_f32_c4                (not supported)
ggml_metal_init: skipping kernel_mul_mv_bf16_f32_1row              (not supported)
ggml_metal_init: skipping kernel_mul_mv_bf16_f32_l4                (not supported)
ggml_metal_init: skipping kernel_mul_mv_bf16_bf16                  (not supported)
ggml_metal_init: skipping kernel_mul_mv_id_bf16_f32                (not supported)
ggml_metal_init: skipping kernel_mul_mm_bf16_f32                   (not supported)
ggml_metal_init: skipping kernel_mul_mm_id_bf16_f16                (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_bf16_h64           (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_bf16_h80           (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_bf16_h96           (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_bf16_h112          (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_bf16_h128          (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_bf16_h192          (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_bf16_hk192_hv128   (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_bf16_h256          (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_bf16_hk576_hv512   (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_vec_bf16_h64       (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_vec_bf16_h96       (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_vec_bf16_h128      (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_vec_bf16_h192      (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_vec_bf16_hk192_hv128 (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_vec_bf16_h256      (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_vec_bf16_hk576_hv512 (not supported)
ggml_metal_init: skipping kernel_cpy_f32_bf16                      (not supported)
ggml_metal_init: skipping kernel_cpy_bf16_f32                      (not supported)
ggml_metal_init: skipping kernel_cpy_bf16_bf16                     (not supported)
llama_context:        CPU  output buffer size =     0.39 MiB
llama_kv_cache_unified:      Metal KV buffer size =   240.00 MiB
llama_kv_cache_unified:        CPU KV buffer size =    30.00 MiB
llama_kv_cache_unified: size =  270.00 MiB (  1024 cells,  27 layers,  1/1 seqs), K (f16):  162.00 MiB, V (f16):  108.00 MiB
llama_context:      Metal compute buffer size =    71.51 MiB
llama_context:        CPU compute buffer size =   200.00 MiB
llama_context: graph nodes  = 1844
llama_context: graph splits = 49 (with bs=512), 3 (with bs=1)
common_init_from_params: added <｜end▁of▁sentence｜> logit bias = -inf
common_init_from_params: setting dry_penalty_last_n to ctx_size = 1024
common_init_from_params: warming up the model with an empty run - please wait ... (--no-warmup to disable)
main: llama threadpool init, n_threads = 8

system_info: n_threads = 8 (n_threads_batch = 8) / 8 | Metal : EMBED_LIBRARY = 1 | CPU : SSE3 = 1 | SSSE3 = 1 | LLAMAFILE = 1 | ACCELERATE = 1 | REPACK = 1 | 

sampler seed: 2023356166
sampler params: 
        repeat_last_n = 64, repeat_penalty = 1.000, frequency_penalty = 0.000, presence_penalty = 0.000
        dry_multiplier = 0.000, dry_base = 1.750, dry_allowed_length = 2, dry_penalty_last_n = 1024
        top_k = 40, top_p = 0.950, min_p = 0.050, xtc_probability = 0.000, xtc_threshold = 0.100, typical_p = 1.000, top_n_sigma = -1.000, temp = 0.700
        mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
sampler chain: logits -> logit-bias -> penalties -> dry -> top-n-sigma -> top-k -> typical -> top-p -> min-p -> xtc -> temp-ext -> dist 
generate: n_ctx = 1024, n_batch = 1024, n_predict = 128, n_keep = 1

llama_perf_sampler_print:    sampling time =      14.27 ms /   189 runs   (    0.08 ms per token, 13242.71 tokens per second)
llama_perf_context_print:        load time =   10132.30 ms
llama_perf_context_print: prompt eval time =    2684.15 ms /    61 tokens (   44.00 ms per token,    22.73 tokens per second)
llama_perf_context_print:        eval time =   12656.66 ms /   127 runs   (   99.66 ms per token,    10.03 tokens per second)
llama_perf_context_print:       total time =   15407.41 ms /   188 tokens
llama_perf_context_print:    graphs reused =        122
ggml_metal_free: deallocating
(2025-08-13 16:41:00) [INFO    ] Response: 0x103010618 /llm/generate 200 0



--------

curl -sS -X POST http://localhost:18080/llm/generate \
  -H "Content-Type: application/json" \
  -d '{
    "backend": "llama",
    "prompt": "helloworld 수준의 간단한 C코드를 짜줘",
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
{
  "engine": "llama",
  "output": "helloworld 수준의 간단한 C코드를 짜줘. 그리고 그 코드를 컴파일하고 실행하는 방법까지 알려줘.\n\n\n```json\n{\n  \"code\": \"#include <stdio.h>\\n\\nint main() {\\n    printf(\\\"Hello, World!\\\\n\\\");\\n    return 0;\\n}\",\n  \"compile_and_run\": \"To compile and run this C code, you can\n\n",
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
  "prompt": "helloworld 수준의 간단한 C코드를 짜줘"
}

[서버 응답]

llama_perf_sampler_print:    sampling time =      14.27 ms /   189 runs   (    0.08 ms per token, 13242.71 tokens per second)
llama_perf_context_print:        load time =   10132.30 ms
llama_perf_context_print: prompt eval time =    2684.15 ms /    61 tokens (   44.00 ms per token,    22.73 tokens per second)
llama_perf_context_print:        eval time =   12656.66 ms /   127 runs   (   99.66 ms per token,    10.03 tokens per second)
llama_perf_context_print:       total time =   15407.41 ms /   188 tokens
llama_perf_context_print:    graphs reused =        122
ggml_metal_free: deallocating
(2025-08-13 16:41:00) [INFO    ] Response: 0x103010618 /llm/generate 200 0
(2025-08-13 16:41:29) [INFO    ] Request: 127.0.0.1:54232 0x12ed2a218 HTTP/1.1 POST /llm/generate
build: 0 (unknown) with Apple clang version 16.0.0 (clang-1600.0.26.6) for x86_64-apple-darwin24.5.0
main: llama backend init
main: load the model and apply lora adapter, if any
llama_model_load_from_file_impl: using device Metal (Apple M2) - 10922 MiB free
llama_model_loader: loaded meta data with 38 key-value pairs and 377 tensors from /Users/mac/Desktop/workspace/miniGPT/ml-engine/models/deepseek-coder-v2-lite-instruct-q4_k_m.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.name str              = DeepSeek-Coder-V2-Lite-Instruct
llama_model_loader: - kv   2:                      deepseek2.block_count u32              = 27
llama_model_loader: - kv   3:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv   4:                 deepseek2.embedding_length u32              = 2048
llama_model_loader: - kv   5:              deepseek2.feed_forward_length u32              = 10944
llama_model_loader: - kv   6:             deepseek2.attention.head_count u32              = 16
llama_model_loader: - kv   7:          deepseek2.attention.head_count_kv u32              = 16
llama_model_loader: - kv   8:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv   9: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  10:                deepseek2.expert_used_count u32              = 6
llama_model_loader: - kv  11:                          general.file_type u32              = 15
llama_model_loader: - kv  12:        deepseek2.leading_dense_block_count u32              = 1
llama_model_loader: - kv  13:                       deepseek2.vocab_size u32              = 102400
llama_model_loader: - kv  14:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  15:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  16:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  17:       deepseek2.expert_feed_forward_length u32              = 1408
llama_model_loader: - kv  18:                     deepseek2.expert_count u32              = 64
llama_model_loader: - kv  19:              deepseek2.expert_shared_count u32              = 2
llama_model_loader: - kv  20:             deepseek2.expert_weights_scale f32              = 1.000000
llama_model_loader: - kv  21:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  22:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  23:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  24: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  25: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.070700
llama_model_loader: - kv  26:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  27:                         tokenizer.ggml.pre str              = deepseek-llm
llama_model_loader: - kv  28:                      tokenizer.ggml.tokens arr[str,102400]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  29:                  tokenizer.ggml.token_type arr[i32,102400]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  30:                      tokenizer.ggml.merges arr[str,99757]   = ["Ġ Ġ", "Ġ t", "Ġ a", "i n", "h e...
llama_model_loader: - kv  31:                tokenizer.ggml.bos_token_id u32              = 100000
llama_model_loader: - kv  32:                tokenizer.ggml.eos_token_id u32              = 100001
llama_model_loader: - kv  33:            tokenizer.ggml.padding_token_id u32              = 100001
llama_model_loader: - kv  34:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  35:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  36:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  37:               general.quantization_version u32              = 2
llama_model_loader: - type  f32:  108 tensors
llama_model_loader: - type q5_0:   14 tensors
llama_model_loader: - type q8_0:   13 tensors
llama_model_loader: - type q4_K:  229 tensors
llama_model_loader: - type q6_K:   13 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = Q4_K - Medium
print_info: file size   = 9.65 GiB (5.28 BPW) 
load: control-looking token: 100002 '<｜fim▁hole｜>' was not control-type; this is probably a bug in the model. its type will be overridden
load: control-looking token: 100004 '<｜fim▁end｜>' was not control-type; this is probably a bug in the model. its type will be overridden
load: control-looking token: 100003 '<｜fim▁begin｜>' was not control-type; this is probably a bug in the model. its type will be overridden
load: special_eos_id is not in special_eog_ids - the tokenizer config may be incorrect
load: printing all EOG tokens:
load:   - 100001 ('<｜end▁of▁sentence｜>')
load: special tokens cache size = 2400
load: token to piece cache size = 0.6661 MB
print_info: arch             = deepseek2
print_info: vocab_only       = 0
print_info: n_ctx_train      = 163840
print_info: n_embd           = 2048
print_info: n_layer          = 27
print_info: n_head           = 16
print_info: n_head_kv        = 16
print_info: n_rot            = 64
print_info: n_swa            = 0
print_info: is_swa_any       = 0
print_info: n_embd_head_k    = 192
print_info: n_embd_head_v    = 128
print_info: n_gqa            = 1
print_info: n_embd_k_gqa     = 3072
print_info: n_embd_v_gqa     = 2048
print_info: f_norm_eps       = 0.0e+00
print_info: f_norm_rms_eps   = 1.0e-06
print_info: f_clamp_kqv      = 0.0e+00
print_info: f_max_alibi_bias = 0.0e+00
print_info: f_logit_scale    = 0.0e+00
print_info: f_attn_scale     = 0.0e+00
print_info: n_ff             = 10944
print_info: n_expert         = 64
print_info: n_expert_used    = 6
print_info: causal attn      = 1
print_info: pooling type     = 0
print_info: rope type        = 0
print_info: rope scaling     = yarn
print_info: freq_base_train  = 10000.0
print_info: freq_scale_train = 0.025
print_info: n_ctx_orig_yarn  = 4096
print_info: rope_finetuned   = unknown
print_info: model type       = 16B
print_info: model params     = 15.71 B
print_info: general.name     = DeepSeek-Coder-V2-Lite-Instruct
print_info: n_layer_dense_lead   = 1
print_info: n_lora_q             = 0
print_info: n_lora_kv            = 512
print_info: n_embd_head_k_mla    = 0
print_info: n_embd_head_v_mla    = 0
print_info: n_ff_exp             = 1408
print_info: n_expert_shared      = 2
print_info: expert_weights_scale = 1.0
print_info: expert_weights_norm  = 0
print_info: expert_gating_func   = softmax
print_info: rope_yarn_log_mul    = 0.0707
print_info: vocab type       = BPE
print_info: n_vocab          = 102400
print_info: n_merges         = 99757
print_info: BOS token        = 100000 '<｜begin▁of▁sentence｜>'
print_info: EOS token        = 100001 '<｜end▁of▁sentence｜>'
print_info: EOT token        = 100001 '<｜end▁of▁sentence｜>'
print_info: PAD token        = 100001 '<｜end▁of▁sentence｜>'
print_info: LF token         = 185 'Ċ'
print_info: FIM PRE token    = 100003 '<｜fim▁begin｜>'
print_info: FIM SUF token    = 100002 '<｜fim▁hole｜>'
print_info: FIM MID token    = 100004 '<｜fim▁end｜>'
print_info: EOG token        = 100001 '<｜end▁of▁sentence｜>'
print_info: max token length = 256
load_tensors: loading model tensors, this can take a while... (mmap = true)

load_tensors: offloading 24 repeating layers to GPU
load_tensors: offloaded 24/28 layers to GPU
load_tensors: Metal_Mapped model buffer size =  8742.55 MiB
load_tensors:   CPU_Mapped model buffer size =  2567.24 MiB
...................................................................................
llama_context: constructing llama_context
llama_context: n_seq_max     = 1
llama_context: n_ctx         = 1024
llama_context: n_ctx_per_seq = 1024
llama_context: n_batch       = 1024
llama_context: n_ubatch      = 512
llama_context: causal_attn   = 1
llama_context: flash_attn    = 0
llama_context: kv_unified    = false
llama_context: freq_base     = 10000.0
llama_context: freq_scale    = 0.025
llama_context: n_ctx_per_seq (1024) < n_ctx_train (163840) -- the full capacity of the model will not be utilized
ggml_metal_init: allocating
ggml_metal_init: found device: Apple M2
ggml_metal_init: picking default device: Apple M2
ggml_metal_load_library: using embedded metal library
ggml_metal_init: GPU name:   Apple M2
ggml_metal_init: GPU family: MTLGPUFamilyApple8  (1008)
ggml_metal_init: GPU family: MTLGPUFamilyCommon3 (3003)
ggml_metal_init: GPU family: MTLGPUFamilyMetal3  (5001)
ggml_metal_init: simdgroup reduction   = true
ggml_metal_init: simdgroup matrix mul. = true
ggml_metal_init: has residency sets    = false
ggml_metal_init: has bfloat            = true
ggml_metal_init: use bfloat            = false
ggml_metal_init: hasUnifiedMemory      = true
ggml_metal_init: recommendedMaxWorkingSetSize  = 11453.25 MB
ggml_metal_init: skipping kernel_get_rows_bf16                     (not supported)
ggml_metal_init: skipping kernel_set_rows_bf16                     (not supported)
ggml_metal_init: skipping kernel_mul_mv_bf16_f32                   (not supported)
ggml_metal_init: skipping kernel_mul_mv_bf16_f32_c4                (not supported)
ggml_metal_init: skipping kernel_mul_mv_bf16_f32_1row              (not supported)
ggml_metal_init: skipping kernel_mul_mv_bf16_f32_l4                (not supported)
ggml_metal_init: skipping kernel_mul_mv_bf16_bf16                  (not supported)
ggml_metal_init: skipping kernel_mul_mv_id_bf16_f32                (not supported)
ggml_metal_init: skipping kernel_mul_mm_bf16_f32                   (not supported)
ggml_metal_init: skipping kernel_mul_mm_id_bf16_f16                (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_bf16_h64           (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_bf16_h80           (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_bf16_h96           (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_bf16_h112          (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_bf16_h128          (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_bf16_h192          (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_bf16_hk192_hv128   (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_bf16_h256          (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_bf16_hk576_hv512   (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_vec_bf16_h64       (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_vec_bf16_h96       (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_vec_bf16_h128      (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_vec_bf16_h192      (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_vec_bf16_hk192_hv128 (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_vec_bf16_h256      (not supported)
ggml_metal_init: skipping kernel_flash_attn_ext_vec_bf16_hk576_hv512 (not supported)
ggml_metal_init: skipping kernel_cpy_f32_bf16                      (not supported)
ggml_metal_init: skipping kernel_cpy_bf16_f32                      (not supported)
ggml_metal_init: skipping kernel_cpy_bf16_bf16                     (not supported)
llama_context:        CPU  output buffer size =     0.39 MiB
llama_kv_cache_unified:      Metal KV buffer size =   240.00 MiB
llama_kv_cache_unified:        CPU KV buffer size =    30.00 MiB
llama_kv_cache_unified: size =  270.00 MiB (  1024 cells,  27 layers,  1/1 seqs), K (f16):  162.00 MiB, V (f16):  108.00 MiB
llama_context:      Metal compute buffer size =    71.51 MiB
llama_context:        CPU compute buffer size =   200.00 MiB
llama_context: graph nodes  = 1844
llama_context: graph splits = 49 (with bs=512), 3 (with bs=1)
common_init_from_params: added <｜end▁of▁sentence｜> logit bias = -inf
common_init_from_params: setting dry_penalty_last_n to ctx_size = 1024
common_init_from_params: warming up the model with an empty run - please wait ... (--no-warmup to disable)
main: llama threadpool init, n_threads = 8

system_info: n_threads = 8 (n_threads_batch = 8) / 8 | Metal : EMBED_LIBRARY = 1 | CPU : SSE3 = 1 | SSSE3 = 1 | LLAMAFILE = 1 | ACCELERATE = 1 | REPACK = 1 | 

sampler seed: 778456501
sampler params: 
        repeat_last_n = 64, repeat_penalty = 1.000, frequency_penalty = 0.000, presence_penalty = 0.000
        dry_multiplier = 0.000, dry_base = 1.750, dry_allowed_length = 2, dry_penalty_last_n = 1024
        top_k = 40, top_p = 0.950, min_p = 0.050, xtc_probability = 0.000, xtc_threshold = 0.100, typical_p = 1.000, top_n_sigma = -1.000, temp = 0.700
        mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
sampler chain: logits -> logit-bias -> penalties -> dry -> top-n-sigma -> top-k -> typical -> top-p -> min-p -> xtc -> temp-ext -> dist 
generate: n_ctx = 1024, n_batch = 1024, n_predict = 128, n_keep = 1

llama_perf_sampler_print:    sampling time =      11.71 ms /   158 runs   (    0.07 ms per token, 13489.29 tokens per second)
llama_perf_context_print:        load time =   10463.08 ms
llama_perf_context_print: prompt eval time =    2568.96 ms /    30 tokens (   85.63 ms per token,    11.68 tokens per second)
llama_perf_context_print:        eval time =   13100.63 ms /   127 runs   (  103.15 ms per token,     9.69 tokens per second)
llama_perf_context_print:       total time =   15725.43 ms /   157 tokens
llama_perf_context_print:    graphs reused =        122
ggml_metal_free: deallocating
(2025-08-13 16:41:55) [INFO    ] Response: 0x12ed2a218 /llm/generate 200 0

