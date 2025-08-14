/**
 * @file main.cpp
 * @brief Main entry point for the ML engine.
 *
 * -----------------------------------------------------------------------------
 * Project Overview
 * -----------------------------------------------------------------------------
 * A compact, production-style **ML/LLM runtime engine** that can:
 *   1) run an HTTP inference server (REST API),
 *   2) train/evaluate classic ML models (e.g., MNIST CNN) from the CLI, and
 *   3) validate CPU-first safety/stability (overflow-safe ops, threading demos)
 *      before moving on to heavier LLM stacks.
 *
 * Lean but realistic architecture:
 *   - engine/       : HTTP server, config loading, lifecycle management
 *   - ml/           : model registry, trainer, datasets, metrics, checkpoints
 *   - src/main.cpp  : mode selection (server / CLI training / demos & benchmarks)
 *
 * Typical users:
 *   - Devs who need a **local, reproducible** engine to test CPU inference,
 *     checkpoints, and dataset plumbing.
 *   - Teams preparing to integrate LLM backends (e.g., llama.cpp) after hardening
 *     build/logging/APIs on commodity machines.
 *
 * -----------------------------------------------------------------------------
 * Key Behaviors & Modes (precedence)
 * -----------------------------------------------------------------------------
 * • Rich help:
 *     ./ml_engine --help
 *     ./ml_engine --help=server|train|llm|bench|all
 *
 * • Benchmarks (no server starts):
 *     ./ml_engine run        # safety demo: overflow-safe ops + thread demo
 *     ./ml_engine f|s|h|c    # f=checked_mult, s=safe handler, h=manual, c=unsafe
 *
 * • CLI training (no server starts):
 *     ./ml_engine train-cli [model]    # default: cnn_mnist
 *   Uses LibTorch MNIST with transforms:
 *     EnsureChannel → Normalize(0.1307, 0.3081) → Stack
 *   Checkpoints/logs under ./runs/<model>.
 *
 * • Server mode (default when no args):
 *     ./ml_engine
 *   Reads ./config/engine-config.json then exposes routes, e.g.:
 *     GET  /health
 *     GET  /ml/models
 *     POST /ml/train        (JSON payload)
 *     POST /ml/train-all    (if implemented)
 *     POST /llm/generate    (if configured)
 *
 * Notes:
 *   - `train-cli` and all bench modes run **standalone** without launching the server.
 *   - The extra request help is available via `--help=server|train|llm|bench`.
 *
 * -----------------------------------------------------------------------------
 * Build & Runtime Notes
 * -----------------------------------------------------------------------------
 * Toolchain: C++17+, CMake ≥ 3.20, AppleClang/Clang/GCC (tested on macOS)
 * Dependencies:
 *   • LibTorch (prebuilt CPU distribution recommended)
 *   • nlohmann_json (header-only)
 * Avoid committing large binaries (LibTorch, GGUF). Use .gitignore or Git LFS.
 *
 * Typical build:
 *   mkdir build && cd build
 *   cmake -DCMAKE_BUILD_TYPE=Release \
 *         -DCMAKE_PREFIX_PATH=../third_party/libtorch \
 *         -DTorch_DIR=../third_party/libtorch/share/cmake/Torch \
 *         -DCMAKE_OSX_ARCHITECTURES=arm64 \
 *         ..
 *   cmake --build . -j
 *
 * -----------------------------------------------------------------------------
 * Usage Quick Reference
 * -----------------------------------------------------------------------------
 * # Server (default, no args)
 * ./ml_engine
 * curl -s http://localhost:18080/health
 * curl -s http://localhost:18080/ml/models
 *
 * # LLM (if configured in engine-config.json)
 * curl -sS -X POST http://localhost:18080/llm/generate \
 *   -H "Content-Type: application/json" \
 *   -d '{"backend":"llama","prompt":"Hello","extra_args":["-m","./models/model.gguf","-n","64"]}'
 *
 * # CLI training (no server)
 * ./ml_engine train-cli cnn_mnist
 *
 * # Safety demo & micro-benchmarks (no server)
 * ./ml_engine run
 * ./ml_engine f | ./ml_engine s | ./ml_engine h | ./ml_engine c
 *
 * -----------------------------------------------------------------------------
 * Why this structure?
 * -----------------------------------------------------------------------------
 * Iterate on CPU first to prove data I/O, logging, config, and checkpoints are
 * correct. MNIST acts as a **canary** for the training loop and serialization,
 * then you can route `/llm/generate` to a local LLM runner with a small switch.
 *
 * =============================================================================
 * 프로젝트 개요 (Korean)
 * =============================================================================
 * 이 엔진은 경량 **ML/LLM 런타임**으로:
 *   1) HTTP 추론 서버(REST API),
 *   2) CLI 기반 고전 ML 학습/검증(MNIST CNN),
 *   3) CPU 중심 안전성/안정성 검증(오버플로우 안전연산, 스레드 데모)
 * 를 제공합니다.
 *
 * 디렉터리:
 *   - engine/  : 서버/설정/수명주기
 *   - ml/      : 레지스트리/트레이너/데이터셋/체크포인트
 *   - main.cpp : 모드 선택(서버/CLI 학습/데모·벤치)
 *
 * -----------------------------------------------------------------------------
 * 실행 모드 (우선순위)
 * -----------------------------------------------------------------------------
 * • 리치 도움말:
 *     ./ml_engine --help
 *     ./ml_engine --help=server|train|llm|bench|all
 *
 * • 벤치마크(서버 미기동):
 *     ./ml_engine run
 *     ./ml_engine f|s|h|c
 *
 * • CLI 학습(서버 미기동):
 *     ./ml_engine train-cli [모델]   # 기본 cnn_mnist
 *   변환 파이프라인:
 *     EnsureChannel → Normalize(0.1307, 0.3081) → Stack
 *   체크포인트/로그: ./runs/<모델>
 *
 * • 서버(인자 없음일 때만):
 *     ./ml_engine
 *   주요 라우트:
 *     GET  /health
 *     GET  /ml/models
 *     POST /ml/train
 *     POST /ml/train-all (구현된 경우)
 *     POST /llm/generate (구성된 경우)
 *
 * 주의:
 *   - `train-cli` 및 벤치마크 모드는 **단독 실행**(서버 X).
 *   - `--help=server|train|llm|bench`로 세부 사용법을 확인하세요.
 *
 * -----------------------------------------------------------------------------
 * 설계 의도
 * -----------------------------------------------------------------------------
 * 우선 CPU에서 데이터 I/O/로깅/설정/체크포인트를 확정(카나리아: MNIST)한 후
 * `/llm/generate`를 로컬 LLM 실행기로 연결하는 전략으로 안정성을 담보합니다.
 *
 * -----------------------------------------------------------------------------
 * Implementation tip
 * -----------------------------------------------------------------------------
 * Keep this file lean—real logic lives in `engine/` and `ml/`, exposed here via
 * clean interfaces so you can swap backends (ONNX/mobile/embedded) with minimal
 * changes to `main.cpp`.
 */

#include "ml/transforms/ensure_channel.hpp"
#include "engine/engine.hpp"
#include "ml/registry.hpp"
#include "ml/trainer.hpp"
#include "log.h"

#include <nlohmann/json.hpp>
#include <filesystem>
#include <iostream>
#include <thread>
#include <mutex>
#include <vector>
#include <atomic>
#include <chrono>
#include <cstring>
#include <climits>      // INT_MAX, LONG_MIN 등
#include <torch/torch.h> // MNIST
#include <limits>
#include <functional>
#include <torch/torch.h>
#include <torch/data/transforms/base.h>
#include <torch/data/example.h>

// 안전 곱셈: 오버플로우 발생 시 0 반환, 정상 시 *res에 저장하고 1 반환
static inline int checked_mult(int* res, int a, int b) {
#if defined(__has_builtin)
#  if __has_builtin(__builtin_mul_overflow)
    return __builtin_mul_overflow(a, b, res) ? 0 : 1;
#  endif
#endif
    // 빌트인이 없는 컴파일러 대비 수동 체크
    if ((b != 0) && (std::abs(a) > std::numeric_limits<int>::max() / std::abs(b))) return 0;
    *res = a * b; 
    return 1;
}

// 심플 벤치마크 함수: f(i)를 누적해 합 반환(컴파일러 최적화 방지용)
template <typename F>
int bench(F&& f) {
    int sum = 0;
    for (int i = 0; i < 100000; ++i) sum += f(i);
    return sum;
}

// 벤치 실행기: f(안전), h(수동 안전), c(무검사)
static inline void run_benchmark(char mode) {
    int result = 0;
    switch (mode) {
        case 'f': // builtin/safe
            std::cout << "[bench] checked_mult (builtin/safe)\n";
            result = bench([](int x){ int r=0; checked_mult(&r, x, 100); return r; });
            break;
        case 'h': // manual
            std::cout << "[bench] manual overflow-check\n";
            result = bench([](int x){
                if (100 != 0 && std::abs(x) > std::numeric_limits<int>::max()/100) return 0;
                return x * 100;
            });
            break;
        case 'c': // unchecked
            std::cout << "[bench] unchecked baseline\n";
            result = bench([](int x){ return x * 100; });
            break;
        default:
            std::cout << "[bench] unknown mode. use f|h|c\n";
            return;
    }
    std::cout << "[bench] result = " << result << std::endl;
}
// ==========================================================================


// === add: MNIST 채널 보정 변환(텐서 변환으로 구현) ==========================
struct EnsureChannel
    : torch::data::transforms::Transform<torch::data::Example<>, torch::data::Example<>> {

    torch::data::Example<> apply(torch::data::Example<> ex) override {
        // ex.data: [H,W] 일 수 있으니 [1,H,W]로 보정
        if (ex.data.dim() == 2) {
            ex.data = ex.data.unsqueeze(0);
        }
        // (라벨은 그대로)
        return ex;
    }
};
// ==========================================================================

// =====[ 오버플로우 안전 연산 핸들러 ]============================================
//  AppleClang/GCC 의 __builtin_*_overflow 사용. 각 정수형별 안전 계산 지원.
// --------------------------------------------------------------------------------
static inline int checked_op_int(
    int* lval, const char* as, intmax_t first, const char* op, intmax_t second) {

    int tmp; int ok = 0;
    if (op[0] == '*') ok = __builtin_mul_overflow(first, second, &tmp) ? 0 : 1;
    else if (op[0] == '/') ok = (second == 0 || (first == INT_MIN && second == -1)) ? 0 : (tmp = static_cast<int>(first / second), 1);
    else if (op[0] == '+') ok = __builtin_add_overflow(first, second, &tmp) ? 0 : 1;
    else if (op[0] == '-') ok = __builtin_sub_overflow(first, second, &tmp) ? 0 : 1;
    if (!ok) return 0;

    if (as[0] == '=') *lval = tmp;
    else if (as[0] == '+') ok = __builtin_add_overflow(*lval, tmp, lval) ? 0 : 1;
    else if (as[0] == '-') ok = __builtin_sub_overflow(*lval, tmp, lval) ? 0 : 1;
    else if (as[0] == '*') ok = __builtin_mul_overflow(*lval, tmp, lval) ? 0 : 1;
    else if (as[0] == '/') ok = (tmp == 0 || (*lval == INT_MIN && tmp == -1)) ? 0 : (*lval = *lval / tmp, 1);
    return ok;
} // end checked_op_int

static inline int checked_op_uint(
    unsigned* lval, const char* as, uintmax_t first, const char* op, uintmax_t second) {

    unsigned tmp; int ok = 0;
    if (op[0] == '*') ok = __builtin_mul_overflow(first, second, &tmp) ? 0 : 1;
    else if (op[0] == '/') ok = (second == 0) ? 0 : (tmp = static_cast<unsigned>(first / second), 1);
    else if (op[0] == '+') ok = __builtin_add_overflow(first, second, &tmp) ? 0 : 1;
    else if (op[0] == '-') ok = (first < second) ? 0 : (tmp = static_cast<unsigned>(first - second), 1);
    if (!ok) return 0;

    if (as[0] == '=') *lval = tmp;
    else if (as[0] == '+') ok = __builtin_add_overflow(*lval, tmp, lval) ? 0 : 1;
    else if (as[0] == '-') ok = (*lval < tmp) ? 0 : (*lval = *lval - tmp, 1);
    else if (as[0] == '*') ok = __builtin_mul_overflow(*lval, tmp, lval) ? 0 : 1;
    else if (as[0] == '/') ok = (tmp == 0) ? 0 : (*lval = *lval / tmp, 1);
    return ok;
} // end checked_op_uint

static inline int checked_op_long(
    long* lval, const char* as, intmax_t first, const char* op, intmax_t second) {

    long tmp; int ok = 0;
    if (op[0] == '*') ok = __builtin_mul_overflow(first, second, &tmp) ? 0 : 1;
    else if (op[0] == '/') ok = (second == 0 || (first == LONG_MIN && second == -1)) ? 0 : (tmp = static_cast<long>(first / second), 1);
    else if (op[0] == '+') ok = __builtin_add_overflow(first, second, &tmp) ? 0 : 1;
    else if (op[0] == '-') ok = __builtin_sub_overflow(first, second, &tmp) ? 0 : 1;
    if (!ok) return 0;

    if (as[0] == '=') *lval = tmp;
    else if (as[0] == '+') ok = __builtin_add_overflow(*lval, tmp, lval) ? 0 : 1;
    else if (as[0] == '-') ok = __builtin_sub_overflow(*lval, tmp, lval) ? 0 : 1;
    else if (as[0] == '*') ok = __builtin_mul_overflow(*lval, tmp, lval) ? 0 : 1;
    else if (as[0] == '/') ok = (tmp == 0 || (*lval == LONG_MIN && tmp == -1)) ? 0 : (*lval = *lval / tmp, 1);
    return ok;
} // end checked_op_long

static inline int checked_op_ulong(
    unsigned long* lval, const char* as, uintmax_t first, const char* op, uintmax_t second) {

    unsigned long tmp; int ok = 0;
    if (op[0] == '*') ok = __builtin_mul_overflow(first, second, &tmp) ? 0 : 1;
    else if (op[0] == '/') ok = (second == 0) ? 0 : (tmp = static_cast<unsigned long>(first / second), 1);
    else if (op[0] == '+') ok = __builtin_add_overflow(first, second, &tmp) ? 0 : 1;
    else if (op[0] == '-') ok = (first < second) ? 0 : (tmp = static_cast<unsigned long>(first - second), 1);
    if (!ok) return 0;

    if (as[0] == '=') *lval = tmp;
    else if (as[0] == '+') ok = __builtin_add_overflow(*lval, tmp, lval) ? 0 : 1;
    else if (as[0] == '-') ok = (*lval < tmp) ? 0 : (*lval = *lval - tmp, 1);
    else if (as[0] == '*') ok = __builtin_mul_overflow(*lval, tmp, lval) ? 0 : 1;
    else if (as[0] == '/') ok = (tmp == 0) ? 0 : (*lval = *lval / tmp, 1);
    return ok;
} // end checked_op_ulong

static inline int checked_op_llong(
    long long* lval, const char* as, intmax_t first, const char* op, intmax_t second) {

    long long tmp; int ok = 0;
    if (op[0] == '*') ok = __builtin_mul_overflow(first, second, &tmp) ? 0 : 1;
    else if (op[0] == '/') ok = (second == 0 || (first == LLONG_MIN && second == -1)) ? 0 : (tmp = static_cast<long long>(first / second), 1);
    else if (op[0] == '+') ok = __builtin_add_overflow(first, second, &tmp) ? 0 : 1;
    else if (op[0] == '-') ok = __builtin_sub_overflow(first, second, &tmp) ? 0 : 1;
    if (!ok) return 0;

    if (as[0] == '=') *lval = tmp;
    else if (as[0] == '+') ok = __builtin_add_overflow(*lval, tmp, lval) ? 0 : 1;
    else if (as[0] == '-') ok = __builtin_sub_overflow(*lval, tmp, lval) ? 0 : 1;
    else if (as[0] == '*') ok = __builtin_mul_overflow(*lval, tmp, lval) ? 0 : 1;
    else if (as[0] == '/') ok = (tmp == 0 || (*lval == LLONG_MIN && tmp == -1)) ? 0 : (*lval = *lval / tmp, 1);
    return ok;
} // end checked_op_llong

static inline int checked_op_ullong(
    unsigned long long* lval, const char* as, uintmax_t first, const char* op, uintmax_t second) {

    unsigned long long tmp; int ok = 0;
    if (op[0] == '*') ok = __builtin_mul_overflow(first, second, &tmp) ? 0 : 1;
    else if (op[0] == '/') ok = (second == 0) ? 0 : (tmp = static_cast<unsigned long long>(first / second), 1);
    else if (op[0] == '+') ok = __builtin_add_overflow(first, second, &tmp) ? 0 : 1;
    else if (op[0] == '-') ok = (first < second) ? 0 : (tmp = static_cast<unsigned long long>(first - second), 1);
    if (!ok) return 0;

    if (as[0] == '=') *lval = tmp;
    else if (as[0] == '+') ok = __builtin_add_overflow(*lval, tmp, lval) ? 0 : 1;
    else if (as[0] == '-') ok = (*lval < tmp) ? 0 : (*lval = *lval - tmp, 1);
    else if (as[0] == '*') ok = __builtin_mul_overflow(*lval, tmp, lval) ? 0 : 1;
    else if (as[0] == '/') ok = (tmp == 0) ? 0 : (*lval = *lval / tmp, 1);
    return ok;
} // end checked_op_ullong
// ================================================================================


// =====[ 벤치마크 유틸 ]=========================================================
static int bench(const std::function<int(int)>& f) {
    int sum = 0;
    for (int i = 0; i < 100000; ++i) sum += f(i);
    return sum;
} // end bench

// ================================================================================


// =====[ 간단 스레드 데모 ]======================================================
//  표준 C++만 사용 (외부 의존성 없음). CPU 환경에서 안전성 검증에 사용.
// --------------------------------------------------------------------------------
static void run_thread_demo() {
    std::cout << "[thread] 데모 시작 (worker 4, 50k increments per worker)\n";
    constexpr int kWorkers = 4;
    constexpr int kIters   = 50000;

    std::atomic<int> counter{0};
    std::mutex io_mtx;
    std::vector<std::thread> ths;

    auto worker = [&](int id) {
        for (int i = 0; i < kIters; ++i) counter.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lk(io_mtx);
        std::cout << "  - worker#" << id << " done\n";
    };

    for (int i = 0; i < kWorkers; ++i) ths.emplace_back(worker, i);
    for (auto& t : ths) t.join();

    std::cout << "[thread] 기대값=" << (kWorkers * kIters) << ", 결과=" << counter.load() << "\n";
} // end run_thread_demo
// ================================================================================

// =====[ 통합 안전 데모 ]========================================================
static void run_safety_demo() {
    // (1) 오버플로우 데모
    {
        int a = 100000, b = 100000, r = 0;
        std::cout << "[safety] 곱셈 오버플로우 체크: " << a << " * " << b << "\n";
        if (!checked_op_int(&r, "=", a, "*", b)) {
            std::cout << "  -> 오버플로우 발생 (정상 차단)\n";
        } else {
            std::cout << "  -> 정상 결과: " << r << "\n";
        }
    }
    // (2) 덧셈/뺄셈 예시
    {
        long long v = LLONG_MAX - 1;
        if (!checked_op_llong(&v, "+", v, "+", 10)) {
            std::cout << "[safety] long long 덧셈 오버플로우 차단 OK\n";
        }
        unsigned u = 1;
        if (!checked_op_uint(&u, "-", 0u, "-", 2u)) {
            std::cout << "[safety] unsigned 감산 언더플로우 차단 OK\n";
        }
    }
    // (3) 스레드 데모
    run_thread_demo();

    // (4) 벤치마크(안전 vs 미안전) 비교 빠르게 한 번
    run_benchmark('s');
    run_benchmark('c');
} // end run_safety_demo
// ================================================================================

// =====[ Rich usage / help printer ]===========================================
static void print_usage(const char* bin, const char* topic = nullptr) {
    auto exe = std::string(bin ? bin : "./ml_engine");

    auto print_general = [&](){
        std::cout <<
        "\nUSAGE\n"
        "  " << exe << " [run | f | s | h | c | train-cli [model] | --help[=TOPIC]]\n"
        "\nMODES\n"
        "  (no args)        Start HTTP server (reads config/engine-config.json)\n"
        "  run              Safety demo (overflow-safe ops + threads)\n"
        "  f|s|h|c          Micro-bench modes (f=checked_mult, s=safe, h=manual, c=unsafe)\n"
        "  train-cli [m]    MNIST training via LibTorch (default model: cnn_mnist)\n"
        "  --help, -h       Show this guide\n"
        "  --help=TOPIC     One of: server, train, llm, bench, all\n";
    };

    auto print_server = [&](){
        std::cout <<
        "\nSERVER MODE\n"
        "  Start:   " << exe << "\n"
        "  Config:  ./config/engine-config.json (port, threads, routes, LLM backend)\n"
        "  Routes (typical):\n"
        "    GET  /health            - health check\n"
        "    GET  /ml/models         - list registered classic ML models\n"
        "    POST /ml/train          - train one model (JSON payload)\n"
        "    POST /ml/train-all      - train all models (if implemented)\n"
        "    POST /llm/generate      - proxy to LLM backend (if configured)\n"
        "\nEXAMPLES\n"
        "  curl -s http://localhost:18080/health\n"
        "  curl -s http://localhost:18080/ml/models | jq .\n"
        "  curl -sS -X POST http://localhost:18080/ml/train \\\n"
        "    -H 'Content-Type: application/json' \\\n"
        "    -d '{\"model\":\"cnn_mnist\",\"epochs\":1,\"batch_size\":64}'\n"
        "  curl -sS -X POST http://localhost:18080/llm/generate \\\n"
        "    -H 'Content-Type: application/json' \\\n"
        "    -d '{\"backend\":\"llama\",\"prompt\":\"Hello\",\"extra_args\":[\"-m\",\"./models/model.gguf\",\"-n\",\"64\"]}'\n"
        "\nTROUBLESHOOTING\n"
        "  • 404 on /ml/predict: endpoint not implemented in this build.\n"
        "  • Port in use: change port in engine-config.json.\n"
        "  • LLM 200/empty: verify backend path & GGUF file exist.\n";
    };

    auto print_train = [&](){
        std::cout <<
        "\nCLI TRAINING (MNIST)\n"
        "  " << exe << " train-cli [model]\n"
        "  Example:\n"
        "    " << exe << " train-cli cnn_mnist\n"
        "  Behavior:\n"
        "    • Loads MNIST (cfg.dataset_root), applies EnsureChannel → Normalize → Stack\n"
        "    • Trains & validates, writes checkpoints under ./runs/<model>\n"
        "  Tips:\n"
        "    • Pre-download IDX files or set cfg.dataset_root accordingly.\n";
    };

    auto print_llm = [&](){
        std::cout <<
        "\nLLM BACKEND (OPTIONAL)\n"
        "  Exposed via POST /llm/generate when configured in engine-config.json.\n"
        "  Example request:\n"
        "    curl -sS -X POST http://localhost:18080/llm/generate \\\n"
        "      -H 'Content-Type: application/json' \\\n"
        "      -d '{\"backend\":\"llama\",\"prompt\":\"Hello\",\"extra_args\":[\"-m\",\"./models/model.gguf\",\"-n\",\"64\"]}'\n"
        "  Ensure the external runner exists and is executable.\n";
    };

    auto print_bench = [&](){
        std::cout <<
        "\nBENCHMARKS & SAFETY DEMOS\n"
        "  " << exe << " run     # prints safety ops demo & thread demo\n"
        "  " << exe << " f|s|h|c # overflow-checked vs manual vs baseline\n";
    };

    // Decide what to print
    if (!topic) {
        print_general();
        std::cout << "\nHINT  Use '--help=server' (or train|llm|bench|all) for more details.\n";
        return;
    }
    std::string t = topic;
    if (t == "server")      { print_general(); print_server(); }
    else if (t == "train")  { print_general(); print_train(); }
    else if (t == "llm")    { print_general(); print_llm(); }
    else if (t == "bench")  { print_general(); print_bench(); }
    else if (t == "all")    { print_general(); print_server(); print_train(); print_llm(); print_bench(); }
    else                    { print_general(); std::cout << "\nUnknown topic: " << t << "\n"; }
}
// ============================================================================

// =====[ main ]==================================================================
int main(int argc, char** argv) {
#ifdef _DEBUG
    start_log_thread();
    atexit(stop_log_thread);
#endif

    // 허용 인자 집합 (1단계 판별용)
    auto is_bench = [](const char* s){
        return std::strcmp(s, "f")==0 || std::strcmp(s, "s")==0 ||
               std::strcmp(s, "h")==0 || std::strcmp(s, "c")==0;
    };
    auto is_help = [](const char* s){
        return std::strcmp(s, "help")==0 || std::strcmp(s, "--help")==0 ||
               std::strcmp(s, "-h")==0  || std::strncmp(s, "--help=", 7)==0;
    };

    // 1) 도움말: 정상 종료
    if (argc > 1 && is_help(argv[1])) {
        if (std::strncmp(argv[1], "--help=", 7) == 0) {
            print_usage(argv[0], argv[1] + 7);
        } else {
            print_usage(argv[0]);
        }
        return EXIT_SUCCESS;
    }

    // 2) 벤치마크: f|s|h|c 만 허용, 그 외엔 에러
    if (argc > 1 && is_bench(argv[1])) {
        if (argc != 2) {
            std::cerr << "[ERROR] Unexpected extra arguments for benchmark mode.\n";
            print_usage(argv[0], "bench");
            return EXIT_FAILURE;
        }
        run_benchmark(argv[1][0]);
        return 0;
    }

    // 3) 안전 데모(run): run만 허용, 그 외엔 에러
    if (argc > 1 && std::strcmp(argv[1], "run") == 0) {
        if (argc != 2) {
            std::cerr << "[ERROR] Unexpected extra arguments for 'run'.\n";
            print_usage(argv[0], "bench");
            return EXIT_FAILURE;
        }
        run_safety_demo();
        return 0;
    }

    // 4) CLI 학습: "train-cli [model]" 만 허용
    if (argc > 1 && std::strcmp(argv[1], "train-cli") == 0) {
        // 허용 형태: train-cli        -> 기본 모델 사용
        //           train-cli <model> -> 지정 모델 사용
        if (argc > 3) {
            std::cerr << "[ERROR] Too many arguments for 'train-cli'.\n";
            print_usage(argv[0], "train");
            return EXIT_FAILURE;
        }
        std::string name = (argc == 3) ? argv[2] : "cnn_mnist";
        ml::TrainConfig cfg;
        cfg.ckpt_dir = "./runs/" + name;

        try {
            auto model = ml::Registry::get().create(name, cfg);

            auto train = torch::data::datasets::MNIST(
                              cfg.dataset_root,
                              torch::data::datasets::MNIST::Mode::kTrain)
                             .map(ml::transforms::EnsureChannel{})
                             .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                             .map(torch::data::transforms::Stack<>());

            auto valid = torch::data::datasets::MNIST(
                              cfg.dataset_root,
                              torch::data::datasets::MNIST::Mode::kTest)
                             .map(ml::transforms::EnsureChannel{})
                             .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                             .map(torch::data::transforms::Stack<>());

            ml::Trainer::fit(model, cfg, train, valid);
            SQLOG_I("CLI training finished: %s", name.c_str());
        } catch (const std::exception& e) {
            SQLOG_E("CLI training failed: %s", e.what());
            return 1;
        }
        return 0;
    }

    // 5) 인자 없음: 서버 모드 (정상)
    if (argc == 1) {
        auto eng = engine::Engine::create();
        if (eng->loadConfig("./config/engine-config.json") != engine::EngineState::Success) return 2;
        if (eng->init() != engine::EngineState::Success) return 3;
        auto ret = eng->run();
        eng->release();
        return ret == engine::EngineState::Success ? 0 : 4;
    }

    // 6) 그 외는 모두 에러: 사용법 출력 후 실패 종료
    std::cerr << "[ERROR] Unknown or invalid argument(s). Use --help for guidance.\n";
    print_usage(argv[0]);
    return EXIT_FAILURE;
} // end of main
// ================================================================================