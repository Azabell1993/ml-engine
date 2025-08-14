#pragma once
#include <string>
#include <vector>
#include <mutex>
#include <optional>
#include <chrono>

// 간단 파라미터 구조체
namespace llm {

struct Params {
    int n_threads = 4;
    int n_ctx = 512;
    float temperature = 0.8f;
    int top_k = 40;
    float top_p = 0.95f;
    std::string backend = "llama";
    std::string llama_exec_path = "../third_party/llama.cpp/build/bin/llama-cli";
    std::vector<std::string> extra_args;   // 백엔드별 추가 인자
    std::chrono::milliseconds timeout{30000};
};

class Engine {
public:
    // 싱글톤(간단)
    static Engine& instance();

    // Thread-safe generate
    std::string generate(const std::string& prompt, const Params& p);

private:
    Engine() = default;
    std::mutex mu_;

    // 백엔드 구현
    std::string run_mock_(const std::string& prompt, const Params& p);
    std::string run_llama_exec_(const std::string& prompt, const Params& p);
    static std::string joinArgs_(const std::vector<std::string>& v, const std::string& sep);

    // 유틸: 외부 프로세스 실행(타임아웃)
    static std::string runWithTimeout_(const std::string& cmd, std::chrono::milliseconds to, int& exit_code);
};

} // namespace llm