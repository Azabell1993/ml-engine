#include "llm/llm_engine.hpp"
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <thread>
#include <future>
#include <iomanip>

namespace llm {

Engine& Engine::instance() {
    static Engine inst;
    return inst;
}

std::string Engine::generate(const std::string& prompt, const Params& p) {
    std::lock_guard<std::mutex> lk(mu_);
    if (p.backend == "llama" && !p.llama_exec_path.empty())
        return run_llama_exec_(prompt, p);
    return run_mock_(prompt, p);
}

std::string Engine::run_mock_(const std::string& prompt, const Params& p) {
    std::ostringstream oss;
    oss << "[mock] codegen for: " << prompt;
    (void)p;
    return oss.str();
}

std::string Engine::run_llama_exec_(const std::string& prompt, const Params& p) {
    // llama.cpp 등 외부 실행 바이너리 호출 예시
    //   main -p "PROMPT" -t n_threads -c n_ctx --temp temperature --top-k top_k --top-p top_p
    std::ostringstream cmd;
    cmd << p.llama_exec_path
        << " -p " << std::quoted(prompt)
        << " -t " << p.n_threads
        << " -c " << p.n_ctx
        << " --temp " << p.temperature
        << " --top-k " << p.top_k
        << " --top-p " << p.top_p;

    for (auto& a : p.extra_args) cmd << " " << a;

    int exit_code = -1;
    auto out = runWithTimeout_(cmd.str(), p.timeout, exit_code);
    if (exit_code != 0) {
        std::ostringstream err;
        err << "[llama-exec failed] exit=" << exit_code << " cmd=" << cmd.str() << "\n" << out;
        return err.str();
    }
    return out;
}

std::string Engine::runWithTimeout_(const std::string& cmd, std::chrono::milliseconds to, int& exit_code) {
    auto fut = std::async(std::launch::async, [cmd, &exit_code]() -> std::string {
        std::string result;
#if defined(_WIN32)
        // 간단화를 위해 Windows는 미지원 예시(필요 시 _popen)
        exit_code = -1;
        return "Windows not implemented";
#else
        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) { exit_code = -1; return "popen failed"; }
        char buffer[4096];
        while (fgets(buffer, sizeof(buffer), pipe)) result += buffer;
        exit_code = pclose(pipe);
        if (WIFEXITED(exit_code)) exit_code = WEXITSTATUS(exit_code);
#endif
        return result;
    });

    if (fut.wait_for(to) == std::future_status::timeout) {
        exit_code = -2;
        return "[timeout]";
    }
    return fut.get();
}

std::string Engine::joinArgs_(const std::vector<std::string>& v, const std::string& sep) {
    std::ostringstream oss;
    for (size_t i=0;i<v.size();++i) { if (i) oss << sep; oss << v[i]; }
    return oss.str();
}

} // namespace llm