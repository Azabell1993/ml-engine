#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Paper-grade benchmark: ChatGPT-4.0 Web vs Local ml-engine
- repeated runs + warmups
- robust stats + validator
- artifacts: CSV/JSON + Markdown table
- figures (matplotlib): latency CDF, latency box, tokens/s bar
"""

import argparse
import json
import math
import os
import platform
import re
import statistics as stats
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import psutil
import requests

# Optional deps for reporting
try:
    import pandas as pd
except Exception as _e:
    pd = None

try:
    import matplotlib.pyplot as plt
except Exception as _e:
    plt = None

try:
    import resource  # macOS/Linux peak RSS
    HAS_RESOURCE = True
except Exception:
    HAS_RESOURCE = False


# ---------------------------
# Services to benchmark
# ---------------------------
SERVICES: List[Dict[str, Any]] = [
    {
        "name": "Local ml-engine",
        "url": "http://localhost:18080/llm/generate",
        "payload": {
            "backend": "llama",
            "prompt": "C++로 helloworld를 출력하는 코드를 단일 블록으로만 답해줘.",
            "llama_exec_path": "/Users/mac/Desktop/workspace/miniGPT/ml-engine/third_party/llama.cpp/build/bin/llama-cli",
            "n_threads": 8,
            "n_ctx": 1024,
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.95,
            "extra_args": [
                "-m", "/Users/mac/Desktop/workspace/miniGPT/ml-engine/models/deepseek-coder-v2-lite-instruct-q4_k_m.gguf",
                "-ngl", "24",
                "-c", "1024",
                "-b", "1024",
                "--simple-io",
                "-no-cnv",
                "-n", "128"
            ]
        }
    },
    {
        "name": "ChatGPT 4.0 Web",
        "url": "https://api.openai.com/v1/chat/completions",
        "payload": {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "C++로 helloworld를 출력하는 코드를 단일 블록으로만 답해줘."}],
            "max_tokens": 128,
            "temperature": 0.7
        },
        # 헤더는 환경변수 OPENAI_API_KEY 를 자동 사용 (없으면 아래 문자열을 직접 채워도 됨)
        "headers": {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'REPLACE_WITH_YOUR_KEY')}"}
    },
]


# ---------------------------
# Utils
# ---------------------------
def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def system_metadata() -> Dict[str, Any]:
    vm = psutil.virtual_memory()
    cpu_freq = psutil.cpu_freq()
    return {
        "timestamp": datetime.now().isoformat(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "processor": platform.processor() or platform.machine(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_freq_current_mhz": cpu_freq.current if cpu_freq else None,
        "cpu_freq_max_mhz": cpu_freq.max if cpu_freq else None,
        "memory_total_gb": round(vm.total / (1024 ** 3), 2),
    }

def get_peak_rss_mb() -> Optional[float]:
    if not HAS_RESOURCE:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF)
    val = usage.ru_maxrss
    if sys.platform == "darwin":
        return val / (1024 ** 2)  # bytes -> MB
    else:
        return val / 1024.0       # KB -> MB


# ---------------------------
# Validator: single C++ Hello World block
# ---------------------------
CPP_BLOCK = re.compile(r"```(?:cpp|c\+\+)?\s*?(.+?)```", re.DOTALL | re.IGNORECASE)
HELLO_SNIPPET = re.compile(
    r'#include\s*<iostream>.*?int\s+main\s*\(\s*\)\s*\{.*?std::cout\s*<<\s*"?Hello,\s*World!?".*?\}',
    re.DOTALL,
)

def validate_cpp_single_block(content: str) -> Tuple[bool, str]:
    blocks = CPP_BLOCK.findall(content or "")
    if len(blocks) != 1:
        return False, f"invalid_code_blocks:{len(blocks)}"
    code = blocks[0]
    if not HELLO_SNIPPET.search(code):
        return False, "hello_world_not_detected"
    return True, "ok"


# ---------------------------
# Token extraction
# ---------------------------
def extract_tokens(service_name: str, payload: Dict[str, Any], output_json: Dict[str, Any], content_text: str) -> Dict[str, Optional[int]]:
    tok = {"completion_tokens": None, "prompt_tokens": None, "total_tokens": None}
    if "ChatGPT" in service_name or "OpenAI" in service_name:
        usage = output_json.get("usage") or {}
        tok["completion_tokens"] = usage.get("completion_tokens")
        tok["prompt_tokens"] = usage.get("prompt_tokens")
        tok["total_tokens"] = usage.get("total_tokens")
        if tok["completion_tokens"] is None:
            tok["completion_tokens"] = len((content_text or "").split())  # fallback
        if tok["total_tokens"] is None and tok["completion_tokens"] is not None and tok["prompt_tokens"] is not None:
            tok["total_tokens"] = tok["completion_tokens"] + tok["prompt_tokens"]
        return tok

    # Local: use -n if provided, else approximate by words
    extra_args = (payload or {}).get("extra_args", [])
    comp_n = None
    try:
        if "-n" in extra_args:
            idx = extra_args.index("-n")
            comp_n = int(extra_args[idx + 1])
    except Exception:
        pass
    if comp_n is None:
        comp_n = len((content_text or "").split())
    tok["completion_tokens"] = comp_n
    tok["total_tokens"] = comp_n
    return tok


# ---------------------------
# Single measurement
# ---------------------------
def measure_once(session: requests.Session, service: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss / (1024 ** 2)
    peak_before = get_peak_rss_mb()
    t0 = time.perf_counter()
    try:
        resp = session.post(
            service["url"],
            json=service["payload"],
            headers=service.get("headers", {}),
            timeout=timeout_s,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        rss_after = proc.memory_info().rss / (1024 ** 2)
        peak_after = get_peak_rss_mb()
        peak_rss = None if (peak_before is None or peak_after is None) else max(peak_before, peak_after)

        out_json: Dict[str, Any]
        try:
            out_json = resp.json()
        except Exception:
            out_json = {"_raw_text": resp.text[:500]}

        # extract content
        content = ""
        if "choices" in out_json:
            content = (out_json.get("choices", [{}])[0] or {}).get("message", {}).get("content", "") or ""
        elif "output" in out_json:
            content = str(out_json.get("output", "") or "")
        else:
            content = out_json.get("_raw_text", "") or ""

        tokens = extract_tokens(service["name"], service.get("payload", {}), out_json, content)
        v_ok, v_reason = validate_cpp_single_block(content)

        return {
            "service": service["name"],
            "ok": bool(resp.ok),
            "status": resp.status_code,
            "latency_ms": latency_ms,
            "rss_delta_mb": (rss_after - rss_before),
            "peak_rss_mb": peak_rss,
            "resp_bytes": len(resp.content or b""),
            "content_bytes": len(content.encode("utf-8")),
            "validation_ok": v_ok,
            "validation_reason": v_reason,
            "output_excerpt": content[:300],
            "tokens": tokens,
        }
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        rss_after = proc.memory_info().rss / (1024 ** 2)
        peak_after = get_peak_rss_mb()
        peak_rss = None if (peak_before is None or peak_after is None) else max(peak_before, peak_after)
        return {
            "service": service["name"],
            "ok": False,
            "status": None,
            "latency_ms": latency_ms,
            "rss_delta_mb": (rss_after - rss_before),
            "peak_rss_mb": peak_rss,
            "resp_bytes": 0,
            "content_bytes": 0,
            "validation_ok": False,
            "validation_reason": f"exception:{type(e).__name__}:{e}",
            "output_excerpt": "",
            "tokens": {"completion_tokens": None, "prompt_tokens": None, "total_tokens": None},
        }


# ---------------------------
# Aggregation helpers
# ---------------------------
def pct(values: List[float], p: float) -> float:
    if not values:
        return math.nan
    values_sorted = sorted(values)
    k = max(0, min(len(values_sorted) - 1, int(round((p/100.0)*(len(values_sorted)-1)))))
    return float(values_sorted[k])

def summarize(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    lat = [r["latency_ms"] for r in runs if r.get("latency_ms") is not None]
    rss = [max(0.0, r["rss_delta_mb"]) for r in runs if r.get("rss_delta_mb") is not None]  # negative clamp
    ok = sum(1 for r in runs if r.get("ok"))
    val_ok = sum(1 for r in runs if r.get("validation_ok"))
    tokps = []
    for r in runs:
        ct = r["tokens"].get("completion_tokens")
        if ct and r["latency_ms"] and r["latency_ms"] > 0:
            tokps.append(1000.0 * float(ct) / float(r["latency_ms"]))
    def S(arr):
        if not arr:
            return {"mean": math.nan, "median": math.nan, "std": math.nan, "min": math.nan, "max": math.nan,
                    "p50": math.nan, "p90": math.nan, "p95": math.nan, "p99": math.nan}
        return {
            "mean": stats.fmean(arr),
            "median": stats.median(arr),
            "std": stats.pstdev(arr) if len(arr) > 1 else 0.0,
            "min": min(arr),
            "max": max(arr),
            "p50": pct(arr, 50),
            "p90": pct(arr, 90),
            "p95": pct(arr, 95),
            "p99": pct(arr, 99),
        }
    return {
        "count": len(runs),
        "success_rate": ok / len(runs) if runs else 0.0,
        "validation_pass_rate": val_ok / len(runs) if runs else 0.0,
        "latency_ms": S(lat),
        "rss_delta_mb": S(rss),
        "tokens_per_sec": S(tokps),
    }


# ---------------------------
# Reporting (tables & figures)
# ---------------------------
def make_tables_and_figures(outdir: str):
    if pd is None:
        print("pandas 가 설치되어 있지 않아 표 생성은 생략합니다. (pip install pandas)")
        return
    csv_path = os.path.join(outdir, "runs.csv")
    if not os.path.exists(csv_path):
        print("runs.csv 가 없어 표/그림 생성을 건너뜁니다.")
        return
    df = pd.read_csv(csv_path)

    # 논문용 요약 표 (row=metric, col=service)
    def pick(df_s, key):
        return df_s[key].mean()

    # tokens/s는 per-run 계산되어 있으므로 평균
    tbl = []
    for svc in sorted(df["service"].unique()):
        d = df[df["service"] == svc]
        latency_mean = d["latency_ms"].mean()
        latency_p50  = d["latency_ms"].quantile(0.50)
        latency_p95  = d["latency_ms"].quantile(0.95)
        success_rate = d["ok"].mean()
        val_rate     = d["validation_ok"].mean()
        mem_mean     = d["rss_delta_mb"].clip(lower=0).mean()
        tps_mean     = d["tokens_per_sec"].dropna().mean()
        tbl.append({
            "service": svc,
            "latency_ms_mean": round(latency_mean, 2),
            "latency_ms_p50": round(latency_p50, 2),
            "latency_ms_p95": round(latency_p95, 2),
            "success_rate": round(success_rate*100, 1),
            "validation_pass_rate": round(val_rate*100, 1),
            "rss_delta_mb_mean": round(mem_mean, 2),
            "tokens_per_sec_mean": round(tps_mean, 2) if not pd.isna(tps_mean) else ""
        })
    df_sum = pd.DataFrame(tbl)
    df_sum.to_csv(os.path.join(outdir, "paper_table.csv"), index=False)

    # Markdown table
    md_lines = ["| Service | Latency mean (ms) | p50 (ms) | p95 (ms) | Success (%) | Valid (%) | RSS Δ mean (MB) | Tokens/s mean |",
                "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in df_sum.iterrows():
        md_lines.append(f"| {r['service']} | {r['latency_ms_mean']} | {r['latency_ms_p50']} | {r['latency_ms_p95']} | {r['success_rate']} | {r['validation_pass_rate']} | {r['rss_delta_mb_mean']} | {r['tokens_per_sec_mean']} |")
    with open(os.path.join(outdir, "paper_table.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # ===== Figures (matplotlib) =====
    if plt is None:
        print("matplotlib 이 설치되어 있지 않아 그림 생성을 생략합니다. (pip install matplotlib)")
        return

    # 1) Latency CDF
    plt.figure()
    for svc in sorted(df["service"].unique()):
        lat = sorted(df[df["service"] == svc]["latency_ms"].values.tolist())
        if not lat:
            continue
        y = [i/ (len(lat)-1 if len(lat) > 1 else 1) for i in range(len(lat))]
        plt.plot(lat, y, label=svc)  # 색 지정 금지 (규정 준수: 기본값)
    plt.xlabel("Latency (ms)")
    plt.ylabel("CDF")
    plt.title("Latency CDF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "latency_cdf.png"), dpi=200)
    plt.close()

    # 2) Latency boxplot
    plt.figure()
    data = [df[df["service"]==svc]["latency_ms"].values for svc in sorted(df["service"].unique())]
    plt.boxplot(data, labels=sorted(df["service"].unique()), showfliers=False)
    plt.ylabel("Latency (ms)")
    plt.title("Latency Distribution (boxplot)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "latency_box.png"), dpi=200)
    plt.close()

    # 3) Tokens/s bar (mean)
    plt.figure()
    svc_list = sorted(df["service"].unique())
    means = []
    for svc in svc_list:
        m = df[df["service"]==svc]["tokens_per_sec"].dropna().mean()
        means.append(m if not pd.isna(m) else 0.0)
    plt.bar(svc_list, means)  # 색 지정 금지
    plt.ylabel("Tokens/s (mean)")
    plt.title("Throughput (mean Tokens/s)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "tokens_per_sec_bar.png"), dpi=200)
    plt.close()

    print("Tables & figures saved:",
          os.path.join(outdir, "paper_table.csv"),
          os.path.join(outdir, "paper_table.md"),
          os.path.join(outdir, "latency_cdf.png"),
          os.path.join(outdir, "latency_box.png"),
          os.path.join(outdir, "tokens_per_sec_bar.png"))


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Benchmark: ChatGPT-4.0 Web vs Local ml-engine")
    parser.add_argument("--runs", type=int, default=9, help="Measured runs per service (excl. warmups)")
    parser.add_argument("--warmups", type=int, default=2, help="Warmup requests per service")
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout (seconds)")
    parser.add_argument("--retry", type=int, default=0, help="Retries per run on failure")
    parser.add_argument("--outdir", type=str, default=f"./bench_results_{now_str()}", help="Output directory")
    parser.add_argument("--save-runs-json", action="store_true",
                            help="If set, save per-run JSON & per-service summary.json")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # system meta
    with open(os.path.join(args.outdir, "system_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(system_metadata(), f, ensure_ascii=False, indent=2)

    # CSV header
    csv_lines = ["service,run_idx,ok,status,latency_ms,rss_delta_mb,peak_rss_mb,resp_bytes,content_bytes,validation_ok,validation_reason,completion_tokens,prompt_tokens,total_tokens,tokens_per_sec"]

    session = requests.Session()

    for svc in SERVICES:
        name = svc["name"]
        svc_dir = os.path.join(args.outdir, re.sub(r"[^A-Za-z0-9_.-]+", "_", name))
        os.makedirs(svc_dir, exist_ok=True)

        # warmups
        for _ in range(args.warmups):
            _ = measure_once(session, svc, args.timeout)

        # measured runs
        runs = []
        for i in range(args.runs):
            attempt = 0
            res = None
            while attempt <= args.retry:
                res = measure_once(session, svc, args.timeout)
                if res["ok"]:
                    break
                attempt += 1
                time.sleep(0.2)
            runs.append(res)

            if args.save_runs_json:
                svc_dir = os.path.join(args.outdir, re.sub(r"[^A-Za-z0-9_.-]+", "_", name))
                os.makedirs(svc_dir, exist_ok=True)

            ct = res["tokens"].get("completion_tokens") or math.nan
            tps = (1000.0 * float(ct) / res["latency_ms"]) if (ct and res["latency_ms"] > 0) else math.nan
            csv_lines.append(",".join([
                name.replace(",", "_"),
                str(i),
                "1" if res["ok"] else "0",
                str(res["status"] if res["status"] is not None else ""),
                f"{res['latency_ms']:.2f}",
                f"{res['rss_delta_mb']:.2f}",
                f"{res['peak_rss_mb']:.2f}" if res["peak_rss_mb"] is not None else "",
                str(res["resp_bytes"]),
                str(res["content_bytes"]),
                "1" if res["validation_ok"] else "0",
                str(res["validation_reason"]).replace(",", " "),
                str(res["tokens"].get("completion_tokens") or ""),
                str(res["tokens"].get("prompt_tokens") or ""),
                str(res["tokens"].get("total_tokens") or ""),
                f"{tps:.2f}" if not math.isnan(tps) else "",
            ]))

        # per-service summary
        summary = summarize(runs)
        if args.save_runs_json:
            with open(os.path.join(svc_dir, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
                
        # console brief
        lat = summary["latency_ms"]
        tps = summary["tokens_per_sec"]
        print(f"\n=== {name} ===")
        print(f"runs={summary['count']} | success={summary['success_rate']*100:.1f}% | valid={summary['validation_pass_rate']*100:.1f}%")
        print(f"latency ms -> mean:{lat['mean']:.2f} p50:{lat['p50']:.2f} p95:{lat['p95']:.2f} p99:{lat['p99']:.2f} min:{lat['min']:.2f} max:{lat['max']:.2f} std:{lat['std']:.2f}")
        if not math.isnan(tps["mean"]):
            print(f"tokens/s -> mean:{tps['mean']:.2f} p50:{tps['p50']:.2f} p95:{tps['p95']:.2f}")

    # write global csv
    with open(os.path.join(args.outdir, "runs.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines))

    # derived tables & figures
    make_tables_and_figures(args.outdir)

    print(f"\nArtifacts saved to: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()