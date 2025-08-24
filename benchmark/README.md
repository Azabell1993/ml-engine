# Benchmark 테스트 방법

## 목차
1. [개요](#개요)
2. [실행 예시](#실행-예시)
3. [인자와 결과 해석](#인자와-결과-해석)
4. [CLI 결과 형태 (예시)](#cli-결과-형태-예시)
5. [산출물 (파일)](#산출물-파일)
6. [해석 팁](#해석-팁)
7. [참고](#참고)

---

## 1. 개요

이 폴더에서는 ml-engine 및 llama.cpp 기반 LLM의 실제 벤치마크(토큰 생성 속도, latency, RAM 사용량 등)를 측정합니다.



---

## 2. 실행 예시

### 2.1 서버 실행

```bash
./build/ml_engine
```

### 2.2 벤치마크용 API 호출 (터미널에서)

```bash
curl -sS -X POST http://localhost:18080/llm/generate \
    -H "Content-Type: application/json" \
    -d '{
        "backend": "llama",
        "prompt": "C++로 helloworld를 출력하는 코드를 단일 블록으로만 답해줘.",
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

---

## 3. 인자와 결과 해석

### 3.1 --runs란?
- 정의: 서비스(로컬 ml-engine, ChatGPT 4.0 Web)별로 측정(run)을 몇 번 반복할지 지정합니다.
- 기본값: 9
- 의미: 같은 조건을 여러 번 반복해 평균/분산/분위수(p50/p95/p99) 등 통계를 안정화합니다.
- 워밍업: --warmups(기본 2회)은 통계에 포함되지 않음 → 캐시 워밍·초기 지연 제거 목적.

### 3.2 사용 예시

```bash
# 빠른 연습용(짧게)
python benchmark_compare.py --runs 3 --warmups 1

# 논문/리포트용(통계 안정성↑)
python benchmark_compare.py --runs 30 --warmups 3
```

### 3.3 권장값 요약

| 목적             | runs   | warmups | 비고                        |
|------------------|--------|---------|-----------------------------|
| 기능 확인 / 디버그 | 2–3    | 1       | 빠른 확인용                 |
| 일반 비교        | 9–15   | 2–3     | 기본 추천                   |
| 논문급 재현성    | 20–50  | 3–5     | 테일 레이턴시(p95/p99) 신뢰도↑ |

---

## 4. CLI 결과 형태 (예시)

> 실제 결과
```
(venv) mac@azabell-mac benchmark % python benchmark_compare.py 

=== Local ml-engine ===
runs=9 | success=100.0% | valid=77.8%
latency ms -> mean:26720.26 p50:23411.17 p95:46359.31 p99:46359.31 min:17452.08 max:46359.31 std:7840.18
tokens/s -> mean:5.12 p50:5.47 p95:7.33

=== ChatGPT 4.0 Web ===
runs=9 | success=100.0% | valid=100.0%
latency ms -> mean:887.69 p50:752.60 p95:2084.21 p99:2084.21 min:643.75 max:2084.21 std:428.23
tokens/s -> mean:41.87 p50:43.85 p95:51.26
/Users/mac/Desktop/workspace/miniGPT/ml-engine/benchmark/benchmark_compare.py:361: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
    plt.boxplot(data, labels=sorted(df["service"].unique()), showfliers=False)
Tables & figures saved: ./bench_results_20250824_221650/paper_table.csv ./bench_results_20250824_221650/paper_table.md ./bench_results_20250824_221650/latency_cdf.png ./bench_results_20250824_221650/latency_box.png ./bench_results_20250824_221650/tokens_per_sec_bar.png

Artifacts saved to: /Users/mac/Desktop/workspace/miniGPT/ml-engine/benchmark/bench_results_20250824_221650
```

---
#### 결과 해석

- **Deepseek gguf 및 llama.cpp 기반 로컬 LLM 성능 분석**  
    - Deepseek gguf 포맷의 모델과 llama.cpp 엔진을 활용하여, 오픈소스 기반 로컬 LLM의 실제 환경 성능을 정량적으로 평가합니다.  
    - Deepseek gguf는 경량화된 모델 포맷과 다양한 하드웨어 지원을 제공하여, 로컬 환경에서 효율적인 LLM 실행이 가능함을 확인할 수 있습니다.  
    - 상용 서비스(ChatGPT)와의 비교를 통해, 하드웨어 사양 및 파라미터 최적화가 성능에 미치는 영향을 체계적으로 분석할 수 있습니다.

- **Local ml-engine (Deepseek gguf + llama.cpp)**  
    - 평균 지연시간(latency)은 약 26.7초로, ChatGPT 4.0 Web 대비 현저히 높게 나타납니다.  
    - p95/p99 분위수(46.4초)에서 테일 레이턴시가 급격히 증가하여, 일부 요청에서 비정상적으로 긴 응답 지연이 발생함을 확인할 수 있습니다.  
    - 토큰 생성 속도는 평균 5.12 tokens/s로, 상용 서비스 대비 처리량이 크게 제한적입니다.  
    - valid 비율(77.8%)이 100%에 미치지 못해, 출력 형식 검증을 일부 통과하지 못한 사례가 존재합니다.

- **ChatGPT 4.0 Web**  
    - 평균 지연시간은 0.89초로 매우 우수하며, p95/p99에서도 2초 내외로 안정적인 응답 특성을 보입니다.  
    - 토큰 생성 속도는 평균 41.87 tokens/s로, 로컬 엔진 대비 약 8배 이상의 처리 성능을 기록합니다.  
    - valid 비율이 100%로, 모든 출력이 요구 조건을 완벽히 충족합니다.

- **종합 평가**  
    - 본 벤치마크는 오픈소스 기반 로컬 LLM의 성능 한계를 객관적으로 진단하고, Deepseek gguf가 제공하는 경량화 및 하드웨어 최적화의 이점을 실무 환경에서 직접 체감할 수 있음을 보여줍니다.  
    - 상용 서비스와의 격차를 수치로 확인함으로써, 실무 적용 시 기대치와 개선 방향을 명확히 설정할 수 있습니다.

#### 각 항목 설명  
	•	runs: 통계에 포함된 측정 횟수(워밍업 제외).    
	•	success (%): HTTP 200 등 정상 응답 비율.  
	•	valid (%): 출력이 “C++ Hello, World!” 단일 코드블록 규칙을 통과한 비율(형식 검증).  
	•	latency ms  
	•	mean: 평균 지연시간  
	•	p50/p95/p99: 지연시간 분위수 (테일 레이턴시 확인용)  
	•	min/max: 최소/최대값  
	•	std: 표준편차(분산 정도)  
	•	tokens/s  
	•	mean / p50 / p95: 초당 생성 토큰 수 (처리량)  
	•	ChatGPT는 usage 기반, 로컬은 -n 값 또는 텍스트 길이로 보수적 추정    


  
## 5. 산출물 (파일)

벤치마크 완료 후 `./bench_results_YYYYMMDD_HHMMSS/` 폴더가 생성됩니다.
- `runs.csv` : 모든 런의 원시 지표(서비스/런별)
- `paper_table.csv`, `paper_table.md` : 논문용 요약 표
- `latency_cdf.png` : 레이턴시 CDF (테일 비교)
- `latency_box.png` : 레이턴시 분포 (Boxplot)
- `tokens_per_sec_bar.png` : 평균 처리량 막대그래프
- `system_metadata.json` : OS/CPU/메모리/파이썬 버전 등 환경 메타

#### 표/그림 생성을 위해서는 pandas, matplotlib가 필요함.

> 설치

```bash
pip install pandas matplotlib
# macOS GUI 이슈 방지
export MPLBACKEND=Agg
```

---

## 6. 해석 팁

- 모델/파라미터 변경 시 `--runs`를 충분히 크게 하여 p95/p99가 수렴하는지 확인하세요.
- 로컬 엔진은 `-ngl`, 스레드 수, 배치 크기(`-b`) 등 하이퍼파라미터에 민감합니다.
- 동일 조건 유지 후 비교해야 공정합니다.
- 결과는 `paper_table.md`와 그림 3종을 README에 바로 삽입하면 됩니다.

---

## 7. 참고

- 다양한 프롬프트, 모델, 파라미터로 반복 테스트하여 평균값 기반으로 기록해야 더 신뢰도 높은 벤치마크가 됩니다.
- 결과는 표로 정리하여 README에 추가하세요.
