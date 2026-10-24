# Optional serving runners: vLLM and llama.cpp

This repo measures int8 vs fp32 with torch dynamic quantisation on a tiny MLP. For real language
models two open-source runners are the usual next step. Neither is installed or run here; this page
is guidance only.

| Runner | Best for | Hardware | Quantisation formats | How it would plug in |
|---|---|---|---|---|
| vLLM | high-throughput serving of many concurrent requests | GPU (CPU backend exists but is slow) | fp16/bf16, int8, int4 (AWQ, GPTQ), fp8 | OpenAI-compatible HTTP server; the agent calls it like any chat endpoint |
| llama.cpp | single-user or edge serving on CPU and Apple silicon | CPU, small GPUs | GGUF k-quants (Q4_K_M, Q5_K_M, Q8_0) | `llama-server` exposes an OpenAI-compatible endpoint |

## Why they are optional here

* vLLM mainly pays off on a GPU, which this lab avoids to stay free.
* llama.cpp would run on CPU, but needs a multi-gigabyte model download and gives
  machine-dependent numbers, so it does not belong in CI.

## How to evaluate either one the same way this repo does

1. Fix a gold test set (here: 1,000 messages) and score the full-precision model.
2. Score the quantised model on the same set; report the accuracy difference, not just speed.
3. Time single requests (p50 and p95) and, separately, throughput under concurrency.
4. Record the machine, thread count and runner version next to the numbers, as `results/bench.json` does.
