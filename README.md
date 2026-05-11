# Hosting Qwen3 with OpenVINO Model Server on Panther Lake

This repo provides:

1) **Model export + INT4 quantization** for **Qwen/Qwen3-8B** into OpenVINO IR (using OVMS **2026.1** demo export script)
2) A **GPU-ready OpenVINO Model Server (OVMS) 2026.1** launcher exposing **OpenAI-compatible** endpoints (`/v3/chat/completions`)
3) A **long-context benchmark client** that measures **TTFT (prefill proxy)** and **decode throughput** for context sizes `2k/8k/16k/32k/64k`

> Why OVMS 2026.1: 2026.1 adds an extra streaming event right after the first token is generated (helps TTFT benchmarking) and reports KV cache allocation/usage in logs. [OpenVINO Model Server 2026.1 release notes](https://github.com/openvinotoolkit/model_server/releases). 

---

## Prerequisites

- Docker
- Intel Arc GPU accessible via `/dev/dri` (dGPU)
- Network access to Hugging Face Hub (set `HF_TOKEN` to avoid rate limiting)

---

## Step 0) Install `uv` (Astral)

Linux/macOS (standalone installer):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open a new shell (or ensure `~/.local/bin` is on `PATH`) then verify:

```bash
uv --version
```

---

## Step 1) Create Python envs (using uv)

We keep two environments:

- `.venv-export` for model export/quantization dependencies
- `.venv` for the benchmark client

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements-benchmark.txt

uv venv .venv-export --python 3.11
```

---

## Step 2) Export + INT4 quantize Qwen3-8B to OpenVINO IR (OVMS 2026.1 tooling)

```bash
bash scripts/export_model.sh
```

### Best-known export knobs for long context (e.g., 64K)

- **KV cache compression**: `KV_CACHE_PRECISION=u8` (INT8 KV cache)
- **Weights**: `WEIGHT_FORMAT=int4`
- **Cache sizing**: increase `CACHE_SIZE` to allow long contexts and/or concurrency.

This repo defaults to `CACHE_SIZE=16` (a starting point for 64K tests) and makes it overridable.

Override example:

```bash
CACHE_SIZE=24 WEIGHT_FORMAT=int4 KV_CACHE_PRECISION=u8 bash scripts/export_model.sh
```

(OVMS docs recommend enabling prefix caching and using INT8 KV cache compression, and sizing cache based on available memory, expected concurrency, and context length.)

---

## Step 3) Launch OVMS 2026.1 (OpenAI-compatible)

```bash
bash scripts/start_ovms.sh
```

Verify readiness:

```bash
curl http://localhost:8000/v1/config
```

---

## Step 4) Run long-context benchmark

```bash
source .venv/bin/activate

python benchmark.py 2k
python benchmark.py 8k
python benchmark.py 16k
python benchmark.py 32k
python benchmark.py 64k
```

Or without activation:

```bash
uv run --python .venv/bin/python benchmark.py 64k
```

### Metrics

- **TTFT** (time-to-first-token) is a proxy for **prefill latency**.
- **Prefill throughput** is estimated as `target_context_tokens / TTFT` (approx; true token count depends on tokenizer).
- **Decode throughput** is reported as streamed **chunks/sec** (chunk != token).

---

## Step 5) Cleanup

Stop OVMS:

```bash
docker ps | grep openvino/model_server | awk '{print $1}' | xargs -r docker stop
```


