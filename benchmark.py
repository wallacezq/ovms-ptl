import time
import requests
import psutil
import json
import sys

OPENAI_API_URL = "http://localhost:8000/v3/chat/completions"
MODEL_NAME = "Qwen/Qwen3-8B"

CONTEXT_MAP = {
    "2k": 2 * 1024,
    "8k": 8 * 1024,
    "16k": 16 * 1024,
    "32k": 32 * 1024,
    "64k": 64 * 1024,
}


def generate_long_prompt(target_tokens: int) -> str:
    # Approximate. True tokens depend on tokenizer.
    base = "OpenVINO long context test. This sentence is repeated to build a large prompt. "
    repeat = max(1, target_tokens // 20)
    return base * repeat


def print_memory():
    mem = psutil.virtual_memory()
    print(f"[SYSTEM] RAM used: {mem.used/1e9:.2f} GB / {mem.total/1e9:.2f} GB")


def run_test(ctx_key: str):
    if ctx_key not in CONTEXT_MAP:
        print("Usage: python benchmark.py [2k|8k|16k|32k|64k]")
        sys.exit(1)

    target_tokens = CONTEXT_MAP[ctx_key]

    print(f"\n=== TEST CONFIG ===\nContext: {ctx_key} ({target_tokens} approx tokens)\nEndpoint: {OPENAI_API_URL}\nModel: {MODEL_NAME}\n")

    prompt = generate_long_prompt(target_tokens)
    prompt += "\n\nSummarize the above content in 3 bullet points:\n"

    print(f"Prompt size (chars): {len(prompt)}")
    print_memory()

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "stream": True,
    }

    start_time = time.time()
    first_token_time = None
    chunk_count = 0

    print("\n=== STREAM OUTPUT (prefill may be silent) ===\n")

    r = requests.post(
        OPENAI_API_URL,
        headers={"Content-Type": "application/json"},
        json=payload,
        stream=True,
        timeout=600,
    )

    if r.status_code != 200:
        print("\n=== ERROR RESPONSE ===")
        print(r.status_code)
        print(r.text)
        return

    for line in r.iter_lines():
        if not line:
            continue
        s = line.decode("utf-8")
        if not s.startswith("data: "):
            continue
        data_str = s[6:]
        if data_str == "[DONE]":
            break

        data_json = json.loads(data_str)
        delta = data_json["choices"][0]["delta"].get("content", "")

        if delta:
            now = time.time()
            if first_token_time is None:
                first_token_time = now
            chunk_count += 1
            print(delta, end="", flush=True)

    end_time = time.time()

    total_latency = end_time - start_time
    ttft = (first_token_time - start_time) if first_token_time else None
    decode_time = (end_time - first_token_time) if first_token_time else None

    print("\n\n=== METRICS ===")
    print(f"Total latency: {total_latency:.2f} sec")

    if ttft is not None:
        print(f"TTFT (prefill proxy): {ttft:.3f} sec")
        print(f"Prefill throughput (approx): {target_tokens / ttft:.2f} tokens/sec")
    else:
        print("TTFT: N/A (no streamed tokens received)")

    if decode_time and chunk_count:
        print(f"Decoding throughput (chunk/sec): {chunk_count / decode_time:.2f}")

    print_memory()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python benchmark.py [2k|8k|16k|32k|64k]")
        sys.exit(1)

    run_test(sys.argv[1])
