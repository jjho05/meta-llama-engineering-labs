#!/usr/bin/env python3
"""
Module: Applied Artificial Intelligence with Open Weights Models
Challenge: Llama Multi-Model Benchmark & Inference Profiling
Author: Ing. Jesús Javier Hernández Olvera
"""

import os
import re
import sys
import time
import pprint
import getpass
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# -----------------------------------------------------------------------------
# 1. Load Environment Variables & Initialize Groq Client
# -----------------------------------------------------------------------------
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

API_KEY = os.environ.get("GROQ_API_KEY")

if not API_KEY:
    print("[Notice] GROQ_API_KEY not found in environment or .env file.")
    API_KEY = getpass.getpass("Enter your GROQ_API_KEY: ").strip()

if not API_KEY:
    print("[Error] A valid Groq API Key is required to run this benchmark.")
    sys.exit(1)

client = Groq(api_key=API_KEY)
print("=" * 100)
print("MODULE: APPLIED AI WITH OPEN WEIGHTS MODELS")
print("CHALLENGE 1: MULTI-MODEL LLM BENCHMARK (SLM 20B vs LLM 120B vs CoT 27B)")
print("Author / Architect: Ing. Jesús Javier Hernández Olvera")
print("=" * 100)

# Dynamic fallback selection based on active Groq endpoints
def get_available_model(client, preferred, fallback):
    try:
        models = [m.id for m in client.models.list().data]
        return preferred if preferred in models else fallback
    except Exception:
        return fallback

def sanitize_output(text):
    if not text: return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

MODEL_LIGHT  = get_available_model(client, "llama-3.1-8b-instant", "openai/gpt-oss-20b")
MODEL_LARGE  = get_available_model(client, "llama-3.3-70b-versatile", "openai/gpt-oss-120b")
MODEL_QWEN   = get_available_model(client, "qwen/qwen3.6-27b", "qwen/qwen3.6-27b")

print(f"1. Lightweight Model (SLM): {MODEL_LIGHT}")
print(f"2. Large Foundation Model:   {MODEL_LARGE}")
print(f"3. CoT Reasoning Model:      {MODEL_QWEN}\n")

# -----------------------------------------------------------------------------
# 2. Define Benchmark Test Queries (Technical Support & Architecture)
# -----------------------------------------------------------------------------
queries = [
    {
        "id": "Q1_SIMPLE",
        "category": "Factual / Simple Query",
        "prompt": "What is the capital of France and what is its official currency? Answer directly in 1 concise sentence without preamble."
    },
    {
        "id": "Q2_TECHNICAL",
        "category": "Technical Architecture",
        "prompt": "Explain the architectural difference between Scaled Dot-Product Attention and Grouped-Query Attention (GQA) in Meta Llama 3. Explain how KV-Cache memory is reduced."
    },
    {
        "id": "Q3_REASONING",
        "category": "Multi-Step Logic / Reasoning",
        "prompt": "A company processes 1,000,000 tokens daily. An INT4 quantized 8B model costs $0.20 per million tokens with 15ms latency. A 70B model costs $0.90 per million with 85ms latency. If 85% of queries are routine and 15% require complex reasoning, calculate the monthly cost and average latency under a tiered routing architecture."
    }
]

models_to_test = [
    {"name": "Lightweight (20B / 8B)", "id": MODEL_LIGHT, "max_tokens": 400},
    {"name": "CoT Reasoning (27B)", "id": MODEL_QWEN, "max_tokens": 800},
    {"name": "Enterprise (120B / 70B)", "id": MODEL_LARGE, "max_tokens": 600}
]

# -----------------------------------------------------------------------------
# 3. Execute Benchmarking Matrix on Groq LPU Hardware
# -----------------------------------------------------------------------------
results = []

for q in queries:
    print("-" * 100)
    print(f"Testing Query [{q['id']}] ({q['category']}):")
    print(f"\"{q['prompt']}\"\n")
    
    for m in models_to_test:
        print(f"  --> Executing on {m['name']} ({m['id']})...")
        t0 = time.time()
        try:
            response = client.chat.completions.create(
                model=m["id"],
                messages=[
                    {"role": "system", "content": "You are a professional AI engineer. Answer with maximum technical precision without emojis."},
                    {"role": "user", "content": q["prompt"]}
                ],
                max_tokens=m["max_tokens"],
                temperature=0.2
            )
            elapsed = time.time() - t0
            usage = response.usage
            completion_tokens = usage.completion_tokens if usage else 0
            prompt_tokens = usage.prompt_tokens if usage else 0
            tps = completion_tokens / elapsed if elapsed > 0 else 0
            clean_text = sanitize_output(response.choices[0].message.content)

            results.append({
                "query_id": q["id"],
                "model_name": m["name"],
                "model_id": m["id"],
                "latency_s": round(elapsed, 3),
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "tokens_per_sec": round(tps, 1),
                "output_preview": clean_text[:120] + "..." if len(clean_text) > 120 else clean_text
            })
            print(f"      Latency: {elapsed:.3f}s | Tokens: {completion_tokens} | Throughput: {tps:.1f} t/s")
        except Exception as e:
            print(f"      [Error executing {m['id']}]: {e}")

# -----------------------------------------------------------------------------
# 4. Display Analytical Performance Matrix
# -----------------------------------------------------------------------------
print("\n" + "=" * 100)
print("FINAL BENCHMARK PERFORMANCE MATRIX (GROQ LPU)")
print("=" * 100)
header_str = "{:<15} | {:<24} | {:<12} | {:<8} | {:<12}".format("Query ID", "Model", "Latency (s)", "Tokens", "Speed (t/s)")
print(header_str)
print("-" * 80)
for r in results:
    row_str = "{:<15} | {:<24} | {:<12.3f} | {:<8} | {:<12.1f}".format(r["query_id"], r["model_name"], r["latency_s"], r["completion_tokens"], r["tokens_per_sec"])
    print(row_str)
print("=" * 100)
print("Benchmark completed successfully.")
