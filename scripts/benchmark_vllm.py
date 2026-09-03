# scripts/benchmark_vllm.py

import os
import time
import litellm

# Dedicated vLLM settings isolated from application AI_MODEL
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "https://dullness-tackiness-sneeze.ngrok-free.dev/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "dummy") # Set to empty if no key is required
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4")

# Configure LiteLLM endpoint credentials
litellm.api_base = VLLM_BASE_URL
litellm.api_key = VLLM_API_KEY

def benchmark(model, messages, n_requests=5):
    total_time = 0
    total_tokens = 0
    
    for i in range(n_requests):
        start = time.time()
        try:
            response = litellm.completion(
                model=model,
                messages=messages,
                max_tokens=50,
                temperature=0
            )
            elapsed = time.time() - start
            total_time += elapsed
            tokens = response.usage.total_tokens
            total_tokens += tokens
            print(f"  Request {i+1}: {elapsed:.2f}s, {tokens} tokens")
        except Exception as e:
            print(f"  Request {i+1} failed: {e}")
            raise e
            
    avg_time = total_time / n_requests
    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
    
    return {
        "avg_time": avg_time,
        "tokens_per_sec": tokens_per_sec,
        "total_time": total_time,
        "total_tokens": total_tokens,
    }

if __name__ == "__main__":
    messages = [
        {"role": "user", "content": "What are the prohibited practices in Article 5 of the EU AI Act?"}
    ]
    
    print("Benchmarking vLLM...")
    
    # Use the isolated local vllm model variable with openai prefix
    vllm_model = f"openai/{LOCAL_MODEL_NAME}"
    
    print(f"Target Model: {vllm_model}")
    print(f"API Base: {litellm.api_base}\n")
    
    results_vllm = benchmark(vllm_model, messages, n_requests=5)
    
    print("\n--- Benchmark Results ---")
    print(f"vLLM Average Time: {results_vllm['avg_time']:.2f}s")
    print(f"vLLM Throughput:   {results_vllm['tokens_per_sec']:.2f} tokens/sec")