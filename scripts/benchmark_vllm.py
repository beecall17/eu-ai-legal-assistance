# scripts/benchmark_vllm_gemini.py

import os
import time
import litellm

# Define models to compare (pulling keys safely from environment variables)
MODELS_TO_TEST = [
    {
        "label": "Local vLLM (Llama-3.1-8B-AWQ)",
        "model": "openai/hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        "api_base": "https://dullness-tackiness-sneeze.ngrok-free.dev/v1",
        "api_key": "dummy"
    },
    {
        "label": "Google Gemini (Gemini 3.6 Flash)",
        "model": os.getenv("AI_MODEL", "gemini/gemini-3.6-flash"),
        "api_base": None,  # Uses Google's default endpoint automatically
        "api_key": os.getenv("GEMINI_API_KEY")
    }
]

def benchmark_model(config, messages, n_requests=3):
    print(f"\n🚀 Benchmarking: {config['label']}")
    print(f"   Model String: {config['model']}")
    
    total_time = 0
    total_tokens = 0
    successful_requests = 0
    
    for i in range(n_requests):
        start = time.time()
        try:
            kwargs = {
                "model": config["model"],
                "messages": messages,
                "max_tokens": 50,
                "temperature": 0
            }
            if config.get("api_base"):
                kwargs["api_base"] = config["api_base"]
            if config.get("api_key"):
                kwargs["api_key"] = config["api_key"]
                
            response = litellm.completion(**kwargs)
            elapsed = time.time() - start
            
            total_time += elapsed
            tokens = response.usage.total_tokens
            total_tokens += tokens
            successful_requests += 1
            print(f"    Request {i+1}: {elapsed:.2f}s | {tokens} tokens")
            
        except Exception as e:
            print(f"    Request {i+1} failed: {e}")
            
    if successful_requests == 0:
        return None
        
    avg_time = total_time / successful_requests
    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
    
    return {
        "label": config["label"],
        "avg_time": avg_time,
        "tokens_per_sec": tokens_per_sec,
        "total_tokens": total_tokens
    }

if __name__ == "__main__":
    messages = [
        {"role": "user", "content": "What are the prohibited practices in Article 5 of the EU AI Act?"}
    ]
    
    print("=== Starting vLLM vs. Gemini Performance Comparison ===")
    results = []
    
    for model_config in MODELS_TO_TEST:
        # Skip if Gemini key isn't provided yet
        if "gemini" in model_config["model"] and not model_config["api_key"]:
            print(f"\n⚠️ Skipping {model_config['label']}: GEMINI_API_KEY environment variable not found.")
            continue
            
        res = benchmark_model(model_config, messages, n_requests=3)
        if res:
            results.append(res)
            
    if results:
        print("\n" + "="*40)
        print("📊 FINAL COMPARISON SUMMARY")
        print("="*40)
        for r in results:
            print(f"• {r['label']}:")
            print(f"  - Average Latency:  {r['avg_time']:.2f}s")
            print(f"  - Throughput Speed: {r['tokens_per_sec']:.2f} tokens/sec")
            print("-" * 40)