## 🔗 Step 2: Local Test Script (`scripts/test_vllm_endpoint.py`)
# get a public URL like `https://abc123.ngrok.io` from notebook/vllm_server.ipynb. Copy that URL to run it locally.
# scripts/test_vllm_endpoint.py

import os
import openai

# Replace with your actual ngrok URL
VLLM_BASE_URL = "https://dullness-tackiness-sneeze.ngrok-free.dev/v1"

client = openai.OpenAI(
    base_url=VLLM_BASE_URL,
    api_key="dummy"  # vLLM ignores this when --api-key dummy is set
)

response = client.chat.completions.create(
    model="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    messages=[
        {"role": "user", "content": "What is the EU AI Act in one sentence?"}
    ],
    max_tokens=50
)

print("Response:", response.choices[0].message.content)