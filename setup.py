# setup_project.py
import os
import sys

# Define the folder structure
structure = {
    "core": ["__init__.py", "extractor.py", "summarizer.py", "mock_retriever.py"],
    "orchestrator": ["__init__.py", "agent.py", "tools.py"],
    "rag": ["__init__.py", "naive.py", "hybrid.py", "advanced.py", "agentic.py", "vector_store.py"],
    "serving": ["__init__.py", "vllm_deploy.py", "quantization.py", "benchmark.py"],
    "deploy": ["__init__.py", "Dockerfile", "docker-compose.yml", "k8s/"],
    "frontend": ["__init__.py", "app.py", "streamlit_ui.py"],
    "notebooks": ["rag_evaluation.ipynb", "serving_benchmarks.ipynb", "ddp_training.ipynb"],
    "config": ["__init__.py", "settings.py"],
    "data": ["raw/", "chunks/", "embeddings/"],
    "tests": ["__init__.py", "test_extractor.py", "test_summarizer.py"],
    "scripts": ["download_models.py", "launch_vllm.sh"]
}

# Define root-level files
root_files = [
    "main.py",
    "requirements.txt",
    "README.md",
    "ROADMAP.md",
    ".env.example",
    ".gitignore"
]

def create_structure(base_path):
    for folder, files in structure.items():
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        for file in files:
            if "/" in file:  # Handle nested directories like k8s/
                nested_path = os.path.join(folder_path, file)
                os.makedirs(nested_path, exist_ok=True)
            else:
                file_path = os.path.join(folder_path, file)
                with open(file_path, 'w') as f:
                    f.write(f"# {folder}/{file}\n")

    # Create root files
    for file in root_files:
        file_path = os.path.join(base_path, file)
        with open(file_path, 'w') as f:
            if file == ".env.example":
                f.write("""# API Keys (get free ones)
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
ANTHROPIC_API_KEY=your_anthropic_key

# Model Routing
AI_MODEL=openai/gpt-4o-mini
FALLBACK_MODELS=["groq/llama-3.1-8b-instant", "gemini/gemini-1.5-flash"]

# Local vLLM (for Phase 3)
VLLM_BASE_URL=http://localhost:8000/v1
LOCAL_MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct

# RAG Settings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMA_DB_PATH=./data/chromadb
""")
            elif file == ".gitignore":
                f.write("""__pycache__/
*.pyc
.env
data/chunks/
data/embeddings/
data/chromadb/
*.log
*.pkl
*.onnx
*.engine
*.bin
.DS_Store
venv/
.venv/
""")
            elif file == "requirements.txt":
                f.write("""# Core
langchain
langchain-community
chromadb
sentence-transformers
rank-bm25

# LLM & Orchestration
litellm
instructor
openai
anthropic
groq
google-generativeai

# Serving & Optimization (Phase 3)
vllm
tensorrt-llm
onnx
onnxruntime
torch
# Note: For CUDA versions, refer to PyTorch website.

# Frontend & API
fastapi
uvicorn
streamlit
pydantic-settings

# Monitoring & Evaluation
locust
pandas
matplotlib
scikit-learn
jupyter

# Distributed Training (Phase 4)
torch
torchvision
torchaudio  # or just torch for DDP
accelerate
deepspeed

# Utilities
python-dotenv
typing-extensions
pydantic
""")
            elif file == "README.md":
                f.write("""# 🧠 Legal AI Assistant

## Overview
This project builds an agentic legal AI assistant capable of structured data extraction, summarization, and advanced RAG, with a focus on MLOps practices.

## Roadmap
See [ROADMAP.md](ROADMAP.md) for the detailed development plan.

## Quick Start
1. Copy `.env.example` to `.env` and add your API keys.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the orchestrator: `python main.py`
""")
            else:
                f.write(f"# {file}\n")

    print(f"✅ Project structure created in: {base_path}")

if __name__ == "__main__":
    # Use current directory or pass a path as argument
    base = sys.argv[1] if len(sys.argv) > 1 else "."
    create_structure(base)