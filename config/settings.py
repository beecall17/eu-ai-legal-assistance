# config/settings.py
import os
from dotenv import load_dotenv
from typing import List

load_dotenv()

# Cloud API Models
AI_MODEL = os.getenv("AI_MODEL", "gemini/gemini-3.6-flash")
FALLBACK_MODELS = eval(os.getenv("FALLBACK_MODELS", '["groq/llama-3.1-8b-instant", "openai/gpt-4o-mini" ]'))


# Local vLLM (Phase 3)
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

# RAG Settings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chromadb")

# API Keys (loaded via .env)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")