# 🧠 Legal AI Assistant - Development Roadmap

> **Project Vision:** Build a production-ready, agentic legal AI assistant capable of structured data extraction (EU AI Act), contextual summarization, and advanced RAG, while mastering MLOps practices like vLLM serving, quantization, and Kubernetes orchestration.

## 🏗️ High-Level Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                           │
│                 (Streamlit / FastAPI + React)                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                      Orchestration Layer                        │
│   (Tool-Calling Agent - Decides: Extract vs Summarize vs RAG)   │
└───────────┬───────────────────────────┬────────────────────────┘
            │                           │
┌───────────▼───────────┐     ┌─────────▼────────────────────────────┐
│    Generator Tools    │     │      Retrieval Tools (RAG)           │
│ - Structured Extractor│     │ - Naive / Hybrid / Advanced / Agentic│
│ - Free-Text Summarizer│     │ - Vector DB (ChromaDB) + Rerankers   │
└───────────┬───────────┘     └───────────┬──────────────────────────┘
            │                           │
┌───────────▼───────────────────────────▼────────────────────────┐
│                        Model Serving                           │
│   Cloud APIs (OpenAI/Groq/Gemini) ↔ Local vLLM / TensorRT      │
│                                +                               │
│       Optimization: AWQ/GPTQ Quantization, ONNX Graph Fusion   │
└────────────────────────────────────────────────────────────────┘

```
---
## 🗺️ Development Phases

### Phase 0: ✅ Core Tooling (The "Brain")
*Goal: Create the fundamental LLM functions that do the actual work.*

- [ ] **Structured Extractor:** Finalize `extract_article_structure` using Instructor + Pydantic.
- [ ] **Summarizer:** Build `generate_summary` using vanilla `litellm.completion` (no structured output).
- [ ] **Mock Context:** Write a `mock_retriever(query)` that returns hardcoded text for testing without RAG.

**Deliverable:** Two pure-Python functions that work flawlessly on a static legal text snippet.

---

### Phase 1: 🧠 Orchestrator (The Agent Router)
*Goal: Build the decision engine that routes user intent to the right tool.*

- [ ] **Tool Definitions:** Convert your two Python functions into OpenAI/Anthropic tool-calling schemas.
- [ ] **Router Logic:** Write `orchestrate(user_query, context_text)`.
- [ ] **Execution Loop:** Parse the model's tool call, execute the corresponding function, and return the result.

**Deliverable:** A script where asking *"What is the risk level?"* calls the Extractor, and *"Explain this to me"* calls the Summarizer.

---

### Phase 2: 🔍 Retrieval-Augmented Generation (RAG) Pipeline
*Goal: Replace mock text with real document retrieval. Compare 4 architectures.*

- [ ] **Data Prep:** Chunk the EU AI Act PDF/text and generate embeddings.
- [ ] **Naive RAG:** Dense vector search (ChromaDB + OpenAI embeddings).
- [ ] **Hybrid RAG:** Dense + Sparse (BM25) retrieval fused via Reciprocal Rank Fusion (RRF).
- [ ] **Advanced RAG:** Add a Cross-Encoder Reranker (e.g., `BAAI/bge-reranker-v2-m3`) on top of the top-k results.
- [ ] **Agentic RAG:** Let the Orchestrator decide *how* to search (e.g., by Article Number vs. by Topic).
- [ ] **Integration:** Swap `mock_retriever` with the actual Vector DB. The Orchestrator now receives real legal text.

**Deliverable:** A benchmark notebook/script comparing retrieval quality (Hit Rate, MRR) across the 4 strategies. Choose the winner.

---

### Phase 3: 🚀 Model Serving & Optimization (MLOps Deep Dive)
*Goal: Move from Cloud APIs to self-hosted, optimized open-source models.*

- [ ] **vLLM Setup:** Deploy `Llama-3.1-8B-Instruct` using vLLM's OpenAI-compatible server.
- [ ] **LiteLLM Integration:** Point `.env` to `hosted_vllm/meta-llama/Llama-3.1-8B-Instruct`.
- [ ] **Quantization:** Deploy an **AWQ** or **GPTQ** 4-bit version of the same model.
- [-] **Graph Compilation (Stretch):** Compile the embedding model (`BGE`) via ONNX Runtime/TensorRT to boost embedding generation speed. --< skipped >

**Deliverable:** Two local endpoints (FP16 vs 4-bit) running side-by-side, with performance metrics.

---

### Phase 4: ⚙️ Distributed Training (DDP) -- skiped for next time.
*Goal: Learn distributed training fundamentals.*

- [ ] **Dataset:** Select a small legal dataset (e.g., LexGLUE for classification) and a small model (e.g., `law-ai/InLegalBERT`).
- [ ] **Baseline:** Train on a single GPU.
- [ ] **DDP Script:** Convert the training script using PyTorch `DistributedDataParallel`.
- [ ] **Launch:** Run it via `torchrun --nproc_per_node=2 train.py`.
- [ ] **Analysis:** Log speedup and gradient synchronization overhead.

**Deliverable:** Training logs showing scaling efficiency from 1 GPU to 2 GPUs (or simulated using cloud credits).

---

### Phase 5: ☸️ Containerization 

- [ ] **Dockerize:** Write a `Dockerfile` for the Orchestrator + FastAPI server.
- [ ] **Dockerize vLLM:** Use the official vLLM Docker image for the model server.

**Deliverable:** A `docker-compose.yml` for local runs, plus K8s YAMLs ready for cloud deployment.

---

---

## 🧪 Experiments & Benchmarks

| Experiment | Phase | Tools | Success Metric |
| :--- | :--- | :--- | :--- |
| **Serving Throughput** | Phase 3 | `locust` / `wrk` | Tokens/Second compared between Flask wrapper and raw vLLM endpoint. |
| **Quantization Trade-off** | Phase 3 | `nvidia-smi`, Perplexity | VRAM usage (FP16 vs. AWQ) and output quality difference. |
| **Graph Compilation** | Phase 3 (Stretch) | ONNX Runtime, TensorRT | Latency reduction in embedding generation (ms). |
| **Distributed Scaling** | Phase 4 | PyTorch DDP logs | Speedup factor (e.g., 1.8x on 2 GPUs vs. 1 GPU). |

---

## 🆓 Free Tier Strategy

| Resource | Use Case | Free Tier Details |
| :--- | :--- | :--- |
| **Groq API** | Fast orchestration/routing | 1,000 requests/day free. |
| **Google Gemini API** | Summarization/Extraction | 1,500 requests/day (Flash) / 100 (Pro). |
| **ChromaDB** | Local Vector DB | 100% Open Source. |
| **Oracle Cloud** | Kubernetes (K8s) | "Always Free" 2 CPU / 12GB RAM. |
| **Cloud GPU Credits** | vLLM / DDP Experiments | Sign-up bonuses from AWS/GCP/RunPod. |

---

## 📅 Suggested Sprint Timeline

- **Sprint 1 (Week 1-2):** Phases 0 & 1 (Core Tools + Orchestrator).
- **Sprint 2 (Week 3-4):** Phase 2 (RAG Pipeline).
- **Sprint 3 (Week 5-6):** Phase 3 (vLLM & Quantization).

---
