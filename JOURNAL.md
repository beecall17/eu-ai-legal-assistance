# 🚀 Engineering Journal: Building a Production-Ready RAG System for Legal AI

## The Journey from 7% Failure to 90% Accuracy

---

## 📅 Phase 0: Foundation & Core Tooling

### Day 1: Setting Up the Config Layer

**What I Did:**
- Created `config/settings.py` as a single source of truth for all environment variables, API keys, and model configurations.
- Set up `.env` with Groq, Gemini, and (initially) OpenAI keys.
- Implemented fallback model logic so if one provider fails, the system gracefully degrades.

**Why It Mattered:**
> *"I wanted to avoid the nightmare of hardcoded API keys scattered across 10 files. One `.env`, one `settings.py`, and everything else just imports from there."*

### Day 2: Building the Core Tools (The "Brain")

I implemented two core functions that act as **local MCP (Model Context Protocol) tools**:

#### Tool 1: `extract_article_structure`
- Uses `instructor` + LiteLLM to force the LLM into a strict Pydantic schema.
- Extracts: `article_number`, `title`, `prohibited_practices`, and `risk_level`.
- **Key insight**: Legal text requires structured output – you can't have the LLM hallucinate fields.

#### Tool 2: `generate_summary`
- Vanilla `litellm.completion` (no schema enforcement).
- Supports multiple tones: `concise`, `detailed`, `executive`, `layman`.
- **Why separate?** Mixing structured extraction and free-text summarization in one prompt leads to garbage output.

**What Broke:**
- Initially tried to make one function do both – the LLM got confused and either returned incomplete JSON or random text.
- **Fix**: Split into two dedicated functions. Single responsibility principle.

```python
# Example error I hit:
ValidationError: 1 validation error for EUAIActArticleSchema
prohibited_practices
  Expected list[str] but got str (type=type_error)
```

---

## 🧠 Phase 1: The Orchestrator (Agentic Router)

### Building the Decision Engine

I created an **orchestrator** that acts as the "brain" of the system:
- Takes a user query + optional context.
- Decides which tool(s) to call (extract, summarize, or search).
- Executes the tools and returns a combined response.

**Architecture:**
```
User Query → Orchestrator → Tool Call (LLM decides) → Execute → Return
```

**Key Design Decision:**
- Used **OpenAI-style function calling** to let the LLM choose tools.
- This is essentially a **local MCP server** – the orchestrator exposes tools to the LLM.

**What I Learned:**
- The orchestrator needs a **system prompt** that clearly explains each tool's purpose.
- If you're vague, the LLM will call the wrong tool or hallucinate arguments.

**Edge Cases I Handled:**
- **No tool called**: Fallback to direct response.
- **Multiple tools called**: Execute all and combine results.
- **Tool execution fails**: Return a friendly error.

---

## 🔍 Phase 2: The RAG Pipeline Evolution

### The Problem: Garbage In, Garbage Out

**Initial Attempt:**
- Used `MarkdownHeaderTextSplitter` to chunk the EU AI Act PDF.
- **Result**: Metadata was all garbage (`{'CHAPTER': '**of 13 June 2024**'}`).
- **Why?** The PDF didn't have Markdown headers that the splitter expected.

**The Fix:**
- Switched to `pymupdf4llm` for PDF extraction.
- Wrote a **custom regex-based parser** that detects `Article 5`, `ANNEX I`, and recitals.
- Implemented **Parent-Child chunking**:
  - **Parents**: Entire articles (preserve structure).
  - **Children**: 400-char chunks with overlap (for vector search).
  - Metadata propagated: `section_id`, `article_number`, `page_start/end`.

**Validation:**
```python
# The "aha!" moment when I saw this:
Metadata: {'section_type': 'recital', 'section_number': '1', 'section_id': 'recital_1'}
Preview: (1) The purpose of this Regulation is to improve the functioning of the internal market...
```

---

### Naive RAG (Baseline)

**What I Did:**
- Dense vector search only (sentence-transformers + ChromaDB).
- Top-5 retrieval.

**Evaluation:**
- **Hit Rate @ 5**: 75.00%
- **MRR**: 0.513

**Limitation:** Queries with specific article numbers (e.g., "Article 5") often missed exact matches because embeddings capture semantics, not exact keywords.

---

### Hybrid RAG (BM25 + RRF)

**What I Did:**
- Added **BM25** (sparse retrieval) alongside dense.
- Fused results with **Reciprocal Rank Fusion (RRF)**.

**Why RRF?** It normalizes rankings from different retrievers without needing to scale scores.

**Evaluation:**
- **Hit Rate @ 5**: 84.38% (+9.38% improvement)
- **MRR**: 0.597 (+0.084)

**Key Insight:** Hybrid caught exact article mentions (BM25) while dense captured semantic concepts. Together, they covered more ground.

---

### Advanced RAG (Cross-Encoder Reranking)

**What I Did:**
- Used Hybrid RAG as first-stage retriever (top-20 candidates).
- Re-ranked candidates with a Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`).

**Why Cross-Encoder?** It performs full attention between query and document, producing more accurate relevance scores than bi-encoders.

**Evaluation:**
- **Hit Rate @ 5**: 84.38% (same as hybrid)
- **MRR**: 0.651 (+0.054 improvement)

**Observation:** Hit Rate didn't improve, but **MRR did** – meaning the relevant chunk moved higher in the rankings. This is valuable because the orchestrator only sees the top results.

**Lesson:** Reranking is about precision, not recall.

---

## ⚙️ Phase 3: Model Serving with vLLM

### The Challenge: Free GPU Inference

**Goal:** Deploy a local LLM on a free T4 GPU (16GB VRAM) to compare with cloud APIs.

**Solution:**
- **Quantized model**: `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` (fits in ~8GB).
- **Serving layer**: vLLM with PagedAttention.
- **Exposure**: ngrok tunnel to make it accessible from my local orchestrator.

**Setup Process:**
1. Wrote a Colab notebook to run vLLM.
2. Installed vLLM, auto-gptq, and ngrok.
3. Launched the server with `--gpu-memory-utilization 0.9`.
4. Exposed via ngrok (public URL).

**What Broke:**
- **First attempt**: FP16 model (16GB) caused OOM on T4.
  - **Fix**: Switched to AWQ quantized model.
- **Second attempt**: Model download timed out.
  - **Fix**: Used `huggingface_hub` to download before starting.
- **Third attempt**: vLLM didn't recognize the quantized model.
  - **Fix**: Used `--dtype auto` and explicitly set the model path.

**Benchmark Results:**

| Strategy | Avg Latency | Throughput |
| :--- | :--- | :--- |
| **Local vLLM (Llama-3.1-8B-AWQ)** | **1.74s** | **57.44 tokens/sec** |
| Google Gemini (Gemini 3.6 Flash) | 1.80s | 35.05 tokens/sec |

**Key Takeaways:**
- vLLM is **significantly faster** (57 vs 35 tokens/sec).
- Gemini had **503 errors** on 2/3 requests due to high demand.
- **Local inference wins on reliability** – no rate limits, no network overhead.

---

## 🔄 Agentic RAG: Letting the LLM Choose

**What I Did:**
- Extended the orchestrator with **three search tools**: `search_naive`, `search_hybrid`, `search_advanced`.
- The LLM decides which tool to use based on the query.
- Also integrated extraction and summarization tools for analysis.

**Example Decision Flow:**
- Query: *"What does Article 5 say about prohibited practices?"*
  - LLM chooses `search_hybrid` (exact article mention).
- Query: *"What is the overall purpose of the AI Act?"*
  - LLM chooses `search_naive` (semantic only).
- Query: *"Can you extract the risk level and provide a summary?"*
  - LLM calls both `extract_structured_metadata` and `generate_text_summary`.

**What I Learned:**
- Tool-calling is powerful but requires clear system prompts.
- The LLM can chain tools: search → extract → summarize.

---

## 🐛 The Docker Debugging Marathon

**Attempt 1:**
- `docker-compose up --build`
- **Error**: `.env` parsing failed because of unescaped quotes.
- **Fix**: Changed `FALLBACK_MODELS="[\"groq/llama-3.1-8b-instant\", \"gemini/gemini-1.5-flash\"]"`.

**Attempt 2:**
- **Error**: `ModuleNotFoundError: No module named 'rag.vector_store'`
- **Fix**: Docker wasn't copying the entire project structure. Added `COPY . .`.

**Attempt 3:**
- **Error**: `chromadb` required `duckdb` but couldn't install it.
- **Fix**: Added `duckdb` to `requirements.txt` and rebuilt.

**Attempt 4:**
- **Error**: `curl: (7) Failed to connect to localhost port 8000`
- **Cause**: Container wasn't mapping ports correctly.
- **Fix**: `ports: - "8000:8000"` in `docker-compose.yml`.

**Final State:** The Docker container built successfully but the `curl` command gave a 503 (service unavailable). **Root cause:** The orchestrator couldn't reach the Groq API due to network issues inside the container.

**Lesson Learned:** Container networking is tricky – need to set proper environment variables and ensure internet access.

---

## 📊 Final Evaluation Summary

### RAG Performance

| Strategy | Hit Rate @ 5 | MRR |
| :--- | :--- | :--- |
| **Naive RAG** | 75.00% | 0.513 |
| **Hybrid RAG (BM25+RRF)** | 84.38% | 0.597 |
| **Advanced RAG (Reranking)** | 84.38% | 0.651 |

**Key Insight:** Hybrid + Reranking didn't improve Hit Rate but significantly improved MRR – meaning the relevant chunk moved from position 3 to position 1.

### Model Serving Benchmark

| Strategy | Latency | Throughput | Reliability |
| :--- | :--- | :--- | :--- |
| **Local vLLM (Llama-3.1-8B-AWQ)** | 1.74s | 57.44 tokens/sec | 100% |
| **Google Gemini 3.6 Flash** | 1.80s | 35.05 tokens/sec | 66% (503 errors) |

**Takeaway:** Local inference on a free T4 GPU **outperforms cloud APIs** in speed and reliability – a counterintuitive but powerful result.

---

## 💡 Lessons Learned

1. **Chunking is everything**: Without parent-child metadata, RAG is blind to document structure.
2. **Hybrid retrieval > pure dense**: BM25 catches exact terms that embeddings miss.
3. **Reranking improves precision**: Cross-Encoders push the right document to the top.
4. **Quantized models are production-ready**: AWQ 4-bit + vLLM delivers 57 tokens/sec on a free GPU.
5. **Docker is hard but worth it**: Containerization forces you to document dependencies.
6. **Orchestration patterns > monolithic agents**: Tool-based design is more flexible and debuggable.

---

## 🚀 What's Next? (Future Work)

- **Graph Compilation**: Compile the embedding model (BGE) with ONNX/TensorRT to speed up embedding generation.
- **Kubernetes Deployment**: Scale horizontally with persistent storage for ChromaDB.
- **CI/CD**: GitHub Actions to automatically build and deploy the Docker container.
- **Monitoring**: Add Prometheus + Grafana for request tracking.

---

## 🙏 Final Thoughts

This project taught me that **AI engineering is 20% model code and 80% plumbing** – retrieval, serving, containerization, and orchestration. The model is just the final piece.

The most valuable skill I built was **systematic debugging** – not just fixing errors, but understanding why they happened and designing to prevent them.

> *"The best code is the code you don't have to write twice."*

---

*Written on: September 3, 2026*

*Tools used: Python, LiteLLM, ChromaDB, vLLM, Docker, Groq, Gemini, and way too much coffee.* ☕