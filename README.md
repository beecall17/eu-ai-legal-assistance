
# 🧠 Legal AI Assistant (EU AI Act)

> A production‑ready, agentic RAG system for navigating and analyzing the EU Artificial Intelligence Act.  
> Combines structured extraction, hybrid search, cross‑encoder reranking, and quantized LLM serving on free GPUs.

![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![vLLM](https://img.shields.io/badge/vLLM-0.6.0-orange)

---

## 🎯 Overview

The **EU AI Legal Assistant** is a complete AI engineering project that demonstrates:

- **Agentic Orchestration** – an LLM decides which tools (extraction, summarization, search) to invoke based on the user’s query.
- **Multi‑Stage RAG** – from naive dense retrieval → hybrid (BM25+RRF) → cross‑encoder reranking, improving MRR from 0.513 to 0.651.
- **Self‑Hosted Inference** – serving a quantized Llama-3.1-8B on a free T4 GPU via vLLM, achieving 57 tokens/sec – faster and more reliable than Gemini/Groq free tiers.
- **Production‑Ready Packaging** – fully containerized with Docker, and Kubernetes manifests for horizontal scaling.

This project was built to learn and showcase the full spectrum of modern AI engineering: from chunking legal documents to deploying a scalable API.

---

## 📊 Performance Highlights

| RAG Strategy | Hit Rate @ 5 | MRR |
| :--- | :--- | :--- |
| Naive (Dense only) | 75.0% | 0.513 |
| Hybrid (BM25 + RRF) | 84.4% | 0.597 |
| Advanced (+ Cross‑Encoder) | 84.4% | **0.651** |

| Serving Backend | Latency | Throughput | Reliability |
| :--- | :--- | :--- | :--- |
| **vLLM (Llama-3.1-8B-AWQ)** | 1.74s | **57.4 tok/s** | 100% |
| Google Gemini 3.6 Flash | 1.80s | 35.1 tok/s | 66% (503 errors) |

> **Key takeaway:** A quantised, self‑hosted LLM on a free GPU can outperform cloud APIs in both speed and reliability.

---
## 📂 Project Structure

```
├── config/              # Settings & environment variables
├── core/                # Extractor, summarizer, mock retriever
├── orchestrator/        # Agentic router + tool definitions
├── rag/                 # Naive, Hybrid, Advanced RAG implementations
├── scripts/             # Ingestion, evaluation, benchmarking
├── notebooks/           # vLLM Colab notebook
├── k8s/                 # Kubernetes manifests (Deployment, Service, PVC)
├── data/                # ChromaDB persistent storage
├── main.py              # FastAPI entry point
├── Dockerfile           # Container definition
├── docker-compose.yml   # Local orchestration
├── requirements.txt
├── .env.example
└── ROADMAP.md / JOURNAL.md
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/eu-ai-legal-assistant.git
cd eu-ai-legal-assistant
```

### 2. Set up environment

```bash
# Copy the example .env file and fill in your API keys
cp .env.example .env
# Edit .env with your Groq/Gemini keys (or use vLLM endpoint)
```

### 3. Install dependencies

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Run the ingestion pipeline (one‑time)

Place the EU AI Act PDF in `data/raw/eu_ai_act.pdf`, then:

```bash
python scripts/ingest_data.py
```

This will chunk the document, generate embeddings, and store them in ChromaDB.

### 5. Start the API server

```bash
# Using the orchestrator directly (CLI)
python main.py   # or uvicorn main:app --reload
```

Then send a query:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"What does Article 5 say about prohibited practices?"}'
```

### 6. Run with Docker (local)

```bash
docker-compose up --build
```

---

## 📖 Documentation

- **[ROADMAP.md](ROADMAP.md)** – step‑by‑step development plan (Phases 0–6).
- **[JOURNAL.md](JOURNAL.md)** – engineering diary: decisions, errors, and breakthroughs.
- **API Reference** (auto‑generated at `/docs` when running FastAPI).

---

## 🧪 Evaluation

Run the full RAG evaluation suite:

```bash
python scripts/evaluate_rag.py
```

This compares Naive, Hybrid, and Advanced RAG on a curated set of 20+ legal queries.

---

## 🔧 Customization

- **Change the LLM**: modify `AI_MODEL` in `.env` (supports OpenAI, Groq, Gemini, or any LiteLLM‑compatible endpoint).
- **Switch RAG strategy**: in `orchestrator/tools.py`, adjust the default used by the agent.
- **Tune chunking**: edit `CHILD_CHUNK_SIZE` and `CHILD_CHUNK_OVERLAP` in `scripts/ingest_data.py`.

---

## 🤝 Contributing

Issues and pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [LiteLLM](https://github.com/BerriAI/litellm) – unified API gateway
- [vLLM](https://vllm.ai/) – high‑throughput LLM serving
- [ChromaDB](https://chromadb.com/) – local vector database
- [Sentence‑Transformers](https://www.sbert.net/) – embedding models
- [PyMuPDF4LLM](https://github.com/pymupdf/pymupdf) – PDF extraction

---

## 📬 Contact

Built by [Bikal Poudel](https://linkedin.com/in/bikalpoudel) – feel free to connect!

**Star this repo** ⭐ if you find it useful – it helps others discover the project.

---

*Happy building!* 🚀
