# main.py
import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

from orchestrator.agent import Orchestrator
from rag.naive import NaiveRAG
from rag.hybrid import HybridRAG
from rag.advanced import AdvancedRAG
from config.settings import AI_MODEL

app = FastAPI(title="EU AI Act Legal Assistant", version="1.0")

# Initialize components (lazy loading recommended)
orchestrator = None
naive = None
hybrid = None
advanced = None

def get_orchestrator():
    global orchestrator
    if orchestrator is None:
        orchestrator = Orchestrator()
    return orchestrator

class ChatRequest(BaseModel):
    query: str
    context: Optional[str] = None
    use_rag: bool = True
    rag_strategy: Optional[str] = None  # "naive", "hybrid", "advanced"

class ChatResponse(BaseModel):
    answer: str
    tool_used: Optional[str] = None

@app.get("/health")
async def health():
    return {"status": "healthy", "model": AI_MODEL}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    If use_rag=True and no context is provided, it retrieves context using the chosen strategy.
    """
    context_text = request.context
    tool_used = None

    # If RAG is enabled and no context provided, fetch it
    if request.use_rag and not context_text:
        strategy = request.rag_strategy or "advanced"
        if strategy == "naive":
            rag = naive or NaiveRAG()
            context_text = rag.retrieve_context(request.query)
            tool_used = f"RAG (Naive)"
        elif strategy == "hybrid":
            rag = hybrid or HybridRAG()
            context_text = rag.retrieve_context(request.query)
            tool_used = f"RAG (Hybrid)"
        else:  # advanced (default)
            rag = advanced or AdvancedRAG()
            context_text = rag.retrieve_context(request.query)
            tool_used = f"RAG (Advanced)"

    # Run orchestrator
    agent = get_orchestrator()
    answer = agent.process_query(request.query, context_text)
    
    return ChatResponse(answer=answer, tool_used=tool_used)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)