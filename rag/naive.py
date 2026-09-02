# rag/naive.py

from typing import List, Dict, Any
from rag.base import BaseRAG
from rag.vector_store import VectorStore
from config.settings import EMBEDDING_MODEL, CHROMA_DB_PATH

class NaiveRAG(BaseRAG):
    """Dense-only retrieval using cosine similarity."""
    
    def __init__(self):
        self.vector_store = VectorStore(
            collection_name="eu_ai_act",
            embedding_model=EMBEDDING_MODEL,
            persist_directory=CHROMA_DB_PATH,
        )
        print(f"✅ NaiveRAG initialized. Chunks: {self.vector_store.count():,}")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Raw retrieval from vector store."""
        return self.vector_store.retrieve(query, top_k=top_k)
    
    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return "No relevant documents found."
        # Combine chunks with a separator
        context = "\n\n---\n\n".join([r['text'] for r in results])
        return context