# rag/naive.py

from rag.vector_store import VectorStore
from config.settings import EMBEDDING_MODEL, CHROMA_DB_PATH

class NaiveRAG:
    def __init__(self):
        self.vector_store = VectorStore(
            collection_name="eu_ai_act",
            embedding_model=EMBEDDING_MODEL,
            persist_directory=CHROMA_DB_PATH
        )
        print("✅ NaiveRAG initialized.")

    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        """Retrieve the top-k chunks and combine them into a single context string."""
        results = self.vector_store.retrieve(query, top_k=top_k)
        if not results:
            return "No relevant documents found."
        
        # Combine chunks with separators
        context = "\n\n---\n\n".join([r['text'] for r in results])
        return context