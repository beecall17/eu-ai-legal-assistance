# rag/hybrid.py

import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from rag.vector_store import VectorStore
from config.settings import EMBEDDING_MODEL, CHROMA_DB_PATH

class HybridRAG:
    def __init__(self):
        # 1. Initialize dense vector store
        self.vector_store = VectorStore(
            collection_name="eu_ai_act",
            embedding_model=EMBEDDING_MODEL,
            persist_directory=CHROMA_DB_PATH
        )
        
        # 2. Fetch all documents for BM25 corpus
        all_docs = self.vector_store.get_all_documents()
        if not all_docs:
            raise RuntimeError("No documents found in vector store. Run ingestion first.")
        
        self.corpus = [doc['text'] for doc in all_docs]
        self.metadatas = [doc['metadata'] for doc in all_docs]
        
        # 3. Build BM25 index (tokenize by splitting on whitespace and punctuation)
        tokenized_corpus = [self._tokenize(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        print(f"✅ HybridRAG initialized with {len(self.corpus)} documents.")
        print(f"📊 Dense + Sparse (BM25) fusion ready.")

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer: lowercases and splits on non-alphanumeric characters."""
        # Keep alphanumeric and apostrophes (for legal terms like "don't" or "AI's")
        text = text.lower()
        tokens = re.findall(r"[a-z0-9']+", text)
        return tokens

    def retrieve_context(
        self, 
        query: str, 
        top_k: int = 5,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5
    ) -> str:
        """
        Retrieve the top-k chunks using Hybrid search (Dense + BM25) with RRF fusion.
        
        Args:
            query: The user's question.
            top_k: Number of final chunks to return.
            dense_weight: Weight for dense scores (default 0.5).
            sparse_weight: Weight for sparse scores (default 0.5).
        
        Returns:
            Combined context string with the top-k fused results.
        """
        # 1. Dense retrieval (get 2x top_k for reranking)
        dense_results = self.vector_store.retrieve(query, top_k=top_k * 2)
        
        # 2. Sparse retrieval (BM25)
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top 2x top_k BM25 results
        bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k * 2]
        sparse_results = [
            {
                'text': self.corpus[i],
                'metadata': self.metadatas[i],
                'bm25_score': bm25_scores[i]
            }
            for i in bm25_indices
        ]
        
        # 3. Reciprocal Rank Fusion (RRF)
        # Build a map of text -> combined score
        rrf_scores = {}
        k = 60  # Standard RRF constant
        
        # Process dense results
        for rank, result in enumerate(dense_results, start=1):
            text = result['text']
            rrf_scores[text] = rrf_scores.get(text, 0) + dense_weight * (1 / (k + rank))
        
        # Process sparse results
        for rank, result in enumerate(sparse_results, start=1):
            text = result['text']
            rrf_scores[text] = rrf_scores.get(text, 0) + sparse_weight * (1 / (k + rank))
        
        # Sort by RRF score and take top_k
        fused_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Build context string
        context_chunks = [text for text, _ in fused_results]
        if not context_chunks:
            return "No relevant documents found."
        
        return "\n\n---\n\n".join(context_chunks)