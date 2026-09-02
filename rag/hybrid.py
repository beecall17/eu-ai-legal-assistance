# rag/hybrid.py

import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from rag.base import BaseRAG
from rag.vector_store import VectorStore
from config.settings import EMBEDDING_MODEL, CHROMA_DB_PATH

class HybridRAG(BaseRAG):
    """Hybrid retrieval: Dense (cosine) + Sparse (BM25) fused with RRF."""
    
    def __init__(self):
        self.vector_store = VectorStore(
            collection_name="eu_ai_act",
            embedding_model=EMBEDDING_MODEL,
            persist_directory=CHROMA_DB_PATH,
        )
        
        # 1. Load all documents for BM25
        all_docs = self.vector_store.get_all_documents()
        if not all_docs:
            raise RuntimeError("No documents found in vector store. Run ingestion first.")
        
        self.corpus = [doc['text'] for doc in all_docs]
        self.metadatas = [doc['metadata'] for doc in all_docs]
        
        # 2. Tokenize and build BM25
        tokenized_corpus = [self._tokenize(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        print(f"✅ HybridRAG initialized. Dense chunks: {self.vector_store.count():,}, BM25 corpus: {len(self.corpus):,}")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer: lower-case, keep alphanumeric and apostrophe."""
        text = text.lower()
        return re.findall(r"[a-z0-9']+", text)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top_k documents using RRF fusion of dense and sparse results.
        """
        # Get 2x top_k from each to have enough candidates for fusion
        dense_results = self.vector_store.retrieve(query, top_k=top_k * 2)
        
        # Sparse (BM25)
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # Get top 2x indices
        sparse_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:top_k * 2]
        
        # Build list of sparse results with score
        sparse_results = [
            {
                'text': self.corpus[i],
                'metadata': self.metadatas[i],
                'bm25_score': bm25_scores[i],
            }
            for i in sparse_indices
        ]
        
        # RRF Fusion
        k = 60  # standard constant
        rrf_scores = {}
        
        # Dense results
        for rank, res in enumerate(dense_results, start=1):
            text = res['text']
            rrf_scores[text] = rrf_scores.get(text, 0) + 0.5 * (1 / (k + rank))
        
        # Sparse results
        for rank, res in enumerate(sparse_results, start=1):
            text = res['text']
            rrf_scores[text] = rrf_scores.get(text, 0) + 0.5 * (1 / (k + rank))
        
        # Sort by RRF score and take top_k
        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Build final results with metadata (fetch from either dense or sparse)
        final_results = []
        for text, score in fused:
            # Try to find metadata from dense or sparse
            meta = None
            for res in dense_results + sparse_results:
                if res['text'] == text:
                    meta = res.get('metadata')
                    break
            final_results.append({
                'text': text,
                'metadata': meta or {},
                'rrf_score': score,
            })
        return final_results
    
    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return "No relevant documents found."
        return "\n\n---\n\n".join([r['text'] for r in results])