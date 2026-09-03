# rag/advanced.py

from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from rag.base import BaseRAG
from rag.hybrid import HybridRAG

class AdvancedRAG(BaseRAG):
    """
    Hybrid retrieval + Cross-Encoder reranking.
    """
    
    def __init__(
        self,
        hybrid_top_k: int = 20,
        rerank_top_k: int = 5,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        # 1. Use HybridRAG as first-stage retriever
        self.hybrid = HybridRAG()
        
        # 2. Load Cross-Encoder
        print(f"🧠 Loading Cross-Encoder: {model_name}")
        self.reranker = CrossEncoder(model_name)
        
        self.hybrid_top_k = hybrid_top_k
        self.rerank_top_k = rerank_top_k
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve using hybrid, then rerank with cross-encoder.
        """
        # Step 1: Get hybrid candidates (more than final top_k)
        candidates = self.hybrid.retrieve(query, top_k=self.hybrid_top_k)
        if not candidates:
            return []
        
        # Step 2: Prepare pairs for cross-encoder: (query, document text)
        pairs = [(query, cand['text']) for cand in candidates]
        scores = self.reranker.predict(pairs)
        
        # Step 3: Sort candidates by score (descending)
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Step 4: Take top_k and add score to metadata
        final = []
        for rank, (cand, score) in enumerate(scored[:top_k], start=1):
            cand['rerank_score'] = float(score)
            cand['rerank_rank'] = rank
            final.append(cand)
        
        return final
    
    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return "No relevant documents found."
        return "\n\n---\n\n".join([r['text'] for r in results])