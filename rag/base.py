# rag/base.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseRAG(ABC):
    """Abstract base class for all RAG retrievers."""
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve raw results (text + metadata + scores) for evaluation.
        
        Returns:
            List of dicts with keys: 'text', 'metadata', 'id', 'distance', 'similarity'
        """
        pass
    
    @abstractmethod
    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        """
        Retrieve and concatenate top-k chunks into a single context string.
        
        This is what the orchestrator uses.
        """
        pass