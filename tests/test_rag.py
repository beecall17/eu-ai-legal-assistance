# tests/test_rag.py

import sys
sys.path.append('.')

from rag.naive import NaiveRAG
from rag.hybrid import HybridRAG

def test_retrieval():
    """Simple sanity test for both retrievers."""
    queries = [
        "What does Article 5 say about prohibited practices?",
        "What is the definition of an AI system?",
    ]
    
    for rag_class, name in [(NaiveRAG, "Naive"), (HybridRAG, "Hybrid")]:
        print(f"\n{'='*60}")
        print(f"{name} RAG Test")
        print('='*60)
        rag = rag_class()
        for q in queries:
            print(f"\nQuestion: {q}")
            context = rag.retrieve_context(q, top_k=2)
            print(f"Context (first 300 chars):\n{context[:300]}...")
            # Optionally print metadata of first result
            results = rag.retrieve(q, top_k=2)
            if results:
                meta = results[0].get('metadata', {})
                print(f"First result section: {meta.get('section_id')} (page {meta.get('page_start')})")

if __name__ == "__main__":
    test_retrieval()