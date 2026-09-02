# test_hybrid_rag.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.hybrid import HybridRAG
from orchestrator.agent import Orchestrator

if __name__ == "__main__":
    # 1. Initialize Hybrid RAG
    hybrid = HybridRAG()
    
    # 2. Test queries
    queries = [
        "What is the risk level of Article 5?",
        "List the prohibited practices in Article 5.",
        "What does the EU AI Act say about social scoring?",
        "Are biometric identification systems prohibited?"
    ]
    
    agent = Orchestrator()
    
    for q in queries:
        print(f"\n{'='*60}")
        print(f"User Query: {q}")
        print("-" * 60)
        
        # Retrieve context
        context = hybrid.retrieve_context(q, top_k=3)
        print(f"📚 Retrieved {len(context.split('---'))} chunks.")
        print(f"=== RETRIEVED CONTEXT ===\n{context[:500]}...\n")
        
        # Orchestrate
        response = agent.process_query(q, context)
        print(f"🤖 Assistant:\n{response}\n")