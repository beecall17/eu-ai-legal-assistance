# test_naive_rag.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.naive import NaiveRAG
from orchestrator.agent import Orchestrator

if __name__ == "__main__":
    # 1. Initialize RAG
    rag = NaiveRAG()
    
    # 2. Query
    query = "What is the risk level of Article 5?"
    context = rag.retrieve_context(query, top_k=3)
    
    print("=== RETRIEVED CONTEXT ===")
    print(context[:500] + "...\n")
    
    # 3. Use the orchestrator with real context
    agent = Orchestrator()
    response = agent.process_query(query, context)
    print("=== FINAL ANSWER ===")
    print(response)