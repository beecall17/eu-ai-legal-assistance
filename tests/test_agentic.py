# test_agentic.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orchestrator.agent import Orchestrator

queries = [
    "What does Article 5 say about prohibited practices?",
    "What is the definition of an AI system?",
    "Explain the transparency obligations in Article 13.",
]

agent = Orchestrator()
for q in queries:
    print(f"\nQuery: {q}")
    response = agent.process_query(q, context_text=None)
    print(f"Response: {response[:200]}...")