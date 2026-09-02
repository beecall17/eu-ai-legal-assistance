# scripts/evaluate_rag.py

import sys
sys.path.append('.')

from rag.naive import NaiveRAG
from rag.hybrid import HybridRAG      # We'll build this next
from rag.advanced import AdvancedRAG  # We'll build this next

# Predefined test queries with their expected article/chunk
TEST_QUERIES = [
    {
        "query": "What are the prohibited practices in Article 5?",
        "expected_keywords": ["subliminal", "vulnerabilities", "social score", "biometric"]
    },
    {
        "query": "What is the risk level of Article 5?",
        "expected_keywords": ["unacceptable", "prohibited"]
    },
    {
        "query": "What are the prohibited practices in Article 5?",
        "expected_keywords": ["subliminal", "vulnerabilities", "social score", "biometric"]
    },
    {
        "query": "What is the risk level of Article 5?",
        "expected_keywords": ["unacceptable", "prohibited"]
    },
    {
        "query": "What does Article 9 require for risk management?",
        "expected_keywords": ["continuous process", "identify", "assess", "mitigate"]
    },
    {
        "query": "What are the data governance obligations in Article 10?",
        "expected_keywords": ["datasets", "representative", "bias-free", "documentation"]
    },
    {
        "query": "What is the purpose of the Quality Management System in Article 11?",
        "expected_keywords": ["compliance", "monitoring", "audits", "procedures"]
    },
    {
        "query": "What transparency obligations are outlined in Article 13?",
        "expected_keywords": ["instructions", "capabilities", "limitations", "inform users"]
    },
    {
        "query": "How does Article 14 ensure human oversight?",
        "expected_keywords": ["supervision", "intervention", "override", "training"]
    },
    {
        "query": "What technical requirements are set in Article 15?",
        "expected_keywords": ["accuracy", "robustness", "cybersecurity", "testing"]
    },
    {
        "query": "What conformity assessment procedures are described in Article 43?",
        "expected_keywords": ["assessment", "compliance", "documentation", "notified bodies"]
    },
    {
        "query": "What penalties are established in Article 71?",
        "expected_keywords": ["fines", "non-compliance", "sanctions", "administrative"]
    },
    {
        "query": "What does Annex I list as high-risk AI systems?",
        "expected_keywords": ["biometric identification", "critical infrastructure", "education", "employment"]
    },
    {
        "query": "What standards are referenced in Annex II?",
        "expected_keywords": ["harmonised standards", "technical specifications", "compliance"]
    }
]


def evaluate_rag(retriever_class, queries):
    retriever = retriever_class()
    hit_count = 0
    mrr_sum = 0
    
    for q in queries:
        results = retriever.vector_store.retrieve(q["query"], top_k=5)
        # Check if any result contains the expected keywords
        for i, result in enumerate(results):
            text = result['text'].lower()
            if any(kw in text for kw in q["expected_keywords"]):
                hit_count += 1
                mrr_sum += 1.0 / (i + 1)
                break
    
    hit_rate = hit_count / len(queries)
    mrr = mrr_sum / len(queries)
    return {"hit_rate": hit_rate, "mrr": mrr}

if __name__ == "__main__":
    results = {}
    for name, cls in [("Naive", NaiveRAG)]:  # Add other strategies later
        results[name] = evaluate_rag(cls, TEST_QUERIES)
        print(f"✅ {name} - Hit Rate: {results[name]['hit_rate']:.2f}, MRR: {results[name]['mrr']:.2f}")