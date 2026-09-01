# test_phase0.py
from core.extractor import extract_article_structure
from core.summarizer import generate_summary
from core.mock_retriever import mock_retriever

if __name__ == "__main__":
    # 1. Mock Retrieval
    text = mock_retriever("Article 5")
    print("=== MOCK RETRIEVAL ===")
    print(text[:200] + "...\n")

    # 2. Extraction
    print("=== EXTRACTION ===")
    structured = extract_article_structure(text)
    print(f"Article: {structured.article_number}")
    print(f"Title: {structured.title}")
    print(f"Risk Level: {structured.risk_level}")
    print(f"Practices: {structured.prohibited_practices}\n")

    # 3. Summarization
    print("=== SUMMARY ===")
    summary = generate_summary(text, tone="executive")
    print(summary)