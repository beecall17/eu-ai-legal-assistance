# core/mock_retriever.py
def mock_retriever(query: str) -> str:
    """
    Simulates a RAG retrieval by returning a hardcoded EU AI Act snippet.
    Used for Phase 0/1 testing before the real RAG pipeline is built.
    
    Args:
        query: The user's query (ignored in mock).
    
    Returns:
        A fixed text snippet from the EU AI Act.
    """
    # Article 5 from the EU AI Act (Prohibited Practices)
    return """
    Article 5
    Prohibited Artificial Intelligence Practices
    
    The following artificial intelligence practices shall be prohibited:
    
    (a) the placing on the market, putting into service or use of an AI system that deploys subliminal techniques beyond a person's consciousness to materially distort a person's behaviour in a manner that causes or is likely to cause that person or another person physical or psychological harm;
    
    (b) the placing on the market, putting into service or use of an AI system that exploits any of the vulnerabilities of a specific group of persons due to their age, disability or a specific social or economic situation, with the objective, or the effect, of materially distorting the behaviour of that person in a manner that causes or is likely to cause that person or another person physical or psychological harm;
    
    (c) the placing on the market, putting into service or use of AI systems for the evaluation or classification of natural persons over a period of time based on their social behaviour or known, inferred or predicted personal or personality characteristics, with the social score leading to detrimental treatment of natural persons or groups of persons;
    
    (d) the use of 'real-time' remote biometric identification systems in publicly accessible spaces for the purpose of law enforcement, unless and in as far as such use is strictly necessary for the searching of a victim of abduction, trafficking in human beings or sexual exploitation, or for the prevention of a specific, substantial and imminent threat to the life or physical safety of natural persons.
    """