# orchestrator/tools.py

from typing import List, Dict, Any, Callable

from rag.naive import NaiveRAG
from rag.hybrid import HybridRAG
from rag.advanced import AdvancedRAG

# Initialize RAG retrieval engines
naive = NaiveRAG()
hybrid = HybridRAG()
advanced = AdvancedRAG()

# Tool schema definitions for the orchestrator
extract_tool: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "extract_structured_metadata",
        "description": "Extract structured information from the legal text, such as article number, title, prohibited practices, and risk level. Use this when the user asks for specific details, lists, risk assessments, or compliance requirements.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The legal text snippet to analyze."
                }
            },
            "required": ["text"]
        }
    }
}

summarize_tool: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "generate_text_summary",
        "description": "Generate a free-text summary of the legal text in a specified tone. Use this when the user asks for an overview, explanation, briefing, or wants to understand the content in plain language.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The legal text snippet to summarize."
                },
                "tone": {
                    "type": "string",
                    "enum": ["concise", "detailed", "executive", "layman"],
                    "description": "The desired tone of the summary. Default is 'concise'."
                }
            },
            "required": ["text"]
        }
    }
}

search_naive_tool: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_naive",
        "description": "Retrieve using semantic similarity only. Good for conceptual or high-level questions.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's question."
                }
            },
            "required": ["query"]
        }
    }
}

search_hybrid_tool: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_hybrid",
        "description": "Retrieve using a combination of semantic similarity and BM25 keyword search. Good for queries containing specific terminology, names, or IDs.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's question."
                }
            },
            "required": ["query"]
        }
    }
}

search_advanced_tool: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_advanced",
        "description": "Retrieve using hybrid search followed by a cross-encoder reranker. Best for complex queries requiring high precision and strict relevance.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's question."
                }
            },
            "required": ["query"]
        }
    }
}

# Complete list of all available tools for easy import and LLM binding
TOOLS: List[Dict[str, Any]] = [
    extract_tool,
    summarize_tool,
    search_naive_tool,
    search_hybrid_tool,
    search_advanced_tool,
]

# Mapping of tool names to their underlying executable functions
TOOL_FUNCTIONS: Dict[str, Callable[[Dict[str, Any]], Any]] = {
    "search_naive": lambda args: naive.retrieve_context(args["query"]),
    "search_hybrid": lambda args: hybrid.retrieve_context(args["query"]),
    "search_advanced": lambda args: advanced.retrieve_context(args["query"]),
}


def get_tool_by_name(name: str) -> Dict[str, Any]:
    """Retrieve a specific tool schema definition by its function name."""
    for tool in TOOLS:
        if tool["function"]["name"] == name:
            return tool
    raise ValueError(f"Tool '{name}' not found")