# orchestrator/tools.py

from typing import List, Dict, Any

# Tool definitions for the orchestrator

extract_tool = {
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

summarize_tool = {
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

# List of all tools for easy import
TOOLS = [extract_tool, summarize_tool]

# Helper to get tool by name
def get_tool_by_name(name: str) -> Dict[str, Any]:
    for tool in TOOLS:
        if tool["function"]["name"] == name:
            return tool
    raise ValueError(f"Tool '{name}' not found")