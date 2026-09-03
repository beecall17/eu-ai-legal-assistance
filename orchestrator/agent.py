# orchestrator/agent.py
import json
from typing import Dict, Any, Optional, Union
from litellm import completion
from config.settings import AI_MODEL
from core.extractor import extract_article_structure
from core.summarizer import generate_summary
from orchestrator.tools import TOOLS, TOOL_FUNCTIONS, get_tool_by_name

class Orchestrator:
    def __init__(self, model: Optional[str] = None):
        self.model = model or AI_MODEL
        self.tools = TOOLS
        self.tool_functions = TOOL_FUNCTIONS

    def _call_llm(self, user_query: str, context_text: Optional[str] = None) -> Dict[str, Any]:
        """Call the LLM with the tools and return the full response."""
        user_content = f"User query: {user_query}"
        if context_text:
            user_content = f"Context from legal text:\n{context_text}\n\n{user_content}"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an advanced legal assistant equipped with tools for extracting structured metadata, "
                    "generating summaries, and searching/retrieving relevant context using different RAG strategies "
                    "(naive, hybrid, advanced search). Based on the user's query, decide which tool(s) to use. "
                    "Use search tools when context needs to be retrieved, and extraction/summary tools when analyzing text."
                )
            },
            {
                "role": "user",
                "content": user_content
            }
        ]

        try:
            response = completion(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",  # Let the model decide
                temperature=0.1,     # Low temperature for consistent routing
                max_retries=3,
            )
            return response
        except Exception as e:
            raise RuntimeError(f"LLM call failed: {e}")

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute the appropriate tool and return the result as a string."""
        if tool_name == "extract_structured_metadata":
            text = arguments.get("text")
            if not text:
                return "Error: Missing 'text' argument for extraction."
            try:
                result = extract_article_structure(text)
                return f"Article: {result.article_number}\nTitle: {result.title}\nRisk Level: {result.risk_level}\nProhibited Practices:\n- " + "\n- ".join(result.prohibited_practices)
            except Exception as e:
                return f"Error during extraction: {e}"

        elif tool_name == "generate_text_summary":
            text = arguments.get("text")
            tone = arguments.get("tone", "concise")
            if not text:
                return "Error: Missing 'text' argument for summarization."
            try:
                summary = generate_summary(text, tone=tone)
                return summary
            except Exception as e:
                return f"Error during summarization: {e}"

        elif tool_name in self.tool_functions:
            try:
                result = self.tool_functions[tool_name](arguments)
                if isinstance(result, (list, dict)):
                    return json.dumps(result, indent=2)
                return str(result)
            except Exception as e:
                return f"Error during tool execution ({tool_name}): {e}"

        else:
            return f"Unknown tool: {tool_name}"

    def process_query(self, user_query: str, context_text: Optional[str] = None) -> str:
        """
        Main entry point: takes user query and optional context, routes to the right tool,
        and returns the final answer.
        """
        # 1. Get LLM response with tool calls
        response = self._call_llm(user_query, context_text)

        # 2. Extract the tool call if present
        message = response.choices[0].message
        tool_calls = message.get("tool_calls", [])

        if tool_calls:
            results = []
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                result = self._execute_tool(tool_name, arguments)
                results.append(f"[{tool_name}]:\n{result}")
            # Combine all results
            return "\n\n".join(results)
        else:
            direct_response = message.content.strip()
            if not direct_response:
                return "I'm not sure how to help with that query. Could you rephrase?"
            return direct_response

# Convenience function for easy use
def orchestrate(user_query: str, context_text: Optional[str] = None) -> str:
    agent = Orchestrator()
    return agent.process_query(user_query, context_text)