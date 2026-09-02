# orchestrator/agent.py
import json
from typing import Dict, Any, Optional, Union
from litellm import completion
from config.settings import AI_MODEL
from core.extractor import extract_article_structure
from core.summarizer import generate_summary
from orchestrator.tools import TOOLS, get_tool_by_name

class Orchestrator:
    def __init__(self, model: Optional[str] = None):
        self.model = model or AI_MODEL
        self.tools = TOOLS

    def _call_llm(self, user_query: str, context_text: str) -> Dict[str, Any]:
        """Call the LLM with the tools and return the full response."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful legal assistant that can either extract structured metadata from legal texts "
                    "or generate summaries. Based on the user's query, decide which tool to use. "
                    "If the user asks for specific details, risk level, or prohibited practices, use the 'extract_structured_metadata' tool. "
                    "If the user asks for an overview, explanation, or summary, use the 'generate_text_summary' tool. "
                    "If the query is ambiguous, prefer the extract tool as it provides more factual data."
                )
            },
            {
                "role": "user",
                "content": f"Context from legal text:\n{context_text}\n\nUser query: {user_query}"
            }
        ]

        try:
            response = completion(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",  # Let the model decide
                temperature=0.1,      # Low temperature for consistent routing
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
                # Convert to a human-readable string (or keep as JSON)
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

        else:
            return f"Unknown tool: {tool_name}"

    def process_query(self, user_query: str, context_text: str) -> str:
        """
        Main entry point: takes user query and context, routes to the right tool,
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
            # No tools called. If the response is too short or generic, we could ask the model to rephrase.
            direct_response = message.content.strip()
            if not direct_response:
                return "I'm not sure how to help with that query. Could you rephrase?"
            return direct_response

# Convenience function for easy use
def orchestrate(user_query: str, context_text: str) -> str:
    agent = Orchestrator()
    return agent.process_query(user_query, context_text)