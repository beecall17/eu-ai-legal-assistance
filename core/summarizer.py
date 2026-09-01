# core/summarizer.py
import os
from typing import Optional
import litellm
from litellm import completion
from config.settings import AI_MODEL, FALLBACK_MODELS

def generate_summary(
    text_snippet: str, 
    tone: str = "concise",
    max_length: Optional[int] = 300
) -> str:
    """
    Generate a free-text summary of a legal article or text snippet.
    
    Args:
        text_snippet: The legal text to summarize.
        tone: 'concise', 'detailed', 'executive', or 'layman'.
        max_length: Approximate word limit for the summary.
    
    Returns:
        A string containing the summary.
    """
    tone_instructions = {
        "concise": "Provide a brief, punchy summary in 3-4 bullet points.",
        "detailed": "Provide a thorough summary covering all key clauses and nuances.",
        "executive": "Provide a high-level summary suitable for a business executive, focusing on implications and risks.",
        "layman": "Explain this legal text in plain, simple English, as if talking to a non-expert."
    }
    
    prompt = f"""
    You are a legal AI assistant specialized in summarizing regulatory texts.
    
    TASK: Summarize the following legal text.
    
    TONE: {tone_instructions.get(tone, tone_instructions["concise"])}
    
    TEXT TO SUMMARIZE:
    {text_snippet}
    
    SUMMARY:
    """
    
    try:
        response = completion(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": "You are a precise legal summarizer."},
                {"role": "user", "content": prompt}
            ],
            fallback_models=FALLBACK_MODELS,
            num_retries=3,              # Automatically retries transient errors and rate limits
            max_tokens=500,
            temperature=0.3,
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        print(f"Error generating summary: {e}")
        return f"Error: Unable to generate summary. Reason: {str(e)}"