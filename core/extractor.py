# core/extractor.py
import os
import traceback
from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field
import instructor
from litellm import completion
from config.settings import AI_MODEL, FALLBACK_MODELS

class RiskLevelEnum(str, Enum):
    UNACCEPTABLE = "Unacceptable"
    HIGH = "High"
    LIMITED = "Limited"
    MINIMAL = "Minimal"
    UNSPECIFIED = "Unspecified"

# Define a strict Pydantic schema for extracting data from legal or policy texts
class EUAIActArticleSchema(BaseModel):
    article_number: str = Field(description="The article number, e.g., 'Article 5'")
    title: str = Field(description="The title of the article")
    prohibited_practices: List[str] = Field(
        description="List of prohibited practices or main core rules outlined in this text"
    )
    risk_level: Optional[RiskLevelEnum] = Field(
        default=RiskLevelEnum.UNSPECIFIED,
        description="Assessed risk level: Unacceptable, High, Limited, or Minimal"
    )

def extract_article_structure(text_snippet: str):
    client = instructor.from_litellm(completion)
    models_to_try = [AI_MODEL] + FALLBACK_MODELS

    for model in models_to_try:
        try:
            print(f"-> Trying model: {model}")
            response = client.chat.completions.create(
                model=model,
                response_model=EUAIActArticleSchema,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a legal AI compliance expert specialized in parsing regulatory frameworks like the EU AI Act.",
                    },
                    {
                        "role": "user",
                        "content": f"Extract the structured details from this text snippet:\n\n{text_snippet}",
                    },
                ],
                num_retries=2,
                timeout=30
            )
            return response
        except Exception as e:
            traceback.print_exc()
            print(f"Model {model} failed: {e}")
            continue
    raise RuntimeError("All models exhausted. No response.")