"""
ai_client/azure.py

Provides a centralized connection to an Azure AI chat model.
"""

import logging
import os
from openai import AzureOpenAI


logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────
# CLIENT
# ───────────────────────────────────────────────────────────

def build_client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=os.environ.get("AZURE_AI_ENDPOINT"),
        api_key=os.environ.get("AZURE_AI_KEY"),
        api_version="2024-05-01-preview",
    )

def generate(
    client: AzureOpenAI,
    messages,
    temperature: float = 0.2,
    max_tokens: int = 5000
) -> str:
    response = client.chat.completions.create(
        model=os.getenv("AZURE_MODEL_ID"),
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content