"""
ai_client/azure.py

Provides a centralized connection to an Azure AI chat model.
"""

import logging
from openai import AzureOpenAI


logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────
# CLIENT
# ───────────────────────────────────────────────────────────

def build_client(endpoint: str, api_key: str) -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version="2024-05-01-preview",
    )

def generate(
    client: AzureOpenAI,
    transcript: str,
    temperature: float = 0.2,
    max_tokens: int = 5000
) -> str:
    response = client.chat.completions.create(
        model=os.getenv("AZURE_MODEL_ID"),
        messages=[
            {"role": "system", "content": CLEANUP_PROMPT},
            {"role": "user", "content": transcript},
        ],
        temperature=0.2,
        max_tokens=5000,
    )

    return response.choices[0].message.content