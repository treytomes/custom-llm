"""
ai_client/azure.py

Provides a centralized connection to an Azure AI chat model.
"""

import logging
from openai import AzureOpenAI


ENDPOINT = os.environ.get("AZURE_AI_ENDPOINT")
API_KEY = os.environ.get("AZURE_AI_KEY")

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────
# CLIENT
# ───────────────────────────────────────────────────────────

def build_client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=ENDPOINT,
        api_key=API_KEY,
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