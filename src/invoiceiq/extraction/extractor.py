"""Groq-backed invoice extractor.

The LLM wiring lives here so document processing, validation, and the API do
not depend on LangChain or on any concrete model name.
"""

from typing import Any

from invoiceiq.config import get_settings
from invoiceiq.schemas import Invoice


def build_llm() -> Any:
    """Create a ``ChatGroq`` client from environment configuration."""
    settings = get_settings()
    api_key, model = settings.require_groq()

    from langchain_groq import ChatGroq

    return ChatGroq(
        api_key=api_key,
        model=model,
        temperature=settings.groq_temperature,
        max_tokens=settings.groq_max_tokens,
    )


class InvoiceExtractor:
    """Turn raw document text into a structured invoice."""

    def __init__(self, llm: Any | None = None) -> None:
        self._llm = llm

    def extract(self, text: str) -> Invoice:
        """Extract the invoice fields from ``text``.

        Implementation is intentionally deferred to the extraction milestone.
        """
        raise NotImplementedError("Invoice extraction will be implemented in the next step.")
