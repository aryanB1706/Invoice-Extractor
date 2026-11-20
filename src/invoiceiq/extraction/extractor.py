"""Groq-backed invoice extractor.

This module owns all LLM access. It receives plain document text and does not
import or depend on the document-processing layer.
"""

import json
from typing import Any

import groq
from langchain_core.exceptions import OutputParserException
from langchain_groq import ChatGroq
from pydantic import ValidationError

from invoiceiq.config import Settings, get_settings
from invoiceiq.extraction.errors import (
    LLMAPIError,
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)
from invoiceiq.extraction.prompts import INVOICE_EXTRACTION_PROMPT
from invoiceiq.schemas import ExtractionResult, Invoice, TokenUsage


def build_llm(settings: Settings | None = None) -> ChatGroq:
    """Create a reusable ``ChatGroq`` client from environment configuration."""
    resolved = settings or get_settings()

    try:
        api_key, model = resolved.require_groq()
    except RuntimeError as exc:
        raise LLMConfigurationError(str(exc)) from exc

    return ChatGroq(
        api_key=api_key,
        model=model,
        temperature=resolved.groq_temperature,
        max_tokens=resolved.groq_max_tokens,
        timeout=resolved.groq_timeout_seconds,
        max_retries=resolved.groq_max_retries,
    )


class InvoiceExtractor:
    """Turn raw document text into a structured invoice."""

    def __init__(self, llm: Any | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._llm = llm if llm is not None else build_llm(self.settings)
        self._structured_llm = self._llm.with_structured_output(
            Invoice,
            method="function_calling",
            include_raw=True,
        )
        self.model_name = self.settings.groq_model or getattr(self._llm, "model_name", None)

    def extract(self, text: str) -> ExtractionResult:
        """Extract invoice fields and token usage from ``text``."""
        if not text or not text.strip():
            raise ValueError("Document text is empty; nothing to extract.")

        messages = INVOICE_EXTRACTION_PROMPT.format_messages(document_text=text)
        result = self._invoke(messages)
        invoice, raw_message = _parse_result(result)

        return ExtractionResult(
            invoice=invoice,
            model=self.model_name,
            usage=_extract_usage(raw_message),
        )

    def extract_invoice(self, text: str) -> Invoice:
        """Return only the assignment-shaped invoice data."""
        return self.extract(text).invoice

    def _invoke(self, messages: list[Any]) -> Any:
        try:
            return self._structured_llm.invoke(messages)
        except groq.RateLimitError as exc:
            raise LLMRateLimitError(
                "Groq rate limit reached. Wait and retry, or reduce request volume."
            ) from exc
        except groq.AuthenticationError as exc:
            raise LLMAuthenticationError("Groq rejected the API key. Check GROQ_API_KEY.") from exc
        except groq.APITimeoutError as exc:
            raise LLMTimeoutError("The Groq request timed out. Try again.") from exc
        except groq.APIConnectionError as exc:
            raise LLMConnectionError("Could not connect to the Groq API.") from exc
        except groq.APIStatusError as exc:
            raise LLMAPIError(f"Groq API error ({exc.status_code}): {exc.message}") from exc
        except groq.APIError as exc:
            raise LLMAPIError(f"Groq API error: {exc}") from exc
        except ValidationError as exc:
            raise LLMResponseError(_format_validation_error(exc)) from exc
        except OutputParserException as exc:
            raise LLMResponseError(f"Could not parse the model response: {exc}") from exc
        except TimeoutError as exc:
            raise LLMTimeoutError("The Groq request timed out. Try again.") from exc


def _parse_result(result: Any) -> tuple[Invoice, Any | None]:
    raw_message: Any | None = None
    parsing_error: Any | None = None
    parsed = result

    if isinstance(result, dict) and ("parsed" in result or "parsing_error" in result):
        parsed = result.get("parsed")
        raw_message = result.get("raw")
        parsing_error = result.get("parsing_error")

    if parsing_error is not None:
        cause = parsing_error if isinstance(parsing_error, BaseException) else None
        raise LLMResponseError(
            f"Could not parse the model response: {parsing_error}"
        ) from cause

    return _coerce_invoice(parsed), raw_message


def _coerce_invoice(parsed: Any) -> Invoice:
    if isinstance(parsed, Invoice):
        return parsed

    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except json.JSONDecodeError as exc:
            raise LLMResponseError(f"Model response is not valid JSON: {exc}") from exc

    if isinstance(parsed, dict):
        try:
            return Invoice.model_validate(parsed)
        except ValidationError as exc:
            raise LLMResponseError(_format_validation_error(exc)) from exc

    raise LLMResponseError("The model did not return a valid invoice response.")


def _format_validation_error(exc: ValidationError) -> str:
    details = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        details.append(f"{location}: {error['msg']}")
    return "Invalid invoice response: " + "; ".join(details)


def _extract_usage(message: Any | None) -> TokenUsage | None:
    if message is None:
        return None

    usage_metadata = getattr(message, "usage_metadata", None)
    if isinstance(usage_metadata, dict) and usage_metadata:
        return TokenUsage(
            prompt_tokens=usage_metadata.get("input_tokens"),
            completion_tokens=usage_metadata.get("output_tokens"),
            total_tokens=usage_metadata.get("total_tokens"),
        )

    response_metadata = getattr(message, "response_metadata", None) or {}
    token_usage = response_metadata.get("token_usage") or response_metadata.get("usage") or {}
    if token_usage:
        return TokenUsage(
            prompt_tokens=token_usage.get("prompt_tokens"),
            completion_tokens=token_usage.get("completion_tokens"),
            total_tokens=token_usage.get("total_tokens"),
        )

    return None


def extract_invoice(document_text: str, extractor: InvoiceExtractor | None = None) -> Invoice:
    """Convert document text into the assignment's invoice structure."""
    return (extractor or InvoiceExtractor()).extract_invoice(document_text)
