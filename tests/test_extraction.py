"""LLM extraction layer tests with a fake structured-output client."""

from datetime import date
from decimal import Decimal
from typing import Any

import groq
import httpx
import pytest
from langchain_core.messages import AIMessage
from pydantic import SecretStr

from invoiceiq.config import Settings
from invoiceiq.extraction import (
    InvoiceExtractor,
    LLMConfigurationError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
    build_llm,
)
from invoiceiq.extraction.prompts import INVOICE_EXTRACTION_PROMPT
from invoiceiq.schemas import Invoice, LineItem


class FakeStructuredLLM:
    """Minimal stand-in for ChatGroq.with_structured_output."""

    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.schema: Any = None
        self.include_raw: bool | None = None
        self.method: str | None = None
        self.received_messages: list[Any] | None = None

    def with_structured_output(
        self,
        schema: Any,
        *,
        method: str | None = None,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> "FakeStructuredLLM":
        self.schema = schema
        self.method = method
        self.include_raw = include_raw
        return self

    def invoke(self, messages: list[Any]) -> Any:
        self.received_messages = messages
        if self.error is not None:
            raise self.error
        return self.response


def _settings(model: str = "test-model") -> Settings:
    return Settings(
        _env_file=None,
        groq_api_key=SecretStr("test-key"),
        groq_model=model,
    )


def _invoice() -> Invoice:
    return Invoice(
        vendor_name="Acme Corporation",
        invoice_number="INV-2026-001",
        invoice_date=date(2026, 1, 15),
        currency="USD",
        subtotal=Decimal("1000.00"),
        tax=Decimal("180.00"),
        total=Decimal("1180.00"),
        line_items=[
            LineItem(
                description="Consulting",
                quantity=Decimal("2"),
                unit_price=Decimal("500.00"),
                amount=Decimal("1000.00"),
            )
        ],
    )


def _rate_limit_error() -> groq.RateLimitError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(429, request=request)
    return groq.RateLimitError("rate limit exceeded", response=response, body=None)


def _timeout_error() -> groq.APITimeoutError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return groq.APITimeoutError(request=request)


def test_build_llm_requires_configuration() -> None:
    with pytest.raises(LLMConfigurationError):
        build_llm(Settings(_env_file=None))


def test_build_llm_uses_environment_settings() -> None:
    llm = build_llm(_settings())

    assert llm.model_name == "test-model"
    assert llm.temperature <= 0.01
    assert llm.max_tokens == 2048
    assert llm.max_retries == 2


def test_prompt_contains_extraction_rules() -> None:
    messages = INVOICE_EXTRACTION_PROMPT.format_messages(document_text="Invoice total 10")
    prompt_text = "\n".join(str(message.content) for message in messages)

    assert "Never invent" in prompt_text
    assert "is_invoice" in prompt_text
    assert "YYYY-MM-DD" in prompt_text
    assert "subtotal" in prompt_text
    assert "tax" in prompt_text
    assert "total" in prompt_text
    assert "Invoice total 10" in prompt_text


def test_extract_returns_invoice_and_token_usage() -> None:
    raw = AIMessage(
        content="",
        usage_metadata={
            "input_tokens": 120,
            "output_tokens": 45,
            "total_tokens": 165,
        },
    )
    llm = FakeStructuredLLM(
        {"raw": raw, "parsed": _invoice(), "parsing_error": None}
    )
    extractor = InvoiceExtractor(llm=llm, settings=_settings())

    result = extractor.extract("Invoice INV-2026-001 from Acme Corporation")

    assert result.invoice == _invoice()
    assert result.model == "test-model"
    assert result.usage is not None
    assert result.usage.prompt_tokens == 120
    assert result.usage.completion_tokens == 45
    assert result.usage.total_tokens == 165
    assert llm.schema is Invoice
    assert llm.method == "function_calling"
    assert llm.include_raw is True
    assert llm.received_messages is not None
    assert "Acme Corporation" in str(llm.received_messages[1].content)


def test_extract_handles_non_invoice_response() -> None:
    invoice = Invoice(
        is_invoice=False,
        extraction_notes="The document is a service agreement, not an invoice.",
    )
    llm = FakeStructuredLLM({"raw": AIMessage(content=""), "parsed": invoice})
    extractor = InvoiceExtractor(llm=llm, settings=_settings())

    result = extractor.extract("Service agreement between Acme and Beta.")

    assert result.invoice.is_invoice is False
    assert "not an invoice" in (result.invoice.extraction_notes or "")
    assert result.invoice.vendor_name is None
    assert result.invoice.total is None


def test_extract_falls_back_to_response_metadata_for_usage() -> None:
    raw = AIMessage(
        content="",
        response_metadata={
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            }
        },
    )
    llm = FakeStructuredLLM({"raw": raw, "parsed": _invoice()})
    extractor = InvoiceExtractor(llm=llm, settings=_settings())

    result = extractor.extract("Invoice text")

    assert result.usage is not None
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 5
    assert result.usage.total_tokens == 15


def test_parsing_error_raises_response_error() -> None:
    llm = FakeStructuredLLM(
        {
            "raw": AIMessage(content="not json"),
            "parsed": None,
            "parsing_error": ValueError("invalid output"),
        }
    )
    extractor = InvoiceExtractor(llm=llm, settings=_settings())

    with pytest.raises(LLMResponseError):
        extractor.extract("Invoice text")


def test_rate_limit_is_mapped_to_domain_error() -> None:
    llm = FakeStructuredLLM(error=_rate_limit_error())
    extractor = InvoiceExtractor(llm=llm, settings=_settings())

    with pytest.raises(LLMRateLimitError):
        extractor.extract("Invoice text")


def test_timeout_is_mapped_to_domain_error() -> None:
    llm = FakeStructuredLLM(error=_timeout_error())
    extractor = InvoiceExtractor(llm=llm, settings=_settings())

    with pytest.raises(LLMTimeoutError):
        extractor.extract("Invoice text")


def test_empty_text_is_rejected() -> None:
    llm = FakeStructuredLLM({"raw": AIMessage(content=""), "parsed": _invoice()})
    extractor = InvoiceExtractor(llm=llm, settings=_settings())

    with pytest.raises(ValueError):
        extractor.extract("   ")
