"""LLM extraction layer tests with a fake structured-output client."""

from datetime import date
from typing import Any

import groq
import httpx
import pytest
from langchain_core.messages import AIMessage
from pydantic import SecretStr

from invoiceiq.config import Settings
from invoiceiq.extraction import (
    InvoiceExtractor,
    LLMAPIError,
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
    build_llm,
    extract_invoice,
)
from invoiceiq.extraction.prompts import INVOICE_EXTRACTION_PROMPT
from invoiceiq.schemas import Invoice, LineItem

REQUIRED_KEYS = {
    "is_invoice",
    "vendor",
    "invoice_number",
    "invoice_date",
    "currency",
    "line_items",
    "tax_total",
    "total",
}


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
        is_invoice=True,
        vendor="Acme Corporation",
        invoice_number="INV-2026-001",
        invoice_date=date(2026, 1, 15),
        currency="USD",
        line_items=[
            LineItem(
                description="Consulting",
                quantity=2,
                amount=1000.0,
            )
        ],
        tax_total=180.0,
        total=1180.0,
    )


def _non_invoice() -> Invoice:
    return Invoice(
        is_invoice=False,
        vendor=None,
        invoice_number=None,
        invoice_date=None,
        currency=None,
        line_items=None,
        tax_total=None,
        total=None,
    )


def _fake_extractor(parsed: Invoice, raw: AIMessage | None = None) -> tuple[
    InvoiceExtractor, FakeStructuredLLM
]:
    llm = FakeStructuredLLM(
        {
            "raw": raw or AIMessage(content=""),
            "parsed": parsed,
            "parsing_error": None,
        }
    )
    return InvoiceExtractor(llm=llm, settings=_settings()), llm


def _rate_limit_error() -> groq.RateLimitError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(429, request=request)
    return groq.RateLimitError("rate limit exceeded", response=response, body=None)


def _timeout_error() -> groq.APITimeoutError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return groq.APITimeoutError(request=request)


def _authentication_error() -> groq.AuthenticationError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(401, request=request)
    return groq.AuthenticationError("invalid api key", response=response, body=None)


def _connection_error() -> groq.APIConnectionError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return groq.APIConnectionError(request=request)


def _api_status_error() -> groq.APIStatusError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(500, request=request)
    return groq.APIStatusError("server error", response=response, body=None)


def test_build_llm_requires_configuration() -> None:
    with pytest.raises(LLMConfigurationError):
        build_llm(Settings(_env_file=None))


def test_build_llm_uses_environment_settings() -> None:
    llm = build_llm(_settings())

    assert llm.model_name == "test-model"
    assert llm.temperature <= 0.01
    assert llm.max_tokens == 8192
    assert llm.max_retries == 2


def test_prompt_contains_exact_extraction_rules() -> None:
    messages = INVOICE_EXTRACTION_PROMPT.format_messages(document_text="Invoice total 10")
    prompt_text = "\n".join(str(message.content) for message in messages)

    assert "Never invent" in prompt_text
    assert "is_invoice" in prompt_text
    assert "line_items" in prompt_text
    assert "tax_total" in prompt_text
    assert "total" in prompt_text
    assert "vendor" in prompt_text
    assert "YYYY-MM-DD" in prompt_text
    assert "bank statements" in prompt_text
    assert "purchase orders" in prompt_text
    assert "cover letters" in prompt_text
    assert "subtotal" not in prompt_text
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
    extractor, llm = _fake_extractor(_invoice(), raw)

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


def test_extract_invoice_function_returns_only_required_fields() -> None:
    extractor, _ = _fake_extractor(_invoice())

    invoice = extract_invoice("Invoice text", extractor=extractor)
    payload = invoice.model_dump(mode="json")

    assert invoice == _invoice()
    assert set(payload) == REQUIRED_KEYS
    assert set(payload["line_items"][0]) == {"description", "quantity", "amount"}


def test_non_invoice_has_null_fields_and_no_line_items() -> None:
    extractor, _ = _fake_extractor(_non_invoice())

    result = extractor.extract("Bank statement for account 12345.")

    assert result.invoice.is_invoice is False
    assert result.invoice.model_dump(mode="json") == {
        "is_invoice": False,
        "vendor": None,
        "invoice_number": None,
        "invoice_date": None,
        "currency": None,
        "line_items": None,
        "tax_total": None,
        "total": None,
    }


def test_json_string_response_is_parsed_and_validated() -> None:
    llm = FakeStructuredLLM(
        {
            "raw": AIMessage(content=""),
            "parsed": _invoice().model_dump_json(),
            "parsing_error": None,
        }
    )
    extractor = InvoiceExtractor(llm=llm, settings=_settings())

    result = extractor.extract("Invoice text")

    assert result.invoice == _invoice()


def test_invalid_schema_fields_raise_response_error() -> None:
    invalid = _invoice().model_dump(mode="json")
    invalid["tax_total"] = "180.00"
    llm = FakeStructuredLLM(
        {"raw": AIMessage(content=""), "parsed": invalid, "parsing_error": None}
    )
    extractor = InvoiceExtractor(llm=llm, settings=_settings())

    with pytest.raises(LLMResponseError):
        extractor.extract("Invoice text")


def test_non_invoice_with_populated_fields_raises_response_error() -> None:
    invalid = _non_invoice().model_dump(mode="json")
    invalid["vendor"] = "Acme"
    llm = FakeStructuredLLM(
        {"raw": AIMessage(content=""), "parsed": invalid, "parsing_error": None}
    )
    extractor = InvoiceExtractor(llm=llm, settings=_settings())

    with pytest.raises(LLMResponseError):
        extractor.extract("Bank statement text")


def test_malformed_json_string_raises_response_error() -> None:
    llm = FakeStructuredLLM(
        {"raw": AIMessage(content=""), "parsed": "{not json", "parsing_error": None}
    )
    extractor = InvoiceExtractor(llm=llm, settings=_settings())

    with pytest.raises(LLMResponseError):
        extractor.extract("Invoice text")


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
    extractor, _ = _fake_extractor(_invoice(), raw)

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
    extractor, _ = _fake_extractor(_invoice())

    with pytest.raises(ValueError):
        extractor.extract("   ")


def test_authentication_error_is_mapped_to_domain_error() -> None:
    llm = FakeStructuredLLM(error=_authentication_error())
    extractor = InvoiceExtractor(llm=llm, settings=_settings())

    with pytest.raises(LLMAuthenticationError):
        extractor.extract("Invoice text")


def test_connection_error_is_mapped_to_domain_error() -> None:
    llm = FakeStructuredLLM(error=_connection_error())
    extractor = InvoiceExtractor(llm=llm, settings=_settings())

    with pytest.raises(LLMConnectionError):
        extractor.extract("Invoice text")


def test_api_status_error_is_mapped_to_domain_error() -> None:
    llm = FakeStructuredLLM(error=_api_status_error())
    extractor = InvoiceExtractor(llm=llm, settings=_settings())

    with pytest.raises(LLMAPIError):
        extractor.extract("Invoice text")


def test_non_object_response_raises_response_error() -> None:
    llm = FakeStructuredLLM(
        {"raw": AIMessage(content=""), "parsed": 42, "parsing_error": None}
    )
    extractor = InvoiceExtractor(llm=llm, settings=_settings())

    with pytest.raises(LLMResponseError):
        extractor.extract("Invoice text")
