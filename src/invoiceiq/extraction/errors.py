"""Errors raised by the LLM extraction layer."""


class LLMError(RuntimeError):
    """Base error for LLM integration failures."""


class LLMConfigurationError(LLMError):
    """Raised when the Groq API key or model is not configured."""


class LLMAuthenticationError(LLMError):
    """Raised when Groq rejects the API key."""


class LLMRateLimitError(LLMError):
    """Raised when the Groq rate limit is reached."""


class LLMTimeoutError(LLMError):
    """Raised when the Groq request times out."""


class LLMConnectionError(LLMError):
    """Raised when Groq cannot be reached."""


class LLMAPIError(LLMError):
    """Raised for other Groq API errors, including unsupported models."""


class LLMResponseError(LLMError):
    """Raised when the model response cannot be parsed into the invoice schema."""
