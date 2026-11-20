"""Configuration loading tests."""

from invoiceiq.config import Settings, get_settings


def test_defaults_load_without_env_file(settings: Settings) -> None:
    assert settings.app_name == "InvoiceIQ"
    assert settings.app_env == "development"
    assert settings.groq_configured is False


def test_cached_settings_returns_same_instance() -> None:
    assert get_settings() is get_settings()
