"""Configuration loading tests."""

from invoiceiq.config import Settings, get_settings


def test_defaults_load_without_env_file(settings: Settings) -> None:
    assert settings.app_name == "InvoiceIQ"
    assert settings.app_env == "development"
    assert settings.groq_configured is False


def test_ocr_render_dpi_defaults_to_300(monkeypatch) -> None:
    monkeypatch.delenv("OCR_RENDER_DPI", raising=False)

    assert Settings(_env_file=None).ocr_render_dpi == 300


def test_cached_settings_returns_same_instance() -> None:
    assert get_settings() is get_settings()
