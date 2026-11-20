"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime settings for InvoiceIQ.

    Every value can be overridden through environment variables or the
    project-level ``.env`` file. Model names and API keys are never hardcoded
    in the pipeline modules.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "InvoiceIQ"
    app_env: str = "development"
    log_level: str = "INFO"

    groq_api_key: SecretStr = SecretStr("")
    groq_model: str = ""
    groq_temperature: float = 0.0
    groq_max_tokens: int = 2048

    tesseract_cmd: str = ""
    ocr_language: str = "eng"
    ocr_min_text_chars: int = 50
    ocr_render_dpi: int = 200

    max_upload_mb: int = 10
    data_dir: Path = PROJECT_ROOT / "data"
    upload_dir: Path = PROJECT_ROOT / "data" / "uploads"
    output_dir: Path = PROJECT_ROOT / "data" / "outputs"

    @property
    def groq_configured(self) -> bool:
        """Return True when both the API key and model are configured."""
        return bool(self.groq_api_key.get_secret_value()) and bool(self.groq_model)

    def require_groq(self) -> tuple[str, str]:
        """Return ``(api_key, model)`` or fail with a helpful error."""
        if not self.groq_configured:
            raise RuntimeError(
                "Groq is not configured. Set GROQ_API_KEY and GROQ_MODEL in your .env file."
            )
        return self.groq_api_key.get_secret_value(), self.groq_model


@lru_cache
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance."""
    return Settings()
