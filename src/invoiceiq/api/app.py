"""FastAPI application instance."""

from fastapi import FastAPI

from invoiceiq import __version__
from invoiceiq.api.routes import router
from invoiceiq.config import get_settings


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()
    application = FastAPI(title=settings.app_name, version=__version__)
    application.include_router(router, prefix="/api/v1")

    @application.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok", "app": settings.app_name, "version": __version__}

    return application


app = create_app()
