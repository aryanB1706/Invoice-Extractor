"""Shared pytest fixtures."""

import pytest

from invoiceiq.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)
