"""get_default_provider() wires settings -> OpenRouterClient."""

from __future__ import annotations

from typing import Any

import pytest

from food_vision.proxy.model_provider import get_default_provider
from food_vision.proxy.openrouter import OpenRouterClient, OpenRouterError


@pytest.fixture(autouse=True)
def env_isolated(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate from the repo's real .env (it carries its own API key)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


def test_get_default_provider_returns_openrouter_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    from food_vision.config.settings import get_settings

    get_settings.cache_clear()
    try:
        provider = get_default_provider()
        assert isinstance(provider, OpenRouterClient)
    finally:
        get_settings.cache_clear()


def test_get_default_provider_raises_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from food_vision.config.settings import get_settings

    get_settings.cache_clear()
    try:
        with pytest.raises(OpenRouterError):
            get_default_provider()
    finally:
        get_settings.cache_clear()
