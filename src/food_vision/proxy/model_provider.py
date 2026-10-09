"""Default model-provider wiring for food-vision.

Food-vision has exactly one provider today: OpenRouter. This module is a
one-line convenience factory so call sites don't need to import
:class:`OpenRouterClient` and :func:`get_settings` separately. If/when a
second provider is needed, add it behind the :class:`ModelProvider`
Protocol in ``openrouter.py`` rather than growing a cache/registry here.
"""

from __future__ import annotations

from food_vision.proxy.openrouter import ModelProvider, OpenRouterClient


def get_default_provider() -> ModelProvider:
    """Return the configured completion provider (OpenRouter today)."""
    return OpenRouterClient()


__all__ = ["ModelProvider", "get_default_provider"]
