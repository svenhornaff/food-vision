"""Default model-provider wiring for food-vision.

The previous version of this module was copy-paste residue from an
unrelated project: it referenced ``ChatOllamaLocal``, ``OpenAIEmbedding``,
``OllamaEmbedding`` and a ``model_defaults`` module that don't exist here,
and was a LangChain-based multi-provider cache registry — none of which the
concept doc calls for (§5: "no agent framework"; no Ollama, no embeddings
anywhere in the concept). It raised ``NameError`` on import.

Today food-vision has exactly one provider: OpenRouter. This module is a
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
