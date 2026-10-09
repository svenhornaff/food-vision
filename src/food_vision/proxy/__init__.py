"""OpenRouter proxy: thin transport layer for chat/vision completions.

See ``openrouter.py`` for the client and docs/dev/food-vision-concept.md
§9 for the request/routing requirements it implements.
"""

from .model_provider import get_default_provider
from .openrouter import (
    CompletionResult,
    JsonSchemaFormat,
    ModelProvider,
    OpenRouterClient,
    OpenRouterError,
    RoutingMode,
    Usage,
)

__all__ = [
    "CompletionResult",
    "JsonSchemaFormat",
    "ModelProvider",
    "OpenRouterClient",
    "OpenRouterError",
    "RoutingMode",
    "Usage",
    "get_default_provider",
]
