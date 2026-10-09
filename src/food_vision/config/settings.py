"""Application settings, loaded from environment variables / ``.env``.

Settings are **not** instantiated at import time. Importing this module (or
anything that transitively imports it, e.g. in tests or tooling) must never
fail just because ``OPENROUTER_API_KEY`` isn't set in the current shell.
Call :func:`get_settings` when a value is actually needed; it's the single
place credential/allowlist validation happens, and it's cached so the
validation cost is paid once per process.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Environment-backed configuration for the food-vision service.

    Only the settings actually consumed today (the OpenRouter proxy +
    logging) are defined here. See docs/dev/food-vision-concept.md §9 for
    the full set of OpenRouter request/routing requirements these map to.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- OpenRouter credentials & endpoint -------------------------------
    OPENROUTER_API_KEY: str = Field(default="")
    OPENROUTER_API_URL: str = "https://openrouter.ai/api/v1"

    # --- Model allowlist (concept §9: "FOOD_VISION_MODELS allowlist;
    # default + fallback configured") -------------------------------------
    FOOD_VISION_MODELS: tuple[str, ...] = Field(default_factory=tuple)
    OPENROUTER_DEFAULT_MODEL: str = ""
    OPENROUTER_FALLBACK_MODELS: tuple[str, ...] = Field(default_factory=tuple)

    # --- Provider routing (concept §9) ------------------------------------
    OPENROUTER_PROVIDER_ORDER: tuple[str, ...] = Field(default_factory=tuple)
    OPENROUTER_ALLOW_FALLBACKS: bool = True
    OPENROUTER_REQUIRE_PARAMETERS: bool = True
    OPENROUTER_DATA_COLLECTION: str = "deny"
    OPENROUTER_ZDR: bool = True

    # --- Determinism ("temperature: 0; fixed seed where supported") ------
    OPENROUTER_TEMPERATURE: float = 0.0
    OPENROUTER_SEED: int | None = None

    # --- Timeouts / retries ("connect 5s / total 30s; retry only
    # 429/5xx with jitter, max 2") -----------------------------------------
    OPENROUTER_CONNECT_TIMEOUT_S: float = 5.0
    OPENROUTER_TOTAL_TIMEOUT_S: float = 30.0
    OPENROUTER_MAX_RETRIES: int = 2

    # --- Logging -----------------------------------------------------------
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = False

    @field_validator("OPENROUTER_DATA_COLLECTION")
    @classmethod
    def _validate_data_collection(cls, v: str) -> str:
        if v not in {"allow", "deny"}:
            raise ValueError("OPENROUTER_DATA_COLLECTION must be 'allow' or 'deny'.")
        return v

    @model_validator(mode="after")
    def _validate_model_allowlist(self) -> Settings:
        """Default/fallback models must be on the allowlist, if one is set.

        An empty allowlist means "not configured yet" and is accepted so a
        minimal dev setup (just an API key) still works; once
        ``FOOD_VISION_MODELS`` is set it becomes the single source of truth
        per concept §9.
        """
        if not self.FOOD_VISION_MODELS:
            return self
        default_model = self.OPENROUTER_DEFAULT_MODEL
        if default_model and default_model not in self.FOOD_VISION_MODELS:
            raise ValueError(
                f"OPENROUTER_DEFAULT_MODEL {default_model!r} is not in FOOD_VISION_MODELS."
            )
        unknown_fallbacks = set(self.OPENROUTER_FALLBACK_MODELS) - set(self.FOOD_VISION_MODELS)
        if unknown_fallbacks:
            raise ValueError(
                f"OPENROUTER_FALLBACK_MODELS not in FOOD_VISION_MODELS: {sorted(unknown_fallbacks)}"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` instance (cached).

    Import-safe: constructing ``Settings`` only happens on first call, not
    on module import, so code that imports this module without needing
    OpenRouter credentials (tests, tooling, ``--help``) keeps working.
    """
    return Settings()
