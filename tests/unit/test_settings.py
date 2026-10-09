"""Settings must stay import-safe and validate the model allowlist lazily."""

from __future__ import annotations

from typing import Any

import pytest

from food_vision.config.settings import Settings, get_settings


def _settings(**overrides: Any) -> Settings:
    """Build a Settings instance, isolated from any real .env on disk
    (the ``env_isolated`` autouse fixture chdirs into a clean tmp_path).
    """
    return Settings(**overrides)


@pytest.fixture(autouse=True)
def env_isolated(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run each test from an empty directory so the repo's real .env
    (with its own model/credential values) never leaks into assertions.
    """
    monkeypatch.chdir(tmp_path)
    for key in (
        "OPENROUTER_API_KEY",
        "OPENROUTER_DEFAULT_MODEL",
        "FOOD_VISION_MODELS",
        "OPENROUTER_FALLBACK_MODELS",
        "OPENROUTER_DATA_COLLECTION",
    ):
        monkeypatch.delenv(key, raising=False)


def test_import_does_not_require_env_vars() -> None:
    """Importing settings.py must never raise, even with no credentials."""
    import food_vision.config.settings  # noqa: F401


def test_settings_constructs_without_api_key() -> None:
    settings = _settings()
    assert settings.OPENROUTER_API_KEY == ""


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()


def test_default_model_must_be_in_allowlist() -> None:
    with pytest.raises(ValueError, match="OPENROUTER_DEFAULT_MODEL"):
        _settings(
            FOOD_VISION_MODELS=("model/a", "model/b"),
            OPENROUTER_DEFAULT_MODEL="model/not-allowed",
        )


def test_fallback_models_must_be_in_allowlist() -> None:
    with pytest.raises(ValueError, match="OPENROUTER_FALLBACK_MODELS"):
        _settings(
            FOOD_VISION_MODELS=("model/a",),
            OPENROUTER_DEFAULT_MODEL="model/a",
            OPENROUTER_FALLBACK_MODELS=("model/not-allowed",),
        )


def test_empty_allowlist_is_accepted() -> None:
    """An unset allowlist means 'not configured yet', not an error."""
    settings = _settings(OPENROUTER_DEFAULT_MODEL="anything/goes")
    assert settings.OPENROUTER_DEFAULT_MODEL == "anything/goes"


def test_data_collection_must_be_allow_or_deny() -> None:
    with pytest.raises(ValueError, match="OPENROUTER_DATA_COLLECTION"):
        _settings(OPENROUTER_DATA_COLLECTION="maybe")


def test_data_collection_defaults_to_deny() -> None:
    assert _settings().OPENROUTER_DATA_COLLECTION == "deny"
