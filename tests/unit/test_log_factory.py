"""configure_logging()/get_logger() must be side-effect-free and idempotent."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from food_vision.utils.log_factory import configure_logging, get_logger


@pytest.fixture(autouse=True)
def clean_root_logger() -> Any:
    """Save/restore root logger state so tests don't leak handlers into
    each other or into pytest's own logging capture.
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    root.handlers.clear()
    yield
    root.handlers.clear()
    root.handlers.extend(saved_handlers)
    root.setLevel(saved_level)


def test_configure_logging_adds_exactly_one_handler() -> None:
    configure_logging()
    root = logging.getLogger()
    assert len(root.handlers) == 1


def test_configure_logging_is_idempotent() -> None:
    configure_logging()
    configure_logging()
    configure_logging()
    root = logging.getLogger()
    assert len(root.handlers) == 1


def test_configure_logging_respects_existing_handlers() -> None:
    """If something else already configured the root logger, don't touch it."""
    root = logging.getLogger()
    sentinel = logging.NullHandler()
    root.addHandler(sentinel)

    configure_logging(level=logging.DEBUG)

    assert root.handlers == [sentinel]


def test_configure_logging_text_format() -> None:
    configure_logging(json_output=False, level=logging.INFO)
    handler = logging.getLogger().handlers[0]
    record = logging.LogRecord(
        "food_vision.test", logging.INFO, __file__, 1, "hello %s", ("world",), None
    )
    formatted = handler.formatter.format(record)  # type: ignore[union-attr]
    assert "hello world" in formatted
    assert not formatted.startswith("{")


def test_configure_logging_json_format() -> None:
    configure_logging(json_output=True, level=logging.INFO)
    handler = logging.getLogger().handlers[0]
    formatted = handler.formatter.format(  # type: ignore[union-attr]
        logging.LogRecord("food_vision.test", logging.INFO, __file__, 1, "hello json", (), None)
    )
    assert '"message":"hello json"' in formatted


def test_get_logger_creates_no_filesystem_side_effects(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    get_logger("food_vision.anything")
    assert not (tmp_path / "logs").exists()


def test_get_logger_returns_standard_logger() -> None:
    logger = get_logger("food_vision.example")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "food_vision.example"
