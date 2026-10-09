"""Centralised logging — one factory, no scattered ``basicConfig`` calls.

Call :func:`configure_logging` once at process startup (CLI entrypoint, test
fixture, future FastAPI ``lifespan``). Every other module imports
:func:`get_logger` instead of calling ``logging.getLogger`` directly, so the
format/level/output target stays controlled from one place.

Design notes (why this differs from a naive singleton-class factory):

- No filesystem side effects. Nothing creates a ``logs/`` directory or a
  file handler as a side effect of importing a module or calling
  ``get_logger()`` — a library/CLI/test process that never calls
  ``configure_logging()`` stays silent-but-functional (root logger's default
  handler takes over) and never touches disk.
- Idempotent. Calling ``configure_logging()`` twice (e.g. tests invoking an
  app factory twice) does not duplicate handlers.
- No credentials, prompts, images or meal content ever belong in a log
  line — see docs/dev/food-vision-concept.md §11. Callers are responsible
  for keeping messages free of secrets; this module only owns formatting.
"""

from __future__ import annotations

import logging
import sys

_JSON_FORMAT = (
    '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}'
)
_TEXT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s — %(message)s"


def configure_logging(*, json_output: bool = False, level: int = logging.INFO) -> None:
    """Configure the root logger once. Call at startup, not per-module.

    Safe to call more than once: if the root logger already has handlers
    (set by this function, a parent process, or a test runner) this is a
    no-op so repeated calls never duplicate output.
    """
    root = logging.getLogger()
    root.setLevel(level)

    if root.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    fmt = _JSON_FORMAT if json_output else _TEXT_FORMAT
    handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a logger. Import and call this everywhere instead of
    ``logging.getLogger`` directly, so call sites don't drift from the
    centralised format/level if that ever changes.
    """
    return logging.getLogger(name)
