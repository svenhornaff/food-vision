"""Vision-model observation: prompts, output schemas, and the adapter.

See ``schema.py``, ``prompts/`` and ``adapter.py``, and
docs/dev/pre-study-web-ui.md §7.3/§7.4. This is the seam that sits behind
``proxy.ModelProvider`` — it knows prompts/schemas/outcome classification,
the proxy knows transport/retries/telemetry.
"""

from __future__ import annotations
