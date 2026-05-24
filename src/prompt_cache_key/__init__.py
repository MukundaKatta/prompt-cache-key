"""prompt-cache-key - stable scope hashes for the Anthropic prompt cache.

Anthropic's prompt cache hits when the prefix (system + tools, up to a
`cache_control` breakpoint) is byte-identical to a previously seen
request. When you have multiple workers serving the same agent, you
want them all to share that warm prefix. Coordinating that requires a
deterministic scope key everyone can compute locally.

`compute_cache_key()` walks the system + tools + model into a canonical
byte stream and returns a SHA-256 hex digest prefixed with the model:

    from prompt_cache_key import compute_cache_key

    key = compute_cache_key(
        model="claude-opus-4-7",
        system=[
            {"type": "text", "text": "..."},
            {"type": "text", "text": "...", "cache_control": {"type": "ephemeral"}},
        ],
        tools=my_tools,
    )
    # "anthropic-cache:claude-opus-4-7:sha256:7c4..."

Anything AFTER the last `cache_control` breakpoint in `system` is
deliberately excluded from the key — that content isn't part of the
cached scope from Anthropic's perspective. `find_breakpoints(blocks)`
returns the indices of `cache_control` markers if you need to inspect
that yourself.

Companion to `prompt-cache-warmer` (which actively warms a cache scope)
and `cachebench` (which measures hit ratios over time). Distinct from
`llm-message-hash` which is for request idempotency, not cache scoping.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

__version__ = "0.1.0"
__all__ = [
    "compute_cache_key",
    "find_breakpoints",
    "scope_blocks",
    "KEY_PREFIX",
]


KEY_PREFIX = "anthropic-cache"


def find_breakpoints(blocks: Iterable[dict[str, Any]] | None) -> list[int]:
    """Return zero-based indices of blocks carrying `cache_control`."""
    if not blocks:
        return []
    return [
        i for i, b in enumerate(blocks)
        if isinstance(b, dict) and b.get("cache_control") is not None
    ]


def scope_blocks(
    blocks: Iterable[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Return only the prefix up to and including the LAST cache_control.

    If `blocks` is empty / None, returns []. If no cache_control marker
    is present, the full block list is returned (cache scope = everything
    in the system prompt; Anthropic treats this as no caching, but the
    helper still returns the full list so the key is stable).
    """
    if not blocks:
        return []
    lst = list(blocks)
    bps = find_breakpoints(lst)
    if not bps:
        return [dict(b) for b in lst]
    last = bps[-1]
    return [dict(b) for b in lst[: last + 1]]


def compute_cache_key(
    *,
    model: str,
    system: str | list[dict[str, Any]] | None = None,
    tools: Iterable[dict[str, Any]] | None = None,
) -> str:
    """Stable scope key for a given model + system + tools.

    Returns `"{KEY_PREFIX}:{model}:sha256:{hex}"`.

    `system` may be a plain string (treated as one text block) or the
    Anthropic content-block list. `tools` is a list of tool definitions
    (`{"name", "description", "input_schema"}` shape).

    The body is a JSON-serialized, sort-keyed canonical view of:
      - model
      - scope_blocks(system)  (prefix up to last cache_control)
      - tools                 (entire list, canonicalized)

    Everything AFTER the last cache_control in `system` is excluded —
    that content isn't part of the cached prefix. Tools always participate
    in the cache scope (Anthropic includes them in the cached prefix).
    """
    body = {
        "model": model,
        "system": scope_blocks(_normalize_system(system)),
        "tools": [dict(t) for t in (tools or [])],
    }
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return f"{KEY_PREFIX}:{model}:sha256:{digest}"


def _normalize_system(system: str | list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if system is None:
        return []
    if isinstance(system, str):
        return [{"type": "text", "text": system}]
    return list(system)
