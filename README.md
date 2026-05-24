# prompt-cache-key

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/prompt-cache-key.svg)](https://pypi.org/project/prompt-cache-key/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Generate stable Anthropic prompt-cache scope keys.** Distributed workers can compute the same key locally and share warm cache. Zero deps.

```python
from prompt_cache_key import compute_cache_key

key = compute_cache_key(
    model="claude-opus-4-7",
    system=[
        {"type": "text", "text": LONG_SYSTEM_PROMPT},
        {"type": "text", "text": "", "cache_control": {"type": "ephemeral"}},
    ],
    tools=my_tools,
)
# "anthropic-cache:claude-opus-4-7:sha256:7c4f8a..."
```

Content **after** the last `cache_control` breakpoint is excluded — that part isn't in the cached scope from Anthropic's perspective. Tools always participate.

## Why

Anthropic's prompt cache hits when the prefix (system + tools, up to a `cache_control` breakpoint) is byte-identical to a recently seen request. If you have multiple workers serving the same agent, they ALL benefit when the warm prefix is shared. But the coordination layer (a queue, a router, a memo) needs a stable key everyone can compute locally without an extra round trip.

`prompt-cache-key` is that key generator. Walk the model, scope-clipped system blocks, and tools into a canonical JSON view; SHA-256 hash; prefix with model so different models can't collide. ~100 LOC, no SDK dependency.

Pair with [`prompt-cache-warmer`](https://github.com/MukundaKatta/prompt-cache-warmer) (warm a scope before user traffic) and [`cachebench`](https://github.com/MukundaKatta/cachebench) (measure hit ratios).

Distinct from [`llm-message-hash-py`](https://github.com/MukundaKatta/llm-message-hash-py) which hashes the FULL request for idempotency/dedup. This lib hashes ONLY the cache-relevant scope.

## Install

```bash
pip install prompt-cache-key
```

## API

```python
from prompt_cache_key import (
    compute_cache_key,    # main entrypoint
    find_breakpoints,     # indices of cache_control markers
    scope_blocks,         # prefix up to (incl.) last cache_control
    KEY_PREFIX,           # "anthropic-cache"
)

compute_cache_key(
    *,
    model: str,
    system: str | list[dict] | None = None,
    tools: Iterable[dict] | None = None,
) -> str
```

`system` accepts either a plain string (treated as one text block) or the Anthropic content-block list. `tools` is a list of tool definitions.

## License

MIT
