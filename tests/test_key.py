"""Tests for prompt_cache_key."""

from __future__ import annotations

import pytest

from prompt_cache_key import (
    KEY_PREFIX,
    compute_cache_key,
    find_breakpoints,
    scope_blocks,
)


# ---- find_breakpoints --------------------------------------------------


def test_find_breakpoints_empty():
    assert find_breakpoints([]) == []
    assert find_breakpoints(None) == []


def test_find_breakpoints_no_markers():
    blocks = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    assert find_breakpoints(blocks) == []


def test_find_breakpoints_single_marker():
    blocks = [
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "c"},
    ]
    assert find_breakpoints(blocks) == [1]


def test_find_breakpoints_multiple_markers():
    blocks = [
        {"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "b"},
        {"type": "text", "text": "c", "cache_control": {"type": "ephemeral"}},
    ]
    assert find_breakpoints(blocks) == [0, 2]


# ---- scope_blocks ---------------------------------------------------


def test_scope_blocks_includes_through_last_breakpoint():
    blocks = [
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "c"},
        {"type": "text", "text": "d", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "e"},
    ]
    out = scope_blocks(blocks)
    texts = [b["text"] for b in out]
    assert texts == ["a", "b", "c", "d"]  # excludes "e"


def test_scope_blocks_no_breakpoint_returns_all():
    blocks = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    out = scope_blocks(blocks)
    assert [b["text"] for b in out] == ["a", "b"]


def test_scope_blocks_empty_returns_empty():
    assert scope_blocks([]) == []
    assert scope_blocks(None) == []


def test_scope_blocks_returns_copies():
    blocks = [{"type": "text", "text": "a"}]
    out = scope_blocks(blocks)
    out[0]["text"] = "MUTATED"
    assert blocks[0]["text"] == "a"


# ---- compute_cache_key ---------------------------------------------


def test_compute_key_returns_prefixed_hex():
    key = compute_cache_key(model="claude-opus-4-7", system="You are helpful.")
    assert key.startswith(f"{KEY_PREFIX}:claude-opus-4-7:sha256:")
    hex_part = key.rsplit(":", 1)[-1]
    assert len(hex_part) == 64
    int(hex_part, 16)


def test_compute_key_stable():
    a = compute_cache_key(model="claude-opus-4-7", system="x")
    b = compute_cache_key(model="claude-opus-4-7", system="x")
    assert a == b


def test_compute_key_changes_with_model():
    a = compute_cache_key(model="claude-opus-4-7", system="x")
    b = compute_cache_key(model="claude-sonnet-4-6", system="x")
    assert a != b


def test_compute_key_changes_with_system():
    a = compute_cache_key(model="claude-opus-4-7", system="x")
    b = compute_cache_key(model="claude-opus-4-7", system="y")
    assert a != b


def test_compute_key_changes_with_tools():
    base_tools = [{"name": "search", "description": "d", "input_schema": {}}]
    a = compute_cache_key(model="claude-opus-4-7", system="x", tools=base_tools)
    b = compute_cache_key(model="claude-opus-4-7", system="x", tools=[])
    assert a != b


def test_compute_key_string_system_equals_single_text_block():
    a = compute_cache_key(model="claude-opus-4-7", system="hello")
    b = compute_cache_key(
        model="claude-opus-4-7",
        system=[{"type": "text", "text": "hello"}],
    )
    assert a == b


def test_compute_key_ignores_content_after_last_breakpoint():
    base = [
        {"type": "text", "text": "stable", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "EXTRA-1"},
    ]
    other = [
        {"type": "text", "text": "stable", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "DIFFERENT"},
    ]
    a = compute_cache_key(model="claude-opus-4-7", system=base)
    b = compute_cache_key(model="claude-opus-4-7", system=other)
    assert a == b  # post-breakpoint content excluded from scope


def test_compute_key_includes_content_before_and_at_breakpoint():
    a = compute_cache_key(
        model="claude-opus-4-7",
        system=[
            {"type": "text", "text": "stable-a", "cache_control": {"type": "ephemeral"}},
        ],
    )
    b = compute_cache_key(
        model="claude-opus-4-7",
        system=[
            {"type": "text", "text": "stable-b", "cache_control": {"type": "ephemeral"}},
        ],
    )
    assert a != b


def test_compute_key_none_system_works():
    key = compute_cache_key(model="claude-opus-4-7")
    assert key.startswith(f"{KEY_PREFIX}:claude-opus-4-7:sha256:")


def test_compute_key_tools_order_matters():
    t1 = [{"name": "a", "description": "", "input_schema": {}}]
    t2 = [
        {"name": "a", "description": "", "input_schema": {}},
        {"name": "b", "description": "", "input_schema": {}},
    ]
    a = compute_cache_key(model="claude-opus-4-7", tools=t1)
    b = compute_cache_key(model="claude-opus-4-7", tools=t2)
    assert a != b


def test_compute_key_no_breakpoint_includes_full_system():
    """When system has no cache_control, the whole thing participates."""
    a = compute_cache_key(model="claude-opus-4-7", system="full prompt A")
    b = compute_cache_key(model="claude-opus-4-7", system="full prompt B")
    assert a != b


def test_key_prefix_constant():
    assert KEY_PREFIX == "anthropic-cache"
