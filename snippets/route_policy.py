"""
The per-request routing rule: local model vs. frontier model.

This runs on every request before any model is invoked. It's pure string work —
no network, no LLM call — so it costs microseconds and cannot fail closed: if
nothing matches, the default is the free local tier. Priority order is the whole
design:

  1. An explicit model hint from the client always wins.
  2. Sensitive content never leaves the box (stays local).
  3. Recency keywords (today's news, current prices) need a model with fresh
     world knowledge, so they escalate to frontier.
  4. Everything else defaults to local — free, private, good enough.

Ambiguous prompts that reach step 4 can optionally escalate to a cheap classifier
model; that's layered on top of this rule, not baked into it. Keeping the hot
path as plain rules is what makes it both fast and auditable — every decision is
logged with its reason for later analysis.

Excerpt from the two-layer agent router. See docs/agent-architecture.md.
"""
from __future__ import annotations


def route(
    model: str,
    messages: list[dict],
    local_alias: str,
    frontier_alias: str,
    recency_keywords: list[str],
    sensitive_prefixes: list[str],
) -> tuple[str, str]:
    """Return (tier, reason) where tier is "local" or "frontier"."""
    model_lower = model.lower().strip()

    # 1. Explicit override — the client knows what it wants
    if model_lower and model_lower in (local_alias.lower(), "tuesday", "local", "qwen"):
        return "local", "explicit_model_hint"
    if model_lower and (
        model_lower in (frontier_alias.lower(), "frontier")
        or model_lower.startswith("claude")
    ):
        return "frontier", "explicit_model_hint"

    # Flatten message content (handles both str and structured content blocks)
    content = " ".join(
        m.get("content", "") if isinstance(m.get("content"), str)
        else " ".join(
            c.get("text", "") for c in m.get("content", []) if isinstance(c, dict)
        )
        for m in messages
    ).lower()

    # 2. Sensitive content → always stays local
    for prefix in sensitive_prefixes:
        if prefix.lower() in content:
            return "local", f"sensitive_content:{prefix}"

    # 3. Recency keywords → needs fresh world knowledge → frontier
    for kw in recency_keywords:
        if kw.lower() in content:
            return "frontier", f"recency_keyword:{kw}"

    # 4. Default — local is free, private, and good enough for most tasks
    return "local", "default"
