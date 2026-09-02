#!/usr/bin/env python3
"""Probe rust.cat OpenAI-compatible router for gpt-5.6-sol / gpt-5.6-luna.

Usage:
  export RUSTCAT_API_KEY='sk-...'
  python scripts/probe_rustcat_router.py

The router is at https://rust.cat/v1 and uses chat.completions (not responses).
The OpenAI Python SDK needs a curl-like User-Agent or Cloudflare blocks it.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from openai import OpenAI

BASE_URL = os.environ.get("RUSTCAT_BASE_URL", "https://rust.cat/v1")
MODELS = ["gpt-5.6-sol", "gpt-5.6-luna"]


def short(obj: Any, limit: int = 500) -> str:
    s = obj if isinstance(obj, str) else json.dumps(obj, default=str)
    return s if len(s) <= limit else s[:limit] + "..."


def make_client(key: str) -> OpenAI:
    # Cloudflare on rust.cat blocks the default OpenAI SDK User-Agent.
    return OpenAI(
        api_key=key,
        base_url=BASE_URL,
        timeout=120,
        max_retries=0,
        default_headers={"User-Agent": "curl/8.5.0"},
    )


def try_list_models(client: OpenAI) -> dict:
    out = {"ok": False, "mode": "models.list"}
    try:
        models = client.models.list()
        ids = [m.id for m in models.data]
        wanted = [m for m in MODELS if m in ids]
        out.update({"ok": True, "count": len(models.data), "wanted_present": wanted})
    except Exception as e:
        out["error"] = repr(e)
    return out


def try_chat(client: OpenAI, model: str, *, reasoning_effort: str | None = None) -> dict:
    out = {"ok": False, "mode": "chat.completions", "model": model}
    if reasoning_effort:
        out["reasoning_effort"] = reasoning_effort
    try:
        request_kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
            "max_completion_tokens": 64,
        }
        if reasoning_effort is not None:
            request_kwargs["reasoning_effort"] = reasoning_effort
        resp = client.chat.completions.create(**request_kwargs)
        text = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        out.update(
            {
                "ok": bool(text),
                "text": text,
                "usage": usage.model_dump() if hasattr(usage, "model_dump") else usage,
            }
        )
    except Exception as e:
        out["error"] = repr(e)
    return out


def main() -> int:
    key = os.environ.get("RUSTCAT_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        print("Set RUSTCAT_API_KEY or OPENAI_API_KEY", file=sys.stderr)
        return 1

    print(f"base_url={BASE_URL}")
    client = make_client(key)

    print("models.list:", short(try_list_models(client)))

    any_ok = False
    for model in MODELS:
        basic = try_chat(client, model)
        print(f"chat {model}:", short(basic))
        any_ok = any_ok or bool(basic.get("ok"))

        reasoning = try_chat(client, model, reasoning_effort="high")
        print(f"chat {model} (reasoning_effort=high):", short(reasoning))
        any_ok = any_ok or bool(reasoning.get("ok"))

    if not any_ok:
        print("\nFAILED: no successful chat.completions call.")
        return 2

    print("\nOK: rust.cat works via chat.completions.")
    print("Cutegen already uses chat.completions for server_type='openai'.")
    print("To plug in (not as-is today):")
    print("  1. OPENAI_API_KEY=<rust.cat key>")
    print("  2. OPENAI_BASE_URL=https://rust.cat/v1")
    print("  3. OPENAI_MODEL=gpt-5 | gpt-5.6-sol | gpt-5.6-luna")
    print("  4. python scripts/test_rustcat_models.py  # verify cutegen path")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
