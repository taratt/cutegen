#!/usr/bin/env python3
"""Test cutegen's OpenAI path against rust.cat for gpt-5 / gpt-5.6-sol / gpt-5.6-luna.

Usage:
  export OPENAI_API_KEY='sk-...'
  export OPENAI_BASE_URL='https://rust.cat/v1'
  python scripts/test_rustcat_models.py

Optional:
  python scripts/test_rustcat_models.py --models gpt-5.6-sol
"""

from __future__ import annotations

import argparse
import os
import sys

from cutegen.llm_api import query_server

DEFAULT_MODELS = ["gpt-5", "gpt-5.6-sol", "gpt-5.6-luna"]


def test_model(model: str) -> dict:
    out = {"model": model, "ok": False}
    try:
        result = query_server(
            prompt="Reply with exactly: pong",
            server_type="openai",
            model_name=model,
            is_reasoning_model=True,
            reasoning_effort="high",
            max_completion_tokens=64,
        )
        text = (result if isinstance(result, str) else result[0] or "").strip()
        out["text"] = text
        out["ok"] = text.lower() == "pong"
    except Exception as e:
        out["error"] = repr(e)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Models to test (default: gpt-5 gpt-5.6-sol gpt-5.6-luna)",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY", file=sys.stderr)
        return 1
    if not os.environ.get("OPENAI_BASE_URL"):
        print(
            "Warning: OPENAI_BASE_URL not set; will hit api.openai.com",
            file=sys.stderr,
        )

    print(f"OPENAI_BASE_URL={os.environ.get('OPENAI_BASE_URL', '(default)')}")
    results = []
    for model in args.models:
        result = test_model(model)
        results.append(result)
        status = "OK" if result.get("ok") else "FAIL"
        detail = result.get("text") or result.get("error", "unknown")
        print(f"[{status}] {model}: {detail}")

    passed = sum(1 for r in results if r.get("ok"))
    print(f"\n{passed}/{len(results)} models passed via cutegen.query_server")
    return 0 if passed == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
