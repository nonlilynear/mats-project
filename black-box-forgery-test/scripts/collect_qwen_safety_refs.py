#!/usr/bin/env python3
"""Collect authentic reasoning-style references from a local Qwen vLLM endpoint.

The input prompts are intentionally abstract and non-actionable. The output is
an auditable JSON artifact containing the visible answer and the reasoning span
returned by the OpenAI-compatible endpoint. No API credentials are read or
written by this script.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read_prompts(path: Path) -> list[dict[str, str]]:
    prompts: list[dict[str, str]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"prompt line {line_number} is not an object")
        reference_id = row.get("reference_id")
        prompt = row.get("prompt")
        if not isinstance(reference_id, str) or not reference_id.strip():
            raise ValueError(f"prompt line {line_number} has no reference_id")
        if reference_id in seen:
            raise ValueError(f"duplicate reference_id: {reference_id}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"prompt {reference_id} is empty")
        seen.add(reference_id)
        prompts.append({"reference_id": reference_id, "prompt": prompt})
    if len(prompts) < 3:
        raise ValueError("at least three safety style prompts are required")
    return prompts


def _request(base_url: str, model: str, prompt: str, *, max_tokens: int, seed: int, timeout: float) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "top_p": 1,
        "seed": seed,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(f"Qwen request failed: {exc}") from exc
    try:
        result = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Qwen returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise RuntimeError("Qwen response is not an object")
    return result


def _extract(result: dict[str, Any]) -> tuple[str, str, str, dict[str, Any]]:
    choices = result.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RuntimeError("Qwen response has no choice")
    choice = choices[0]
    message = choice.get("message") or {}
    if not isinstance(message, dict):
        raise RuntimeError("Qwen response message is malformed")
    answer = message.get("content") or ""
    thinking = message.get("reasoning_content") or message.get("reasoning") or ""
    finish_reason = choice.get("finish_reason") or "unknown"
    usage = result.get("usage") or {}
    if not isinstance(answer, str) or not isinstance(thinking, str) or not isinstance(usage, dict):
        raise RuntimeError("Qwen response fields are malformed")
    return thinking, answer, str(finish_reason), usage


def collect(args: argparse.Namespace) -> dict[str, Any]:
    prompts = _read_prompts(args.prompts)
    references: list[dict[str, Any]] = []
    for task_index, item in enumerate(prompts):
        raw = _request(
            args.base_url,
            args.model,
            item["prompt"],
            max_tokens=args.max_tokens,
            seed=args.seed,
            timeout=args.timeout,
        )
        thinking, answer, finish_reason, usage = _extract(raw)
        references.append(
            {
                "reference_id": item["reference_id"],
                "task_index": task_index,
                "prompt": item["prompt"],
                "prompt_sha256": _sha256_text(item["prompt"]),
                "thinking": thinking,
                "thinking_sha256": _sha256_text(thinking) if thinking else None,
                "answer": answer,
                "answer_sha256": _sha256_text(answer) if answer else None,
                "finish_reason": finish_reason,
                "complete": bool(thinking.strip() and answer.strip() and finish_reason == "stop"),
                "usage": usage,
            }
        )

    artifact = {
        "schema_version": "1.0",
        "source": "live_base_qwen_safety_style_collection",
        "model": args.model,
        "endpoint": args.base_url,
        "decoding": {
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "temperature": 0,
            "top_p": 1,
            "thinking_enabled": True,
        },
        "complete": all(reference["complete"] for reference in references),
        "references": references,
    }
    _atomic_write(args.output, json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return artifact


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    result.add_argument("--model", default="Qwen/Qwen3.6-27B")
    result.add_argument("--prompts", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--max-tokens", type=int, default=4096)
    result.add_argument("--seed", type=int, default=123)
    result.add_argument("--timeout", type=float, default=180.0)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    artifact = collect(arguments)
    print(
        json.dumps(
            {
                "complete": artifact["complete"],
                "model": artifact["model"],
                "references": len(artifact["references"]),
                "output": str(arguments.output),
            },
            sort_keys=True,
        )
    )
