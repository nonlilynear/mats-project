#!/usr/bin/env python3
"""Judge completed target-chat outputs with the same Qwen endpoint.

This is intentionally a separate, provisional self-judge artifact.  It does
not rerun target scenarios and it never replaces the raw generation records.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from black_box_forgery.inference import OpenAICompatibleBackend, make_request
from black_box_forgery.judging import parse_chat_label
from black_box_forgery.schemas import Condition, DecodingParams


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_judge_messages(path: Path) -> list[dict[str, str]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"judge prompt must be a nonempty message list: {path}")
    messages: list[dict[str, str]] = []
    for index, message in enumerate(payload):
        if not isinstance(message, Mapping):
            raise ValueError(f"judge prompt message {index} must be an object")
        role, content = message.get("role"), message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise ValueError(f"invalid judge prompt message {index}")
        messages.append({"role": str(role), "content": content})
    return messages


def _source_key(record: Mapping[str, Any]) -> str:
    return json.dumps(record["request_key"], sort_keys=True, separators=(",", ":"))


def _completed_chat_records(run_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((run_dir / "generation_records" / "records").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        condition = record.get("request_key", {}).get("condition")
        if record.get("status") == "complete" and condition in {"raw_chat", "cot_chat"}:
            record["_source_path"] = str(path)
            records.append(record)
    return sorted(
        records,
        key=lambda record: (
            str(record["request_key"].get("condition")),
            str(record["request_key"].get("dataset_item_id")),
        ),
    )


def _append(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--chat-prompts", type=Path, required=True)
    parser.add_argument(
        "--judge-prompt",
        type=Path,
        default=Path("prompts/upstream/harm-classifier.yaml"),
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:18000")
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--model-revision", default="6a9e13bd6fc8f0983b9b99948120bc37f49c13e9")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or args.run_dir / "analysis" / "qwen-chat-self-judgments.jsonl"
    judge_messages = _load_judge_messages(args.judge_prompt)
    prompt_rows = {
        str(row.get("id")): row
        for row in _read_jsonl(args.chat_prompts)
        if isinstance(row, Mapping) and row.get("id") is not None
    }
    source_records = _completed_chat_records(args.run_dir)
    prior: dict[str, dict[str, Any]] = {}
    if output.exists():
        for row in _read_jsonl(output):
            if row.get("source_request_key") is not None:
                prior[str(row["source_request_key"])] = row

    backend = OpenAICompatibleBackend(
        base_url=args.base_url,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
        allow_network=True,
    )
    decoding = DecodingParams(
        seed=123,
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=512,
        max_turns=1,
        thinking_enabled=False,
    )
    attempted = completed = skipped = errors = 0
    for source in source_records:
        source_key = _source_key(source)
        old = prior.get(source_key)
        if old is not None and old.get("status") == "complete":
            skipped += 1
            continue
        request_key = source["request_key"]
        item_id = str(request_key["dataset_item_id"])
        prompt_row = prompt_rows.get(item_id)
        if prompt_row is None:
            raise ValueError(f"missing frozen harmful prompt for {item_id}")
        harmful_prompt = str(prompt_row["prompt"])
        response_text = str(source.get("output_text") or "")
        final_messages = [
            *judge_messages,
            {
                "role": "user",
                "content": json.dumps(
                    {"harmful_prompt": harmful_prompt, "llm_response": response_text},
                    ensure_ascii=False,
                ),
            },
        ]
        request = make_request(
            run_id="qwen-chat-self-judge-20260905",
            model_id=args.model,
            model_revision=args.model_revision,
            dataset_item_id=item_id,
            condition=Condition(request_key["condition"]),
            messages=final_messages,
            decoding=decoding,
            metadata={
                "judge_type": "same_model_self_judge",
                "source_run": str(args.run_dir),
                "source_request_key": source_key,
                "judge_prompt_path": str(args.judge_prompt),
                "judge_prompt_sha256": _sha256_text(args.judge_prompt.read_text(encoding="utf-8")),
                "source_output_sha256": _sha256_text(response_text),
            },
        )
        attempted += 1
        started = datetime.now(timezone.utc)
        try:
            result = backend.complete(request)
            parsed = parse_chat_label(result.output_text)
            row = {
                "schema_version": 1,
                "status": "complete",
                "source_run": str(args.run_dir),
                "source_path": source["_source_path"],
                "source_request_key": source_key,
                "dataset_item_id": item_id,
                "condition": request_key["condition"],
                "model": source.get("model_id", request_key.get("victim_model_id", args.model)),
                "judge_model": args.model,
                "judge_model_revision": args.model_revision,
                "judge_output_text": result.output_text,
                "judge_thinking_text": result.thinking_text,
                "automated_raw": result.output_text,
                "automated_label": parsed.label,
                "automated_valid": parsed.valid,
                "automated_error": parsed.error,
                "label": parsed.label,
                "valid": parsed.valid,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "stop_reason": result.stop_reason,
                "started_at": started.isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "metadata": result.metadata,
            }
            completed += 1
        except Exception as exc:
            errors += 1
            row = {
                "schema_version": 1,
                "status": "error",
                "source_run": str(args.run_dir),
                "source_path": source["_source_path"],
                "source_request_key": source_key,
                "dataset_item_id": item_id,
                "condition": request_key["condition"],
                "model": source.get("model_id", request_key.get("victim_model_id", args.model)),
                "judge_model": args.model,
                "error": f"{type(exc).__name__}: {exc}",
                "started_at": started.isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
        _append(output, row)
        print(
            json.dumps(
                {"condition": row["condition"], "item_id": item_id, "status": row["status"]},
                sort_keys=True,
            ),
            flush=True,
        )

    print(
        json.dumps(
            {
                "output": str(output),
                "source_records": len(source_records),
                "attempted": attempted,
                "completed": completed,
                "skipped": skipped,
                "errors": errors,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
