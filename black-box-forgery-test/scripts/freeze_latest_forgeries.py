#!/usr/bin/env python3
"""Freeze the latest mechanically valid auxiliary forgery for each request."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data = (json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode()
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--results", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block", choices=("chat", "agent"), required=True)
    args = parser.parse_args()

    requests = _read_jsonl(args.requests)
    expected = {
        str(row["request_id"]): row
        for row in requests
        if row.get("metadata", {}).get("block") == args.block
    }
    if not expected:
        raise ValueError(f"no {args.block} requests found")

    latest: dict[str, dict[str, Any]] = {}
    for results_path in args.results:
        for result in _read_jsonl(results_path):
            request_id = str(result.get("request_id", ""))
            if request_id not in expected:
                continue
            if result.get("status") != "complete" or result.get("valid") is not True:
                continue
            # Result logs are append-only. Later valid rows supersede earlier
            # valid rows, including rows from a later --results file.
            result = dict(result)
            result["_source_results_path"] = str(results_path)
            latest[request_id] = result

    missing = sorted(set(expected) - set(latest))
    if missing:
        raise ValueError(f"cannot freeze forgeries: no valid result for {missing}")

    frozen = []
    for request_id, request in expected.items():
        result = latest[request_id]
        output_text = str(result["output_text"])
        metadata = request["metadata"]
        frozen.append(
            {
                "item_id": str(metadata["dataset_item_id"]),
                "forgery": output_text,
                "source_request_id": request_id,
                "source_attempt": int(result["attempt"]),
                "source_result_key": str(result["request_key"]),
                "source_results_path": str(result["_source_results_path"]),
                "source_output_sha256": hashlib.sha256(output_text.encode()).hexdigest(),
                "source_mechanical_valid": True,
                "postprocessed_tag_wrapper": False,
                "postprocess_reason": None,
                "prompt_template_sha256": str(metadata["prompt_template_sha256"]),
                "style_sha256": str(metadata["style_sha256"]),
            }
        )

    _atomic_write(args.output, frozen)
    print(json.dumps({"output": str(args.output), "block": args.block, "count": len(frozen)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
