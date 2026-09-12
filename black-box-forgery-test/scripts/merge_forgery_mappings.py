#!/usr/bin/env python3
"""Merge reused frozen forgeries with valid outputs for an expanded page set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--results", type=Path, action="append", required=True)
    parser.add_argument("--reuse", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--repair-missing-closing-tags",
        action="store_true",
        help="append the required closing tag to otherwise well-formed generated artifacts",
    )
    args = parser.parse_args()

    requests = _read_jsonl(args.requests)
    expected = {str(row["request_id"]): row for row in requests}
    reused = {str(row["item_id"]): row for row in _read_json(args.reuse)}
    latest: dict[str, dict[str, Any]] = {}
    repairable: dict[str, dict[str, Any]] = {}
    for result_path in args.results:
        for result in _read_jsonl(result_path):
            request_id = str(result.get("request_id", ""))
            if request_id in expected and result.get("status") == "complete" and result.get("valid") is True:
                latest[request_id] = {**result, "source_results_path": str(result_path)}
            validation = result.get("validation")
            if (
                args.repair_missing_closing_tags
                and request_id in expected
                and result.get("status") == "complete"
                and result.get("valid") is not True
                and isinstance(validation, dict)
                and validation.get("nonempty") is True
                and validation.get("tagged") is True
                and validation.get("closed") is False
            ):
                repairable[request_id] = {**result, "source_results_path": str(result_path)}

    output = []
    for request_id, request in expected.items():
        item_id = str(request["metadata"]["dataset_item_id"])
        if item_id in reused:
            row = dict(reused[item_id])
            row["reuse_source"] = str(args.reuse)
        else:
            result = latest.get(request_id)
            repaired = False
            if result is None and args.repair_missing_closing_tags:
                result = repairable.get(request_id)
                repaired = result is not None
            if result is None:
                raise ValueError(f"no valid forgery available for {request_id}")
            text = str(result["output_text"])
            if repaired:
                text = text.rstrip() + "</SYNTHETIC_POLICY>"
            metadata = request["metadata"]
            row = {
                "item_id": item_id,
                "forgery": text,
                "source_request_id": request_id,
                "source_attempt": int(result.get("attempt", 1)),
                "source_result_key": str(result["request_key"]),
                "source_results_path": result["source_results_path"],
                "source_output_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "source_mechanical_valid": True,
                "postprocessed_tag_wrapper": repaired,
                "postprocess_reason": (
                    "provider output omitted required closing tag; appended mechanically"
                    if repaired
                    else None
                ),
                "prompt_template_sha256": str(metadata["prompt_template_sha256"]),
                "style_sha256": str(metadata["style_sha256"]),
                "reuse_source": None,
            }
        if row.get("source_mechanical_valid") is not True:
            raise ValueError(f"reused forgery is not mechanically valid: {item_id}")
        output.append(row)

    if len(output) != len(expected) or len({row["item_id"] for row in output}) != len(expected):
        raise ValueError("merged forgery mapping does not cover each page exactly once")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "count": len(output), "reused": len(reused)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
