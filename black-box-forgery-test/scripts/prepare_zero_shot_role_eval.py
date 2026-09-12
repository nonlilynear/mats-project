#!/usr/bin/env python3
"""Build a frozen zero-shot role-probe validation set from SecOPD outputs.

Clean agent traces provide genuine user, input, and assistant spans. Raw,
non-forged SecOPD chat traces provide genuine model-generated reasoning spans,
because the saved agent traces contain no nonempty ``thinking_text``. The chat
records are used only to validate the model-level CoT probe, not as experiment
outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paired_records(request_dir: Path, generation_dir: Path, condition: str) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for generation_path in sorted(generation_dir.glob(f"*__{condition}__*.json")):
        request_path = request_dir / generation_path.name
        if not request_path.is_file():
            raise ValueError(f"missing request record for {generation_path.name}")
        pairs.append((request_path, generation_path))
    return pairs


def _copy_messages(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("request record has no messages")
    return json.loads(json.dumps(value, ensure_ascii=False))


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for request_path, generation_path in _paired_records(
        args.agent_request_dir, args.agent_generation_dir, "clean_agent"
    ):
        request = _read(request_path)
        generation = _read(generation_path)
        output = generation.get("output_text")
        if generation.get("status") != "complete" or not isinstance(output, str) or not output.strip():
            raise ValueError(f"incomplete clean agent generation: {generation_path}")
        messages = _copy_messages(request.get("messages"))
        role_positions = {message.get("role"): index for index, message in enumerate(messages)}
        if "user" not in role_positions or "input" not in role_positions:
            raise ValueError(f"clean agent request lacks user/input roles: {request_path}")
        assistant_index = len(messages)
        messages.append({"role": "assistant", "reasoning_content": "", "content": output})
        page_id = str(generation.get("request_key", {}).get("dataset_item_id", "unknown"))
        rows.append(
            {
                "schema_version": "1.0",
                "case_id": f"zero_shot_agent__{page_id}",
                "source": "clean_agent",
                "messages": messages,
                "targets": [
                    {"role": "user", "message_index": role_positions["user"], "field": "content"},
                    {"role": "input", "message_index": role_positions["input"], "field": "content"},
                    {"role": "assistant", "message_index": assistant_index, "field": "content"},
                ],
                "provenance": {
                    "request_record": str(request_path),
                    "request_sha256": _sha(request_path),
                    "generation_record": str(generation_path),
                    "generation_sha256": _sha(generation_path),
                    "condition": "clean_agent",
                    "forged_text_used_as_ground_truth": False,
                },
            }
        )

    for request_path, generation_path in _paired_records(
        args.chat_request_dir, args.chat_generation_dir, "raw_chat"
    ):
        request = _read(request_path)
        generation = _read(generation_path)
        reasoning = generation.get("thinking_text")
        output = generation.get("output_text")
        if (
            generation.get("status") != "complete"
            or not isinstance(reasoning, str)
            or not reasoning.strip()
            or not isinstance(output, str)
            or not output.strip()
        ):
            raise ValueError(f"raw SecOPD trace lacks genuine reasoning/output: {generation_path}")
        messages = _copy_messages(request.get("messages"))
        assistant_index = len(messages)
        messages.append(
            {"role": "assistant", "reasoning_content": reasoning, "content": output}
        )
        item_id = str(generation.get("request_key", {}).get("dataset_item_id", "unknown"))
        rows.append(
            {
                "schema_version": "1.0",
                "case_id": f"zero_shot_native_cot__{item_id}",
                "source": "raw_nonforged_native_reasoning",
                "messages": messages,
                # User/assistant recovery is evaluated on the clean agent
                # traces; this source exists only to supply genuine SecOPD CoT.
                "targets": [
                    {"role": "cot", "message_index": assistant_index, "field": "reasoning_content"}
                ],
                "provenance": {
                    "request_record": str(request_path),
                    "request_sha256": _sha(request_path),
                    "generation_record": str(generation_path),
                    "generation_sha256": _sha(generation_path),
                    "condition": "raw_chat",
                    "used_only_for_model_level_cot_validation": True,
                    "forged_text_used_as_ground_truth": False,
                },
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent-request-dir",
        type=Path,
        default=ROOT / "runs/secopd-agent-qwen-style-18-smoke-20260908/request_records/records",
    )
    parser.add_argument(
        "--agent-generation-dir",
        type=Path,
        default=ROOT / "runs/secopd-agent-qwen-style-18-smoke-20260908/agent_generation_records/records",
    )
    parser.add_argument(
        "--chat-request-dir",
        type=Path,
        default=ROOT / "runs/secopd-chat-qwen-style-smoke-20260906/request_records/records",
    )
    parser.add_argument(
        "--chat-generation-dir",
        type=Path,
        default=ROOT / "runs/secopd-chat-qwen-style-smoke-20260906/generation_records/records",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = build_rows(args)
    source_counts = {
        source: sum(row["source"] == source for row in rows)
        for source in sorted({row["source"] for row in rows})
    }
    if source_counts != {"clean_agent": 6, "raw_nonforged_native_reasoning": 12}:
        raise ValueError(f"unexpected zero-shot source counts: {source_counts}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "output_sha256": _sha(args.output),
                "row_count": len(rows),
                "source_counts": source_counts,
                "target_roles": sorted({target["role"] for row in rows for target in row["targets"]}),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"prepare_zero_shot_role_eval.py: error: {exc}", file=sys.stderr)
        raise SystemExit(2)
