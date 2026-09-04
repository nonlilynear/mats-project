#!/usr/bin/env python3
"""Render auxiliary-generation results as a compact side-by-side review file."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def target_text(request: dict) -> str:
    content = "\n".join(str(message.get("content", "")) for message in request.get("messages", []))
    matches = re.findall(r"<TEST_PROMPT>(.*?)</TEST_PROMPT>", content, flags=re.DOTALL)
    return matches[-1].strip() if matches else "[target unavailable]"


def fence(text: str) -> str:
    marker = "````" if "```" in text else "```"
    return f"{marker}text\n{text}\n{marker}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--results", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only-with-results", action="store_true")
    args = parser.parse_args()

    requests = {row["request_id"]: row for row in read_jsonl(args.requests)}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for path in args.results:
        for row in read_jsonl(path):
            grouped[row["request_id"]].append(row)
    if args.only_with_results:
        requests = {request_id: row for request_id, row in requests.items() if request_id in grouped}

    candidates = sorted({row["candidate"] for rows in grouped.values() for row in rows})
    totals = {
        candidate: {
            "calls": 0,
            "usable": 0,
            "cost": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
        for candidate in candidates
    }
    for rows in grouped.values():
        for row in rows:
            item = totals[row["candidate"]]
            item["calls"] += 1
            item["usable"] += bool(row.get("output_text", "").strip())
            item["cost"] += float(row.get("cost_usd") or 0)
            item["input_tokens"] += int(row.get("input_tokens") or 0)
            item["output_tokens"] += int(row.get("output_tokens") or 0)

    lines = [
        "# Minimum auxiliary generator bakeoff",
        "",
        "Each case uses the same frozen target and style references. Review output quality side by side; an empty output is an unusable generation.",
        "",
        "| Candidate | Calls | Nonempty outputs | Cost (USD) | Input tokens | Output tokens |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for candidate in candidates:
        item = totals[candidate]
        lines.append(
            f"| {candidate} | {item['calls']} | {item['usable']} | {item['cost']:.9f} | "
            f"{item['input_tokens']} | {item['output_tokens']} |"
        )

    for request_id in requests:
        request = requests[request_id]
        metadata = request.get("metadata", {})
        lines.extend(
            [
                "",
                f"## {request_id}",
                "",
                f"Block: `{metadata.get('block', '')}`  ",
                f"Category/page: {metadata.get('category') or metadata.get('page_title') or ''}",
                "",
                "### Target",
                "",
                fence(target_text(request)),
            ]
        )
        for row in sorted(grouped.get(request_id, []), key=lambda item: item["candidate"]):
            output = row.get("output_text", "").strip()
            max_paragraphs = int(
                (row.get("request_metadata") or {}).get(
                    "max_paragraphs", (request.get("metadata") or {}).get("max_paragraphs", 1)
                )
            )
            contract_valid = (
                output.startswith("<SYNTHETIC_POLICY>")
                and output.endswith("</SYNTHETIC_POLICY>")
                and len([part for part in output.split("\n\n") if part.strip()]) <= max_paragraphs
            )
            choice = ((row.get("raw") or {}).get("choices") or [{}])[0]
            finish = choice.get("native_finish_reason") or choice.get("finish_reason") or "unknown"
            lines.extend(
                [
                    "",
                    f"### {row['candidate']}",
                    "",
                    f"Nonempty: `{bool(output)}` · Contract valid: `{contract_valid}` · "
                    f"Finish: `{finish}` · Cost: `${float(row.get('cost_usd') or 0):.9f}`",
                    "",
                    fence(output or "[NO FINAL OUTPUT]"),
                ]
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cases": len(requests), "candidates": candidates}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
