#!/usr/bin/env python3
"""Build frozen auxiliary forgery requests for an arbitrary page subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from black_box_forgery.data import assign_injections, canonical_json, load_fixture_pages, load_injection_templates, sha256_path, sha256_text
from prepare_minimum_aux_requests import _load_prompt_messages, _single_turn_messages, _style_context, _target_message


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages-index", type=Path, required=True)
    parser.add_argument("--forgery-prompt", type=Path, required=True)
    parser.add_argument("--injections", type=Path, required=True)
    parser.add_argument("--style-references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-paragraphs", type=int, default=16)
    args = parser.parse_args()

    pages = load_fixture_pages(args.pages_index)
    prompt_messages = _load_prompt_messages(args.forgery_prompt)
    style_context, style_sha256 = _style_context(args.style_references)
    injections = load_injection_templates(args.injections)["base_injections"]
    assignments = {
        assignment.sample_id: assignment
        for assignment in assign_injections(
            [page.sample_id for page in pages],
            injections,
            seed=20260903,
            injection_type="standard",
        )
    }
    rows = []
    for page in pages:
        assignment = assignments[page.sample_id]
        rows.append(
            {
                "schema_version": "1.0",
                "request_id": f"agent__{page.sample_id}",
                "task": "generate_forgery",
                "input_text": f"Generate the frozen agent forgery for page assignment {page.sample_id}.",
                "messages": [
                    *prompt_messages,
                    {"role": "user", "content": _target_message(assignment.prompt, style_context)},
                ],
                "max_tokens": args.max_tokens,
                "metadata": {
                    "block": "agent",
                    "rubric": "agent",
                    "dataset_item_id": page.sample_id,
                    "page_title": page.title,
                    "target_sha256": sha256_text(assignment.prompt),
                    "injection_index": injections.index(assignment.prompt),
                    "injection_sha256": assignment.source_sha256,
                    "style_sha256": style_sha256,
                    "prompt_template_sha256": sha256_path(args.forgery_prompt),
                    "max_paragraphs": args.max_paragraphs,
                },
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "purpose": "benign_rabbit_agent_injection_2x2",
        "request_count": len(rows),
        "agent_count": len(rows),
        "requests_path": str(args.output),
        "requests_sha256": sha256_path(args.output),
        "pages_index": str(args.pages_index),
        "pages_index_sha256": sha256_path(args.pages_index),
        "forgery_prompt": str(args.forgery_prompt),
        "forgery_prompt_sha256": sha256_path(args.forgery_prompt),
        "injections": str(args.injections),
        "injections_sha256": sha256_path(args.injections),
        "style_references": str(args.style_references),
        "style_sha256": style_sha256,
        "max_tokens": args.max_tokens,
        "max_paragraphs": args.max_paragraphs,
        "page_ids": [page.sample_id for page in pages],
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"requests": len(rows), "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
