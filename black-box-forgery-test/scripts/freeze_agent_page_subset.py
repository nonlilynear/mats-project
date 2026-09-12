#!/usr/bin/env python3
"""Freeze a reproducible expanded agent-page subset around the prior six."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

from black_box_forgery.data import canonical_json, deterministic_select, load_fixture_pages, sha256_path


PRIOR_SIX = (
    "66872153",
    "66872497",
    "66875836",
    "66885891",
    "66886217",
    "66886867",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages-index", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()

    pages = load_fixture_pages(args.pages_index)
    by_id = {page.sample_id: page for page in pages}
    missing = sorted(set(PRIOR_SIX) - set(by_id))
    if missing:
        raise ValueError(f"prior six pages are missing from source index: {missing}")
    if args.count < len(PRIOR_SIX) or args.count > len(pages):
        raise ValueError("count must include the prior six and fit within the source index")

    extensions = deterministic_select(
        [page for page in pages if page.sample_id not in PRIOR_SIX],
        args.count - len(PRIOR_SIX),
        args.seed,
        "benign-rabbit-agent-pages-extension",
    )
    selected = [by_id[page_id] for page_id in PRIOR_SIX] + extensions
    if len({page.sample_id for page in selected}) != args.count:
        raise ValueError("expanded page selection contains duplicate IDs")

    args.output_index.parent.mkdir(parents=True, exist_ok=True)
    with args.output_index.open("w", encoding="utf-8") as stream:
        for page in selected:
            stream.write(canonical_json(page.model_dump(mode="json")) + "\n")
            fixture_path = str(page.provenance.metadata.get("fixture_path", ""))
            source_fixture = args.pages_index.parent / fixture_path
            target_fixture = args.output_index.parent / fixture_path
            if not source_fixture.is_file():
                raise ValueError(f"source fixture is missing: {source_fixture}")
            target_fixture.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_fixture, target_fixture)
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "purpose": "benign_rabbit_agent_injection_2x2",
        "seed": args.seed,
        "source_index": str(args.pages_index),
        "source_index_sha256": sha256_path(args.pages_index),
        "selection": {
            "count": args.count,
            "prior_ids_preserved": list(PRIOR_SIX),
            "extension_count": len(extensions),
            "extension_namespace": "benign-rabbit-agent-pages-extension",
        },
        "page_ids": [page.sample_id for page in selected],
        "page_titles": {page.sample_id: page.title for page in selected},
        "page_hashes": {page.sample_id: page.provenance.sha256 for page in selected},
        "index": str(args.output_index),
        "index_sha256": sha256_path(args.output_index),
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"count": len(selected), "index": str(args.output_index), "manifest": str(args.output_manifest)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
