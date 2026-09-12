#!/usr/bin/env python3
"""Materialize a reproducible neutral corpus for role-probe training.

This samples neutral pretraining text once from C4 and Dolma3, then writes a
local JSONL snapshot. The snapshot is the input to ``train_role_probes.py``;
the attack pages and injection transcripts are never used here.

Example:

  uv run --extra role-probes python scripts/materialize_probe_texts.py \
    --output data/probe/neutral.jsonl \
    --manifest data/probe/neutral.manifest.json \
    --allow-network
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from black_box_forgery.role_probes import RoleProbeError  # noqa: E402


DATASET_REVISION = "3a8349c"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--num-texts", type=int, default=250)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="required because this command streams public datasets",
    )
    return parser


def _load_streams() -> list[tuple[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise RoleProbeError(
            "install the role-probes extra before materializing the corpus"
        ) from exc

    return [
        (
            "c4",
            load_dataset(
                "allenai/c4",
                "en",
                split="validation",
                streaming=True,
            ),
        ),
        (
            "dolma3",
            load_dataset(
                "allenai/dolma3_mix-150B-1025",
                split="train",
                revision=DATASET_REVISION,
                streaming=True,
            ),
        ),
    ]


def _sample_stream(dataset: Iterable[Any], count: int, seed: int) -> list[str]:
    shuffled = dataset.shuffle(seed=seed, buffer_size=50_000)
    texts: list[str] = []
    seen: set[str] = set()
    for row in shuffled:
        text = row.get("text") if isinstance(row, dict) else None
        if not isinstance(text, str) or not text.strip():
            continue
        text = text.strip()
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        texts.append(text)
        if len(texts) >= count:
            break
    if len(texts) < count:
        raise RoleProbeError(
            f"stream returned only {len(texts)} unique text rows; needed {count}"
        )
    return texts


def _write_snapshot(
    output: Path,
    manifest: Path,
    rows: list[dict[str, str]],
    *,
    num_texts: int,
    seed: int,
    allocations: dict[str, int],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    file_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest_payload = {
        "format": "neutral_probe_texts_v1",
        "num_texts": num_texts,
        "seed": seed,
        "sampling": "streamed shuffled validation/train rows, exact-text deduplication",
        "allocations": allocations,
        "datasets": {
            "c4": {
                "id": "allenai/c4",
                "config": "en",
                "split": "validation",
            },
            "dolma3": {
                "id": "allenai/dolma3_mix-150B-1025",
                "split": "train",
                "revision": DATASET_REVISION,
            },
        },
        "jsonl_sha256": file_hash,
    }
    manifest.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.allow_network:
        raise RoleProbeError("pass --allow-network to stream C4 and Dolma3")
    if args.num_texts < 2:
        raise RoleProbeError("--num-texts must be at least two")

    c4_count = args.num_texts // 2
    dolma3_count = args.num_texts - c4_count
    streams = _load_streams()
    rows: list[dict[str, str]] = []
    allocations = {"c4": c4_count, "dolma3": dolma3_count}
    for (source, dataset), count in zip(streams, (c4_count, dolma3_count)):
        rows.extend({"text": text, "source": source} for text in _sample_stream(dataset, count, args.seed))

    _write_snapshot(
        args.output,
        args.manifest,
        rows,
        num_texts=args.num_texts,
        seed=args.seed,
        allocations=allocations,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(args.output),
                "manifest": str(args.manifest),
                "num_texts": len(rows),
                "allocations": allocations,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RoleProbeError as exc:
        raise SystemExit(f"error: {exc}") from exc
