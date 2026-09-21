#!/usr/bin/env python3
"""Filter pairs whose items occupy exactly one Qwen tokenizer token.

The contextual check is span-aware. For example, with ``newline`` context,
the newline itself may be a separate token; that separator is ignored and we
count only tokens whose character spans overlap the item.

Example:
    python filter_pairs.py --backend tinker

Set TINKER_API_KEY before using the Tinker backend. The HF backend is useful
for validating the same tokenizer locally when the model files are available.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from pathlib import Path
from typing import Any


MODEL_NAME = "Qwen/Qwen3.6-27B"
PREFIXES = {
    "standalone": "",
    "space": " ",
    "newline": "\n",
    "double-newline": "\n\n",
}


def load_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")

        fields = {field.strip().lower(): field for field in reader.fieldnames}
        item1_field = fields.get("item1") or fields.get("item 1")
        item2_field = fields.get("item2") or fields.get("item 2")
        if item1_field is None or item2_field is None:
            raise ValueError("Input must have item1/item2 or item 1/item 2 columns")

        pairs = []
        for row in reader:
            item1 = (row.get(item1_field) or "").strip()
            item2 = (row.get(item2_field) or "").strip()
            if not item1 or not item2:
                raise ValueError(f"Blank pair item in row: {row}")
            pairs.append({"item1": item1, "item2": item2})
        return pairs


async def load_tinker_tokenizer(model_name: str) -> Any:
    import tinker

    service_client = tinker.ServiceClient()
    sampling_client = await service_client.create_sampling_client_async(
        base_model=model_name
    )
    return sampling_client.get_tokenizer()


def load_hf_tokenizer(model_name: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def tokenize_with_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]], list[str]]:
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except (NotImplementedError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "This tokenizer must expose offset mappings. Use a fast Qwen tokenizer "
            "or install/use the matching tokenizer.json locally."
        ) from exc

    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
        offsets = offsets[0]

    input_ids = [int(token_id) for token_id in input_ids]
    offsets = [(int(start), int(end)) for start, end in offsets]
    token_strings = tokenizer.convert_ids_to_tokens(input_ids)
    return input_ids, offsets, token_strings


def count_overlapping_tokens(
    offsets: list[tuple[int, int]], item_start: int, item_end: int
) -> int:
    """Count tokens touching the item span, excluding separator-only tokens."""

    return sum(
        token_start < item_end and item_start < token_end
        for token_start, token_end in offsets
        if token_end > token_start
    )


def inspect_item(tokenizer: Any, item: str, context_names: list[str]) -> dict[str, Any]:
    contexts: dict[str, Any] = {}
    for context_name in context_names:
        prefix = PREFIXES[context_name]
        text = prefix + item
        token_ids, offsets, token_strings = tokenize_with_offsets(tokenizer, text)
        item_start = len(prefix)
        item_end = item_start + len(item)
        item_token_count = count_overlapping_tokens(offsets, item_start, item_end)
        contexts[context_name] = {
            "prefix": repr(prefix),
            "token_ids": token_ids,
            "token_strings": token_strings,
            "item_token_count": item_token_count,
            "one_token": item_token_count == 1,
        }

    return {
        "item": item,
        "valid": all(context["one_token"] for context in contexts.values()),
        "contexts": contexts,
    }


def write_tsv(path: Path, pairs: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item1", "item2"], delimiter="\t")
        writer.writeheader()
        writer.writerows(pairs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("pairs.tsv"))
    parser.add_argument("--output", type=Path, default=Path("filtered_pairs.tsv"))
    parser.add_argument("--diagnostics", type=Path, default=Path("token_diagnostics.json"))
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--backend", choices=["tinker", "hf"], default="tinker")
    parser.add_argument(
        "--contexts",
        nargs="+",
        choices=list(PREFIXES),
        default=["standalone", "space", "newline"],
        help="Contexts in which each item must occupy exactly one token",
    )
    args = parser.parse_args()

    pairs = load_tsv(args.input)
    if args.backend == "tinker":
        tokenizer = asyncio.run(load_tinker_tokenizer(args.model))
    else:
        tokenizer = load_hf_tokenizer(args.model)

    unique_items = {item for pair in pairs for item in pair.values()}
    reports = {
        item: inspect_item(tokenizer, item, args.contexts)
        for item in sorted(unique_items)
    }

    filtered = [
        pair
        for pair in pairs
        if reports[pair["item1"]]["valid"] and reports[pair["item2"]]["valid"]
    ]

    write_tsv(args.output, filtered)
    args.diagnostics.write_text(
        json.dumps(
            {
                "model": args.model,
                "backend": args.backend,
                "contexts": args.contexts,
                "pairs_seen": len(pairs),
                "pairs_kept": len(filtered),
                "items": reports,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Kept {len(filtered)}/{len(pairs)} pairs")
    print(f"Wrote {args.output} and {args.diagnostics}")


if __name__ == "__main__":
    main()
