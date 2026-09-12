#!/usr/bin/env python3
"""Split saved malicious-forgery traces into command and CoT components.

The original activation export retained per-token probe probabilities for the
entire generated forgery, so correcting its semantic segmentation does not
require another model forward pass. This script uses the target tokenizer's
offset mapping to partition those saved probabilities without duplication or
loss, and records the source export hash for auditability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from black_box_forgery.rendering import (  # noqa: E402
    escape_control_tokens,
    prompt_sha256,
    render_qwen36_upstream_template,
)
from black_box_forgery.role_probes import RoleProbeError  # noqa: E402
from project_role_probes import (  # noqa: E402
    _as_list,
    _rendered_span_for_raw_span,
    _token_span,
    read_jsonl,
)


PARTITION_VERSION = "disjoint_embedded_command_and_forged_cot_v2"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(roles: dict[str, list[float]]) -> dict[str, Any]:
    role_order = list(roles)
    lengths = {len(values) for values in roles.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise RoleProbeError("cannot summarize empty or misaligned role arrays")
    token_count = next(iter(lengths))
    winners: list[str] = []
    top_probabilities: list[float] = []
    entropies: list[float] = []
    for token_index in range(token_count):
        values = {role: float(roles[role][token_index]) for role in role_order}
        winner = max(values, key=values.get)
        winners.append(winner)
        top_probabilities.append(values[winner])
        entropies.append(-sum(value * math.log(max(value, 1e-12)) for value in values.values()))
    mean_entropy = sum(entropies) / token_count
    return {
        "token_count": token_count,
        "mean_probability": {
            role: sum(float(value) for value in roles[role]) / token_count for role in role_order
        },
        "top_role_fraction": {
            role: sum(winner == role for winner in winners) / token_count for role in role_order
        },
        "mean_top_probability": sum(top_probabilities) / token_count,
        "mean_entropy_nats": mean_entropy,
        "normalized_entropy": mean_entropy / math.log(len(role_order)),
    }


def _slice_roles(
    roles: dict[str, list[float]],
    positions: list[int],
) -> dict[str, list[float]]:
    return {
        role: [float(values[position]) for position in positions]
        for role, values in roles.items()
    }


def resegment_row(
    projection: dict[str, Any],
    source: dict[str, Any],
    tokenizer: Any,
) -> dict[str, Any]:
    if projection.get("scenario") != "malicious_cot_forged":
        return projection
    spans = source.get("spans")
    messages = source.get("messages")
    if not isinstance(spans, dict) or not isinstance(messages, list):
        raise RoleProbeError(f"malformed source row for {projection.get('case_id')}")
    input_index = int(spans["input_message_index"])
    raw_input = messages[input_index].get("content")
    if not isinstance(raw_input, str):
        raise RoleProbeError("projection source input content must be text")

    rendered = render_qwen36_upstream_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if prompt_sha256(rendered) != projection.get("rendered_prompt_sha256"):
        raise RoleProbeError(
            f"rendered prompt changed for {projection.get('case_id')}; cannot reuse activations"
        )
    encoding = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = _as_list(encoding["offset_mapping"])
    escaped_input = escape_control_tokens(raw_input)
    forgery_chars = _rendered_span_for_raw_span(
        rendered,
        escaped_input,
        raw_input,
        int(spans["forgery_char_start"]),
        int(spans["forgery_char_end"]),
    )
    command_chars = _rendered_span_for_raw_span(
        rendered,
        escaped_input,
        raw_input,
        int(spans["malicious_command_char_start"]),
        int(spans["malicious_command_char_end"]),
    )
    forgery_indices = _token_span(offsets, *forgery_chars)
    command_indices = _token_span(offsets, *command_chars)
    forgery_positions = {token_index: position for position, token_index in enumerate(forgery_indices)}
    try:
        command_positions = [forgery_positions[token_index] for token_index in command_indices]
    except KeyError as exc:
        raise RoleProbeError("command token falls outside the saved forgery trace") from exc
    command_position_set = set(command_positions)
    prelude_positions = [
        position
        for position in range(len(forgery_indices))
        if position not in command_position_set and position < min(command_positions)
    ]
    remainder_positions = [
        position
        for position in range(len(forgery_indices))
        if position not in command_position_set and position > max(command_positions)
    ]

    trace = projection.get("token_trace")
    if not isinstance(trace, dict):
        raise RoleProbeError(f"missing token trace for {projection.get('case_id')}")
    segments = trace.get("segments")
    if not isinstance(segments, list):
        raise RoleProbeError("token trace segments must be a list")
    old_forged = [segment for segment in segments if segment.get("name") == "forged_cot"]
    if len(old_forged) != 1:
        raise RoleProbeError("expected exactly one saved forged-CoT segment")
    old_roles = old_forged[0].get("roles")
    if not isinstance(old_roles, dict) or any(
        not isinstance(values, list) or len(values) != len(forgery_indices)
        for values in old_roles.values()
    ):
        raise RoleProbeError("saved forged-CoT probabilities do not align with tokenizer offsets")
    command_roles = _slice_roles(old_roles, command_positions)
    prelude_roles = _slice_roles(old_roles, prelude_positions)
    remainder_roles = _slice_roles(old_roles, remainder_positions)
    retained = [
        segment
        for segment in segments
        if segment.get("name") not in {"forged_cot_prelude", "malicious_command", "forged_cot"}
    ]
    retained.extend(
        [
            {
                "name": "forged_cot_prelude",
                "source_role": "cot",
                "token_count": len(prelude_positions),
                "roles": prelude_roles,
            },
            {
                "name": "malicious_command",
                "source_role": "injection",
                "token_count": len(command_positions),
                "roles": command_roles,
            },
            {
                "name": "forged_cot",
                "source_role": "cot",
                "token_count": len(remainder_positions),
                "roles": remainder_roles,
            },
        ]
    )
    trace["segments"] = retained
    trace["segment_partition"] = PARTITION_VERSION
    trace["partition_audit"] = {
        "original_forgery_token_count": len(forgery_indices),
        "forged_cot_prelude_token_count": len(prelude_positions),
        "malicious_command_token_count": len(command_positions),
        "forged_cot_remainder_token_count": len(remainder_positions),
        "partition_is_disjoint_and_exhaustive": (
            len(prelude_positions) + len(command_positions) + len(remainder_positions)
            == len(forgery_indices)
        ),
        "malicious_command_span_method": spans["malicious_command_span_method"],
    }
    projection.setdefault("rendered_spans", {})["malicious_command"] = {
        "char_start": command_chars[0],
        "char_end": command_chars[1],
        "token_start": min(command_indices),
        "token_end": max(command_indices) + 1,
    }
    command_summary = _summary(command_roles)
    command_summary.update(
        {
            "raw_char_start": int(spans["malicious_command_char_start"]),
            "raw_char_end": int(spans["malicious_command_char_end"]),
            "span_method": spans["malicious_command_span_method"],
        }
    )
    projection.setdefault("projections", {})["malicious_command"] = command_summary
    projection["projections"]["forged_cot_prelude"] = _summary(prelude_roles)
    projection["projections"]["forged_cot_remainder"] = _summary(remainder_roles)
    projection["trace_resegmentation"] = {
        "version": PARTITION_VERSION,
        "activation_recomputed": False,
        "saved_per_token_probabilities_reused": True,
    }
    return projection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-jsonl", type=Path, required=True)
    parser.add_argument("--source-projections-jsonl", type=Path, required=True)
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        from transformers import AutoTokenizer  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised on pod
        raise RoleProbeError("resegmentation requires transformers") from exc
    inputs = read_jsonl(args.inputs_jsonl)
    projections = read_jsonl(args.source_projections_jsonl)
    by_case = {row.get("case_id"): row for row in inputs}
    if len(by_case) != len(inputs) or {row.get("case_id") for row in projections} != set(by_case):
        raise RoleProbeError("input and projection case IDs do not match exactly")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    outputs = [resegment_row(row, by_case[row.get("case_id")], tokenizer) for row in projections]
    partitioned = [
        row for row in outputs if row.get("trace_resegmentation", {}).get("version") == PARTITION_VERSION
    ]
    if len(partitioned) != 10:
        raise RoleProbeError(f"expected 10 partitioned malicious forgeries, found {len(partitioned)}")
    if any(
        not row["token_trace"]["partition_audit"]["partition_is_disjoint_and_exhaustive"]
        for row in partitioned
    ):
        raise RoleProbeError("a malicious command/forgery partition is not exact")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "projections.jsonl"
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in outputs),
        encoding="utf-8",
    )
    metadata = json.loads(args.source_metadata.read_text(encoding="utf-8"))
    metadata.update(
        {
            "input_path": str(args.inputs_jsonl),
            "input_sha256": _sha(args.inputs_jsonl),
            "source_projections_jsonl": str(args.source_projections_jsonl),
            "source_projections_sha256": _sha(args.source_projections_jsonl),
            "trace_resegmentation": PARTITION_VERSION,
            "activation_recomputed": False,
            "partitioned_row_count": len(partitioned),
            "output_sha256": _sha(output_path),
        }
    )
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "output_sha256": _sha(output_path),
                "row_count": len(outputs),
                "partitioned_row_count": len(partitioned),
                "activation_recomputed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RoleProbeError) as exc:
        print(f"resegment_malicious_forged_role_traces.py: error: {exc}", file=sys.stderr)
        raise SystemExit(2)
