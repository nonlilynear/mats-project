#!/usr/bin/env python3
"""Project a frozen SecOPD role probe onto real agent prompts.

This is an activation-only measurement pass: it does not generate text or
execute tools.  For every prompt it renders the checked-in Qwen template,
aligns the page/injection/forged-CoT character spans to tokenizer offsets,
captures the selected pre-MLP activations, and writes aggregate role
probabilities for each span.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from black_box_forgery.rendering import (  # noqa: E402
    escape_control_tokens,
    local_template_contract,
    prompt_sha256,
    render_qwen36_upstream_template,
)
from black_box_forgery.role_probes import (  # noqa: E402
    RoleProbeError,
    TrainedRoleProbe,
    forward_pre_mlp_hidden_states,
)


SPAN_NAMES = ("page", "injection", "forgery", "malicious_command")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise RoleProbeError(f"expected an object at {path}:{line_number}")
        rows.append(row)
    return rows


def load_probe(path: Path, layer: int) -> tuple[TrainedRoleProbe, str]:
    with path.open("rb") as handle:
        artifact = pickle.load(handle)
    if not isinstance(artifact, list):
        raise RoleProbeError(f"probe artifact must contain a list: {path}")
    matches = [probe for probe in artifact if getattr(probe, "layer_index", None) == layer]
    if len(matches) != 1:
        available = [getattr(probe, "layer_index", None) for probe in artifact]
        raise RoleProbeError(f"expected one probe for layer {layer}; available={available}")
    return matches[0], hashlib.sha256(path.read_bytes()).hexdigest()


def _as_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _rendered_span_for_raw_span(
    rendered: str,
    escaped_input: str,
    raw_input: str,
    raw_start: int,
    raw_end: int,
) -> tuple[int, int]:
    if rendered.count(escaped_input) != 1:
        raise RoleProbeError(
            "input message content must occur exactly once in the rendered prompt; "
            f"found {rendered.count(escaped_input)} matches"
        )
    content_start = rendered.find(escaped_input)
    escaped_start = len(escape_control_tokens(raw_input[:raw_start]))
    escaped_end = len(escape_control_tokens(raw_input[:raw_end]))
    content_end = content_start + len(escaped_input)
    if not (0 <= escaped_start <= escaped_end <= len(escaped_input)):
        raise RoleProbeError("raw span is outside the input message content")
    return content_start + escaped_start, content_start + escaped_end


def _rendered_message_content_span(rendered: str, content: str) -> tuple[int, int]:
    """Locate one complete message's escaped content in the rendered prompt."""

    escaped = escape_control_tokens(content)
    matches: list[int] = []
    start = 0
    while True:
        match = rendered.find(escaped, start)
        if match < 0:
            break
        matches.append(match)
        start = match + max(1, len(escaped))
    if len(matches) != 1:
        raise RoleProbeError(
            "message content must occur exactly once in the rendered prompt; "
            f"found {len(matches)} matches"
        )
    return matches[0], matches[0] + len(escaped)


def _token_span(offsets: list[Any], char_start: int, char_end: int) -> list[int]:
    indices: list[int] = []
    for index, pair in enumerate(offsets):
        if hasattr(pair, "tolist"):
            pair = pair.tolist()
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        token_start, token_end = int(pair[0]), int(pair[1])
        # Ignore empty special-token offsets and retain tokens that overlap the
        # character span, including a token crossing the span boundary.
        if token_end > token_start and token_start < char_end and token_end > char_start:
            indices.append(index)
    if not indices:
        raise RoleProbeError(f"character span [{char_start}, {char_end}) mapped to no tokens")
    return indices


def _span_summary(probabilities: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np  # type: ignore

    roles = tuple(probabilities)
    matrix = np.stack([np.asarray(probabilities[role], dtype=np.float64) for role in roles], axis=1)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise RoleProbeError("cannot summarize an empty activation projection")
    top_indices = matrix.argmax(axis=1)
    top_probability = matrix.max(axis=1)
    entropy = -(matrix * np.log(np.clip(matrix, 1e-12, 1.0))).sum(axis=1)
    top_role_fraction = {
        role: float(np.mean(top_indices == index)) for index, role in enumerate(roles)
    }
    mean_probability = {
        role: float(matrix[:, index].mean()) for index, role in enumerate(roles)
    }
    return {
        "token_count": int(matrix.shape[0]),
        "mean_probability": mean_probability,
        "top_role_fraction": top_role_fraction,
        "mean_top_probability": float(top_probability.mean()),
        "mean_entropy_nats": float(entropy.mean()),
        "normalized_entropy": float(entropy.mean() / np.log(len(roles))),
    }


def _load_transformers(model_id: str, revision: str | None, tokenizer_id: str | None, device_name: str) -> tuple[Any, Any, str]:
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised on pod
        raise RoleProbeError("projection requires torch and transformers") from exc
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id or model_id, use_fast=True)
    model_kwargs: dict[str, Any] = {}
    if revision:
        model_kwargs["revision"] = revision
    if torch.cuda.is_available() and device_name.startswith("cuda"):
        model_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.to(device_name).eval()
    return model, tokenizer, device_name


def project_row(
    row: dict[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    probe: TrainedRoleProbe,
    layer: int,
    allow_hidden_state_fallback: bool,
    save_token_traces: bool = False,
    trace_page_tail_tokens: int = 200,
) -> dict[str, Any]:
    import torch  # type: ignore

    messages = row.get("messages")
    spans = row.get("spans")
    if not isinstance(messages, list) or not isinstance(spans, dict):
        raise RoleProbeError(f"malformed projection input {row.get('case_id')}")
    input_index = int(spans["input_message_index"])
    if input_index < 0 or input_index >= len(messages):
        raise RoleProbeError(f"invalid input message index for {row.get('case_id')}")
    input_message = messages[input_index]
    if not isinstance(input_message, dict) or input_message.get("role") != "input":
        raise RoleProbeError(f"projection target is not an input message for {row.get('case_id')}")
    raw_input = input_message.get("content")
    if not isinstance(raw_input, str):
        raise RoleProbeError(f"input message content is not text for {row.get('case_id')}")

    rendered = render_qwen36_upstream_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    escaped_input = escape_control_tokens(raw_input)
    if rendered.count(escaped_input) != 1:
        raise RoleProbeError(
            f"could not uniquely locate input content for {row.get('case_id')}: "
            f"matches={rendered.count(escaped_input)}"
        )
    content_start = rendered.find(escaped_input)
    content_end = content_start + len(escaped_input)

    encoding = tokenizer(
        rendered,
        add_special_tokens=False,
        return_attention_mask=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"]
    attention_mask = encoding.get("attention_mask")
    offsets = _as_list(encoding["offset_mapping"][0])
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    token_count = int(input_ids.shape[1])
    input_device = next(model.parameters()).device
    input_ids = input_ids.to(input_device)
    attention_mask = attention_mask.to(input_device)

    token_spans: dict[str, list[int]] = {}
    rendered_spans: dict[str, dict[str, int]] = {}
    for name in SPAN_NAMES:
        raw_start = spans.get(f"{name}_char_start")
        raw_end = spans.get(f"{name}_char_end")
        if raw_start is None or raw_end is None:
            continue
        rendered_start, rendered_end = _rendered_span_for_raw_span(
            rendered,
            escaped_input,
            raw_input,
            int(raw_start),
            int(raw_end),
        )
        token_indices = _token_span(offsets, rendered_start, rendered_end)
        token_spans[name] = token_indices
        rendered_spans[name] = {
            "char_start": rendered_start,
            "char_end": rendered_end,
            "token_start": min(token_indices),
            "token_end": max(token_indices) + 1,
        }

    activations = forward_pre_mlp_hidden_states(
        model,
        input_ids,
        attention_mask,
        layers=(layer,),
        allow_hidden_state_fallback=allow_hidden_state_fallback,
    )[layer][0]
    projections: dict[str, Any] = {}
    for name, indices in token_spans.items():
        values = activations[indices]
        probabilities = probe.classifier.predict_proba(values)
        named = {
            role: probabilities[:, index]
            for role, index in probe.roles_map.items()
        }
        projections[name] = _span_summary(named)
        projections[name]["raw_char_start"] = int(spans[f"{name}_char_start"])
        projections[name]["raw_char_end"] = int(spans[f"{name}_char_end"])

    token_trace: dict[str, Any] | None = None
    if save_token_traces:
        if trace_page_tail_tokens <= 0:
            raise RoleProbeError("trace_page_tail_tokens must be positive")
        user_indices = [
            index for index, message in enumerate(messages) if message.get("role") == "user"
        ]
        if not user_indices:
            raise RoleProbeError(f"no user task message for {row.get('case_id')}")
        user_message = messages[user_indices[0]]
        user_content = user_message.get("content")
        if not isinstance(user_content, str):
            raise RoleProbeError(f"user task content is not text for {row.get('case_id')}")
        user_char_start, user_char_end = _rendered_message_content_span(rendered, user_content)
        user_token_indices = _token_span(offsets, user_char_start, user_char_end)

        # The page can be tens of thousands of tokens long. As in the paper's
        # role-trace figures, keep a fixed context window so the injected
        # suffix remains visible and comparable across pages.
        page_indices = token_spans.get("page", [])[-trace_page_tail_tokens:]
        scenario = str(row.get("scenario", ""))
        segment_indices: list[tuple[str, str, list[int]]] = [
            ("user_task", "user", user_token_indices),
            ("page_context", "input", page_indices),
        ]
        if scenario.endswith("cot_forged"):
            # The malicious forged payload is itself the complete appended
            # suffix in this dataset, so it must not be plotted twice as both
            # a command and a forged-CoT segment. Benign forged prompts do
            # have a separate direct contradiction before the forged trace.
            if scenario.startswith("benign"):
                injection_start = int(spans["injection_char_start"])
                forgery_start = int(spans["forgery_char_start"])
                if forgery_start > injection_start:
                    direct_start, direct_end = _rendered_span_for_raw_span(
                        rendered,
                        escaped_input,
                        raw_input,
                        injection_start,
                        forgery_start,
                    )
                    direct_indices = _token_span(offsets, direct_start, direct_end)
                    segment_indices.append(("benign_contradiction", "injection", direct_indices))
            if scenario.startswith("malicious") and "malicious_command" in token_spans:
                command_indices = token_spans["malicious_command"]
                command_set = set(command_indices)
                # The source command is nested inside the generated reasoning
                # artifact. Partition rather than duplicate those tokens.
                forged_prelude_indices = [
                    index
                    for index in token_spans.get("forgery", [])
                    if index not in command_set and index < min(command_indices)
                ]
                forged_remainder_indices = [
                    index
                    for index in token_spans.get("forgery", [])
                    if index not in command_set and index > max(command_indices)
                ]
                segment_indices.extend(
                    (
                        ("forged_cot_prelude", "cot", forged_prelude_indices),
                        ("malicious_command", "injection", command_indices),
                        ("forged_cot", "cot", forged_remainder_indices),
                    )
                )
            elif "forgery" in token_spans:
                segment_indices.append(("forged_cot", "cot", token_spans["forgery"]))
        else:
            segment_indices.append((
                "malicious_command" if scenario.startswith("malicious") else "benign_contradiction",
                "injection",
                token_spans.get("injection", []),
            ))

        segments: list[dict[str, Any]] = []
        for segment_name, source_role, indices in segment_indices:
            if not indices:
                continue
            segment_probabilities = probe.classifier.predict_proba(activations[indices])
            segments.append(
                {
                    "name": segment_name,
                    "source_role": source_role,
                    "token_count": len(indices),
                    "roles": {
                        role: [float(value) for value in segment_probabilities[:, role_index]]
                        for role, role_index in probe.roles_map.items()
                    },
                }
            )
        token_trace = {
            "page_context_tail_tokens": trace_page_tail_tokens,
            "segment_partition": (
                "disjoint_embedded_command_and_forged_cot_v2"
                if scenario.startswith("malicious") and scenario.endswith("cot_forged")
                else "standard_v1"
            ),
            "segments": segments,
        }

    del activations, input_ids, attention_mask
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    source_metadata = row.get("provenance", {}).get("source_request_metadata", {})
    source_hash = source_metadata.get("rendered_prompt_sha256") if isinstance(source_metadata, dict) else None
    output = {
        "schema_version": "1.0",
        "case_id": row.get("case_id"),
        "scenario": row.get("scenario"),
        "page_id": row.get("page_id"),
        "probe_layer": layer,
        "probe_roles": list(probe.role_space),
        "activation_source": "pre_mlp_post_attention_layernorm",
        "rendered_prompt_sha256": prompt_sha256(rendered),
        "source_rendered_prompt_sha256": source_hash,
        "source_rendered_prompt_matches": source_hash is None or source_hash == prompt_sha256(rendered),
        "rendered_token_count": token_count,
        "input_content_char_count": len(raw_input),
        "rendered_input_char_span": {"char_start": content_start, "char_end": content_end},
        "rendered_spans": rendered_spans,
        "projections": projections,
    }
    if token_trace is not None:
        output["token_trace"] = token_trace
    if source_hash is not None and source_hash != output["rendered_prompt_sha256"]:
        raise RoleProbeError(
            f"rendered prompt hash mismatch for {row.get('case_id')}: "
            f"source={source_hash}, local={output['rendered_prompt_sha256']}"
        )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-jsonl", type=Path, required=True)
    parser.add_argument("--probe-artifact", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--layer", type=int, default=56)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--save-token-traces",
        action="store_true",
        help="save per-token role probabilities for plotting",
    )
    parser.add_argument(
        "--trace-page-tail-tokens",
        type=int,
        default=200,
        help="number of page tokens retained in each token trace",
    )
    parser.add_argument("--allow-hidden-state-fallback", action="store_true")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise RoleProbeError("--limit must be positive")
    if args.trace_page_tail_tokens <= 0:
        raise RoleProbeError("--trace-page-tail-tokens must be positive")

    rows = read_jsonl(args.inputs_jsonl)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise RoleProbeError("no projection inputs selected")
    probe, probe_sha256 = load_probe(args.probe_artifact, args.layer)
    model, tokenizer, device = _load_transformers(
        args.model, args.revision, args.tokenizer, args.device
    )

    outputs: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        print(
            f"[{index}/{len(rows)}] projecting {row.get('case_id')} "
            f"(layer={args.layer})",
            flush=True,
        )
        outputs.append(
            project_row(
                row,
                model=model,
                tokenizer=tokenizer,
                probe=probe,
                layer=args.layer,
                allow_hidden_state_fallback=args.allow_hidden_state_fallback,
                save_token_traces=args.save_token_traces,
                trace_page_tail_tokens=args.trace_page_tail_tokens,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "projections.jsonl"
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in outputs),
        encoding="utf-8",
    )
    metadata = {
        "schema_version": "1.0",
        "input_path": str(args.inputs_jsonl),
        "input_sha256": hashlib.sha256(args.inputs_jsonl.read_bytes()).hexdigest(),
        "probe_artifact": str(args.probe_artifact),
        "probe_sha256": probe_sha256,
        "probe_layer": args.layer,
        "probe_roles": list(probe.role_space),
        "probe_test_accuracy": probe.test_accuracy,
        "model": args.model,
        "model_revision": args.revision,
        "tokenizer": args.tokenizer or args.model,
        "device": device,
        "activation_source": "pre_mlp_post_attention_layernorm",
        "template_contract": local_template_contract(),
        "row_count": len(outputs),
        "token_traces": args.save_token_traces,
        "trace_page_tail_tokens": args.trace_page_tail_tokens if args.save_token_traces else None,
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RoleProbeError) as exc:
        print(f"project_role_probes.py: error: {exc}", file=sys.stderr)
        raise SystemExit(2)
