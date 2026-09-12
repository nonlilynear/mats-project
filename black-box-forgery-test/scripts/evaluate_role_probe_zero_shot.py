#!/usr/bin/env python3
"""Evaluate a frozen SecOPD role probe on genuine, correctly tagged traces.

This implements the paper's second probe-validity criterion: zero-shot role
identification in real, unmodified conversations. It reports soft role
probabilities and hard argmax recovery for genuine content tokens only. The
primary summary uses the paper's temporal-plot truncation convention (100
user tokens, 200 tokens for other roles); an all-token sensitivity summary is
also retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
from typing import Any, Iterable


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


PAPER_TOKEN_CAPS = {"user": 100}
DEFAULT_OTHER_TOKEN_CAP = 200


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RoleProbeError(f"expected an object at {path}:{line_number}")
        rows.append(value)
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


def _token_span(offsets: list[Any], char_start: int, char_end: int) -> list[int]:
    indices: list[int] = []
    for index, pair in enumerate(offsets):
        if hasattr(pair, "tolist"):
            pair = pair.tolist()
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        token_start, token_end = int(pair[0]), int(pair[1])
        if token_end > token_start and token_start < char_end and token_end > char_start:
            indices.append(index)
    if not indices:
        raise RoleProbeError(f"character span [{char_start}, {char_end}) mapped to no tokens")
    return indices


def locate_target_char_spans(
    rendered: str,
    messages: list[dict[str, Any]],
    targets: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Locate target fields relative to their structural message delimiter.

    Exact model-control tokens in content are escaped before rendering, so an
    exact ``<|im_start|>ROLE`` occurrence unambiguously starts a real message.
    Locating within that block prevents quoted/repeated content from matching
    a different turn or a copy inside the model's reasoning.
    """

    message_starts: dict[int, int] = {}
    cursor = 0
    for message_index, message in enumerate(messages):
        role = message.get("role")
        if not isinstance(role, str):
            raise RoleProbeError(f"message {message_index} has no textual role")
        marker = f"<|im_start|>{role}\n"
        marker_start = rendered.find(marker, cursor)
        if marker_start < 0:
            raise RoleProbeError(
                f"could not locate structural marker for message {message_index} ({role})"
            )
        message_starts[message_index] = marker_start + len(marker)
        cursor = marker_start + len(marker)

    located: list[dict[str, Any]] = []
    for target_index, target in enumerate(targets):
        role = target.get("role")
        field = target.get("field")
        message_index = int(target.get("message_index", -1))
        if message_index not in message_starts or not (0 <= message_index < len(messages)):
            raise RoleProbeError(f"target {target_index} has invalid message index")
        message = messages[message_index]
        if message.get("role") == "assistant":
            expected_role = "cot" if field == "reasoning_content" else "assistant"
            if role != expected_role:
                raise RoleProbeError(
                    f"target role/field mismatch: role={role!r}, field={field!r}"
                )
            reasoning = message.get("reasoning_content", "")
            content = message.get("content", "")
            if not isinstance(reasoning, str) or not isinstance(content, str):
                raise RoleProbeError("assistant reasoning/content must be text")
            escaped_reasoning = escape_control_tokens(reasoning).strip()
            escaped_content = escape_control_tokens(content).strip()
            assistant_start = message_starts[message_index]
            prefix = "<think>\n"
            if rendered[assistant_start : assistant_start + len(prefix)] != prefix:
                raise RoleProbeError("assistant target does not start with a thinking block")
            reasoning_start = assistant_start + len(prefix)
            if rendered[reasoning_start : reasoning_start + len(escaped_reasoning)] != escaped_reasoning:
                raise RoleProbeError("rendered reasoning does not match the source trace")
            content_start = reasoning_start + len(escaped_reasoning) + len("\n</think>\n\n")
            if rendered[content_start : content_start + len(escaped_content)] != escaped_content:
                raise RoleProbeError("rendered assistant content does not match the source trace")
            if field == "reasoning_content":
                char_start, text = reasoning_start, escaped_reasoning
            elif field == "content":
                char_start, text = content_start, escaped_content
            else:
                raise RoleProbeError(f"unsupported assistant target field: {field!r}")
        else:
            if field != "content" or role != message.get("role"):
                raise RoleProbeError(
                    f"target does not match message role/field: role={role!r}, field={field!r}"
                )
            raw_content = message.get("content")
            if not isinstance(raw_content, str):
                raise RoleProbeError("message target content must be text")
            text = escape_control_tokens(raw_content).strip()
            char_start = message_starts[message_index]
            if rendered[char_start : char_start + len(text)] != text:
                raise RoleProbeError("rendered message content does not match the source trace")
        if not text:
            raise RoleProbeError(f"target {target_index} has empty rendered content")
        located.append(
            {
                **target,
                "char_start": char_start,
                "char_end": char_start + len(text),
                "rendered_char_count": len(text),
            }
        )
    return located


def _load_transformers(
    model_id: str,
    revision: str | None,
    tokenizer_id: str | None,
    device_name: str,
) -> tuple[Any, Any, str]:
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised on pod
        raise RoleProbeError("zero-shot evaluation requires torch and transformers") from exc
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


def evaluate_row(
    row: dict[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    probe: TrainedRoleProbe,
    layer: int,
    allow_hidden_state_fallback: bool,
) -> dict[str, Any]:
    import numpy as np  # type: ignore
    import torch  # type: ignore

    messages = row.get("messages")
    targets = row.get("targets")
    if not isinstance(messages, list) or not messages or not isinstance(targets, list):
        raise RoleProbeError(f"malformed zero-shot row: {row.get('case_id')}")
    rendered = render_qwen36_upstream_template(
        messages,
        add_generation_prompt=False,
        enable_thinking=True,
    )
    located = locate_target_char_spans(rendered, messages, targets)
    encoding = tokenizer(
        rendered,
        add_special_tokens=False,
        return_attention_mask=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"]
    attention_mask = encoding.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    offsets = _as_list(encoding["offset_mapping"][0])
    input_device = next(model.parameters()).device
    input_ids = input_ids.to(input_device)
    attention_mask = attention_mask.to(input_device)

    activations = forward_pre_mlp_hidden_states(
        model,
        input_ids,
        attention_mask,
        layers=(layer,),
        allow_hidden_state_fallback=allow_hidden_state_fallback,
    )[layer][0]
    target_outputs: list[dict[str, Any]] = []
    for target in located:
        indices = _token_span(offsets, int(target["char_start"]), int(target["char_end"]))
        probabilities = np.asarray(
            probe.classifier.predict_proba(activations[indices]), dtype=np.float64
        )
        if probabilities.shape != (len(indices), len(probe.role_space)):
            raise RoleProbeError(
                f"unexpected probability shape {probabilities.shape} for {len(indices)} tokens"
            )
        role = str(target["role"])
        if role not in probe.roles_map:
            raise RoleProbeError(f"target role absent from probe: {role!r}")
        hard = probabilities.argmax(axis=1)
        true_index = probe.roles_map[role]
        cap = PAPER_TOKEN_CAPS.get(role, DEFAULT_OTHER_TOKEN_CAP)
        modes: dict[str, Any] = {}
        for mode, keep in (("paper_capped", min(cap, len(indices))), ("all_tokens", len(indices))):
            selected = probabilities[:keep]
            selected_hard = hard[:keep]
            modes[mode] = {
                "token_count": int(keep),
                "mean_probability": {
                    predicted_role: float(selected[:, predicted_index].mean())
                    for predicted_role, predicted_index in probe.roles_map.items()
                },
                "top_role_fraction": {
                    predicted_role: float((selected_hard == predicted_index).mean())
                    for predicted_role, predicted_index in probe.roles_map.items()
                },
                "hard_accuracy": float((selected_hard == true_index).mean()),
                "mean_true_role_probability": float(selected[:, true_index].mean()),
                # Retained temporarily for exact aggregation, removed before JSON output.
                "_probabilities": selected,
            }
        target_outputs.append(
            {
                "role": role,
                "message_index": int(target["message_index"]),
                "field": target["field"],
                "full_token_start": int(min(indices)),
                "full_token_end": int(max(indices) + 1),
                "full_token_count": len(indices),
                "paper_token_cap": cap,
                "modes": modes,
            }
        )

    rendered_token_count = int(input_ids.shape[1])
    del activations, input_ids, attention_mask
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "schema_version": "1.0",
        "case_id": row.get("case_id"),
        "source": row.get("source"),
        "rendered_prompt_sha256": prompt_sha256(rendered),
        "rendered_token_count": rendered_token_count,
        "targets": target_outputs,
        "provenance": row.get("provenance"),
    }


def summarize(
    outputs: list[dict[str, Any]],
    probe: TrainedRoleProbe,
    mode: str,
) -> dict[str, Any]:
    import numpy as np  # type: ignore

    role_order = tuple(probe.role_space)
    evaluated_roles = [role for role in role_order if any(
        target["role"] == role for output in outputs for target in output["targets"]
    )]
    per_role: dict[str, Any] = {}
    hard_rows: list[list[float]] = []
    soft_rows: list[list[float]] = []
    all_correct: list[Any] = []
    for true_role in evaluated_roles:
        target_modes = [
            target["modes"][mode]
            for output in outputs
            for target in output["targets"]
            if target["role"] == true_role
        ]
        probabilities = np.concatenate(
            [target_mode["_probabilities"] for target_mode in target_modes], axis=0
        )
        hard = probabilities.argmax(axis=1)
        true_index = probe.roles_map[true_role]
        hard_fraction = [float((hard == index).mean()) for index in range(len(role_order))]
        mean_probability = [float(probabilities[:, index].mean()) for index in range(len(role_order))]
        hard_rows.append(hard_fraction)
        soft_rows.append(mean_probability)
        correct = hard == true_index
        all_correct.append(correct)
        per_role[true_role] = {
            "example_count": len(target_modes),
            "token_count": int(probabilities.shape[0]),
            "hard_accuracy": float(correct.mean()),
            "mean_true_role_probability": float(probabilities[:, true_index].mean()),
            "per_example_mean_hard_accuracy": float(
                np.mean([target_mode["hard_accuracy"] for target_mode in target_modes])
            ),
            "per_example_mean_true_role_probability": float(
                np.mean([
                    target_mode["mean_true_role_probability"] for target_mode in target_modes
                ])
            ),
            "mean_probability": dict(zip(role_order, mean_probability)),
            "top_role_fraction": dict(zip(role_order, hard_fraction)),
        }
    role_hard = [per_role[role]["hard_accuracy"] for role in evaluated_roles]
    role_soft = [per_role[role]["mean_true_role_probability"] for role in evaluated_roles]
    example_hard = [
        target["modes"][mode]["hard_accuracy"]
        for output in outputs
        for target in output["targets"]
    ]
    example_soft = [
        target["modes"][mode]["mean_true_role_probability"]
        for output in outputs
        for target in output["targets"]
    ]
    return {
        "mode": mode,
        "evaluated_roles": evaluated_roles,
        "unevaluated_probe_roles": [role for role in role_order if role not in evaluated_roles],
        "probe_role_order": list(role_order),
        "token_count": int(sum(len(values) for values in all_correct)),
        "target_span_count": len(example_hard),
        "micro_hard_accuracy": float(np.concatenate(all_correct).mean()),
        "macro_role_hard_accuracy": float(np.mean(role_hard)),
        "macro_role_mean_true_role_probability": float(np.mean(role_soft)),
        "macro_target_span_hard_accuracy": float(np.mean(example_hard)),
        "macro_target_span_mean_true_role_probability": float(np.mean(example_soft)),
        "per_role": per_role,
        "confusion_matrix": {
            "true_role_order": evaluated_roles,
            "predicted_role_order": list(role_order),
            "hard_argmax_fraction": hard_rows,
            "mean_probability": soft_rows,
        },
    }


def strip_temporary_arrays(outputs: list[dict[str, Any]]) -> None:
    for output in outputs:
        for target in output["targets"]:
            for values in target["modes"].values():
                values.pop("_probabilities", None)


def plot_confusion(summary: dict[str, Any], output_path: Path, *, layer: int) -> None:
    import matplotlib.pyplot as plt  # type: ignore
    import numpy as np  # type: ignore

    matrix_info = summary["confusion_matrix"]
    true_roles = matrix_info["true_role_order"]
    predicted_roles = matrix_info["predicted_role_order"]
    matrices = (
        ("Hard argmax fraction", np.asarray(matrix_info["hard_argmax_fraction"])),
        ("Mean probe probability", np.asarray(matrix_info["mean_probability"])),
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)
    for axis, (title, matrix) in zip(axes, matrices):
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto")
        axis.set_title(title, fontsize=13, weight="bold")
        axis.set_xticks(range(len(predicted_roles)), predicted_roles, rotation=35, ha="right")
        axis.set_yticks(range(len(true_roles)), true_roles)
        axis.set_xlabel("Probe prediction")
        axis.set_ylabel("True architectural role")
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = float(matrix[row_index, column_index])
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.1%}",
                    ha="center",
                    va="center",
                    color="white" if value >= 0.55 else "#172033",
                    fontsize=9,
                )
    fig.colorbar(image, ax=axes, shrink=0.85, label="Fraction / probability")
    fig.suptitle(
        f"SecOPD layer-{layer} role probe: zero-shot role recovery",
        fontsize=15,
        weight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


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
    parser.add_argument("--allow-hidden-state-fallback", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)

    rows = read_jsonl(args.inputs_jsonl)
    if not rows:
        raise RoleProbeError("zero-shot input set is empty")
    if any(row.get("provenance", {}).get("forged_text_used_as_ground_truth") is not False for row in rows):
        raise RoleProbeError("every zero-shot row must attest that forged text is not ground truth")
    probe, probe_sha256 = load_probe(args.probe_artifact, args.layer)
    model, tokenizer, device = _load_transformers(
        args.model, args.revision, args.tokenizer, args.device
    )

    outputs: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        print(f"[{index}/{len(rows)}] evaluating {row.get('case_id')}", flush=True)
        outputs.append(
            evaluate_row(
                row,
                model=model,
                tokenizer=tokenizer,
                probe=probe,
                layer=args.layer,
                allow_hidden_state_fallback=args.allow_hidden_state_fallback,
            )
        )

    summaries = {
        mode: summarize(outputs, probe, mode) for mode in ("paper_capped", "all_tokens")
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.plot:
        plot_confusion(
            summaries["paper_capped"],
            args.output_dir / "confusion_matrix.png",
            layer=args.layer,
        )
    strip_temporary_arrays(outputs)
    cases_path = args.output_dir / "per_case.jsonl"
    cases_path.write_text(
        "".join(json.dumps(output, ensure_ascii=False, sort_keys=True) + "\n" for output in outputs),
        encoding="utf-8",
    )
    results = {
        "schema_version": "1.0",
        "validity_test": "zero_shot_role_identification_in_real_unmodified_traces",
        "input_path": str(args.inputs_jsonl),
        "input_sha256": hashlib.sha256(args.inputs_jsonl.read_bytes()).hexdigest(),
        "probe_artifact": str(args.probe_artifact),
        "probe_sha256": probe_sha256,
        "probe_layer": args.layer,
        "probe_roles": list(probe.role_space),
        "probe_neutral_heldout_accuracy": probe.test_accuracy,
        "model": args.model,
        "model_revision": args.revision,
        "tokenizer": args.tokenizer or args.model,
        "device": device,
        "activation_source": "pre_mlp_post_attention_layernorm",
        "template_contract": local_template_contract(),
        "row_count": len(outputs),
        "source_counts": {
            source: sum(output["source"] == source for output in outputs)
            for source in sorted({output["source"] for output in outputs})
        },
        "forged_text_used_as_ground_truth": False,
        "primary_mode": "paper_capped",
        "paper_token_caps": {"user": 100, "other_roles": DEFAULT_OTHER_TOKEN_CAP},
        "summaries": summaries,
    }
    results_path = args.output_dir / "results.json"
    results_path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata = {
        "results_sha256": hashlib.sha256(results_path.read_bytes()).hexdigest(),
        "per_case_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "confusion_matrix_sha256": (
            hashlib.sha256((args.output_dir / "confusion_matrix.png").read_bytes()).hexdigest()
            if args.plot
            else None
        ),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RoleProbeError) as exc:
        print(f"evaluate_role_probe_zero_shot.py: error: {exc}", file=sys.stderr)
        raise SystemExit(2)
