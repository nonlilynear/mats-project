#!/usr/bin/env python3
"""Train SecOPD-adapted Qwen role probes.

Examples:

  # Use a frozen local neutral-text JSONL (one object with a ``text`` field
  # per line) and a local/Hugging Face checkpoint:
  python scripts/train_role_probes.py \
    --model pybbb/Qwen3.6-27B-SecOPD \
    --texts-jsonl data/probe/neutral.jsonl \
    --output-dir runs/role-probes/secopd

  # For an explicit reproduction-data fetch, use --dataset both and opt in:
  python scripts/train_role_probes.py --dataset both --allow-network ...

The script never uses target pages, injections, or attack outcomes to train a
probe.  Those are projection/evaluation inputs, not role-probe training data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from black_box_forgery.rendering import local_template_contract  # noqa: E402
from black_box_forgery.role_probes import (  # noqa: E402
    DEFAULT_ROLE_SPACE,
    ProbeTrainingConfig,
    RoleProbeError,
    build_probe_examples,
    save_probe_artifact,
    train_role_probes,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local path")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--tokenizer", default=None, help="optional tokenizer ID/path")
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--texts-jsonl", type=Path, default=None)
    parser.add_argument(
        "--dataset",
        choices=("none", "c4", "dolma3", "both"),
        default="none",
        help="optional streamed pretraining source; requires --allow-network",
    )
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-texts", type=int, default=250)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument(
        "--fit-backend",
        choices=("sklearn", "torch"),
        default="sklearn",
        help="probe fitter; torch uses a GPU-native softmax linear classifier",
    )
    parser.add_argument("--fit-epochs", type=int, default=10)
    parser.add_argument("--fit-batch-size", type=int, default=8192)
    parser.add_argument("--fit-learning-rate", type=float, default=0.05)
    parser.add_argument("--fit-device", default="auto")
    parser.add_argument(
        "--layers",
        default="auto",
        help="comma-separated zero-based decoder layers, or auto (every fourth layer)",
    )
    parser.add_argument(
        "--roles",
        default=",".join(DEFAULT_ROLE_SPACE),
        help="comma-separated role classes; default includes SecOPD input",
    )
    parser.add_argument(
        "--allow-hidden-state-fallback",
        action="store_true",
        help="use post-block hidden_states if pre-MLP hooks are unavailable",
    )
    parser.add_argument("--device", default="auto")
    return parser


def _read_jsonl(path: Path, limit: int) -> list[str]:
    if not path.is_file():
        raise RoleProbeError(f"neutral-text JSONL does not exist: {path}")
    texts: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RoleProbeError(f"invalid JSON at {path}:{line_number}") from exc
        text = row.get("text") if isinstance(row, dict) else row
        if isinstance(text, str) and text.strip():
            texts.append(text)
        if len(texts) >= limit:
            break
    if len(texts) < 2:
        raise RoleProbeError("neutral-text JSONL must contain at least two usable rows")
    return texts


def _stream_hf_texts(dataset_name: str, count: int, seed: int) -> list[str]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise RoleProbeError(
            "--dataset requires the 'datasets' package; install the role-probes extra"
        ) from exc
    if dataset_name == "c4":
        datasets = [("c4", load_dataset("allenai/c4", "en", split="validation", streaming=True))]
    elif dataset_name == "dolma3":
        datasets = [
            (
                "dolma3",
                load_dataset(
                    "allenai/dolma3_mix-150B-1025",
                    split="train",
                    revision="3a8349c",
                    streaming=True,
                ),
            )
        ]
    else:
        datasets = [
            ("c4", load_dataset("allenai/c4", "en", split="validation", streaming=True)),
            (
                "dolma3",
                load_dataset(
                    "allenai/dolma3_mix-150B-1025",
                    split="train",
                    revision="3a8349c",
                    streaming=True,
                ),
            ),
        ]
    # The public GPT-OSS notebook samples an even C4/Dolma3 mix. Keep the
    # allocation deterministic; the actual source and text hash are recorded
    # in the output metadata.
    allocations = [count] if len(datasets) == 1 else [count // 2, count - count // 2]
    texts: list[str] = []
    for (_, dataset), allocation in zip(datasets, allocations):
        shuffled = dataset.shuffle(seed=seed, buffer_size=50_000)
        dataset_count = 0
        for row in shuffled:
            text = row.get("text") if isinstance(row, dict) else None
            if isinstance(text, str) and text.strip():
                texts.append(text)
                dataset_count += 1
            if dataset_count >= allocation:
                break
    if len(texts) < 2:
        raise RoleProbeError("streamed datasets returned fewer than two text rows")
    return texts[:count]


def _parse_layers(value: str, model: Any) -> list[int]:
    depth = int(getattr(getattr(model, "config", None), "num_hidden_layers", 0))
    if value == "auto":
        if depth <= 0:
            raise RoleProbeError("--layers=auto requires config.num_hidden_layers")
        return list(range(0, depth, 4)) or [depth - 1]
    layers = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not layers:
        raise RoleProbeError("--layers must contain at least one layer index")
    return layers


def _load_transformers(model_id: str, revision: str | None, tokenizer_id: str | None, tokenizer_revision: str | None, device_name: str) -> tuple[Any, Any, Any]:
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise RoleProbeError(
            "training requires torch and transformers; install the role-probes extra"
        ) from exc
    tokenizer_kwargs: dict[str, Any] = {"use_fast": True}
    if tokenizer_revision:
        tokenizer_kwargs["revision"] = tokenizer_revision
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id or model_id, **tokenizer_kwargs)
    model_kwargs: dict[str, Any] = {}
    if revision:
        model_kwargs["revision"] = revision
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device_name).eval()
    return model, tokenizer, device_name


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.num_texts < 2:
        raise RoleProbeError("--num-texts must be at least two")
    if args.dataset != "none" and not args.allow_network:
        raise RoleProbeError("--dataset requires --allow-network")
    if args.dataset != "none" and args.texts_jsonl is not None:
        raise RoleProbeError("choose --texts-jsonl or --dataset, not both")
    if args.texts_jsonl is not None:
        texts = _read_jsonl(args.texts_jsonl, args.num_texts)
        text_source = str(args.texts_jsonl)
    elif args.dataset != "none":
        texts = _stream_hf_texts(args.dataset, args.num_texts, args.seed)
        text_source = f"hf:{args.dataset}"
    else:
        raise RoleProbeError("provide --texts-jsonl or an explicitly enabled --dataset")

    model, tokenizer, device = _load_transformers(
        args.model,
        args.revision,
        args.tokenizer,
        args.tokenizer_revision,
        args.device,
    )
    roles = tuple(role.strip() for role in args.roles.split(",") if role.strip())
    examples = build_probe_examples(
        texts,
        tokenizer,
        roles=roles,
        sequence_length=args.sequence_length,
        seed=args.seed,
    )
    layers = _parse_layers(args.layers, model)
    probes, training_metadata = train_role_probes(
        model,
        tokenizer,
        examples,
        layers=layers,
        batch_size=args.batch_size,
        config=ProbeTrainingConfig(
            seed=args.seed,
            sequence_length=args.sequence_length,
            c=args.c,
            max_iter=args.max_iter,
        ),
        allow_hidden_state_fallback=args.allow_hidden_state_fallback,
        fit_backend=args.fit_backend,
        fit_epochs=args.fit_epochs,
        fit_batch_size=args.fit_batch_size,
        fit_learning_rate=args.fit_learning_rate,
        fit_device=args.fit_device,
    )
    output_metadata = {
        **training_metadata,
        "model": args.model,
        "model_revision": args.revision,
        "tokenizer": args.tokenizer or args.model,
        "tokenizer_revision": args.tokenizer_revision,
        "device": device,
        "text_source": text_source,
        "text_source_sha256": hashlib.sha256(
            "\n".join(texts).encode("utf-8")
        ).hexdigest(),
        "template_contract": local_template_contract(),
        "adaptation": {
            "paper_roles": ["system", "user", "cot", "assistant", "tool"],
            "added_role": "input",
            "input_role_is_not_tool": True,
            "tool_serialization": "qwen_tool_message_as_tool_response_in_user_block",
        },
    }
    paths = save_probe_artifact(args.output_dir, probes, output_metadata)
    print(json.dumps({"ok": True, "paths": paths, "training": training_metadata}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RoleProbeError, OSError, ValueError) as exc:
        print(f"train_role_probes.py: error: {exc}", file=sys.stderr)
        raise SystemExit(2)
