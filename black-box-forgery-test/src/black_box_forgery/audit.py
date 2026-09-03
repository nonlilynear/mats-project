"""Deterministic manual-audit sampling and blinding.

Sampling is performed by model/condition strata and is independent of input
ordering.  Flags that should receive 100% review are unioned into the sample
after the random allocation.  Blinding returns fresh dictionaries, leaving
the raw run records untouched.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
import hmac
import json
import math
import argparse
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


DEFAULT_STRATUM_KEYS = ("model", "condition")
DEFAULT_UNUSUAL_FIELDS = (
    "invalid",
    "invalid_parse",
    "parse_error",
    "truncated",
    "disputed",
    "unusual",
    "partial_compliance",
    "unexpected_tool_sequence",
    "automated_disagreement",
)
DEFAULT_HIDDEN_FIELDS = frozenset(
    {
        "model",
        "model_id",
        "model_name",
        "victim_model",
        "victim_model_id",
        "checkpoint",
        "checkpoint_id",
        "condition",
        "condition_id",
        "attack_condition",
        "attack_type",
        "block",
        "cell",
        "audit_stratum",
        "candidate",
        "judge_model",
        "auxiliary_model",
        "run_id",
        "victim_revision",
        "model_revision",
        "tokenizer_revision",
        # Human labels must never be exposed in the blinded queue or used to
        # overwrite the separate human-label artifact.
        "human_label",
        "human_raw",
        "human_valid",
        "human_error",
    }
)


def _deep_blind(value: Any, hidden_lower: set[str]) -> Any:
    """Recursively remove identity and human-label fields.

    Persisted records commonly place model and condition inside
    ``request_key``.  A top-level-only filter therefore is not a meaningful
    blind.  For request keys we retain only the dataset item identifier, which
    is useful to reviewers and does not reveal the checkpoint or condition.
    """

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.lower() in hidden_lower:
                continue
            if key_text == "request_key" and isinstance(item, Mapping):
                dataset_item_id = item.get("dataset_item_id")
                if dataset_item_id is not None:
                    result["dataset_item_id"] = _deep_blind(dataset_item_id, hidden_lower)
                continue
            result[key_text] = _deep_blind(item, hidden_lower)
        return result
    if isinstance(value, (list, tuple)):
        return [_deep_blind(item, hidden_lower) for item in value]
    return value


def _mapping(record: Any) -> dict[str, Any]:
    if isinstance(record, Mapping):
        return dict(record)
    if is_dataclass(record) and not isinstance(record, type):
        return asdict(record)
    if hasattr(record, "__dict__"):
        return dict(vars(record))
    raise TypeError(f"audit record must be a mapping or dataclass, got {type(record)!r}")


def _record_id(record: Mapping[str, Any], index: int) -> str:
    for key in ("episode_id", "sample_id", "item_id", "request_id", "id"):
        value = record.get(key)
        if value is not None and str(value):
            return str(value)
    return f"row-{index:08d}"


def _stable_stratum(record: Mapping[str, Any], keys: Sequence[str]) -> str:
    values = [record.get(key, "<missing>") for key in keys]
    return json.dumps(values, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))


def _truthy_unusual(record: Mapping[str, Any], fields: Sequence[str]) -> bool:
    for field in fields:
        value = record.get(field)
        if value is True:
            return True
        if field in {"invalid", "invalid_parse", "parse_error"} and value not in (None, False, "", 0, [], {}):
            return True
    # Invalid automated labels are unusual even if callers only provide the
    # parsed fields rather than an explicit invalid flag.
    if record.get("automated_valid") is False or record.get("human_valid") is False:
        return True
    return False


def _score(seed: int | str, stratum: str, record_id: str) -> int:
    payload = f"{seed}\x00{stratum}\x00{record_id}".encode("utf-8")
    return int.from_bytes(sha256(payload).digest(), "big")


@dataclass(frozen=True)
class AuditSelection:
    """One selected record plus reproducibility metadata."""

    audit_id: str
    episode_id: str
    stratum: str
    reasons: tuple[str, ...]
    record: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.record)
        result.update(
            {
                "audit_id": self.audit_id,
                "audit_stratum": self.stratum,
                "audit_reasons": list(self.reasons),
            }
        )
        return result


def _selection_id(seed: int | str, record_id: str) -> str:
    digest = sha256(f"audit\x00{seed}\x00{record_id}".encode("utf-8")).hexdigest()[:20]
    return f"audit-{digest}"


def select_audit_sample(
    records: Iterable[Any],
    *,
    fraction: float = 0.10,
    seed: int | str = 20260903,
    stratify_by: Sequence[str] = DEFAULT_STRATUM_KEYS,
    unusual_fields: Sequence[str] = DEFAULT_UNUSUAL_FIELDS,
    include_unusual: bool = True,
    include_ids: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Select a deterministic stratified audit sample.

    ``ceil(fraction * stratum_size)`` records are selected from each stratum;
    a positive fraction always selects at least one record from a non-empty
    stratum.  Explicitly unusual/disputed records and ``include_ids`` are then
    added.  Selection uses a hash score, so reordering the input does not
    change the result.
    """

    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")
    materialized = [_mapping(item) for item in records]
    indexed = [(_record_id(item, i), i, item) for i, item in enumerate(materialized)]
    by_stratum: dict[str, list[tuple[str, int, dict[str, Any]]]] = {}
    for record_id, index, item in indexed:
        by_stratum.setdefault(_stable_stratum(item, stratify_by), []).append((record_id, index, item))

    selected: dict[str, tuple[dict[str, Any], str, set[str], int]] = {}
    for stratum, rows in by_stratum.items():
        count = math.ceil(fraction * len(rows)) if fraction else 0
        if fraction > 0 and rows:
            count = max(1, count)
        ranked = sorted(rows, key=lambda row: (_score(seed, stratum, row[0]), row[0]))
        for record_id, index, item in ranked[:count]:
            selected[record_id] = (item, stratum, {"stratified_random"}, index)

    requested = {str(value) for value in include_ids}
    for record_id, index, item in indexed:
        reasons: set[str] = set()
        if include_unusual and _truthy_unusual(item, unusual_fields):
            reasons.add("unusual_or_invalid")
        if record_id in requested:
            reasons.add("explicit")
        if reasons:
            if record_id in selected:
                selected[record_id][2].update(reasons)
            else:
                selected[record_id] = (item, _stable_stratum(item, stratify_by), reasons, index)

    results: list[AuditSelection] = []
    for record_id, (item, stratum, reasons, index) in selected.items():
        results.append(
            AuditSelection(
                audit_id=_selection_id(seed, record_id),
                episode_id=record_id,
                stratum=stratum,
                reasons=tuple(sorted(reasons)),
                record=item,
            )
        )
    # Stable output order is useful for review and deterministic JSONL files.
    results.sort(key=lambda item: item.audit_id)
    return [item.to_dict() for item in results]


def stratified_audit_sample(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    """Alias with a name matching the experiment-plan terminology."""

    return select_audit_sample(*args, **kwargs)


def _blind_digest(source_id: str, seed: int | str, secret: str | bytes | None) -> str:
    payload = f"{seed}\x00{source_id}".encode("utf-8")
    if secret is None:
        return sha256(payload).hexdigest()[:24]
    key = secret.encode("utf-8") if isinstance(secret, str) else secret
    return hmac.new(key, payload, sha256).hexdigest()[:24]


def blind_record(
    record: Mapping[str, Any] | Any,
    *,
    seed: int | str = 20260903,
    secret: str | bytes | None = None,
    hidden_fields: Iterable[str] = DEFAULT_HIDDEN_FIELDS,
) -> dict[str, Any]:
    """Return an opaque audit copy with model/condition identity removed."""

    source = _mapping(record)
    source_id = _record_id(source, 0)
    hidden = {str(field).lower() for field in hidden_fields}
    clean = _deep_blind(source, hidden)
    if not isinstance(clean, dict):  # defensive; source is always a mapping
        raise TypeError("blinded audit record must remain a mapping")
    # A serialized request-key value often embeds model and condition.  The
    # opaque audit ID is the reviewer-visible join key instead.
    clean.pop("episode_id", None)
    clean.pop("audit_id", None)
    clean["audit_id"] = f"blind-{_blind_digest(source_id, seed, secret)}"
    clean["blinded"] = True
    return clean


def blind_records(
    records: Iterable[Mapping[str, Any] | Any],
    *,
    seed: int | str = 20260903,
    secret: str | bytes | None = None,
    hidden_fields: Iterable[str] = DEFAULT_HIDDEN_FIELDS,
) -> list[dict[str, Any]]:
    """Blind records and sort by opaque ID for a stable reviewer queue."""

    blinded = [blind_record(record, seed=seed, secret=secret, hidden_fields=hidden_fields) for record in records]
    blinded.sort(key=lambda item: str(item["audit_id"]))
    return blinded


def write_blinded_jsonl(path: str | Path, records: Iterable[Mapping[str, Any] | Any], **kwargs: Any) -> int:
    """Write a blinded queue; returns the number of records written."""

    rows = blind_records(records, **kwargs)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")
    return len(rows)


def _cli_join_for_audit(run_dir: Path) -> list[dict[str, Any]]:
    from .offline_io import find_records, record_id

    judgments = find_records(run_dir, ("judgment", "judgments"))
    generations = find_records(run_dir, ("generation", "generations"))
    by_id = {record_id(row, index): dict(row) for index, row in enumerate(generations)}
    rows: list[dict[str, Any]] = []
    source = judgments or generations
    for index, value in enumerate(source):
        row = dict(by_id.get(record_id(value, index), {}))
        row.update(dict(value))
        row.setdefault("episode_id", record_id(value, index))
        rows.append(row)
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Export a deterministic blinded audit queue from a local run."""

    parser = argparse.ArgumentParser(prog="bbf export-audit")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--secret", default=None, help="optional local blinding secret")
    args = parser.parse_args(list(argv) if argv is not None else None)
    from .offline_io import write_jsonl_atomic

    rows = _cli_join_for_audit(args.run_dir)
    selected = select_audit_sample(rows, fraction=args.fraction, seed=args.seed)
    blinded = blind_records(selected, seed=args.seed, secret=args.secret)
    output = args.output or args.run_dir / "review" / "audit_queue.jsonl"
    write_jsonl_atomic(output, blinded)
    manifest = {
        "schema_version": 1,
        "source_run": str(args.run_dir),
        "output": str(output),
        "fraction": args.fraction,
        "seed": args.seed,
        "records_attempted": len(rows),
        "records_selected": len(selected),
        "records_exported": len(blinded),
        "human_labels_path": str(args.run_dir / "human_labels.jsonl"),
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(manifest_path)
    print(json.dumps(manifest, sort_keys=True))
    return 0


__all__ = [
    "AuditSelection",
    "DEFAULT_STRATUM_KEYS",
    "blind_record",
    "blind_records",
    "select_audit_sample",
    "stratified_audit_sample",
    "write_blinded_jsonl",
    "main",
]
