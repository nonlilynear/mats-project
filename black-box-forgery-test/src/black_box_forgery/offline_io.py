"""Small JSON/JSONL readers used by the CPU-only Stage A CLI pipelines."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


def plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Read JSON, JSONL, or optional zstandard-compressed JSONL."""

    target = Path(path)
    if target.name.endswith(".jsonl.zst"):
        try:
            import zstandard as zstd  # type: ignore
        except ImportError as exc:
            raise RuntimeError("zstandard is required to read .jsonl.zst artifacts") from exc
        with target.open("rb") as stream:
            raw = zstd.ZstdDecompressor().stream_reader(stream).read()
        text = raw.decode("utf-8")
        values = [json.loads(line) for line in text.splitlines() if line.strip()]
    elif target.suffix.lower() == ".jsonl":
        values = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif target.suffix.lower() == ".json":
        payload = json.loads(target.read_text(encoding="utf-8"))
        values = payload if isinstance(payload, list) else [payload]
    else:
        raise ValueError(f"unsupported record file: {target}")
    if not all(isinstance(value, Mapping) for value in values):
        raise ValueError(f"records in {target} must be JSON objects")
    return [dict(value) for value in values]


def write_jsonl_atomic(path: str | Path, records: Iterable[Mapping[str, Any]]) -> Path:
    """Write canonical sorted JSONL through an atomic sibling replacement."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    return target


def _record_id(record: Mapping[str, Any], index: int = 0) -> str:
    for key in ("episode_id", "sample_id", "item_id", "request_id", "id"):
        if record.get(key) is not None:
            return str(record[key])
    request_key = record.get("request_key")
    if isinstance(request_key, Mapping):
        # Pydantic's RequestKey exposes ``value`` as a property but does not
        # serialize that property. Reconstruct a collision-free episode key
        # for files containing multiple conditions per dataset item.
        parts = (
            request_key.get("run_id"),
            request_key.get("victim_model_id", request_key.get("model_id")),
            request_key.get("victim_revision", request_key.get("model_revision")),
            request_key.get("dataset_item_id"),
            request_key.get("condition"),
            request_key.get("prompt_hash"),
            request_key.get("decoding_seed"),
        )
        if all(part is not None for part in parts):
            return "__".join(str(getattr(part, "value", part)) for part in parts)
        for key in ("dataset_item_id", "request_id", "value"):
            if request_key.get(key) is not None:
                return str(request_key[key])
    return f"row-{index:08d}"


def record_id(record: Mapping[str, Any], index: int = 0) -> str:
    return _record_id(record, index)


def nested_field(record: Mapping[str, Any], field: str) -> Any:
    if field in record:
        value = record[field]
    else:
        request_key = record.get("request_key")
        aliases = {
            "model": ("victim_model_id", "model_id"),
            "condition": ("condition",),
            "episode_id": ("dataset_item_id", "request_id", "value"),
        }
        names = aliases.get(field, (field,))
        if isinstance(request_key, Mapping):
            value = next((request_key[name] for name in names if name in request_key), None)
        else:
            value = next((getattr(request_key, name) for name in names if hasattr(request_key, name)), None)
    return getattr(value, "value", value)


def find_records(
    run_dir: str | Path,
    names: Iterable[str],
    *,
    include_attempts: bool = False,
) -> list[dict[str, Any]]:
    """Load a named artifact once, preferring canonical top-level files.

    Attempt logs are excluded by default because they contain duplicate
    retries.  A judging pass may opt in when an episode has only an error
    attempt and therefore still needs an explicit invalid judgment.
    """

    base = Path(run_dir)
    candidates: list[Path] = []
    wanted = tuple(names)
    for name in wanted:
        for suffix in (".jsonl", ".jsonl.zst", ".json"):
            candidate = base / f"{name}{suffix}"
            if candidate.is_file():
                candidates.append(candidate)
                break
    # The scaffold's RunArtifactStore uses <kind>_records/records/*.json.
    for path in sorted(base.rglob("*.json")):
        relative = path.relative_to(base).as_posix()
        lower = relative.lower()
        if not include_attempts and "attempt" in path.name.lower():
            continue
        if any(name.lower() in lower for name in wanted) and path.name != "manifest.json":
            candidates.append(path)
    for path in sorted(base.rglob("*.jsonl")) + sorted(base.rglob("*.jsonl.zst")):
        relative = path.relative_to(base).as_posix().lower()
        if not include_attempts and "attempt" in path.name.lower():
            continue
        if any(name.lower() in relative for name in wanted):
            candidates.append(path)
    seen: set[Path] = set()
    result: list[dict[str, Any]] = []
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        result.extend(read_records(path))
    return result


def first_existing(run_dir: str | Path, names: Iterable[str]) -> Optional[Path]:
    base = Path(run_dir)
    for name in names:
        for suffix in (".jsonl", ".jsonl.zst", ".json"):
            candidate = base / f"{name}{suffix}"
            if candidate.is_file():
                return candidate
    return None


__all__ = [
    "find_records",
    "first_existing",
    "nested_field",
    "plain",
    "read_records",
    "record_id",
    "write_jsonl_atomic",
]
