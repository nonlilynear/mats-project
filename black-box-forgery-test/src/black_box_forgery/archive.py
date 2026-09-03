"""Run-archive manifests, hashes, and completeness verification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


# ``manifest.json`` is reserved for the canonical experimental RunManifest.
# The integrity seal must live in a separate file: an integrity manifest cannot
# hash itself, while the experimental manifest is expected to be updated with
# run metadata and schema-validated independently.
EXPERIMENT_MANIFEST_NAME = "manifest.json"
ARCHIVE_MANIFEST_NAME = "archive_manifest.json"
# Keep the historical archive API spelling, but point it at the non-canonical
# integrity seal so callers cannot accidentally overwrite RunManifest data.
MANIFEST_NAME = ARCHIVE_MANIFEST_NAME


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading an artifact into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


hash_file = sha256_file


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _line_records(path: Path) -> tuple[Optional[list[dict[str, Any]]], Optional[str]]:
    """Read JSONL for validation; return ``None`` when optional zstd is absent."""

    if path.name.endswith(".jsonl.zst"):
        try:
            import zstandard as zstd  # type: ignore
        except ImportError:
            return None, "zstandard is not installed"
        try:
            with path.open("rb") as raw:
                stream = zstd.ZstdDecompressor().stream_reader(raw)
                text = stream.read().decode("utf-8")
        except Exception as exc:  # corruption or decoding error
            return None, f"cannot decompress JSONL: {exc}"
    elif path.name.endswith(".jsonl"):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:
            return None, f"cannot read JSONL: {exc}"
    else:
        return None, None
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except (TypeError, ValueError) as exc:
            return None, f"invalid JSON on line {line_number}: {exc}"
        if not isinstance(value, dict):
            return None, f"JSONL line {line_number} is not an object"
        records.append(value)
    return records, None


def _record_count(path: Path) -> tuple[Optional[int], Optional[str], Optional[list[dict[str, Any]]]]:
    records, error = _line_records(path)
    return (len(records) if records is not None else None), error, records


@dataclass(frozen=True)
class ManifestEntry:
    path: str
    size_bytes: int
    sha256: str
    record_count: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArchiveManifest:
    schema_version: int
    created_at: str
    files: dict[str, ManifestEntry]
    required_files: tuple[str, ...] = ()
    expected_records: dict[str, int] | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "files": {path: entry.to_dict() for path, entry in sorted(self.files.items())},
            "required_files": list(self.required_files),
            "expected_records": dict(self.expected_records or {}),
            "metadata": dict(self.metadata or {}),
        }


def create_manifest(
    root: str | Path,
    *,
    required_files: Optional[Iterable[str]] = None,
    expected_records: Optional[Mapping[str, int]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    exclude: Iterable[str] = (MANIFEST_NAME,),
) -> dict[str, Any]:
    """Build a SHA-256 manifest for all regular files below ``root``.

    JSONL record counts are included when readable.  The manifest itself is
    excluded by default because a manifest cannot contain its own final hash.
    """

    base = Path(root)
    if not base.is_dir():
        raise NotADirectoryError(base)
    excluded = {str(value).replace(os.sep, "/") for value in exclude}
    entries: dict[str, ManifestEntry] = {}
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = _relative(path, base)
        if relative in excluded:
            continue
        count, _error, _records = _record_count(path)
        entries[relative] = ManifestEntry(relative, path.stat().st_size, sha256_file(path), count)
    required = tuple(sorted(str(value).replace(os.sep, "/") for value in (required_files or ())))
    expected = {str(key).replace(os.sep, "/"): int(value) for key, value in (expected_records or {}).items()}
    manifest = ArchiveManifest(
        schema_version=1,
        created_at=datetime.now(timezone.utc).isoformat(),
        files=entries,
        required_files=required,
        expected_records=expected,
        metadata=dict(metadata or {}),
    )
    return manifest.to_dict()


build_manifest = create_manifest


def write_manifest(
    root: str | Path,
    manifest: Optional[Mapping[str, Any]] = None,
    *,
    path: str | Path | None = None,
    **kwargs: Any,
) -> Path:
    """Write a manifest atomically and return its path."""

    base = Path(root)
    target = Path(path) if path is not None else base / MANIFEST_NAME
    value = dict(manifest or create_manifest(base, **kwargs))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return target


def _load_manifest(path: Path) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError:
        return None, "manifest is missing"
    except (OSError, ValueError) as exc:
        return None, f"manifest is unreadable: {exc}"
    if not isinstance(value, dict) or not isinstance(value.get("files"), (dict, list)):
        return None, "manifest has no valid files mapping"
    return value, None


def _entries(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    source = manifest.get("files", {})
    if isinstance(source, dict):
        result: dict[str, Mapping[str, Any]] = {}
        for path, entry in source.items():
            if isinstance(entry, Mapping):
                result[str(path).replace(os.sep, "/")] = entry
        return result
    result = {}
    for entry in source:
        if isinstance(entry, Mapping) and entry.get("path") is not None:
            result[str(entry["path"]).replace(os.sep, "/")] = entry
    return result


def _completed_record_keys(directory: Path) -> set[str]:
    """Return completed-record filenames from an AtomicRecordStore directory."""

    if not directory.is_dir():
        return set()
    return {path.stem for path in directory.glob("*.json") if path.is_file() and not path.is_symlink()}


@dataclass(frozen=True)
class ArchiveVerification:
    valid: bool
    manifest_present: bool
    checked_files: int
    missing: tuple[str, ...] = ()
    corrupted: tuple[str, ...] = ()
    invalid_records: tuple[str, ...] = ()
    incomplete: tuple[str, ...] = ()
    untracked: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return not self.missing and not self.incomplete

    def __bool__(self) -> bool:
        return self.valid

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"complete": self.complete}


def verify_archive(
    root: str | Path,
    manifest: Mapping[str, Any] | str | Path | None = None,
    *,
    required_files: Optional[Iterable[str]] = None,
    expected_records: Optional[Mapping[str, int | Sequence[str]]] = None,
    check_untracked: bool = False,
) -> ArchiveVerification:
    """Verify manifest hashes, required artifacts, and JSONL completeness.

    Hash/size mismatches are reported as ``corrupted``.  Missing expected
    records are reported as ``incomplete``.  The function never raises for a
    bad archive, making it suitable as a CI/CLI verification command; invalid
    arguments still raise ``ValueError`` where appropriate.
    """

    base = Path(root)
    manifest_path: Path
    value: Optional[dict[str, Any]]
    error: Optional[str]
    if isinstance(manifest, (str, Path)):
        manifest_path = Path(manifest)
        value, error = _load_manifest(manifest_path)
    elif manifest is None and base.is_file() and base.name.endswith(".json"):
        manifest_path = base
        base = base.parent
        value, error = _load_manifest(manifest_path)
    else:
        manifest_path = base / MANIFEST_NAME
        if manifest is None:
            value, error = _load_manifest(manifest_path)
        else:
            value, error = dict(manifest), None
    if value is None:
        return ArchiveVerification(False, False, 0, errors=(error or "invalid manifest",))

    entries = _entries(value)
    missing: set[str] = set()
    corrupted: set[str] = set()
    invalid_records: set[str] = set()
    incomplete: set[str] = set()
    errors: list[str] = []
    for relative, entry in entries.items():
        candidate = base / relative
        try:
            candidate.relative_to(base)
        except ValueError:
            corrupted.add(relative)
            errors.append(f"manifest path escapes archive root: {relative}")
            continue
        if not candidate.is_file() or candidate.is_symlink():
            missing.add(relative)
            continue
        expected_size = entry.get("size_bytes", entry.get("size"))
        expected_hash = entry.get("sha256", entry.get("hash"))
        if expected_size is not None and candidate.stat().st_size != int(expected_size):
            corrupted.add(relative)
        if expected_hash and sha256_file(candidate) != str(expected_hash):
            corrupted.add(relative)
        records, parse_error = _line_records(candidate)
        if parse_error and candidate.name.endswith((".jsonl", ".jsonl.zst")):
            # A missing optional zstandard dependency prevents completeness
            # checking but does not make a byte-identical archive corrupt.
            if "not installed" not in parse_error:
                invalid_records.add(relative)
                errors.append(f"{relative}: {parse_error}")
        expected_count = entry.get("record_count")
        if expected_count is not None and records is not None and len(records) != int(expected_count):
            incomplete.add(relative)

    required = {str(path).replace(os.sep, "/") for path in value.get("required_files", ())}
    required.update(str(path).replace(os.sep, "/") for path in (required_files or ()))
    for path in required:
        if path not in entries or not (base / path).is_file():
            missing.add(path)

    expected = dict(value.get("expected_records", {}) or {})
    expected.update({str(path).replace(os.sep, "/"): count for path, count in (expected_records or {}).items()})
    for relative, requirement in expected.items():
        candidate = base / relative
        if candidate.is_dir():
            actual_keys = _completed_record_keys(candidate)
            if isinstance(requirement, int):
                if len(actual_keys) != requirement:
                    incomplete.add(relative)
            else:
                expected_ids = {str(item) for item in requirement}
                if not expected_ids.issubset(actual_keys):
                    incomplete.add(relative)
            continue
        if not candidate.is_file():
            missing.add(relative)
            continue
        records, parse_error = _line_records(candidate)
        if records is None:
            if parse_error:
                errors.append(f"{relative}: {parse_error}")
            incomplete.add(relative)
            continue
        if isinstance(requirement, int):
            if len(records) != requirement:
                incomplete.add(relative)
        else:
            expected_ids = {str(item) for item in requirement}
            actual_ids = {
                str(record.get("episode_id", record.get("item_id", record.get("id"))))
                for record in records
                if record.get("episode_id", record.get("item_id", record.get("id"))) is not None
            }
            if not expected_ids.issubset(actual_ids):
                incomplete.add(relative)

    # RunArtifactStore puts a richer request/generation completeness contract
    # in seal metadata because those records live as one JSON object per file,
    # not as a JSONL stream.  Verify it here as well as in the schema facade so
    # callers of this low-level archive module get the same answer.
    completeness = (value.get("metadata") or {}).get("expected_completeness", {})
    record_directories = {
        "requests": "request_records/records",
        "generations": "generation_records/records",
        "agent_generations": "agent_generation_records/records",
        "judgments": "judgment_records/records",
    }
    if isinstance(completeness, Mapping):
        for name, requirement in completeness.items():
            directory_name = record_directories.get(str(name))
            if directory_name is None or not isinstance(requirement, Mapping):
                errors.append(f"invalid expected completeness entry: {name}")
                continue
            actual_keys = _completed_record_keys(base / directory_name)
            expected_count = requirement.get("count")
            if expected_count is not None and len(actual_keys) != int(expected_count):
                incomplete.add(directory_name)
            expected_keys = {str(key) for key in requirement.get("keys", ())}
            if not expected_keys.issubset(actual_keys):
                incomplete.add(directory_name)

    untracked: set[str] = set()
    if check_untracked and base.is_dir():
        tracked = set(entries) | {MANIFEST_NAME}
        for path in base.rglob("*"):
            if path.is_file() and not path.is_symlink() and _relative(path, base) not in tracked:
                untracked.add(_relative(path, base))

    valid = not (missing or corrupted or invalid_records or incomplete or errors)
    return ArchiveVerification(
        valid,
        True,
        len(entries),
        tuple(sorted(missing)),
        tuple(sorted(corrupted)),
        tuple(sorted(invalid_records)),
        tuple(sorted(incomplete)),
        tuple(sorted(untracked)),
        tuple(errors),
    )


verify_manifest = verify_archive


__all__ = [
    "ArchiveManifest",
    "ArchiveVerification",
    "ARCHIVE_MANIFEST_NAME",
    "EXPERIMENT_MANIFEST_NAME",
    "MANIFEST_NAME",
    "ManifestEntry",
    "build_manifest",
    "canonical_json",
    "create_manifest",
    "hash_file",
    "sha256_bytes",
    "sha256_file",
    "verify_archive",
    "verify_manifest",
    "write_manifest",
]
