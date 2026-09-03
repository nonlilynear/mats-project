"""Atomic, resumable artifact storage.

Completed records are immutable files keyed by :class:`RequestKey`.  Every
attempt is additionally appended to an attempt log, so a retry cannot erase
the original failure or completion.  Writes use a temporary sibling followed
by ``os.replace`` and an advisory lock for multi-process runners.
"""

import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Type, TypeVar

from pydantic import BaseModel

from .data import canonical_json, sha256_path
from .schemas import (
    AgentEpisodeRecord,
    EpisodeRequest,
    GenerationRecord,
    JudgmentRecord,
    RequestKey,
    RunManifest,
    model_to_dict,
)


T = TypeVar("T", bound=BaseModel)


class StorageError(RuntimeError):
    """Base error for artifact persistence problems."""


class DuplicateRecordError(StorageError):
    """Raised when a different completed payload uses an existing key."""


class ArchiveVerificationError(StorageError):
    """Raised when required archive invariants do not hold."""


_LOCKS: Dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.RLock())


@contextmanager
def _advisory_lock(lock_path: Path) -> Iterator[None]:
    """Lock writes across processes where ``fcntl`` is available."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = lock_path.open("a+")
    try:
        try:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        except ImportError:  # pragma: no cover - Windows fallback
            pass
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        except ImportError:  # pragma: no cover
            pass
        stream.close()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(Path(path), (canonical_json(_as_jsonable(value)) + "\n").encode("utf-8"))


def _as_jsonable(value: Any) -> Dict[str, Any]:
    if isinstance(value, BaseModel):
        return model_to_dict(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"expected Pydantic model or mapping, got {type(value)!r}")


class AtomicRecordStore:
    """Immutable completed records plus append-only attempt history."""

    def __init__(self, root: Path, record_type: Optional[Type[T]] = None) -> None:
        self.root = Path(root)
        self.records_dir = self.root / "records"
        self.attempts_path = self.root / "attempts.jsonl"
        self.lock_path = self.root / ".store.lock"
        self.record_type = record_type
        self.records_dir.mkdir(parents=True, exist_ok=True)

    def has(self, key: str) -> bool:
        return (self.records_dir / f"{_safe_filename(key)}.json").exists()

    def get(self, key: str) -> Optional[T]:
        path = self.records_dir / f"{_safe_filename(key)}.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return self.record_type.model_validate(payload) if self.record_type else payload  # type: ignore

    def completed_keys(self) -> set[str]:
        return {path.stem for path in self.records_dir.glob("*.json")}

    def write_completed(self, key: str, record: Any) -> bool:
        """Persist a completion; return False for an identical idempotent write."""
        safe_key = _safe_filename(key)
        destination = self.records_dir / f"{safe_key}.json"
        payload = _as_jsonable(record)
        encoded = (canonical_json(payload) + "\n").encode("utf-8")
        with _lock_for(self.lock_path), _advisory_lock(self.lock_path):
            if destination.exists():
                existing = destination.read_bytes()
                if existing == encoded:
                    return False
                raise DuplicateRecordError(f"completed record already exists for {key}")
            atomic_write_bytes(destination, encoded)
        return True

    def append_attempt(self, record: Any) -> None:
        payload = _as_jsonable(record)
        line = (canonical_json(payload) + "\n").encode("utf-8")
        with _lock_for(self.lock_path), _advisory_lock(self.lock_path):
            existing = self.attempts_path.read_bytes() if self.attempts_path.exists() else b""
            atomic_write_bytes(self.attempts_path, existing + line)

    def iter_completed(self) -> Iterator[Any]:
        for path in sorted(self.records_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            yield self.record_type.model_validate(payload) if self.record_type else payload  # type: ignore

    def iter_attempts(self) -> Iterator[Any]:
        if not self.attempts_path.exists():
            return
        with self.attempts_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    payload = json.loads(line)
                    yield self.record_type.model_validate(payload) if self.record_type else payload  # type: ignore

    def verify(self) -> Dict[str, Any]:
        errors = []
        for path in sorted(self.records_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if self.record_type:
                    self.record_type.model_validate(payload)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                errors.append({"path": str(path), "error": str(exc)})
        return {
            "ok": not errors,
            "record_count": len(list(self.records_dir.glob("*.json"))),
            "attempt_log": str(self.attempts_path) if self.attempts_path.exists() else None,
            "errors": errors,
        }


class RunArtifactStore:
    """Convenience facade for the canonical ``runs/<run-id>`` layout."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._requests = AtomicRecordStore(self.run_dir / "request_records", EpisodeRequest)
        self._generations = AtomicRecordStore(self.run_dir / "generation_records", GenerationRecord)
        self._agent_generations = AtomicRecordStore(
            self.run_dir / "agent_generation_records", AgentEpisodeRecord
        )
        self._judgments = AtomicRecordStore(self.run_dir / "judgment_records", JudgmentRecord)

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def archive_manifest_path(self) -> Path:
        """Path of the separate SHA-256 integrity seal."""
        return self.run_dir / "archive_manifest.json"

    def write_manifest(self, manifest: RunManifest) -> None:
        if manifest.run_id != self.run_dir.name:
            raise StorageError("manifest run_id must match the run directory name")
        atomic_write_json(self.manifest_path, manifest)

    def seal_archive(
        self,
        *,
        required_files: Optional[Iterable[str]] = None,
        expected_records: Optional[Mapping[str, int]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        """Write an integrity manifest without replacing ``RunManifest``.

        The seal excludes itself from its file list and therefore remains
        stable under byte-level verification.  Call this only after all run
        artifacts for the current batch have been flushed.
        """
        from .archive import ARCHIVE_MANIFEST_NAME, create_manifest, write_manifest

        supplied_metadata = dict(metadata or {})
        expected_completeness = dict(supplied_metadata.get("expected_completeness") or {})
        stores = {
            "requests": self._requests,
            "generations": self._generations,
            "agent_generations": self._agent_generations,
            "judgments": self._judgments,
        }
        for name, store in stores.items():
            keys = sorted(store.completed_keys())
            expected_completeness.setdefault(name, {"count": len(keys), "keys": keys})
        supplied_metadata["expected_completeness"] = expected_completeness
        integrity = create_manifest(
            self.run_dir,
            required_files=required_files,
            expected_records=expected_records,
            metadata=supplied_metadata,
            exclude=(ARCHIVE_MANIFEST_NAME,),
        )
        return write_manifest(
            self.run_dir,
            integrity,
            path=self.archive_manifest_path,
        )

    def read_manifest(self) -> RunManifest:
        if not self.manifest_path.exists():
            raise FileNotFoundError(self.manifest_path)
        return RunManifest.model_validate_json(self.manifest_path.read_text(encoding="utf-8"))

    def write_request(self, request: EpisodeRequest) -> bool:
        return self._requests.write_completed(request.request_key.value, request)

    def iter_requests(self) -> Iterator[EpisodeRequest]:
        return self._requests.iter_completed()

    def write_generation(self, record: GenerationRecord) -> bool:
        return self.write_generation_attempt(record)

    def write_generation_attempt(self, record: GenerationRecord, *, terminal: bool = True) -> bool:
        """Append one immutable generation attempt.

        ``terminal=False`` is used for a retryable response (currently the
        first 4096-token chat truncation).  The attempt remains in the
        append-only history, but does not occupy the completed-record slot
        that the terminal retry must claim.
        """
        self._generations.append_attempt(record)
        if terminal and record.status.value in {"complete", "invalid", "truncated", "limit_terminated"}:
            return self._generations.write_completed(record.request_key.value, record)
        return False

    def write_agent_generation(self, record: AgentEpisodeRecord) -> bool:
        self._agent_generations.append_attempt(record)
        if record.status.value in {"complete", "invalid", "truncated", "limit_terminated"}:
            return self._agent_generations.write_completed(record.request_key.value, record)
        return False

    def write_judgment(self, record: JudgmentRecord) -> bool:
        # A judgment may be revised by a human, so the generation attempt is
        # part of its identity and a new judge can be recorded separately.
        key = f"{record.request_key.value}__attempt-{record.generation_attempt}"
        return self._judgments.write_completed(key, record)

    def is_complete(self, request_key: RequestKey) -> bool:
        return self._generations.has(request_key.value) or self._agent_generations.has(
            request_key.value
        )

    def verification(self, *, require_manifest: bool = True) -> Dict[str, Any]:
        errors = []
        if require_manifest and not self.manifest_path.exists():
            errors.append(f"missing {self.manifest_path.name}")
        stores = {
            "requests": self._requests.verify(),
            "generations": self._generations.verify(),
            "agent_generations": self._agent_generations.verify(),
            "judgments": self._judgments.verify(),
        }
        for name, result in stores.items():
            if not result["ok"]:
                errors.append(f"invalid records in {name}")
        if self.manifest_path.exists():
            try:
                run_manifest = self.read_manifest()
                expected_requests = run_manifest.counts.get("expected_requests")
                if expected_requests is not None:
                    actual_requests = int(stores["requests"]["record_count"])
                    actual_terminal = int(stores["generations"]["record_count"]) + int(
                        stores["agent_generations"]["record_count"]
                    )
                    if actual_requests != expected_requests:
                        errors.append(
                            f"incomplete requests: expected {expected_requests}, got {actual_requests}"
                        )
                    if actual_terminal != expected_requests:
                        errors.append(
                            f"incomplete generations: expected {expected_requests}, got {actual_terminal}"
                        )
            except Exception as exc:
                errors.append(f"invalid manifest completeness metadata: {exc}")
        # A seal records the expected logical counts and key sets.  This
        # lightweight check catches a run that was sealed before all request
        # or generation records had been flushed, even when every remaining
        # JSON object is individually valid.
        if self.archive_manifest_path.exists():
            try:
                import json

                seal = json.loads(self.archive_manifest_path.read_text(encoding="utf-8"))
                expected = (seal.get("metadata") or {}).get("expected_completeness", {})
                actual_dirs = {
                    "requests": self._requests,
                    "generations": self._generations,
                    "agent_generations": self._agent_generations,
                    "judgments": self._judgments,
                }
                for name, requirement in expected.items():
                    store = actual_dirs.get(name)
                    if store is None or not isinstance(requirement, Mapping):
                        errors.append(f"invalid expected completeness entry: {name}")
                        continue
                    expected_count = requirement.get("count")
                    actual_keys = store.completed_keys()
                    if expected_count is not None and len(actual_keys) != int(expected_count):
                        errors.append(
                            f"incomplete {name}: expected {expected_count} completed records, got {len(actual_keys)}"
                        )
                    expected_keys = {str(key) for key in requirement.get("keys", ())}
                    if not expected_keys.issubset(actual_keys):
                        errors.append(f"incomplete {name}: expected keys are missing")
            except (OSError, ValueError, TypeError) as exc:
                errors.append(f"invalid archive completeness metadata: {exc}")
        return {"ok": not errors, "run_dir": str(self.run_dir), "errors": errors, "stores": stores}


def verify_archive(run_dir: Path, *, require_manifest: bool = True) -> Dict[str, Any]:
    """Verify a run's JSON records and manifest without modifying anything."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return {"ok": False, "run_dir": str(run_dir), "errors": ["run directory is missing"]}
    result = RunArtifactStore(run_dir).verification(require_manifest=require_manifest)
    if result["ok"] and require_manifest:
        try:
            RunArtifactStore(Path(run_dir)).read_manifest()
        except Exception as exc:  # schema and JSON errors are surfaced together
            result["ok"] = False
            result["errors"].append(f"invalid manifest: {exc}")
    return result


def _safe_filename(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in str(value))
    if not safe or safe in {".", ".."}:
        raise StorageError("record key cannot be empty or a path component")
    return safe
