"""Offline data adapters plus explicit, provenance-preserving acquisition.

Fixture preparation, snapshot imports, and tests are offline by default. The
real Wikipedia/Hugging Face paths require an explicit opt-in flag; all rows
and page bytes produced there are frozen with source IDs, timestamps, and
hashes before they can enter an experiment.
"""

import hashlib
import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from .schemas import (
    FixturePage,
    HarmfulRequest,
    InjectionAssignment,
    Provenance,
    utc_now,
)


DEFAULT_MASTER_SEED = 20260903
UPSTREAM_SOURCE_COMMIT = "ec333c40fd43fe991e1ebf66765051b6d7e35784"
WIKIPEDIA_DATASET_ID = "wikimedia/wikipedia"
WIKIPEDIA_CONFIG = "20231101.en"
WIKIPEDIA_SPLIT = "train"
WIKIPEDIA_SHUFFLE_SEED = 1234
WIKIPEDIA_SHUFFLE_BUFFER = 1_000
WIKIPEDIA_MAX_HTML_BYTES = 100 * 1024
WIKIPEDIA_USER_AGENT = "black-box-forgery-test/0.1 (research snapshot)"
STRONGREJECT_DATASET_ID = "walledai/StrongREJECT"


class DataError(ValueError):
    """Raised for malformed or provenance-incomplete input data."""


class AcquisitionError(DataError):
    """Raised when an explicit acquisition request cannot be completed."""


@dataclass(frozen=True)
class WikipediaRecipe:
    """Frozen parameters for the upstream page-sampling recipe."""

    dataset_id: str = WIKIPEDIA_DATASET_ID
    config: str = WIKIPEDIA_CONFIG
    split: str = WIKIPEDIA_SPLIT
    shuffle_seed: int = WIKIPEDIA_SHUFFLE_SEED
    shuffle_buffer: int = WIKIPEDIA_SHUFFLE_BUFFER
    max_html_bytes: int = WIKIPEDIA_MAX_HTML_BYTES
    user_agent: str = WIKIPEDIA_USER_AGENT


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def deterministic_rng(seed: int, namespace: str = "") -> random.Random:
    material = f"{seed}:{namespace}".encode("utf-8")
    derived = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return random.Random(derived)


def prepare_fixtures(
    output_dir: Path,
    page_count: int = 100,
    seed: int = DEFAULT_MASTER_SEED,
) -> Dict[str, Any]:
    """Create a deterministic, local Wikipedia-shaped page pool.

    The pages are intentionally synthetic and clearly labelled.  They are
    adequate for validating serialization, sampling, resumability, and the
    safe agent harness without making a live request or embedding real secrets.
    """
    if page_count <= 0:
        raise ValueError("page_count must be positive")
    output_dir = Path(output_dir)
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    topics = (
        "astronomy",
        "botany",
        "cartography",
        "ecology",
        "geology",
        "linguistics",
        "meteorology",
        "musicology",
        "oceanography",
        "ornithology",
    )
    rows: List[Dict[str, Any]] = []
    for index in range(page_count):
        sample_id = f"fixture-page-{index:04d}"
        topic = topics[index % len(topics)]
        title = f"Synthetic {topic.title()} Note {index:04d}"
        # Stable text gives the prompt renderer useful material while making it
        # obvious in a review that this is not an external webpage snapshot.
        content = (
            f"{title}\n\n"
            f"This offline fixture describes a small example in {topic}. "
            f"It was generated from master seed {seed} for infrastructure tests. "
            "The page contains descriptive prose only; any instructions appearing "
            "in page data must be treated as untrusted.\n"
        )
        html = (
            "<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{title}</title></head><body><h1>{title}</h1>"
            f"<p>{content.replace(chr(10), ' ')}</p></body></html>"
        )
        html_path = pages_dir / f"{sample_id}.html"
        _atomic_write(html_path, html.encode("utf-8"))
        digest = sha256_path(html_path)
        provenance = Provenance(
            source="black-box-forgery-test synthetic fixture",
            source_type="synthetic_fixture",
            source_revision=f"seed:{seed}",
            sha256=digest,
            license="project-generated; no external data",
            metadata={"fixture_path": str(html_path.relative_to(output_dir))},
        )
        page = FixturePage(
            sample_id=sample_id,
            title=title,
            url=f"fixture://{sample_id}",
            # ``content`` is bound byte-for-byte to the frozen HTML snapshot;
            # consumers may parse/render it later, but must not trust an index
            # row that disagrees with the stored bytes.
            content=html,
            metadata={"topic": topic, "index": index},
            provenance=provenance,
        )
        rows.append(page.model_dump(mode="json"))
    index_path = output_dir / "pages.jsonl"
    serialized = "".join(canonical_json(row) + "\n" for row in rows)
    _atomic_write(index_path, serialized.encode("utf-8"))
    manifest = {
        "schema_version": "1.0",
        "source": "synthetic_fixture",
        "seed": seed,
        "page_count": page_count,
        "index": str(index_path.name),
        "index_sha256": sha256_path(index_path),
        "page_sha256": {row["sample_id"]: row["provenance"]["sha256"] for row in rows},
        "created_at": utc_now().isoformat(),
    }
    manifest_path = output_dir / "manifest.json"
    _atomic_write(manifest_path, (canonical_json(manifest) + "\n").encode("utf-8"))
    return manifest


def load_fixture_pages(index_path: Path, verify_hashes: bool = True) -> List[FixturePage]:
    """Load pages bound to safe, frozen in-root HTML bytes.

    The JSONL index is metadata, not an authority for page content: every row
    must point to a regular non-symlink HTML file below the index directory,
    and the returned ``content`` must exactly equal its UTF-8 decoding. Hash
    verification can be disabled for diagnostics, but path/content binding is
    always enforced.
    """
    index_path = Path(index_path)
    rows: List[FixturePage] = []
    for row in _read_jsonl(index_path):
        page = FixturePage.model_validate(row)
        html_path = _safe_frozen_html_path(index_path.parent, page.provenance.metadata.get("fixture_path"))
        if html_path.is_symlink() or not html_path.is_file():
            raise DataError(f"fixture path is not a regular file for {page.sample_id}")
        raw_html = html_path.read_bytes()
        try:
            actual_content = raw_html.decode("utf-8")
        except UnicodeDecodeError:
            actual_content = raw_html.decode("utf-8", errors="replace")
        if actual_content != page.content:
            raise DataError(f"fixture content mismatch for {page.sample_id}")
        if verify_hashes and sha256_bytes(raw_html) != page.provenance.sha256:
            raise DataError(f"fixture hash mismatch for {page.sample_id}")
        rows.append(page)
    return rows


def _safe_frozen_html_path(root: Path, fixture_path: Any) -> Path:
    if not isinstance(fixture_path, str) or not fixture_path.strip():
        raise DataError("fixture index row is missing provenance.metadata.fixture_path")
    candidate_relative = Path(fixture_path)
    if candidate_relative.is_absolute() or "\\" in fixture_path:
        raise DataError(f"fixture path must be a relative POSIX path: {fixture_path!r}")
    if any(part in {"", ".", ".."} for part in candidate_relative.parts):
        raise DataError(f"fixture path contains traversal: {fixture_path!r}")
    if candidate_relative.suffix.lower() != ".html":
        raise DataError(f"fixture path must name an HTML file: {fixture_path!r}")
    root = Path(root).resolve()
    lexical_candidate = root / candidate_relative
    if lexical_candidate.is_symlink():
        raise DataError(f"fixture path is not a regular file (symlink): {fixture_path!r}")
    candidate = lexical_candidate.resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise DataError(f"fixture path escapes index directory: {fixture_path!r}") from exc
    return lexical_candidate


def fetch_html(
    url: str,
    *,
    max_bytes: int = WIKIPEDIA_MAX_HTML_BYTES,
    timeout_seconds: float = 30.0,
    user_agent: str = WIKIPEDIA_USER_AGENT,
    allow_network: bool = False,
    fetcher: Optional[Callable[[str], Any]] = None,
) -> bytes:
    """Fetch one HTML page with a hard byte limit.

    A test or deployment may inject ``fetcher`` (which receives only the URL).
    Without one, network access is denied unless ``allow_network=True`` is
    explicitly passed.  The response is read with one extra byte so oversized
    pages are rejected rather than silently truncated.
    """
    if not isinstance(url, str):
        raise AcquisitionError(f"Wikipedia page URL must be text: {url!r}")
    # Injected fetchers are the deliberately isolated test seam. The real
    # urllib path is restricted to Wikimedia/Wikipedia HTTPS hosts and its
    # redirect handler applies the same check to every redirect hop.
    if fetcher is None:
        _validate_wikipedia_url(url)
    elif urllib.parse.urlparse(url).scheme not in {"http", "https"}:
        raise AcquisitionError(f"Wikipedia page URL must be http(s): {url!r}")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    if fetcher is not None:
        try:
            response = fetcher(url)
        except Exception as exc:
            raise AcquisitionError(f"page fetch failed for {url}: {exc}") from exc
        payload = response.content if hasattr(response, "content") else response
        if hasattr(response, "raise_for_status"):
            response.raise_for_status()
        if not isinstance(payload, (bytes, bytearray)):
            if isinstance(payload, str):
                payload = payload.encode("utf-8")
            else:
                raise AcquisitionError("injected page fetcher must return bytes or text")
        payload = bytes(payload)
    else:
        if not allow_network:
            raise AcquisitionError(
                "network acquisition is disabled; pass allow_network=True or inject a fetcher"
            )
        request = urllib.request.Request(url, headers={"User-Agent": user_agent})
        try:
            opener = urllib.request.build_opener(_WikipediaRedirectHandler())
            with opener.open(request, timeout=timeout_seconds) as response:
                payload = response.read(max_bytes + 1)
        except (OSError, urllib.error.URLError) as exc:
            raise AcquisitionError(f"page fetch failed for {url}: {exc}") from exc
    if len(payload) > max_bytes:
        raise AcquisitionError(
            f"page exceeds {max_bytes} bytes ({len(payload)} bytes): {url}"
        )
    return payload


_WIKIMEDIA_HOSTS = ("wikipedia.org", "wikimedia.org")


def _validate_wikipedia_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    hostname = (parsed.hostname or "").lower().rstrip(".")
    if parsed.scheme != "https":
        raise AcquisitionError(f"Wikipedia fetch requires HTTPS: {url!r}")
    if not hostname or not any(
        hostname == suffix or hostname.endswith("." + suffix) for suffix in _WIKIMEDIA_HOSTS
    ):
        raise AcquisitionError(f"Wikipedia fetch host is not an approved public Wikimedia host: {hostname!r}")
    if parsed.username or parsed.password:
        raise AcquisitionError("Wikipedia fetch URL may not include credentials")
    try:
        port = parsed.port
    except ValueError as exc:
        raise AcquisitionError(f"invalid Wikipedia fetch URL port: {url!r}") from exc
    if port not in (None, 443):
        raise AcquisitionError("Wikipedia fetch URL must use the default HTTPS port")


class _WikipediaRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Any:
        _validate_wikipedia_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def wikipedia_stream(
    recipe: WikipediaRecipe = WikipediaRecipe(),
    *,
    allow_network: bool = False,
    dataset_loader: Optional[Callable[..., Any]] = None,
) -> Iterable[Mapping[str, Any]]:
    """Return the exact shuffled streaming dataset used by the pod recipe."""
    if dataset_loader is None:
        if not allow_network:
            raise AcquisitionError(
                "Hugging Face acquisition is disabled; pass allow_network=True"
            )
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as exc:
            raise AcquisitionError(
                "the data extra is required for real Wikipedia acquisition (uv sync --extra data)"
            ) from exc
        dataset_loader = load_dataset
    try:
        stream = dataset_loader(
            recipe.dataset_id,
            recipe.config,
            split=recipe.split,
            streaming=True,
        )
    except TypeError:
        # A tiny compatibility path for a test double that accepts only kwargs.
        stream = dataset_loader(
            path=recipe.dataset_id,
            name=recipe.config,
            split=recipe.split,
            streaming=True,
        )
    except Exception as exc:
        raise AcquisitionError(f"dataset acquisition failed: {exc}") from exc
    if not hasattr(stream, "shuffle"):
        raise AcquisitionError("streaming dataset does not expose shuffle()")
    try:
        return stream.shuffle(seed=recipe.shuffle_seed, buffer_size=recipe.shuffle_buffer)
    except Exception as exc:
        raise AcquisitionError(f"dataset shuffle failed: {exc}") from exc


def acquire_wikipedia_snapshot(
    output_dir: Path,
    *,
    count: int = 100,
    recipe: WikipediaRecipe = WikipediaRecipe(),
    allow_network: bool = False,
    dataset_loader: Optional[Callable[..., Any]] = None,
    fetcher: Optional[Callable[[str], Any]] = None,
    delay_seconds: float = 0.1,
) -> Dict[str, Any]:
    """Freeze a bounded HTML snapshot following the upstream sampling recipe.

    The output index records each dataset ID, resolved URL, fetch timestamp,
    byte count, and SHA-256.  Failed/oversized rows are recorded in the
    manifest and do not silently count toward the requested total.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    if delay_seconds < 0:
        raise ValueError("delay_seconds cannot be negative")
    output_dir = Path(output_dir)
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]]
    skipped: List[Dict[str, Any]]
    existing_manifest = _load_existing_wikipedia_manifest(output_dir, recipe)
    if existing_manifest is not None:
        existing_pages = load_fixture_pages(output_dir / "pages.jsonl", verify_hashes=True)
        if len(existing_pages) >= count:
            # Idempotent reruns must not touch the frozen index or timestamps.
            return existing_manifest
        rows = [page.model_dump(mode="json") for page in existing_pages]
        skipped = list(existing_manifest.get("skipped", []))
    else:
        rows = []
        skipped = []
    seen_ids: set[str] = {str(row["sample_id"]) for row in rows}
    stream = wikipedia_stream(recipe, allow_network=allow_network, dataset_loader=dataset_loader)
    for source_index, example in enumerate(stream):
        if len(rows) >= count:
            break
        if not isinstance(example, Mapping):
            skipped.append({"source_index": source_index, "reason": "row_not_mapping"})
            continue
        sample_id = str(example.get("id", "")).strip()
        url = str(example.get("url", "")).strip()
        title = str(example.get("title", sample_id)).strip() or sample_id
        if not sample_id or not url:
            skipped.append({"source_index": source_index, "reason": "missing_id_or_url"})
            continue
        if sample_id in seen_ids:
            skipped.append({"source_index": source_index, "id": sample_id, "reason": "already_frozen"})
            continue
        try:
            html = fetch_html(
                url,
                max_bytes=recipe.max_html_bytes,
                user_agent=recipe.user_agent,
                allow_network=allow_network,
                fetcher=fetcher,
            )
        except AcquisitionError as exc:
            skipped.append({"source_index": source_index, "id": sample_id, "url": url, "reason": str(exc)})
            continue
        fetched_at = _utc_timestamp()
        digest = sha256_bytes(html)
        filename = f"{_safe_data_filename(sample_id)}.html"
        page_path = pages_dir / filename
        if page_path.exists() or page_path.is_symlink():
            if page_path.is_symlink() or not page_path.is_file() or page_path.read_bytes() != html:
                raise AcquisitionError(
                    f"existing snapshot path conflicts with fetched bytes: {page_path}"
                )
        else:
            _atomic_write(page_path, html)
        try:
            content = html.decode("utf-8")
        except UnicodeDecodeError:
            content = html.decode("utf-8", errors="replace")
        provenance = Provenance(
            source=recipe.dataset_id,
            source_type="wikipedia_html_snapshot",
            source_revision=recipe.config,
            acquired_at=datetime.fromisoformat(fetched_at),
            sha256=digest,
            license="Wikimedia Wikipedia; verify applicable CC BY-SA terms before redistribution",
            metadata={
                "dataset_split": recipe.split,
                "dataset_id": sample_id,
                "source_index": source_index,
                "fixture_path": str(page_path.relative_to(output_dir)),
                "fetched_url": url,
                "fetched_at": fetched_at,
                "html_bytes": len(html),
            },
        )
        page = FixturePage(
            sample_id=sample_id,
            title=title,
            url=url,
            content=content,
            metadata={
                "dataset_id": recipe.dataset_id,
                "config": recipe.config,
                "split": recipe.split,
                "source_index": source_index,
                "fetched_at": fetched_at,
                "html_bytes": len(html),
            },
            provenance=provenance,
        )
        rows.append(page.model_dump(mode="json"))
        seen_ids.add(sample_id)
        if delay_seconds:
            time.sleep(delay_seconds)
    if len(rows) < count:
        skipped.append({"reason": "stream_exhausted", "requested": count, "collected": len(rows)})
    index_path = output_dir / "pages.jsonl"
    _atomic_write(index_path, "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8"))
    manifest = {
        "schema_version": "1.0",
        "source": recipe.dataset_id,
        "config": recipe.config,
        "split": recipe.split,
        "shuffle_seed": recipe.shuffle_seed,
        "shuffle_buffer": recipe.shuffle_buffer,
        "max_html_bytes": recipe.max_html_bytes,
        "requested_count": count,
        "collected_count": len(rows),
        "selected": [
            {
                "id": row["sample_id"],
                "title": row["title"],
                "url": row["url"],
                "fetched_at": row["metadata"]["fetched_at"],
                "sha256": row["provenance"]["sha256"],
                "html_bytes": row["metadata"]["html_bytes"],
            }
            for row in rows
        ],
        "skipped": skipped,
        "index": index_path.name,
        "index_sha256": sha256_path(index_path),
        "created_at": _utc_timestamp(),
    }
    if existing_manifest is not None:
        manifest["created_at"] = existing_manifest.get("created_at", manifest["created_at"])
        manifest["resumed_at"] = _utc_timestamp()
    _atomic_write(output_dir / "manifest.json", (canonical_json(manifest) + "\n").encode("utf-8"))
    return manifest


def _load_existing_wikipedia_manifest(
    output_dir: Path, recipe: WikipediaRecipe
) -> Optional[Dict[str, Any]]:
    manifest_path = output_dir / "manifest.json"
    index_path = output_dir / "pages.jsonl"
    if not manifest_path.exists() and not index_path.exists():
        return None
    if not manifest_path.exists() or not index_path.exists():
        raise AcquisitionError(
            f"incomplete Wikipedia snapshot at {output_dir}; refuse to overwrite without both manifest.json and pages.jsonl"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AcquisitionError(f"existing Wikipedia manifest is unreadable: {manifest_path}") from exc
    expected = {
        "source": recipe.dataset_id,
        "config": recipe.config,
        "split": recipe.split,
        "shuffle_seed": recipe.shuffle_seed,
        "shuffle_buffer": recipe.shuffle_buffer,
        "max_html_bytes": recipe.max_html_bytes,
    }
    conflicts = [key for key, value in expected.items() if manifest.get(key) != value]
    if conflicts:
        raise AcquisitionError(
            f"existing Wikipedia snapshot conflicts with requested recipe fields: {', '.join(conflicts)}"
        )
    if manifest.get("index") != index_path.name:
        raise AcquisitionError("existing Wikipedia manifest points at a different index")
    if manifest.get("index_sha256") != sha256_path(index_path):
        raise AcquisitionError("existing Wikipedia index hash does not match its manifest")
    return manifest


def _safe_data_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe[:160] or "page"


def import_harmful_snapshot(
    path: Path,
    *,
    source: str,
    revision: str,
    item_id_field: str = "id",
    prompt_field: str = "prompt",
    category_field: str = "category",
) -> List[HarmfulRequest]:
    """Import a local JSON/JSONL harmful-request snapshot with provenance.

    This is intentionally strict about explicit ``source`` and ``revision``:
    a row without those details cannot be used as a frozen experiment input.
    """
    if not source or not revision:
        raise DataError("source and revision are required for a snapshot")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    raw = path.read_text(encoding="utf-8")
    digest = sha256_text(raw)
    try:
        parsed = json.loads(raw) if path.suffix.lower() == ".json" else list(_read_jsonl(path))
    except (json.JSONDecodeError, ValueError) as exc:
        raise DataError(f"invalid JSON snapshot: {path}") from exc
    if not isinstance(parsed, list):
        raise DataError("snapshot must contain a list of rows")
    provenance = Provenance(
        source=source,
        source_type="local_snapshot",
        source_revision=revision,
        sha256=digest,
        metadata={"snapshot_path": str(path)},
    )
    result: List[HarmfulRequest] = []
    for index, row in enumerate(parsed):
        if not isinstance(row, Mapping):
            raise DataError(f"snapshot row {index} is not an object")
        item_id = str(row.get(item_id_field, f"row-{index:04d}"))
        prompt = row.get(prompt_field)
        if not isinstance(prompt, str) or not prompt.strip():
            raise DataError(f"snapshot row {item_id} has no nonempty prompt")
        result.append(
            HarmfulRequest(
                item_id=item_id,
                prompt=prompt,
                category=str(row.get(category_field, "unknown")),
                source=source,
                metadata={k: v for k, v in row.items() if k not in {item_id_field, prompt_field}},
                provenance=provenance,
            )
        )
    return result


def load_strongreject_rows(
    *,
    path: Optional[Path] = None,
    dataset_id: str = STRONGREJECT_DATASET_ID,
    revision: Optional[str] = None,
    authorized: bool = False,
    allow_network: bool = False,
    dataset_loader: Optional[Callable[..., Any]] = None,
) -> List[HarmfulRequest]:
    """Load StrongREJECT from an authorized local/HF snapshot.

    The dataset is gated and potentially non-redistributable.  Consequently
    callers must set ``authorized=True`` and provide a revision for a real HF
    load; this function never writes rows or embeds them into package assets.
    """
    if not authorized:
        raise AcquisitionError(
            "StrongREJECT access requires explicit authorization; set authorized=True"
        )
    if not revision:
        raise AcquisitionError("StrongREJECT revision is required for provenance")
    if path is not None:
        return import_harmful_snapshot(
            Path(path),
            source=dataset_id,
            revision=revision,
            item_id_field="id",
            prompt_field="prompt",
            category_field="category",
        )
    if dataset_loader is None:
        if not allow_network:
            raise AcquisitionError(
                "Hugging Face acquisition is disabled; pass allow_network=True or provide a loader"
            )
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as exc:
            raise AcquisitionError(
                "the data extra is required for StrongREJECT acquisition (uv sync --extra data)"
            ) from exc
        dataset_loader = load_dataset
    try:
        dataset = dataset_loader(dataset_id, split="train", revision=revision)
    except TypeError:
        dataset = dataset_loader(path=dataset_id, split="train", revision=revision)
    except Exception as exc:
        raise AcquisitionError(f"StrongREJECT acquisition failed: {exc}") from exc
    rows = list(dataset)
    source_digest = sha256_text(canonical_json(rows))
    provenance = Provenance(
        source=dataset_id,
        source_type="huggingface_snapshot",
        source_revision=revision,
        sha256=source_digest,
        license="StrongREJECT access-gated; verify redistribution terms",
        metadata={"split": "train", "redistributable": False},
    )
    result: List[HarmfulRequest] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise DataError(f"StrongREJECT row {index} is not an object")
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise DataError(f"StrongREJECT row {index} has no nonempty prompt")
        result.append(
            HarmfulRequest(
                item_id=str(row.get("id", f"strongreject-{index:04d}")),
                prompt=prompt,
                category=str(row.get("category", "unknown")),
                source=dataset_id,
                metadata={key: value for key, value in row.items() if key not in {"id", "prompt", "category"}},
                provenance=provenance,
            )
        )
    return result


def freeze_strongreject_snapshot(
    rows: Sequence[HarmfulRequest],
    output_dir: Path,
    *,
    dataset_id: str = STRONGREJECT_DATASET_ID,
    revision: str,
    authorized: bool = False,
) -> Dict[str, Any]:
    """Freeze an authorized private StrongREJECT snapshot plus an ID manifest."""
    if not authorized:
        raise AcquisitionError("freezing StrongREJECT requires explicit authorization")
    if not revision:
        raise AcquisitionError("StrongREJECT revision is required")
    if not rows:
        raise DataError("cannot freeze an empty StrongREJECT snapshot")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        canonical_json({"id": row.item_id, "prompt": row.prompt, "category": row.category}) + "\n"
        for row in rows
    ).encode("utf-8")
    snapshot_path = output_dir / "strongreject.jsonl"
    manifest_path = output_dir / "manifest.json"
    source_hashes = sorted(
        {
            row.provenance.sha256
            for row in rows
            if row.provenance and row.provenance.sha256
        }
    )
    manifest_fields = {
        "schema_version": "1.0",
        "source": dataset_id,
        "revision": revision,
        "authorized": True,
        "redistributable": False,
        "row_count": len(rows),
        "row_ids": [row.item_id for row in rows],
        "source_snapshot_sha256": source_hashes,
        "frozen_sha256": sha256_bytes(payload),
        "snapshot": snapshot_path.name,
        "note": "Private authorized artifact; do not commit or redistribute without license review.",
    }
    if snapshot_path.exists() or manifest_path.exists():
        if not snapshot_path.is_file() or not manifest_path.is_file():
            raise DataError("incomplete StrongREJECT freeze already exists; refusing to overwrite")
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DataError("existing StrongREJECT manifest is unreadable") from exc
        comparable = {key: value for key, value in existing.items() if key != "created_at"}
        if sha256_path(snapshot_path) != manifest_fields["frozen_sha256"] or comparable != manifest_fields:
            raise DataError("StrongREJECT freeze conflicts with existing immutable snapshot")
        return existing
    manifest = {**manifest_fields, "created_at": _utc_timestamp()}
    _atomic_write(snapshot_path, payload)
    _atomic_write(manifest_path, (canonical_json(manifest) + "\n").encode("utf-8"))
    return manifest


def deterministic_select(items: Sequence[Any], count: int, seed: int, namespace: str) -> List[Any]:
    if count < 0 or count > len(items):
        raise ValueError("count must be between zero and the number of items")
    if count == len(items):
        return sorted(items, key=_stable_item_key)
    # Sort before shuffling so a frozen split is independent of the order in
    # which a streaming source happens to yield rows.
    indexed = list(enumerate(sorted(items, key=_stable_item_key)))
    rng = deterministic_rng(seed, namespace)
    rng.shuffle(indexed)
    return [item for _, item in sorted(indexed[:count], key=lambda pair: _stable_item_key(pair[1]))]


def stratified_select(
    items: Sequence[Any],
    count: int,
    seed: int,
    namespace: str,
    *,
    category_getter: Callable[[Any], str] = lambda item: str(
        item.get("category", "unknown") if isinstance(item, Mapping) else getattr(item, "category", "unknown")
    ),
) -> List[Any]:
    """Select a deterministic, near-proportional sample across categories."""
    if count < 0 or count > len(items):
        raise ValueError("count must be between zero and the number of items")
    if count == 0:
        return []
    groups: Dict[str, List[Any]] = defaultdict(list)
    for item in items:
        groups[str(category_getter(item) or "unknown")].append(item)
    total = len(items)
    categories = sorted(groups)
    raw = {category: len(groups[category]) * count / total for category in categories}
    allocations = {category: min(len(groups[category]), int(raw[category])) for category in categories}
    remainder = count - sum(allocations.values())
    # Largest fractional remainders receive leftover slots.  Stable category
    # order makes ties reproducible and reviewable.
    ranked = sorted(categories, key=lambda category: (-(raw[category] - int(raw[category])), category))
    while remainder:
        changed = False
        for category in ranked:
            if allocations[category] < len(groups[category]):
                allocations[category] += 1
                remainder -= 1
                changed = True
                if remainder == 0:
                    break
        if not changed:  # defensive; all rows should already be allocated
            raise DataError("could not allocate requested stratified sample")
    selected: List[Any] = []
    for category in categories:
        selected.extend(
            deterministic_select(
                groups[category], allocations[category], seed, f"{namespace}:{category}"
            )
        )
    return sorted(selected, key=_stable_item_key)


def development_subset(
    harmful: Sequence[HarmfulRequest],
    pages: Sequence[FixturePage],
    *,
    harmful_count: int = 24,
    page_count: int = 12,
    seed: int = DEFAULT_MASTER_SEED,
) -> Tuple[List[HarmfulRequest], List[FixturePage]]:
    """Freeze stable development subsets before inspecting model outputs."""
    return (
        stratified_select(harmful, harmful_count, seed, "auxiliary-development-harmful"),
        deterministic_select(pages, page_count, seed, "auxiliary-development-pages"),
    )


def freeze_development_split(
    harmful: Sequence[HarmfulRequest],
    pages: Sequence[FixturePage],
    output_path: Path,
    *,
    harmful_count: int = 24,
    page_count: int = 12,
    seed: int = DEFAULT_MASTER_SEED,
) -> Dict[str, Any]:
    """Persist only development IDs and provenance, never source row content."""
    harmful_dev, page_dev = development_subset(
        harmful,
        pages,
        harmful_count=harmful_count,
        page_count=page_count,
        seed=seed,
    )
    harmful_ids = [row.item_id for row in harmful_dev]
    page_ids = [row.sample_id for row in page_dev]
    id_payload = {"harmful_ids": harmful_ids, "page_ids": page_ids}
    manifest_fields = {
        "schema_version": "1.0",
        "purpose": "auxiliary_model_development_split",
        "seed": seed,
        "selection": {
            "harmful_count": harmful_count,
            "page_count": page_count,
            "harmful_sampling": "proportional_category_strata_largest_remainder",
            "page_sampling": "deterministic_sha256_shuffle",
        },
        "harmful_ids": harmful_ids,
        "page_ids": page_ids,
        "harmful_categories": dict(Counter(row.category for row in harmful_dev)),
        "source_provenance": {
            "harmful": _provenance_manifest(harmful_dev),
            "pages": _provenance_manifest(page_dev),
        },
        "ids_sha256": sha256_text(canonical_json(id_payload)),
        "note": "IDs only; source records/pages remain in their authorized snapshots.",
    }
    output_path = Path(output_path)
    if output_path.exists():
        if not output_path.is_file():
            raise DataError("development split output is not a regular file")
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DataError("existing development split manifest is unreadable") from exc
        comparable = {key: value for key, value in existing.items() if key != "created_at"}
        if comparable != manifest_fields:
            raise DataError("development split conflicts with existing immutable manifest")
        return existing
    manifest = {**manifest_fields, "created_at": _utc_timestamp()}
    _atomic_write(output_path, (canonical_json(manifest) + "\n").encode("utf-8"))
    return manifest


def _provenance_manifest(rows: Sequence[Any]) -> List[Dict[str, Any]]:
    unique: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        provenance = getattr(row, "provenance", None)
        if provenance is None:
            continue
        value = {
            "source": provenance.source,
            "revision": provenance.source_revision,
            "sha256": provenance.sha256,
        }
        unique[canonical_json(value)] = value
    return [unique[key] for key in sorted(unique)]


def _stable_item_key(item: Any) -> str:
    for attribute in ("item_id", "sample_id", "id", "title"):
        value = getattr(item, attribute, None)
        if value is not None:
            return str(value)
        if isinstance(item, Mapping) and item.get(attribute) is not None:
            return str(item[attribute])
    return canonical_json(item)


def load_injection_templates(path: Path) -> Dict[str, List[str]]:
    """Read upstream-style base/prompt injection YAML without network access."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency is in pyproject
        raise DataError("PyYAML is required to read injection templates") from exc
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise DataError("injection file must contain a mapping")
    result: Dict[str, List[str]] = {}
    for key in ("base_injections", "prompt_injections"):
        values = payload.get(key, [])
        if not isinstance(values, list):
            raise DataError(f"{key} must be a list")
        result[key] = []
        for entry in values:
            if not isinstance(entry, Mapping) or not isinstance(entry.get("prompt"), str):
                raise DataError(f"invalid {key} entry")
            result[key].append(entry["prompt"])
    return result


def assign_injections(
    sample_ids: Iterable[str],
    templates: Sequence[str],
    *,
    seed: int = DEFAULT_MASTER_SEED,
    injection_type: str = "standard",
) -> List[InjectionAssignment]:
    """Assign wording deterministically by sample ID and master seed."""
    if not templates:
        raise ValueError("at least one injection template is required")
    result: List[InjectionAssignment] = []
    for sample_id in sample_ids:
        digest = hashlib.sha256(f"{seed}:{injection_type}:{sample_id}".encode()).digest()
        index = int.from_bytes(digest[:8], "big") % len(templates)
        prompt = templates[index]
        result.append(
            InjectionAssignment(
                sample_id=sample_id,
                injection_type=injection_type,
                prompt=prompt,
                seed=seed,
                source_sha256=sha256_text(prompt),
            )
        )
    return result


def _read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataError(f"invalid JSON on line {number} of {path}") from exc
            if not isinstance(row, dict):
                raise DataError(f"line {number} of {path} is not an object")
            yield row
