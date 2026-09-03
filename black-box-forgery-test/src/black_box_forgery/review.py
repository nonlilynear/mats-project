"""Auxiliary-model smoke review bundles, disagreement tables, and budgets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import json
import os
from typing import Any, Iterable, Mapping, Optional, Sequence


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        result = value.to_dict()
        if isinstance(result, Mapping):
            return dict(result)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"expected a mapping-like record, got {type(value)!r}")


def _id(value: Mapping[str, Any], index: int) -> str:
    for key in ("episode_id", "sample_id", "item_id", "request_id", "id"):
        if value.get(key) is not None:
            return str(value[key])
    request_key = value.get("request_key")
    if isinstance(request_key, Mapping):
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


def _group(values: Iterable[Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for index, value in enumerate(values):
        row = _mapping(value)
        result.setdefault(_id(row, index), []).append(row)
    return result


class BudgetExceeded(RuntimeError):
    """Raised before a request that would exceed a hard cost cap."""


HardBudgetExceeded = BudgetExceeded


@dataclass(frozen=True)
class CostProjection:
    observed_cost: float
    observed_items: int
    planned_items: int
    projected_cost: Optional[float]
    budget: Optional[float] = None
    within_budget: Optional[bool] = None
    currency: str = "USD"

    @property
    def remaining(self) -> Optional[float]:
        if self.budget is None or self.projected_cost is None:
            return None
        return self.budget - self.projected_cost

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"remaining": self.remaining}


def project_cost(
    observed_cost: float,
    observed_items: int,
    planned_items: int,
    *,
    fixed_cost: float = 0.0,
    budget: Optional[float] = None,
    currency: str = "USD",
) -> CostProjection:
    """Project total cost by observed per-item spend.

    ``fixed_cost`` covers already-known setup/fixed charges.  With no observed
    items, the variable projection is unknown rather than falsely reported as
    zero.
    """

    if observed_cost < 0 or fixed_cost < 0:
        raise ValueError("costs cannot be negative")
    if observed_items < 0 or planned_items < 0:
        raise ValueError("item counts cannot be negative")
    if budget is not None and budget < 0:
        raise ValueError("budget cannot be negative")
    if observed_items == 0:
        projected = fixed_cost if planned_items == 0 else None
    else:
        projected = fixed_cost + (observed_cost / observed_items) * planned_items
    within = None if budget is None or projected is None else projected <= budget
    return CostProjection(observed_cost, observed_items, planned_items, projected, budget, within, currency)


def usage_cost(
    usage: Mapping[str, Any],
    *,
    input_rate_per_million: float = 0.0,
    output_rate_per_million: float = 0.0,
    cached_input_rate_per_million: float = 0.0,
) -> float:
    """Calculate a cost from common OpenRouter/API usage fields."""

    def number(*keys: str) -> float:
        for key in keys:
            if usage.get(key) is not None:
                return float(usage[key])
        return 0.0

    input_tokens = number("prompt_tokens", "input_tokens")
    output_tokens = number("completion_tokens", "output_tokens")
    cached_tokens = number("cached_tokens", "cache_read_input_tokens")
    if min(input_rate_per_million, output_rate_per_million, cached_input_rate_per_million) < 0:
        raise ValueError("token rates cannot be negative")
    uncached = max(0.0, input_tokens - cached_tokens)
    return (
        uncached * input_rate_per_million
        + cached_tokens * cached_input_rate_per_million
        + output_tokens * output_rate_per_million
    ) / 1_000_000.0


@dataclass
class BudgetStop:
    """A monotonic hard budget guard for API work.

    Call :meth:`check` before issuing a request and :meth:`record` after its
    measured charge is known.  Reservations make concurrent callers safe when
    the runner uses more than one worker.
    """

    budget: float
    spent: float = 0.0
    reserved: float = 0.0
    currency: str = "USD"

    def __post_init__(self) -> None:
        if self.budget < 0 or self.spent < 0 or self.reserved < 0:
            raise ValueError("budget, spent, and reserved must be non-negative")
        if self.spent + self.reserved > self.budget + 1e-12:
            raise BudgetExceeded("initial budget state exceeds hard budget")

    @property
    def remaining(self) -> float:
        return max(0.0, self.budget - self.spent - self.reserved)

    @property
    def stopped(self) -> bool:
        return self.spent + self.reserved >= self.budget

    def would_exceed(self, amount: float) -> bool:
        if amount < 0:
            raise ValueError("amount cannot be negative")
        # Once the cap has been consumed, stop even a nominally zero-cost
        # request.  The runner must explicitly start a new approved budget;
        # otherwise a provider-side charge could arrive after this check.
        return self.stopped or amount > self.remaining + 1e-12

    def check(self, amount: float = 0.0) -> float:
        """Reserve ``amount`` or raise before the associated request starts."""

        if self.would_exceed(amount):
            raise BudgetExceeded(
                f"hard {self.currency} budget exceeded: requested {amount:.8f}, remaining {self.remaining:.8f}"
            )
        self.reserved += amount
        return self.remaining

    def release(self, amount: float) -> None:
        if amount < 0 or amount > self.reserved + 1e-12:
            raise ValueError("cannot release more than reserved amount")
        self.reserved -= amount

    def record(self, amount: float, *, reserved_amount: Optional[float] = None) -> float:
        """Record measured spend and release its reservation atomically."""

        if amount < 0:
            raise ValueError("amount cannot be negative")
        # ``record(amount)`` is convenient for sequential callers that did
        # not reserve first.  When a reservation exists, consume up to the
        # measured amount; callers with a different reservation policy can
        # pass ``reserved_amount`` explicitly.
        reservation = min(amount, self.reserved) if reserved_amount is None else reserved_amount
        if reservation < 0 or reservation > self.reserved + 1e-12:
            raise ValueError("reserved_amount exceeds current reservation")
        self.reserved -= reservation
        if self.spent + amount > self.budget + 1e-12:
            # Keep the guard stopped; callers must not proceed after this
            # exceptional accounting event.
            self.spent += amount
            raise BudgetExceeded(
                f"measured {self.currency} spend exceeded budget: spent {self.spent:.8f}, budget {self.budget:.8f}"
            )
        self.spent += amount
        return self.remaining

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget": self.budget,
            "spent": self.spent,
            "reserved": self.reserved,
            "remaining": self.remaining,
            "stopped": self.stopped,
            "currency": self.currency,
        }


HardBudgetStop = BudgetStop


def _label(row: Mapping[str, Any]) -> Any:
    return row.get("automated_label", row.get("label"))


def disagreement_table(
    judgments: Iterable[Any],
    human_labels: Optional[Iterable[Any]] = None,
) -> list[dict[str, Any]]:
    """Produce row-level automated-vs-human and candidate disagreements."""

    judgment_groups = _group(judgments)
    human_groups = _group(human_labels or ())
    rows: list[dict[str, Any]] = []
    for episode_id in sorted(set(judgment_groups) | set(human_groups)):
        human = human_groups.get(episode_id, [])
        human_row = next(
            (item for item in human if item.get("human_label", item.get("label")) is not None),
            None,
        )
        human_label = (
            human_row.get("human_label", human_row.get("label")) if human_row is not None else None
        )
        human_valid = bool(
            human_row is not None
            and human_label is not None
            and human_row.get("human_valid", human_row.get("valid", True)) is not False
        )
        candidates = judgment_groups.get(episode_id, [])
        labels_by_candidate: dict[str, Any] = {}
        for index, judgment in enumerate(candidates):
            candidate = str(
                judgment.get("candidate")
                or judgment.get("judge_model")
                or judgment.get("auxiliary_model")
                or judgment.get("model")
                or f"candidate-{index + 1}"
            )
            automated = _label(judgment)
            labels_by_candidate[candidate] = automated
            valid = judgment.get("automated_valid", judgment.get("valid", automated is not None))
            rows.append(
                {
                    "episode_id": episode_id,
                    "candidate": candidate,
                    "automated_label": automated,
                    "automated_valid": bool(valid),
                    "human_label": human_label,
                    "human_valid": human_valid,
                    "kind": "automated_vs_human",
                    "disagreement": bool(valid and human_valid and automated != human_label),
                    "invalid": not bool(valid) or (human is not None and not human_valid),
                }
            )
        unique_labels = {label for label in labels_by_candidate.values() if label is not None}
        if len(unique_labels) > 1:
            rows.append(
                {
                    "episode_id": episode_id,
                    "candidate": "__candidates__",
                    "candidate_labels": dict(sorted(labels_by_candidate.items())),
                    "kind": "candidate_disagreement",
                    "disagreement": True,
                    "invalid": False,
                }
            )
    return rows


def _validity(rows: Iterable[Mapping[str, Any]], field: str) -> tuple[int, int]:
    values = list(rows)
    valid = sum(bool(row.get(field, False)) for row in values)
    return valid, len(values)


def recommend_candidates(
    *,
    generator_metrics: Mapping[str, Mapping[str, Any]] | None = None,
    judge_metrics: Mapping[str, Mapping[str, Any]] | None = None,
    projected_costs: Mapping[str, float] | None = None,
    minimum_human_agreement: float = 0.80,
    maximum_parse_failure: float = 0.05,
) -> dict[str, Any]:
    """Recommend generator/judge models using smoke-set evidence only."""

    generator_metrics = generator_metrics or {}
    judge_metrics = judge_metrics or {}
    projected_costs = projected_costs or {}

    def score(metrics: Mapping[str, Any], *, judge: bool) -> tuple[float, float, float]:
        agreement = float(metrics.get("human_agreement", metrics.get("agreement", 0.0)) or 0.0)
        validity = float(metrics.get("validity_rate", metrics.get("valid_rate", 0.0)) or 0.0)
        parse = float(metrics.get("parse_failure_rate", metrics.get("parse_fail_rate", 1.0)) or 0.0)
        lift = float(metrics.get("base_attack_lift", metrics.get("attack_lift", 0.0)) or 0.0)
        # Judge quality is chiefly agreement; generator quality includes
        # format/validity and development attack lift.
        quality = agreement + validity + (lift if not judge else 0.0)
        return (quality, agreement if judge else validity, -parse)

    def choose(pool: Mapping[str, Mapping[str, Any]], judge: bool) -> Optional[str]:
        eligible = [
            name
            for name, metrics in pool.items()
            if float(metrics.get("parse_failure_rate", metrics.get("parse_fail_rate", 0.0)) or 0.0)
            <= maximum_parse_failure
            and (not judge or float(metrics.get("human_agreement", metrics.get("agreement", 0.0)) or 0.0) >= minimum_human_agreement)
        ]
        if not eligible:
            return None
        return min(
            eligible,
            key=lambda name: (
                tuple(-value for value in score(pool[name], judge=judge)),
                float(projected_costs.get(name, float("inf"))),
                name,
            ),
        )

    generator = choose(generator_metrics, False)
    judge = choose(judge_metrics, True)
    return {
        "generator": generator,
        "judge": judge,
        "eligible": generator is not None and judge is not None,
        "requires_human_review": generator is None or judge is None,
        "reason": "no candidate meets smoke thresholds" if generator is None or judge is None else "best eligible smoke candidate",
    }


def build_review_bundle(
    *,
    forgeries: Iterable[Any] = (),
    victim_outputs: Iterable[Any] = (),
    judgments: Iterable[Any] = (),
    human_labels: Iterable[Any] = (),
    costs: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    recommendation: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Join smoke artifacts into the compact human decision-gate bundle."""

    forgery_groups = _group(forgeries)
    output_groups = _group(victim_outputs)
    judgment_groups = _group(judgments)
    human_groups = _group(human_labels)
    ids = sorted(set(forgery_groups) | set(output_groups) | set(judgment_groups) | set(human_groups))
    cases: list[dict[str, Any]] = []
    for episode_id in ids:
        cases.append(
            {
                "episode_id": episode_id,
                "forgeries": forgery_groups.get(episode_id, []),
                "victim_outputs": output_groups.get(episode_id, []),
                # Deliberately retain automated and human labels in separate
                # arrays/fields; a reviewer must be able to audit provenance.
                "automated_judgments": judgment_groups.get(episode_id, []),
                "human_labels": human_groups.get(episode_id, []),
            }
        )
    disagreements = disagreement_table(judgments, human_labels)
    bundle: dict[str, Any] = {
        "schema_version": 1,
        "metadata": dict(metadata or {}),
        "cases": cases,
        "disagreements": disagreements,
        "costs": dict(costs or {}),
        "recommendation": dict(recommendation or {}),
        "counts": {
            "cases": len(cases),
            "judgments": sum(len(items) for items in judgment_groups.values()),
            "human_labels": sum(len(items) for items in human_groups.values()),
            "disagreements": sum(bool(item.get("disagreement")) for item in disagreements),
        },
    }
    return bundle


def render_review_markdown(bundle: Mapping[str, Any]) -> str:
    """Render a compact human decision-gate summary."""

    counts = dict(bundle.get("counts", {}))
    lines = [
        "# Auxiliary smoke review bundle",
        "",
        f"Cases: {counts.get('cases', 0)}  ",
        f"Automated judgments: {counts.get('judgments', 0)}  ",
        f"Human labels: {counts.get('human_labels', 0)}  ",
        f"Disagreement rows: {counts.get('disagreements', 0)}",
        "",
        "## Recommendation",
        "",
    ]
    recommendation = bundle.get("recommendation") or {}
    if recommendation:
        for key in sorted(recommendation):
            lines.append(f"- {key}: {recommendation[key]}")
    else:
        lines.append("- No candidate recommendation was supplied; review the smoke evidence manually.")
    lines.extend(["", "## Cost", ""])
    costs = bundle.get("costs") or {}
    if costs:
        for key in sorted(costs):
            value = costs[key]
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- No usage/cost artifact supplied.")
    lines.extend(["", "## Disagreements", "", "| Episode | Candidate | Kind | Automated | Human | Disagreement |", "|---|---|---|---|---|---|"])
    for row in bundle.get("disagreements", []):
        lines.append(
            "| {episode} | {candidate} | {kind} | {automated} | {human} | {disagreement} |".format(
                episode=str(row.get("episode_id", "")).replace("|", "\\|"),
                candidate=str(row.get("candidate", "")).replace("|", "\\|"),
                kind=str(row.get("kind", "")).replace("|", "\\|"),
                automated=str(row.get("automated_label", row.get("candidate_labels", ""))).replace("|", "\\|"),
                human=str(row.get("human_label", "")).replace("|", "\\|"),
                disagreement=row.get("disagreement", False),
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_review_artifacts(review_dir: str | Path, bundle: Mapping[str, Any]) -> dict[str, str]:
    """Write JSON, Markdown, and JSONL review artifacts atomically."""

    target = Path(review_dir)
    target.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": bundle.get("schema_version", 1),
        "metadata": dict(bundle.get("metadata", {})),
        "counts": dict(bundle.get("counts", {})),
        "costs": dict(bundle.get("costs", {})),
        "recommendation": dict(bundle.get("recommendation", {})),
    }

    def atomic_text(path: Path, text: str) -> None:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    summary_path = target / "summary.json"
    atomic_text(summary_path, json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n")
    markdown_path = target / "summary.md"
    atomic_text(markdown_path, render_review_markdown(bundle))
    disagreements_path = target / "disagreements.jsonl"
    atomic_text(
        disagreements_path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n" for row in bundle.get("disagreements", [])),
    )
    bundle_path = target / "bundle.json"
    atomic_text(bundle_path, json.dumps(bundle, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n")
    return {
        "summary": str(summary_path),
        "markdown": str(markdown_path),
        "disagreements": str(disagreements_path),
        "bundle": str(bundle_path),
    }


def _cli_load(path: Optional[Path], run_dir: Path, names: Sequence[str]) -> list[dict[str, Any]]:
    from .offline_io import find_records, read_records

    if path is not None:
        return read_records(path)
    return find_records(run_dir, names)


def _cli_cost_summary(rows: Sequence[Mapping[str, Any]], planned_items: Optional[int], budget: Optional[float]) -> dict[str, Any]:
    actual = 0.0
    priced_items = 0
    candidate_costs: dict[str, float] = {}
    for row in rows:
        value = row.get("cost_usd", row.get("cost", row.get("total_cost_usd")))
        if value is None and isinstance(row.get("usage"), Mapping):
            # Usage records without rates cannot be priced; preserve them in
            # the bundle rather than inventing a cost.
            value = row.get("usage", {}).get("cost_usd")
        if value is None:
            continue
        cost = float(value)
        actual += cost
        priced_items += 1
        candidate = str(row.get("candidate") or row.get("model") or "all")
        candidate_costs[candidate] = candidate_costs.get(candidate, 0.0) + cost
    planned = len(rows) if planned_items is None else planned_items
    projection = project_cost(actual, priced_items, planned, budget=budget) if priced_items else None
    result: dict[str, Any] = {
        "observed_items": len(rows),
        "planned_items": planned,
        "observed_cost_usd": actual if priced_items else None,
        "candidate_costs_usd": candidate_costs,
        "projection": projection.to_dict() if projection is not None else None,
        "records": [dict(row) for row in rows],
    }
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Build the offline human review bundle from synchronized artifacts."""

    parser = argparse.ArgumentParser(prog="bbf build-review-bundle")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--forgeries", type=Path, default=None)
    parser.add_argument("--victim-outputs", type=Path, default=None)
    parser.add_argument("--judgments", type=Path, default=None)
    parser.add_argument("--human-labels", type=Path, default=None)
    parser.add_argument("--costs", type=Path, default=None)
    parser.add_argument("--recommendation", type=Path, default=None)
    parser.add_argument("--planned-items", type=int, default=None)
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument("--refresh-seal", action="store_true", help="explicitly refresh archive_manifest.json after writing outputs")
    args = parser.parse_args(list(argv) if argv is not None else None)
    from .offline_io import read_records

    run_dir = args.run_dir
    output_dir = args.output_dir or run_dir / "review"
    forgeries = _cli_load(args.forgeries, run_dir, ("forgery", "forgeries", "auxiliary_requests"))
    victim_outputs = _cli_load(args.victim_outputs, run_dir, ("generation", "generations"))
    judgments = _cli_load(args.judgments, run_dir, ("judgment", "judgments"))
    human_labels = _cli_load(args.human_labels, run_dir, ("human_labels", "human-labels"))
    cost_rows = _cli_load(args.costs, run_dir, ("cost", "costs", "usage"))
    recommendation_rows = read_records(args.recommendation) if args.recommendation is not None else []
    recommendation = recommendation_rows[0] if recommendation_rows else {}
    costs = _cli_cost_summary(cost_rows, args.planned_items, args.budget) if cost_rows else {}
    archive_seal = run_dir / "archive_manifest.json"
    output_inside_run = False
    try:
        output_dir.resolve().relative_to(run_dir.resolve())
        output_inside_run = True
    except ValueError:
        pass
    metadata = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "offline": True,
        "archive_seal_present_before_build": archive_seal.exists(),
        "archive_seal_refresh_requested": bool(args.refresh_seal),
        "output_inside_run": output_inside_run,
    }
    bundle = build_review_bundle(
        forgeries=forgeries,
        victim_outputs=victim_outputs,
        judgments=judgments,
        human_labels=human_labels,
        costs=costs,
        metadata=metadata,
        recommendation=recommendation,
    )
    if args.refresh_seal and not output_inside_run:
        raise ValueError("--refresh-seal requires --output-dir inside --run-dir")
    paths = write_review_artifacts(output_dir, bundle)
    seal_path: Optional[Path] = None
    if args.refresh_seal:
        from .storage import RunArtifactStore

        seal_path = RunArtifactStore(run_dir).seal_archive(
            required_files=("manifest.json",),
            metadata={"sealed_by": "bbf build-review-bundle", "derived_outputs": sorted(paths.values())},
        )
    result = {
        "output_dir": str(output_dir),
        "artifacts": paths,
        "cases": len(bundle["cases"]),
        "disagreements": bundle["counts"]["disagreements"],
        "archive_seal": str(seal_path) if seal_path else ("stale" if archive_seal.exists() else "not_present"),
        "archive_seal_refreshed": bool(seal_path),
    }
    print(json.dumps(result, sort_keys=True))
    return 0


def write_review_bundle(path: str | Path, bundle: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(bundle, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        handle.write("\n")


__all__ = [
    "BudgetExceeded",
    "BudgetStop",
    "CostProjection",
    "HardBudgetExceeded",
    "HardBudgetStop",
    "build_review_bundle",
    "disagreement_table",
    "project_cost",
    "recommend_candidates",
    "render_review_markdown",
    "usage_cost",
    "write_review_artifacts",
    "write_review_bundle",
    "main",
]
