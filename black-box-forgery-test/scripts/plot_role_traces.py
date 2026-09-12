#!/usr/bin/env python3
"""Plot Figure-26-style Userness/CoTness traces for SecOPD prompts.

The projection runner stores fixed-size token traces for the user task, the
tail of the page/input context, and the appended injection/forgery.  This
script writes both representative-page traces and a page-averaged trace in
which each segment is resampled to a common horizontal width.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCENARIO_ORDER = (
    "malicious_standard",
    "malicious_cot_forged",
    "benign_contradiction_standard",
    "benign_contradiction_cot_forged",
)
SEGMENT_ORDER = (
    "user_task",
    "page_context",
    "forged_cot_prelude",
    "malicious_command",
    "benign_contradiction",
    "forged_cot",
)
SEGMENT_LABELS = {
    "user_task": "User",
    "page_context": "Page / input",
    "forged_cot_prelude": "Forged CoT prelude",
    "malicious_command": "Malicious command",
    "benign_contradiction": "Benign contradiction",
    "forged_cot": "Forged CoT",
}
SEGMENT_COLORS = {
    "user_task": "#2f8bdc",
    "page_context": "#8e63b8",
    "forged_cot_prelude": "#ef9b27",
    "malicious_command": "#ed6a83",
    "benign_contradiction": "#ed6a83",
    "forged_cot": "#ef9b27",
}
METRICS = ("user", "cot")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _require_plot_dependencies() -> tuple[Any, Any]:
    try:
        import matplotlib  # type: ignore
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised on pod
        raise RuntimeError("plotting requires matplotlib and numpy") from exc
    return plt, np


def _segments(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    trace = row.get("token_trace")
    if not isinstance(trace, dict):
        raise ValueError(f"missing token_trace for {row.get('case_id')}")
    result: dict[str, dict[str, Any]] = {}
    for segment in trace.get("segments", []):
        if isinstance(segment, dict) and isinstance(segment.get("name"), str):
            result[segment["name"]] = segment
    scenario = str(row.get("scenario", ""))
    if scenario == "malicious_cot_forged":
        # Older trace files stored the complete forged suffix under both
        # names. Keep only the semantically correct forged-CoT segment. New
        # v2 traces partition the embedded command and forgery disjointly.
        partition = trace.get("segment_partition")
        if partition != "disjoint_embedded_command_and_forged_cot_v2":
            result.pop("malicious_command", None)
    elif scenario == "benign_contradiction_cot_forged":
        # Older trace files stored direct contradiction + forged CoT together
        # under ``benign_contradiction``. Split its role arrays at the known
        # absolute token boundary recorded by the projection runner.
        injection = result.get("benign_contradiction")
        spans = row.get("rendered_spans", {})
        injection_span = spans.get("injection", {})
        forgery_span = spans.get("forgery", {})
        if injection and isinstance(injection_span, dict) and isinstance(forgery_span, dict):
            direct_count = int(forgery_span.get("token_start", 0)) - int(
                injection_span.get("token_start", 0)
            )
            direct_count = max(0, min(direct_count, int(injection.get("token_count", 0))))
            injection["token_count"] = direct_count
            injection["roles"] = {
                role: values[:direct_count] for role, values in injection.get("roles", {}).items()
            }
    return result


def _plot_representative(rows: list[dict[str, Any]], output_dir: Path, page_id: str) -> list[Path]:
    plt, np = _require_plot_dependencies()
    selected = {
        row["scenario"]: row
        for row in rows
        if row.get("page_id") == page_id and row.get("scenario") in SCENARIO_ORDER
    }
    missing = [scenario for scenario in SCENARIO_ORDER if scenario not in selected]
    if missing:
        raise ValueError(f"representative page {page_id} is missing scenarios: {missing}")

    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharey=True, constrained_layout=True)
    for column, scenario in enumerate(SCENARIO_ORDER):
        row = selected[scenario]
        segments = _segments(row)
        scenario_label = scenario.replace("_", " ").title()
        for axis_row, metric in enumerate(METRICS):
            ax = axes[axis_row, column]
            cursor = 0
            centers: list[float] = []
            labels: list[str] = []
            for segment_name in SEGMENT_ORDER:
                segment = segments.get(segment_name)
                if not segment:
                    continue
                values = np.asarray(segment["roles"].get(metric, []), dtype=float)
                if values.size == 0:
                    continue
                x = np.arange(cursor, cursor + values.size)
                ax.plot(
                    x,
                    values,
                    color=SEGMENT_COLORS[segment_name],
                    marker="o",
                    markersize=2.2,
                    linewidth=1.0,
                    alpha=0.88,
                )
                centers.append(float(cursor + (values.size - 1) / 2))
                labels.append(SEGMENT_LABELS[segment_name])
                cursor += values.size
                if cursor < 10_000:
                    ax.axvline(cursor - 0.5, color="#d0d0d0", linewidth=0.8, zorder=0)
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["0%", "100%"])
            ax.grid(axis="y", color="#eeeeee", linewidth=0.7)
            ax.set_xticks(centers)
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
            if column == 0:
                ax.set_ylabel("Userness" if metric == "user" else "CoTness", fontsize=12)
            if axis_row == 0:
                title = {
                    "malicious_standard": "Malicious\nNo CoT forgery",
                    "malicious_cot_forged": "Malicious\nCoT forgery",
                    "benign_contradiction_standard": "Benign contradiction\nNo CoT forgery",
                    "benign_contradiction_cot_forged": "Benign contradiction\nCoT forgery",
                }[scenario]
                ax.set_title(title, fontsize=12, fontweight="bold")
    fig.suptitle(
        f"SecOPD layer-56 role projections — representative page {page_id}\n"
        "pre-generation prompt trace",
        fontsize=16,
        fontweight="bold",
    )
    output = output_dir / "role_traces_representative.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return [output]


def _resample(values: Any, bins: int, np: Any) -> Any:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.full(bins, np.nan)
    if values.size == 1:
        return np.full(bins, values[0])
    old_x = np.linspace(0, 1, values.size)
    new_x = np.linspace(0, 1, bins)
    return np.interp(new_x, old_x, values)


def _plot_aggregate(rows: list[dict[str, Any]], output_dir: Path, bins: int) -> tuple[Path, dict[str, Any]]:
    plt, np = _require_plot_dependencies()
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharey=True, constrained_layout=True)
    summary: dict[str, Any] = {}
    for column, scenario in enumerate(SCENARIO_ORDER):
        scenario_rows = [row for row in rows if row.get("scenario") == scenario]
        segment_names = [name for name in SEGMENT_ORDER if any(name in _segments(row) for row in scenario_rows)]
        summary[scenario] = {}
        for axis_row, metric in enumerate(METRICS):
            ax = axes[axis_row, column]
            cursor = 0
            for segment_name in segment_names:
                series = []
                for row in scenario_rows:
                    segment = _segments(row).get(segment_name)
                    if segment:
                        series.append(_resample(segment["roles"].get(metric, []), bins, np))
                if not series:
                    continue
                matrix = np.stack(series)
                mean = np.nanmean(matrix, axis=0)
                low = np.nanpercentile(matrix, 10, axis=0)
                high = np.nanpercentile(matrix, 90, axis=0)
                x = np.linspace(cursor, cursor + 1, bins)
                color = SEGMENT_COLORS[segment_name]
                ax.plot(x, mean, color=color, linewidth=2.2)
                ax.fill_between(x, low, high, color=color, alpha=0.14)
                summary[scenario].setdefault(segment_name, {})[metric] = {
                    "mean": float(np.nanmean(matrix)),
                    "p10": float(np.nanpercentile(matrix, 10)),
                    "p90": float(np.nanpercentile(matrix, 90)),
                }
                cursor += 1
                if cursor < len(segment_names):
                    ax.axvline(cursor, color="#d0d0d0", linewidth=0.8, zorder=0)
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["0%", "100%"])
            ax.set_xticks(np.arange(len(segment_names)) + 0.5)
            ax.set_xticklabels([SEGMENT_LABELS[name] for name in segment_names], rotation=35, ha="right", fontsize=9)
            ax.grid(axis="y", color="#eeeeee", linewidth=0.7)
            if column == 0:
                ax.set_ylabel("Userness" if metric == "user" else "CoTness", fontsize=12)
            if axis_row == 0:
                ax.set_title(scenario.replace("_", " ").title(), fontsize=12, fontweight="bold")
    fig.suptitle(
        "SecOPD layer-56 role projections — 10-page normalized segment average\n"
        "lines: mean; bands: 10th–90th percentile",
        fontsize=16,
        fontweight="bold",
    )
    output = output_dir / "role_traces_aggregate.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projections-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--representative-page", default="66872153")
    parser.add_argument("--bins", type=int, default=80)
    args = parser.parse_args(argv)
    if args.bins < 2:
        raise ValueError("--bins must be at least two")
    rows = _read_rows(args.projections_jsonl)
    if not rows or any("token_trace" not in row for row in rows):
        raise ValueError("projection JSONL must contain token traces")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = _plot_representative(rows, args.output_dir, args.representative_page)
    aggregate_path, summary = _plot_aggregate(rows, args.output_dir, args.bins)
    outputs.append(aggregate_path)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"outputs": [str(path) for path in outputs], "row_count": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
