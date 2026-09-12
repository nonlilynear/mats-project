from __future__ import annotations

import json

import pytest

from black_box_forgery.cli import main as cli_main
from black_box_forgery.analysis import (
    compare_paired_records,
    mcnemar_test,
    summarize_cell,
    summarize_clean_utility,
    wilson_interval,
)
from black_box_forgery.auxiliary import (
    AuxiliaryModelSpec,
    AuxiliaryRequest,
    AuxiliaryResponse,
    OpenRouterBackend,
    FireworksBackend,
    ScriptedAuxiliaryBackend,
    classify_text,
    run_auxiliary_job,
    resolve_candidate_specs,
    snapshot_openrouter_metadata,
)
from black_box_forgery.archive import create_manifest, verify_archive, write_manifest
from black_box_forgery.audit import blind_records, select_audit_sample
from black_box_forgery.judging import JudgmentRecord, parse_agent_label, parse_chat_label
from black_box_forgery.review import BudgetExceeded, BudgetStop, build_review_bundle, project_cost


def test_strict_labels_and_separate_human_provenance() -> None:
    assert parse_chat_label(" HARMFUL_RESPONSE ").label == "HARMFUL_RESPONSE"
    assert not parse_chat_label("The answer is HARMFUL_RESPONSE").valid
    assert not parse_chat_label("harmful_response").valid
    assert parse_agent_label("ATTEMPTED_SUCCESSFUL").valid
    record = JudgmentRecord.from_automated("e1", "REFUSAL")
    audited = record.with_human_label("REDIRECTION")
    assert audited.automated_label == "REFUSAL"
    assert audited.human_label == "REDIRECTION"


def test_stratified_sampling_is_order_independent_and_blinded() -> None:
    rows = [
        {"episode_id": f"e{i}", "model": "base" if i % 2 else "defended", "condition": "raw"}
        for i in range(10)
    ]
    a = select_audit_sample(rows, fraction=0.2, seed=11)
    b = select_audit_sample(list(reversed(rows)), fraction=0.2, seed=11)
    assert [row["episode_id"] for row in a] == [row["episode_id"] for row in b]
    blinded = blind_records(a, seed=11)
    assert all("model" not in row and "condition" not in row for row in blinded)
    assert len({row["audit_id"] for row in blinded}) == len(blinded)


def test_blinding_removes_nested_model_condition_and_raw_episode_key() -> None:
    record = {
        "episode_id": "run__base__rev__item__cot_agent__hash__123",
        "request_key": {
            "run_id": "run-base",
            "victim_model_id": "secret/base",
            "victim_revision": "rev",
            "dataset_item_id": "item-7",
            "condition": "cot_agent",
        },
        "output_text": "review me",
        "human_label": "DENIED",
    }
    [blinded] = blind_records([record], seed=11)
    serialized = json.dumps(blinded)
    assert "secret/base" not in serialized
    assert "cot_agent" not in serialized
    assert "run-base" not in serialized
    assert "DENIED" not in serialized
    assert "episode_id" not in blinded
    assert blinded["dataset_item_id"] == "item-7"


def test_clean_utility_inverts_unrelated_sensitive_action() -> None:
    summary = summarize_clean_utility(
        [
            {"unrelated_sensitive_action": False},
            {"unrelated_sensitive_action": True},
            {},
        ]
    )
    assert summary["sensitive_action_free_rate"] == 0.5


def test_analysis_reports_invalid_denominator_and_intervals() -> None:
    rows = [
        {"episode_id": "1", "model": "base", "condition": "raw", "automated_label": "HARMFUL_RESPONSE"},
        {"episode_id": "2", "model": "base", "condition": "raw", "automated_label": "REFUSAL"},
        {"episode_id": "3", "model": "base", "condition": "raw", "automated_label": "not-a-label", "automated_valid": False},
    ]
    result = summarize_cell(rows, rubric="chat")
    assert result.numerator == 1
    assert result.denominator == 2
    assert result.attempted == 3
    assert result.invalid == 1
    assert result.ci_low is not None and result.ci_high is not None
    assert wilson_interval(0, 0)[0] != wilson_interval(0, 0)[0]  # NaN means undefined


def test_paired_difference_and_mcnemar() -> None:
    result = compare_paired_records(
        [{"episode_id": "1", "automated_label": "HARMFUL_RESPONSE"}, {"episode_id": "2", "automated_label": "REFUSAL"}],
        [{"episode_id": "1", "automated_label": "REFUSAL"}, {"episode_id": "2", "automated_label": "REFUSAL"}],
        rubric="chat",
        n_resamples=100,
    )
    assert result["n_paired_valid"] == 2
    assert result["difference"]["estimate"] == 0.5
    assert mcnemar_test([1, 1, 0], [0, 1, 0]).p_value == 1.0


def test_review_budget_projection_and_bundle() -> None:
    assert project_cost(2, 2, 10, budget=5).within_budget is False
    guard = BudgetStop(1.0)
    guard.check(0.6)
    guard.record(0.5, reserved_amount=0.6)
    with pytest.raises(BudgetExceeded):
        guard.check(0.6)
    exhausted = BudgetStop(0.5)
    exhausted.record(0.5)
    with pytest.raises(BudgetExceeded):
        exhausted.check(0.0)
    bundle = build_review_bundle(
        forgeries=[{"episode_id": "e1", "candidate": "a", "text": "x"}],
        victim_outputs=[{"episode_id": "e1", "output": "y"}],
        judgments=[{"episode_id": "e1", "candidate": "a", "automated_label": "REFUSAL"}],
        human_labels=[{"episode_id": "e1", "human_label": "HARMFUL_RESPONSE"}],
    )
    assert bundle["counts"]["cases"] == 1
    assert bundle["counts"]["disagreements"] == 1


def test_archive_verification_catches_corruption_and_missing(tmp_path) -> None:
    (tmp_path / "generations.jsonl").write_text(
        json.dumps({"episode_id": "e1"}) + "\n", encoding="utf-8"
    )
    manifest = create_manifest(
        tmp_path,
        required_files=["generations.jsonl"],
        expected_records={"generations.jsonl": 1},
    )
    write_manifest(tmp_path, manifest)
    assert verify_archive(tmp_path).valid
    (tmp_path / "generations.jsonl").write_text("broken\n", encoding="utf-8")
    result = verify_archive(tmp_path)
    assert not result.valid
    assert "generations.jsonl" in result.corrupted or "generations.jsonl" in result.invalid_records


def test_cpu_cli_judge_audit_analyze_pipeline_preserves_human_label(tmp_path, capsys) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "generations.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"episode_id": "e1", "model": "base", "condition": "raw_chat", "label": "HARMFUL_RESPONSE"},
                {"episode_id": "e2", "model": "base", "condition": "raw_chat", "label": "REFUSAL"},
                {"episode_id": "e3", "model": "sec", "condition": "raw_chat", "label": "HARMFUL_RESPONSE"},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    assert cli_main(["judge", "--run-dir", str(run_dir)]) == 0
    judgments = [json.loads(line) for line in (run_dir / "judgments.jsonl").read_text().splitlines()]
    judgments[0]["human_label"] = "REFUSAL"
    judgments[0]["human_valid"] = True
    (run_dir / "judgments.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in judgments), encoding="utf-8"
    )
    # Re-running judge retains human fields from its existing canonical output.
    assert cli_main(["judge", "--run-dir", str(run_dir)]) == 0
    judged_again = [json.loads(line) for line in (run_dir / "judgments.jsonl").read_text().splitlines()]
    assert judged_again[0]["human_label"] == "REFUSAL"
    assert cli_main(["export-audit", "--run-dir", str(run_dir), "--fraction", "1"]) == 0
    queue = [json.loads(line) for line in (run_dir / "review/audit_queue.jsonl").read_text().splitlines()]
    assert len(queue) == 3
    assert all("model" not in row and "condition" not in row for row in queue)
    assert all("human_label" not in row for row in queue)
    assert cli_main(["analyze", "--run-dir", str(run_dir), "--bootstrap-resamples", "20"]) == 0
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["cells"][0]["denominator"] == 2
    assert metrics["human_agreement"]["n"] == 1


def test_cpu_cli_build_review_bundle_and_explicit_seal_refresh(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "forgeries.jsonl").write_text(
        json.dumps({"episode_id": "e1", "candidate": "aux-a", "forgery": "one paragraph"}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "generations.jsonl").write_text(
        json.dumps({"episode_id": "e1", "model": "base", "condition": "cot_chat", "output_text": "answer"}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "judgments.jsonl").write_text(
        json.dumps({"episode_id": "e1", "candidate": "aux-a", "automated_label": "HARMFUL_RESPONSE", "automated_valid": True}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "human_labels.jsonl").write_text(
        json.dumps({"episode_id": "e1", "human_label": "REFUSAL", "human_valid": True}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "costs.jsonl").write_text(
        json.dumps({"candidate": "aux-a", "cost_usd": 1.0}) + "\n",
        encoding="utf-8",
    )
    assert cli_main(
        ["build-review-bundle", "--run-dir", str(run_dir), "--planned-items", "4", "--budget", "5", "--refresh-seal"]
    ) == 0
    review_dir = run_dir / "review"
    for name in ("summary.json", "summary.md", "disagreements.jsonl", "bundle.json"):
        assert (review_dir / name).exists()
    summary = json.loads((review_dir / "summary.json").read_text())
    assert summary["counts"]["cases"] == 1
    assert summary["counts"]["disagreements"] == 1
    assert (run_dir / "archive_manifest.json").exists()
    assert verify_archive(run_dir).valid

    # Derived artifacts do not silently refresh an existing seal.
    (run_dir / "human_labels.jsonl").write_text(
        json.dumps({"episode_id": "e1", "human_label": "HARMFUL_RESPONSE", "human_valid": True}) + "\n",
        encoding="utf-8",
    )
    assert cli_main(["build-review-bundle", "--run-dir", str(run_dir)]) == 0
    assert not verify_archive(run_dir).valid
    assert cli_main(["build-review-bundle", "--run-dir", str(run_dir), "--refresh-seal"]) == 0
    assert verify_archive(run_dir).valid


def test_auxiliary_cli_uses_frozen_inputs_and_resumes(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    forgeries = run_dir / "forgeries.jsonl"
    outputs = run_dir / "victim_outputs.jsonl"
    forgeries.write_text(json.dumps({"episode_id": "f1", "prompt": "frozen prompt"}) + "\n")
    outputs.write_text(json.dumps({"episode_id": "v1", "condition": "raw_chat", "output_text": "victim"}) + "\n")
    summary = run_dir / "summary.json"
    results = run_dir / "results.jsonl"
    assert cli_main(
        [
            "auxiliary-smoke",
            "--output",
            str(summary),
            "--results",
            str(results),
            "--forgeries",
            str(forgeries),
            "--victim-outputs",
            str(outputs),
            "--candidate",
            "glm",
        ]
    ) == 0
    payload = json.loads(summary.read_text())
    assert payload["offline"] is True
    assert payload["requests"] == 2
    assert len(results.read_text().splitlines()) == 2
    assert cli_main(
        [
            "auxiliary-smoke",
            "--output",
            str(summary),
            "--results",
            str(results),
            "--forgeries",
            str(forgeries),
            "--victim-outputs",
            str(outputs),
            "--candidate",
            "glm",
        ]
    ) == 0
    resumed = json.loads(summary.read_text())
    assert resumed["comparisons"][0]["skipped"] == 2
    assert len(results.read_text().splitlines()) == 2
    assert classify_text("The answer is REFUSAL") is None


def test_auxiliary_cli_filters_exact_request_ids(tmp_path) -> None:
    requests = tmp_path / "requests.jsonl"
    requests.write_text(
        "\n".join(
            json.dumps({"request_id": request_id, "task": "generate_forgery", "input_text": request_id})
            for request_id in ("one", "two", "three")
        ) + "\n"
    )
    summary = tmp_path / "summary.json"
    results = tmp_path / "results.jsonl"
    assert cli_main([
        "auxiliary-smoke", "--requests", str(requests), "--request-id", "three",
        "--request-id", "one", "--candidate", "glm", "--output", str(summary),
        "--results", str(results),
    ]) == 0
    assert [json.loads(line)["request_id"] for line in results.read_text().splitlines()] == ["three", "one"]


def test_openrouter_backend_is_pinned_and_metadata_transport_is_mocked(tmp_path) -> None:
    calls = []

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(self.payload).encode()

    def transport(request, timeout):
        calls.append((request, timeout))
        if request.full_url.endswith("/models"):
            return Response({"data": [{"id": "candidate/one", "pricing": {"prompt": "1"}}]})
        return Response(
            {
                "model": "resolved/candidate-one",
                "provider": "provider-one",
                "choices": [{"message": {"content": "REFUSAL"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4, "cost": 0.25},
            }
        )

    spec = AuxiliaryModelSpec("candidate/one", provider="provider-one")
    backend = OpenRouterBackend(spec, api_key="do-not-persist", allow_network=True, urlopen=transport)
    response = backend.complete(AuxiliaryRequest("j1", "judge", "victim", metadata={"rubric": "chat"}))
    assert response.resolved_model == "resolved/candidate-one"
    assert response.provider == "provider-one"
    assert response.input_tokens == 12 and response.output_tokens == 4
    assert response.cost_usd == 0.25
    request_body = json.loads(calls[0][0].data.decode())
    assert request_body["provider"] == {"allow_fallbacks": False, "require_parameters": True, "order": ["provider-one"]}
    snapshot = snapshot_openrouter_metadata(
        [spec],
        api_key="do-not-persist",
        api_key_env="TEST_OPENROUTER_KEY",
        base_url="https://router.invalid/api/v1",
        allow_network=True,
        urlopen=transport,
        output_path=tmp_path / "metadata.json",
    )
    serialized = (tmp_path / "metadata.json").read_text()
    assert snapshot["api_key_env"] == "TEST_OPENROUTER_KEY"
    assert "do-not-persist" not in serialized


def test_fireworks_backend_uses_pinned_model_and_estimates_cost() -> None:
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({
                "model": "accounts/fireworks/models/glm-5p3-flash",
                "choices": [{"message": {"content": "<SYNTHETIC_POLICY>test</SYNTHETIC_POLICY>"}}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 100},
            }).encode()

    def transport(request, timeout):
        calls.append(request)
        return Response()

    spec = AuxiliaryModelSpec(
        "z-ai/glm-5.3-flash",
        resolved_model="accounts/fireworks/models/glm-5p3-flash",
        input_price_per_million=0.15,
        output_price_per_million=0.50,
    )
    backend = FireworksBackend(spec, api_key="do-not-persist", allow_network=True, urlopen=transport)
    response = backend.complete(AuxiliaryRequest("g1", "generate_forgery", "target"))
    body = json.loads(calls[0].data.decode())
    assert body["model"] == "accounts/fireworks/models/glm-5p3-flash"
    assert "provider" not in body
    assert response.provider == "Fireworks"
    assert response.cost_usd == pytest.approx(0.0002)


def test_fireworks_candidate_aliases_are_pinned_and_priced() -> None:
    specs = resolve_candidate_specs(["nemotron", "deepseek-v4-flash"])
    assert [spec.resolved_model for spec in specs] == [
        "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b",
        "accounts/fireworks/models/deepseek-v4-flash-0731",
    ]
    assert [(spec.input_price_per_million, spec.output_price_per_million) for spec in specs] == [
        (0.05, 0.20),
        (0.22, 0.66),
    ]


def test_frozen_auxiliary_request_preserves_multi_message_prompt(tmp_path) -> None:
    request_path = tmp_path / "requests.jsonl"
    request_path.write_text(
        json.dumps(
            {
                "request_id": "forge-1",
                "task": "generate_forgery",
                "input_text": "review label",
                "messages": [
                    {"role": "system", "content": "system prompt"},
                    {"role": "user", "content": "target prompt"},
                ],
                "metadata": {"block": "chat"},
                "max_tokens": 512,
            }
        )
        + "\n"
    )
    from black_box_forgery.auxiliary import build_smoke_requests

    [request] = build_smoke_requests(requests_path=request_path)
    assert list(request.messages) == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "target prompt"},
    ]
    assert request.input_text == "review label"
    assert request.metadata["block"] == "chat"
    assert request.max_tokens == 512


def test_frozen_auxiliary_request_uses_nested_agent_rubric(tmp_path) -> None:
    request_path = tmp_path / "agent-requests.jsonl"
    request_path.write_text(
        json.dumps(
            {
                "request_id": "agent__page-1",
                "task": "generate_forgery",
                "input_text": "generate agent forgery",
                "messages": [{"role": "user", "content": "target prompt"}],
                "metadata": {"block": "agent", "rubric": "agent"},
            }
        )
        + "\n"
    )
    from black_box_forgery.auxiliary import build_smoke_requests

    [request] = build_smoke_requests(requests_path=request_path)
    assert request.metadata["block"] == "agent"
    assert request.metadata["rubric"] == "agent"


def test_auxiliary_job_hard_budget_stops_before_next_request(tmp_path) -> None:
    class ChargedBackend:
        model = "charged"

        def __init__(self):
            self.calls = 0

        def complete(self, request):
            self.calls += 1
            return AuxiliaryResponse(
                output_text="REFUSAL",
                model=self.model,
                request_id=request.request_id,
                cost_usd=0.6,
            )

    from black_box_forgery.review import BudgetExceeded, BudgetStop

    backend = ChargedBackend()
    guard = BudgetStop(1.0)
    requests = [
        AuxiliaryRequest("j1", "judge", "x", metadata={"rubric": "chat"}),
        AuxiliaryRequest("j2", "judge", "y", metadata={"rubric": "chat"}),
    ]
    with pytest.raises(BudgetExceeded):
        run_auxiliary_job(
            backend,
            requests,
            candidate="charged",
            output_path=tmp_path / "results.jsonl",
            budget_stop=guard,
            request_cost_cap=0.8,
        )
    assert backend.calls == 1
