import json

import pytest

from black_box_forgery.inference import (
    BackendError,
    OpenAICompatibleAgentModel,
    OpenAICompatibleBackend,
    fetch_endpoint_template_contract,
    make_request,
)
from black_box_forgery.rendering import (
    PromptMismatchError,
    local_template_contract,
    validate_template_contract,
)
from black_box_forgery.schemas import Condition


def _request():
    return make_request(
        run_id="integrity",
        model_id="qwen/test",
        model_revision="rev",
        dataset_item_id="item-1",
        condition=Condition.RAW_CHAT,
        messages=[{"role": "user", "content": "frozen prompt"}],
    )


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def _completion_response():
    return {
        "id": "test",
        "model": "served/qwen",
        "choices": [{"message": {"content": "safe"}, "finish_reason": "stop"}],
    }


def test_template_attestation_is_exact_and_requires_endpoint_proof():
    contract = local_template_contract()
    with pytest.raises(PromptMismatchError, match="verification_method"):
        validate_template_contract(contract)

    attestation = {**contract, "verification_method": "endpoint_contract"}
    assert validate_template_contract(attestation) == attestation

    with pytest.raises(PromptMismatchError, match="template_sha256"):
        validate_template_contract({**attestation, "template_sha256": "wrong"})


def test_chat_response_marks_server_template_unverified(monkeypatch):
    monkeypatch.setattr(
        "black_box_forgery.inference.urllib.request.urlopen",
        lambda request, timeout: _Response(_completion_response()),
    )
    backend = OpenAICompatibleBackend(
        base_url="http://localhost:18000",
        model="qwen/test",
        allow_network=True,
    )
    response = backend.complete(_request())
    integrity = response.metadata
    assert integrity["wire_template_status"] == "server_template_unverified"
    assert integrity["wire_prompt_sha256"] is None
    assert integrity["local_rendered_prompt_sha256"]


def test_strict_backend_requires_a_valid_server_attestation():
    with pytest.raises(BackendError, match="verification is required"):
        OpenAICompatibleBackend(
            base_url="http://localhost:18000",
            model="qwen/test",
            allow_network=True,
            require_template_verification=True,
        )


def test_agent_response_preserves_integrity_metadata(monkeypatch):
    monkeypatch.setattr(
        "black_box_forgery.inference.urllib.request.urlopen",
        lambda request, timeout: _Response(_completion_response()),
    )
    model = OpenAICompatibleAgentModel(
        base_url="http://localhost:18000",
        model="qwen/test",
        allow_network=True,
    )
    response = model.complete(
        messages=[{"role": "user", "content": "task"}],
        tools=[],
        max_new_tokens=32,
        rendered_prompt="local reference prompt",
    )
    integrity = response["_bbf_prompt_integrity"]
    assert integrity["wire_template_status"] == "server_template_unverified"
    assert integrity["wire_prompt_sha256"] is None


def test_endpoint_probe_is_explicit_and_validates_attestation(monkeypatch):
    contract = {**local_template_contract(), "verification_method": "endpoint_contract"}
    monkeypatch.setattr(
        "black_box_forgery.inference.urllib.request.urlopen",
        lambda request, timeout: _Response(contract),
    )
    with pytest.raises(BackendError, match="network backend disabled"):
        fetch_endpoint_template_contract(base_url="http://localhost:18000")
    assert fetch_endpoint_template_contract(
        base_url="http://localhost:18000", allow_network=True
    ) == contract
