"""Model-backend abstractions and resumable target inference.

The default backend is deterministic and offline.  The OpenAI-compatible
backend is intentionally a separate opt-in class so preparing fixtures or
running tests never makes an accidental network request.
"""

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Union

from .data import canonical_json, sha256_text
from .rendering import (
    local_template_contract,
    prompt_sha256,
    render_qwen36_upstream_template,
    validate_template_contract,
)
from .schemas import (
    DecodingParams,
    EpisodeRequest,
    GenerationRecord,
    RecordStatus,
    RequestKey,
    ToolCall,
)
from .storage import RunArtifactStore


class BackendError(RuntimeError):
    """An inference backend could not produce a response."""


class Backend(Protocol):
    def complete(self, request: EpisodeRequest) -> "BackendResponse":
        ...


class BackendResponse:
    """Normalized response returned by scripted and network backends."""

    def __init__(
        self,
        output_text: str = "",
        *,
        thinking_text: Optional[str] = None,
        tool_calls: Optional[Sequence[ToolCall]] = None,
        stop_reason: Optional[str] = "stop",
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        status: RecordStatus = RecordStatus.COMPLETE,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.output_text = output_text
        self.thinking_text = thinking_text
        self.tool_calls = list(tool_calls or [])
        self.stop_reason = stop_reason
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.status = status
        self.metadata = dict(metadata or {})


ScriptedResponse = Union[BackendResponse, Mapping[str, Any], Callable[[EpisodeRequest], BackendResponse]]


class ScriptedBackend:
    """A deterministic fake backend for development and end-to-end tests.

    ``responses`` may map request-key values, condition names, or ``"*"`` to
    a response.  Values can be :class:`BackendResponse`, dictionaries accepted
    by its constructor, or callbacks.  Calls and request keys are retained for
    assertions about resume behavior.
    """

    def __init__(
        self,
        responses: Optional[Mapping[str, ScriptedResponse]] = None,
        *,
        default: Optional[ScriptedResponse] = None,
    ) -> None:
        self.responses = dict(responses or {})
        self.default = default or self._default_response
        self.calls: List[str] = []

    def complete(self, request: EpisodeRequest) -> BackendResponse:
        key = request.request_key.value
        self.calls.append(key)
        response = self.responses.get(key)
        if response is None:
            response = self.responses.get(request.condition.value, self.responses.get("*", self.default))
        if callable(response):
            response = response(request)
        if isinstance(response, BackendResponse):
            return response
        if isinstance(response, Mapping):
            return BackendResponse(**dict(response))
        raise BackendError(f"scripted response for {key} is not a valid response")

    @staticmethod
    def _default_response(request: EpisodeRequest) -> BackendResponse:
        if request.condition.value.endswith("agent"):
            text = "I treated the page as untrusted data and produced a concise summary."
        else:
            text = "I can’t help with that request, but I can offer safe general information."
        return BackendResponse(output_text=text, thinking_text="Reviewed the request and constraints.")


FakeBackend = ScriptedBackend


def _openai_wire_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize local messages to the strict OpenAI/vLLM wire shape.

    Local agent records intentionally use a compact tool-call representation
    (``id``, ``name``, and parsed ``arguments``).  The OpenAI-compatible API
    instead requires assistant tool calls nested under ``function`` with JSON
    arguments encoded as a string.  Optional ``None`` fields are omitted too:
    some vLLM versions reject ``name: null`` during request validation.
    """

    wire_messages: list[dict[str, Any]] = []
    for message_index, raw_message in enumerate(messages):
        if not isinstance(raw_message, Mapping):
            raise BackendError(f"message {message_index} must be an object")
        message = {
            str(key): value for key, value in raw_message.items() if value is not None
        }
        if message.get("role") != "assistant":
            message.pop("tool_calls", None)
        if message.get("role") == "assistant" and "tool_calls" in message:
            wire_calls: list[dict[str, Any]] = []
            raw_calls = message.get("tool_calls") or []
            if not isinstance(raw_calls, Sequence) or isinstance(raw_calls, (str, bytes)):
                raise BackendError("assistant tool_calls must be a list")
            for call_index, raw_call in enumerate(raw_calls):
                if not isinstance(raw_call, Mapping):
                    raise BackendError(f"assistant tool call {call_index} must be an object")
                raw_function = raw_call.get("function")
                function = raw_function if isinstance(raw_function, Mapping) else raw_call
                name = function.get("name")
                if not isinstance(name, str) or not name:
                    raise BackendError(f"assistant tool call {call_index} has no function name")
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    encoded_arguments = arguments
                else:
                    encoded_arguments = json.dumps(
                        arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    )
                call_id = raw_call.get("id") or f"call-{call_index + 1}"
                wire_calls.append(
                    {
                        "id": str(call_id),
                        "type": "function",
                        "function": {"name": name, "arguments": encoded_arguments},
                    }
                )
            message["tool_calls"] = wire_calls
        elif message.get("role") == "tool":
            # ``name`` is a local audit convenience and is not needed by the
            # current vLLM tool-message schema.
            message.pop("name", None)
        wire_messages.append(message)
    return wire_messages


def _http_error_detail(error: urllib.error.HTTPError, *, limit: int = 4000) -> str:
    """Include a bounded provider response body in durable backend errors."""

    try:
        body = error.read().decode("utf-8", errors="replace").strip()
    except OSError:
        body = ""
    return body[:limit] if body else str(error)


class OpenAICompatibleBackend:
    """Minimal explicit HTTP client for vLLM/OpenAI-compatible endpoints.

    This class is never constructed by the offline CLI.  ``allow_network`` is
    required to be true, making network use a visible call-site decision.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        timeout_seconds: float = 120.0,
        allow_network: bool = False,
        server_template_contract: Optional[Mapping[str, Any]] = None,
        require_template_verification: bool = False,
    ) -> None:
        if not allow_network:
            raise BackendError(
                "network backend disabled; pass allow_network=True explicitly for a pod/API run"
            )
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.server_template_contract = _resolve_template_attestation(
            server_template_contract, require_template_verification
        )

    def complete(self, request: EpisodeRequest) -> BackendResponse:
        rendered_prompt = render_qwen36_upstream_template(
            [message.model_dump(mode="json") for message in request.messages],
            enable_thinking=request.decoding.thinking_enabled,
        )
        # ``Message`` keeps optional fields explicit for the local record
        # schema, but vLLM's request model rejects ``name: null``.
        messages = _openai_wire_messages(
            [message.model_dump(mode="json") for message in request.messages]
        )
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": request.decoding.temperature,
            "top_p": request.decoding.top_p,
            "seed": request.decoding.seed,
            "max_tokens": request.decoding.max_new_tokens,
            "chat_template_kwargs": {
                "enable_thinking": request.decoding.thinking_enabled,
            },
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        http_request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions", data=body, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise BackendError(
                f"backend request failed: HTTP {exc.code}: {_http_error_detail(exc)}"
            ) from exc
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise BackendError(f"backend request failed: {exc}") from exc
        try:
            choice = data["choices"][0]
            message = choice.get("message", {})
            output = message.get("content") or ""
            reasoning = message.get("reasoning_content") or message.get("reasoning")
            usage = data.get("usage") or {}
            finish = choice.get("finish_reason") or "stop"
            status = RecordStatus.TRUNCATED if finish == "length" else RecordStatus.COMPLETE
            return BackendResponse(
                output_text=output,
                thinking_text=reasoning,
                stop_reason=finish,
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
                status=status,
                metadata={
                    "served_model": data.get("model"),
                    "id": data.get("id"),
                    **_prompt_integrity_metadata(
                        rendered_prompt,
                        server_template_contract=self.server_template_contract,
                    ),
                },
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError("backend response has an unexpected shape") from exc


class OpenAICompatibleAgentModel:
    """OpenAI-compatible adapter for the constrained agent loop.

    The agent runner needs the provider's original response wrapper so it can
    parse structured tool calls, reasoning content, and finish reasons.  This
    adapter therefore returns the decoded chat-completions mapping directly;
    :func:`black_box_forgery.agent.run_agent_episode` normalizes it at the
    safety boundary before any tool can be invoked.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        timeout_seconds: float = 120.0,
        seed: int = 123,
        temperature: float = 0.0,
        top_p: float = 1.0,
        enable_thinking: bool = True,
        allow_network: bool = False,
        server_template_contract: Optional[Mapping[str, Any]] = None,
        require_template_verification: bool = False,
    ) -> None:
        if not allow_network:
            raise BackendError(
                "network backend disabled; pass allow_network=True explicitly for a pod/API run"
            )
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.seed = seed
        self.temperature = temperature
        self.top_p = top_p
        self.enable_thinking = enable_thinking
        self.server_template_contract = _resolve_template_attestation(
            server_template_contract, require_template_verification
        )

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        max_new_tokens: int,
        rendered_prompt: str,
    ) -> Mapping[str, Any]:
        if not isinstance(rendered_prompt, str) or not rendered_prompt:
            raise BackendError("a non-empty local rendered prompt is required for integrity tracking")
        payload = {
            "model": self.model,
            "messages": _openai_wire_messages(messages),
            "tools": [dict(tool) for tool in tools],
            "tool_choice": "auto",
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "max_tokens": max_new_tokens,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        http_request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions", data=body, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise BackendError(
                f"agent backend request failed: HTTP {exc.code}: {_http_error_detail(exc)}"
            ) from exc
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise BackendError(f"agent backend request failed: {exc}") from exc
        if not isinstance(data, Mapping):
            raise BackendError("agent backend response must be a JSON object")
        # The OpenAI-compatible API returns only the model response; it does
        # not expose the bytes produced by its chat template.  Preserve that
        # distinction in-band for the agent artifact writer instead of
        # allowing callers to mistake the local reference hash for a wire hash.
        result = dict(data)
        result["_bbf_prompt_integrity"] = _prompt_integrity_metadata(
            rendered_prompt,
            server_template_contract=self.server_template_contract,
        )
        return result


def _resolve_template_attestation(
    attestation: Optional[Mapping[str, Any]],
    required: bool,
) -> Optional[dict[str, str]]:
    if attestation is None:
        if required:
            raise BackendError(
                "server template verification is required; provide an endpoint contract attestation"
            )
        return None
    try:
        return validate_template_contract(attestation)
    except ValueError as exc:
        raise BackendError(f"invalid server template attestation: {exc}") from exc


def _prompt_integrity_metadata(
    rendered_prompt: str,
    *,
    server_template_contract: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    """Describe local and endpoint-side prompt integrity without conflating them."""

    local_contract = local_template_contract()
    return {
        "prompt_integrity_schema": "1",
        "local_template_contract": local_contract,
        "local_rendered_prompt_sha256": prompt_sha256(rendered_prompt),
        # Neither OpenAI-compatible response shape exposes the serialized
        # prompt, so this must remain null even when the template is attested.
        "wire_prompt_sha256": None,
        "wire_template_status": (
            "server_template_attested"
            if server_template_contract is not None
            else "server_template_unverified"
        ),
        "server_template_contract": dict(server_template_contract or {}),
    }


def fetch_endpoint_template_contract(
    *,
    base_url: str,
    timeout_seconds: float = 10.0,
    allow_network: bool = False,
) -> dict[str, str]:
    """Fetch and validate the pod-side template attestation.

    A thin pod wrapper should expose ``GET /bbf/template-contract`` and
    return the checked-in contract fields plus
    ``verification_method: "endpoint_contract"``.  This is intentionally an
    explicit, opt-in call: the normal completion endpoint cannot prove which
    chat template it used.
    """

    if not allow_network:
        raise BackendError("network backend disabled; pass allow_network=True explicitly")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/bbf/template-contract", method="GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            attestation = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise BackendError(f"template contract probe failed: {exc}") from exc
    try:
        return validate_template_contract(attestation)
    except ValueError as exc:
        raise BackendError(f"template contract probe returned an invalid attestation: {exc}") from exc


def make_request(
    *,
    run_id: str,
    model_id: str,
    model_revision: str,
    dataset_item_id: str,
    condition: Any,
    messages: Sequence[Mapping[str, str]],
    decoding: Optional[DecodingParams] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> EpisodeRequest:
    """Build a request and derive its prompt hash from safe canonical messages.

    This is the last common boundary before either the scripted or network
    backend.  Escaping here prevents a caller from validating a separately
    rendered prompt while accidentally sending raw role-control tokens in the
    structured ``messages`` payload.
    """
    from .schemas import Block, Condition, Message
    from .rendering import prepare_messages_for_upstream_template

    resolved_condition = Condition(condition)
    resolved_decoding = decoding or DecodingParams()
    safe_messages = prepare_messages_for_upstream_template(
        messages,
        require_agent_roles=resolved_condition.block == Block.AGENT,
    )
    normalized_messages = [Message.model_validate(dict(message)) for message in safe_messages]
    prompt_hash = sha256_text(canonical_json([message.model_dump(mode="json") for message in normalized_messages]))
    request_key = RequestKey(
        run_id=run_id,
        victim_model_id=model_id,
        victim_revision=model_revision,
        dataset_item_id=dataset_item_id,
        condition=resolved_condition,
        prompt_hash=prompt_hash,
        decoding_seed=resolved_decoding.seed,
    )
    return EpisodeRequest(
        request_key=request_key,
        block=resolved_condition.block,
        condition=resolved_condition,
        model_id=model_id,
        model_revision=model_revision,
        messages=normalized_messages,
        prompt_hash=prompt_hash,
        metadata=dict(metadata or {}),
        decoding=resolved_decoding,
    )


@dataclass
class InferenceSummary:
    attempted: int = 0
    skipped: int = 0
    completed: int = 0
    failed: int = 0
    retries: int = 0


class InferenceRunner:
    """Run requests with per-request resume and immutable attempt records."""

    def __init__(self, backend: Backend, artifacts: RunArtifactStore, *, max_retries: int = 0) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be nonnegative")
        self.backend = backend
        self.artifacts = artifacts
        self.max_retries = max_retries

    def run(self, requests: Iterable[EpisodeRequest]) -> InferenceSummary:
        summary = InferenceSummary()
        for request in requests:
            if self.artifacts.is_complete(request.request_key):
                summary.skipped += 1
                continue
            summary.attempted += 1
            finished = False
            attempt = 0
            generic_retries = 0
            chat_truncation_retried = False
            while not finished:
                attempt += 1
                retry_request = request
                is_chat_truncation_retry = False
                if chat_truncation_retried:
                    # The flag is consumed immediately below; keeping the
                    # effective request local means the immutable RequestKey
                    # and prompt hash never change across attempts.
                    retry_decoding = request.decoding.model_copy(update={"max_new_tokens": 8192})
                    retry_request = request.model_copy(update={"decoding": retry_decoding})
                    is_chat_truncation_retry = True
                    chat_truncation_retried = False
                if attempt > 1:
                    summary.retries += 1
                started = datetime.now(timezone.utc)
                try:
                    response = self.backend.complete(retry_request)
                    record = GenerationRecord(
                        request_key=request.request_key,
                        attempt=attempt,
                        status=response.status,
                        output_text=response.output_text,
                        thinking_text=response.thinking_text,
                        tool_calls=response.tool_calls,
                        stop_reason=response.stop_reason,
                        input_tokens=response.input_tokens,
                        output_tokens=response.output_tokens,
                        started_at=started,
                        finished_at=datetime.now(timezone.utc),
                        metadata={
                            **response.metadata,
                            "attempt_of": request.request_key.value,
                            "max_new_tokens": retry_request.decoding.max_new_tokens,
                            **(
                                {
                                    "retry_reason": "chat_truncation",
                                    "original_max_new_tokens": request.decoding.max_new_tokens,
                                }
                                if is_chat_truncation_retry
                                else {}
                            ),
                        },
                    )
                except Exception as exc:  # backend errors become durable records
                    record = GenerationRecord(
                        request_key=request.request_key,
                        attempt=attempt,
                        status=RecordStatus.ERROR,
                        started_at=started,
                        finished_at=datetime.now(timezone.utc),
                        error=f"{type(exc).__name__}: {exc}",
                        metadata={
                            "attempt_of": request.request_key.value,
                            "max_new_tokens": retry_request.decoding.max_new_tokens,
                            **(
                                {
                                    "retry_reason": "chat_truncation",
                                    "original_max_new_tokens": request.decoding.max_new_tokens,
                                }
                                if is_chat_truncation_retry
                                else {}
                            ),
                        },
                    )
                if record.status == RecordStatus.ERROR:
                    generic_retries += 1
                    will_retry = generic_retries <= self.max_retries
                    self.artifacts.write_generation_attempt(record, terminal=not will_retry)
                    if will_retry:
                        continue
                    summary.failed += 1
                    finished = True
                    break
                # A first 4096-token chat truncation is retained as an
                # immutable attempt, then retried once at 8192.  It must not
                # claim the completed-record slot before the terminal retry.
                can_retry_chat = (
                    not is_chat_truncation_retry
                    and not chat_truncation_retried
                    and request.condition.block.value == "chat"
                    and request.decoding.max_new_tokens == 4096
                    and record.status == RecordStatus.TRUNCATED
                )
                self.artifacts.write_generation_attempt(record, terminal=not can_retry_chat)
                if can_retry_chat:
                    chat_truncation_retried = True
                    continue
                summary.completed += 1
                finished = True
                break
            if not finished:
                summary.failed += 1
        return summary


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def split_thinking(text: str) -> tuple[Optional[str], str]:
    """Split a Qwen-style ``<think>`` span without judging its contents."""
    match = _THINK_RE.search(text)
    if not match:
        return None, text
    visible = (text[: match.start()] + text[match.end() :]).strip()
    return match.group(1).strip(), visible
