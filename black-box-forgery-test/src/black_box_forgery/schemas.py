"""Versioned data contracts used by the local runner and artifact store.

The schemas intentionally describe records rather than model-specific client
objects.  A record can therefore be produced by a real vLLM/OpenAI-compatible
client or by the deterministic scripted backend used in local smoke tests.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


SCHEMA_VERSION = "1.0"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class BBFModel(BaseModel):
    """Common Pydantic configuration for persisted records."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)


class Block(str, Enum):
    CHAT = "chat"
    AGENT = "agent"


class Condition(str, Enum):
    RAW_CHAT = "raw_chat"
    COT_CHAT = "cot_chat"
    CLEAN_AGENT = "clean_agent"
    STANDARD_AGENT = "standard_agent"
    COT_AGENT = "cot_agent"

    @property
    def block(self) -> Block:
        return Block.CHAT if self in (self.RAW_CHAT, self.COT_CHAT) else Block.AGENT


class ChatOutcome(str, Enum):
    HARMFUL_RESPONSE = "HARMFUL_RESPONSE"
    REFUSAL = "REFUSAL"
    REDIRECTION = "REDIRECTION"


class AgentOutcome(str, Enum):
    UNNOTICED = "UNNOTICED"
    DENIED = "DENIED"
    ATTEMPTED_UNSUCCESSFUL = "ATTEMPTED_UNSUCCESSFUL"
    ATTEMPTED_SUCCESSFUL = "ATTEMPTED_SUCCESSFUL"


class RecordStatus(str, Enum):
    COMPLETE = "complete"
    ERROR = "error"
    INVALID = "invalid"
    TRUNCATED = "truncated"
    LIMIT_TERMINATED = "limit_terminated"


class Message(BBFModel):
    role: str
    content: str
    name: Optional[str] = None

    @field_validator("role")
    @classmethod
    def valid_role(cls, value: str) -> str:
        if value not in {"system", "user", "input", "assistant", "tool"}:
            raise ValueError("message role must be system, user, input, assistant, or tool")
        return value


class DecodingParams(BBFModel):
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int = 123
    max_new_tokens: int = Field(default=4096, gt=0)
    max_turns: int = Field(default=8, gt=0)
    thinking_enabled: bool = True
    extra: Dict[str, Any] = Field(default_factory=dict)


class RequestKey(BBFModel):
    """Identity of a logical request; attempts never change this identity."""

    run_id: str = Field(min_length=1)
    victim_model_id: str = Field(min_length=1)
    victim_revision: str = Field(min_length=1)
    dataset_item_id: str = Field(min_length=1)
    condition: Condition
    prompt_hash: str = Field(min_length=8)
    decoding_seed: int

    @property
    def value(self) -> str:
        """Stable, filesystem-safe request key."""
        parts = (
            self.run_id,
            self.victim_model_id,
            self.victim_revision,
            self.dataset_item_id,
            self.condition.value,
            self.prompt_hash,
            str(self.decoding_seed),
        )
        return "__".join(_safe_key_part(part) for part in parts)


def _safe_key_part(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value)


class EpisodeRequest(BBFModel):
    schema_version: str = SCHEMA_VERSION
    request_key: RequestKey
    block: Block
    condition: Condition
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    messages: List[Message] = Field(default_factory=list)
    prompt_hash: str = Field(min_length=8)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    decoding: DecodingParams = Field(default_factory=DecodingParams)

    @field_validator("condition")
    @classmethod
    def condition_matches_block(cls, value: Condition, info: Any) -> Condition:
        block = info.data.get("block")
        if block is not None and value.block != block:
            raise ValueError("condition does not belong to the declared block")
        return value


class ToolCall(BBFModel):
    id: str
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolEvent(BBFModel):
    event_id: str
    step: int = Field(ge=0)
    kind: str
    name: Optional[str] = None
    arguments: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Any] = None
    timestamp: datetime = Field(default_factory=utc_now)


class GenerationRecord(BBFModel):
    schema_version: str = SCHEMA_VERSION
    request_key: RequestKey
    attempt: int = Field(default=1, ge=1)
    status: RecordStatus
    output_text: str = ""
    thinking_text: Optional[str] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)
    stop_reason: Optional[str] = None
    input_tokens: Optional[int] = Field(default=None, ge=0)
    output_tokens: Optional[int] = Field(default=None, ge=0)
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime = Field(default_factory=utc_now)
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentEpisodeRecord(GenerationRecord):
    events: List[ToolEvent] = Field(default_factory=list)
    agent_outcome: Optional[AgentOutcome] = None
    fetched_page: bool = False
    summary_nonempty: bool = False
    unrelated_sensitive_action: bool = False


class JudgmentRecord(BBFModel):
    schema_version: str = SCHEMA_VERSION
    request_key: RequestKey
    generation_attempt: int = Field(ge=1)
    label: Optional[str] = None
    valid: bool = True
    judge_model: Optional[str] = None
    judge_revision: Optional[str] = None
    rationale: Optional[str] = None
    raw_judgment: Optional[str] = None
    timestamp: datetime = Field(default_factory=utc_now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Provenance(BBFModel):
    source: str
    source_type: str = "local_fixture"
    source_revision: Optional[str] = None
    acquired_at: datetime = Field(default_factory=utc_now)
    sha256: Optional[str] = None
    license: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FixturePage(BBFModel):
    sample_id: str
    title: str
    url: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    provenance: Provenance


class HarmfulRequest(BBFModel):
    item_id: str
    prompt: str
    category: str = "unknown"
    source: str = "local_fixture"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    provenance: Optional[Provenance] = None


class InjectionAssignment(BBFModel):
    sample_id: str
    injection_type: str
    prompt: str
    seed: int
    source_sha256: Optional[str] = None


class RunManifest(BBFModel):
    schema_version: str = SCHEMA_VERSION
    run_id: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    master_seed: int = 20260903
    victim_model_id: Optional[str] = None
    victim_revision: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    config_path: Optional[str] = None
    git_commit: Optional[str] = None
    dirty_worktree_sha256: Optional[str] = None
    source_revisions: Dict[str, str] = Field(default_factory=dict)
    artifact_hashes: Dict[str, str] = Field(default_factory=dict)
    # Frozen-input and rendering provenance is explicit so a run can be
    # audited without reconstructing it from free-form notes.
    page_ids: List[str] = Field(default_factory=list)
    page_hashes: Dict[str, str] = Field(default_factory=dict)
    template_hashes: Dict[str, str] = Field(default_factory=dict)
    injection_hashes: Dict[str, str] = Field(default_factory=dict)
    paired_prompt_validation: Dict[str, Any] = Field(default_factory=dict)
    decoding: DecodingParams = Field(default_factory=DecodingParams)
    environment: Dict[str, Any] = Field(default_factory=dict)
    counts: Dict[str, int] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)


def model_to_dict(model: BaseModel) -> Dict[str, Any]:
    """Return JSON-compatible data for any persisted schema model."""
    return model.model_dump(mode="json")


def model_from_dict(model_type: Any, value: Dict[str, Any]) -> Any:
    """Small wrapper used by storage to keep validation at artifact boundaries."""
    return model_type.model_validate(value)
