"""Safe, deterministic rendering for the Qwen ``input``-role experiment.

The target checkpoints must receive exactly the same serialized prompt.  This
module intentionally has no tokenizer or Jinja dependency: it mirrors the
small, stable part of the copied chat template and makes the security
boundary explicit in Python.  The checked-in Jinja file is used by the pod
runner, while this renderer is the local validation/reference implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


QWEN_IM_START = "<|im_start|>"
QWEN_IM_END = "<|im_end|>"
QWEN_ASSISTANT = "<|assistant|>"
QWEN_USER = "<|user|>"
QWEN_INPUT = "<|input|>"
QWEN_SYSTEM = "<|system|>"
QWEN_TOOL = "<|tool|>"
QWEN_END_OF_TEXT = "<|endoftext|>"

# Roles supported by the frozen template.  ``input`` is deliberately kept as
# a role rather than smuggling untrusted page data into a user message.
ALLOWED_ROLES = frozenset({"system", "user", "input", "assistant", "tool"})
AGENT_DATA_ROLE = "input"
AGENT_TASK_ROLE = "user"

# U+2063 (INVISIBLE SEPARATOR) splits a textual control token without changing
# the visible text.  Breaking every ``<|`` sequence also handles future Qwen
# control-token names not listed here.  This is done before serialization and
# is therefore applied to both trusted and untrusted content.
_CONTROL_PREFIX = "<|"
_BROKEN_CONTROL_PREFIX = "<\u2063|"
_KNOWN_CONTROL_TOKEN_RE = re.compile(
    r"(?:<\|(?:im_start|im_end|assistant|user|input|system|tool|endoftext)\|>)"
    r"|(?:</s>|<s>|<\|end\|>)"
)


class PromptContractError(ValueError):
    """Raised when a message sequence violates the role/template contract."""


class PromptMismatchError(PromptContractError):
    """Raised when paired checkpoints would receive different prompt bytes."""


@dataclass(frozen=True)
class PromptMessage:
    """A minimal role/content message accepted by the frozen template."""

    role: str
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


def _message_parts(message: PromptMessage | Mapping[str, object]) -> tuple[str, str]:
    if isinstance(message, PromptMessage):
        role, content = message.role, message.content
    elif isinstance(message, Mapping):
        role, content = message.get("role"), message.get("content")
    else:
        raise PromptContractError("messages must be PromptMessage objects or mappings")
    if not isinstance(role, str) or role not in ALLOWED_ROLES:
        raise PromptContractError(f"unsupported message role: {role!r}")
    if not isinstance(content, str):
        raise PromptContractError(f"message content for {role!r} must be text")
    if "\x00" in content:
        raise PromptContractError("NUL bytes are not valid prompt content")
    return role, content


def escape_control_tokens(content: str) -> str:
    """Escape Qwen role delimiters and other special-token spellings.

    Content is data, even when it comes from a trusted task.  In particular,
    an attacker must not be able to close an ``input`` message and open an
    assistant message by writing a control token into a webpage.  The
    invisible separator keeps ordinary rendered text readable while ensuring
    the tokenizer cannot see the exact special-token spelling.
    """

    if not isinstance(content, str):
        raise PromptContractError("prompt content must be text")
    escaped = content.replace(_CONTROL_PREFIX, _BROKEN_CONTROL_PREFIX)
    # These tokens do not all begin with ``<|`` and are worth breaking too.
    escaped = escaped.replace("</s>", "<\u2063/s>")
    escaped = escaped.replace("<s>", "<\u2063s>")
    return escaped


def contains_unescaped_control_token(content: str) -> bool:
    """Return whether *content* still contains a recognized special token."""

    return bool(_KNOWN_CONTROL_TOKEN_RE.search(content))


def validate_message_sequence(
    messages: Sequence[PromptMessage | Mapping[str, object]],
    *,
    require_agent_roles: bool = False,
) -> tuple[PromptMessage, ...]:
    """Validate and normalize messages before they reach a chat template.

    For an agent prompt, there must be a trusted ``user`` task followed by an
    ``input`` page.  The helper does not attempt to infer trust from arbitrary
    message text; callers should construct agent messages with
    :func:`build_agent_messages`.
    """

    normalized = tuple(PromptMessage(*_message_parts(message)) for message in messages)
    if not normalized:
        raise PromptContractError("at least one message is required")
    if require_agent_roles:
        roles = [message.role for message in normalized]
        if AGENT_TASK_ROLE not in roles or AGENT_DATA_ROLE not in roles:
            raise PromptContractError("agent prompts require user and input roles")
        input_positions = [index for index, role in enumerate(roles) if role == AGENT_DATA_ROLE]
        user_positions = [index for index, role in enumerate(roles) if role == AGENT_TASK_ROLE]
        if min(input_positions) <= min(user_positions):
            raise PromptContractError("input data must follow the trusted user task")
    return normalized


def build_agent_messages(
    trusted_task: str,
    webpage: str,
    *,
    system: str | None = None,
) -> tuple[PromptMessage, ...]:
    """Build the canonical agent message sequence.

    ``trusted_task`` is the application instruction.  ``webpage`` is always a
    distinct ``input`` message, including when it contains forged instructions.
    """

    if not isinstance(trusted_task, str) or not trusted_task.strip():
        raise PromptContractError("trusted task must be non-empty text")
    if not isinstance(webpage, str):
        raise PromptContractError("webpage must be text")
    messages: list[PromptMessage] = []
    if system is not None:
        if not isinstance(system, str) or not system.strip():
            raise PromptContractError("system prompt must be non-empty text when supplied")
        messages.append(PromptMessage("system", system))
    messages.extend((PromptMessage(AGENT_TASK_ROLE, trusted_task), PromptMessage(AGENT_DATA_ROLE, webpage)))
    return tuple(messages)


def _escape_nested_text(value: Any) -> Any:
    """Escape textual fields nested in tool-call/message metadata.

    The upstream template interpolates ``tool_calls`` as JSON.  Escaping only
    the top-level ``content`` would therefore leave a model-produced
    ``<|im_end|>`` inside tool-call arguments able to alter the wire format.
    """

    if isinstance(value, str):
        return escape_control_tokens(value)
    if isinstance(value, Mapping):
        return {str(key): _escape_nested_text(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_escape_nested_text(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_escape_nested_text(item) for item in value)
    return value


def prepare_messages_for_upstream_template(
    messages: Sequence[PromptMessage | Mapping[str, object]],
    *,
    require_agent_roles: bool = False,
) -> list[dict[str, Any]]:
    """Return escaped dictionaries safe to pass to the copied Jinja template.

    The upstream template is intentionally preserved byte-for-byte and does
    not escape interpolated content.  Pod code should call this function (or
    use the equivalent preprocessing step) before invoking vLLM's template;
    otherwise a webpage containing a special token could alter role framing.
    """

    normalized = validate_message_sequence(messages, require_agent_roles=require_agent_roles)
    prepared: list[dict[str, Any]] = []
    for original, message in zip(messages, normalized):
        item: dict[str, Any] = {
            "role": message.role,
            "content": escape_control_tokens(message.content),
        }
        # Preserve assistant/tool metadata needed by the exact template while
        # escaping every nested string before interpolation.
        if isinstance(original, Mapping):
            for key, value in original.items():
                if key not in {"role", "content"}:
                    item[str(key)] = _escape_nested_text(value)
        prepared.append(item)
    return prepared


def validate_agent_prompt(messages: Sequence[PromptMessage | Mapping[str, object]]) -> tuple[PromptMessage, ...]:
    """Explicit role-contract spelling for validators and CLI checks."""

    return validate_message_sequence(messages, require_agent_roles=True)


def serialize_qwen_messages(
    messages: Sequence[PromptMessage | Mapping[str, object]],
    *,
    add_generation_prompt: bool = True,
    enable_thinking: bool = True,
) -> str:
    """Render messages using the frozen Qwen3.6 input-role wire format.

    The function deliberately escapes *all* message content before joining it
    with role delimiters.  The resulting string is deterministic and suitable
    for byte-level paired-prompt assertions.
    """

    normalized = validate_message_sequence(messages)
    rendered: list[str] = []
    for message in normalized:
        content = escape_control_tokens(message.content)
        rendered.append(f"{QWEN_IM_START}{message.role}\n{content}{QWEN_IM_END}\n")
    if add_generation_prompt:
        rendered.append(f"{QWEN_IM_START}assistant\n")
        if enable_thinking:
            rendered.append("<think>\n")
        else:
            # Match the copied upstream template's explicit empty thinking
            # span when thinking is disabled.
            rendered.append("<think>\n\n</think>\n\n")
    result = "".join(rendered)
    # Only delimiters introduced by this renderer may remain in the prompt;
    # an exact control token in a content segment would indicate a regression.
    for message in normalized:
        if contains_unescaped_control_token(escape_control_tokens(message.content)):
            raise PromptContractError("control token escaped incorrectly")
    return result


def render_qwen36_chat_template(
    messages: Sequence[PromptMessage | Mapping[str, object]],
    *,
    add_generation_prompt: bool = True,
    enable_thinking: bool = True,
) -> str:
    """Named alias used by runners and tests."""

    return serialize_qwen_messages(
        messages,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )


def _upstream_json(value: Any) -> str:
    """Approximate Jinja's HTML-safe ``tojson`` filter for the offline path."""

    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("'", "\\u0027")
    )


def _render_upstream_compat(
    messages: Sequence[Mapping[str, Any]],
    *,
    tools: Sequence[Mapping[str, Any]],
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> str:
    """Render the checked-in template's finite message grammar without Jinja2.

    Jinja2 is intentionally not a mandatory runtime dependency for the CPU
    scaffold.  When it is installed, :func:`render_qwen36_upstream_template`
    executes the exact checked-in source.  This compatibility implementation
    follows that source's branches so local agent smoke remains fully offline,
    including tool definitions and multi-turn tool responses.
    """

    rendered: list[str] = []
    if tools:
        rendered.append(f"{QWEN_IM_START}system\n")
        if messages and messages[0].get("role") == "system":
            rendered.append(f"{messages[0].get('content', '')}\n\n")
        rendered.append(
            "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
        )
        for tool in tools:
            rendered.extend(("\n", _upstream_json(tool)))
        rendered.append(
            "\n</tools>\n\nFor each function call, return a json object with function name and arguments "
            "within <tool_call></tool_call> XML tags:\n<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
            f"{QWEN_IM_END}\n"
        )
    elif messages and messages[0].get("role") == "system":
        rendered.append(
            f"{QWEN_IM_START}system\n{messages[0].get('content', '')}{QWEN_IM_END}\n"
        )

    last_query_index = len(messages) - 1
    for reverse_index, message in enumerate(reversed(messages)):
        index = len(messages) - 1 - reverse_index
        if (
            message.get("role") == "user"
            and isinstance(message.get("content"), str)
            and not (
                message["content"].startswith("<tool_response>")
                and message["content"].endswith("</tool_response>")
            )
        ):
            last_query_index = index
            break

    for index, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content", "") if isinstance(message.get("content", ""), str) else ""
        if role in {"user", "input"} or (role == "system" and index != 0):
            rendered.append(f"{QWEN_IM_START}{role}\n{content}{QWEN_IM_END}\n")
            continue
        if role == "assistant":
            assistant_content = content
            reasoning_content = ""
            if isinstance(message.get("reasoning_content"), str):
                reasoning_content = message["reasoning_content"]
            elif "</think>" in assistant_content:
                reasoning_content = (
                    assistant_content.split("</think>")[0]
                    .rstrip("\n")
                    .split("<think>")[-1]
                    .lstrip("\n")
                )
                assistant_content = assistant_content.split("</think>")[-1].lstrip("\n")
            rendered.append(f"{QWEN_IM_START}assistant\n")
            if index > last_query_index and (index == len(messages) - 1 or reasoning_content):
                rendered.append(
                    "<think>\n"
                    + reasoning_content.strip("\n")
                    + "\n</think>\n\n"
                    + assistant_content.lstrip("\n")
                )
            else:
                rendered.append(assistant_content)
            for call_index, raw_call in enumerate(message.get("tool_calls") or ()):
                if (call_index == 0 and assistant_content) or call_index > 0:
                    rendered.append("\n")
                call = raw_call.get("function") if isinstance(raw_call, Mapping) and raw_call.get("function") else raw_call
                call = call if isinstance(call, Mapping) else {}
                name = call.get("name", "")
                arguments = call.get("arguments", {})
                rendered.append('<tool_call>\n{"name": "' + str(name) + '", "arguments": ')
                rendered.append(arguments if isinstance(arguments, str) else _upstream_json(arguments))
                rendered.append("}\n</tool_call>")
            rendered.append(f"{QWEN_IM_END}\n")
            continue
        if role == "tool":
            if index == 0 or messages[index - 1].get("role") != "tool":
                rendered.append(f"{QWEN_IM_START}user")
            rendered.append(f"\n<tool_response>\n{content}\n</tool_response>")
            if index == len(messages) - 1 or messages[index + 1].get("role") != "tool":
                rendered.append(f"{QWEN_IM_END}\n")

    if add_generation_prompt:
        rendered.append(f"{QWEN_IM_START}assistant\n")
        rendered.append("<think>\n" if enable_thinking else "<think>\n\n</think>\n\n")
    return "".join(rendered)


def render_qwen36_upstream_template(
    messages: Sequence[PromptMessage | Mapping[str, object]],
    *,
    tools: Sequence[Mapping[str, Any]] | None = None,
    add_generation_prompt: bool = True,
    enable_thinking: bool = True,
) -> str:
    """Render the exact checked-in Qwen template, including tool turns.

    If Jinja2 is present (for example in a pod/dev environment), this executes
    the copied source directly.  The dependency-free compatibility branch
    follows the same frozen grammar for local CPU-only tests.
    """

    validate_message_sequence(messages)
    prepared = prepare_messages_for_upstream_template(messages)
    tool_defs = list(tools or ())
    try:
        from jinja2 import Environment, StrictUndefined  # type: ignore
    except ImportError:
        return _render_upstream_compat(
            prepared,
            tools=tool_defs,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
    source = template_path().read_text(encoding="utf-8")
    environment = Environment(undefined=StrictUndefined, autoescape=False)
    return environment.from_string(source).render(
        messages=prepared,
        tools=tool_defs,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )


# Compatibility aliases make the exact path discoverable to pod adapters.
render_with_upstream_template = render_qwen36_upstream_template
render_qwen36_chat_template_exact = render_qwen36_upstream_template


def render_agent_prompt(
    trusted_task: str,
    webpage: str,
    *,
    system: str | None = None,
    enable_thinking: bool = True,
) -> str:
    """Render the canonical input-role agent prompt."""

    return serialize_qwen_messages(
        build_agent_messages(trusted_task, webpage, system=system),
        enable_thinking=enable_thinking,
    )


def prompt_sha256(rendered_prompt: str) -> str:
    """Return the canonical SHA-256 used in request keys and manifests."""

    if not isinstance(rendered_prompt, str):
        raise TypeError("rendered prompt must be text")
    return hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest()


def assert_paired_byte_identical(*rendered_prompts: str) -> str:
    """Assert that all paired prompts are byte-identical and return their hash."""

    if len(rendered_prompts) < 2:
        raise PromptMismatchError("at least two paired prompts are required")
    if any(not isinstance(prompt, str) for prompt in rendered_prompts):
        raise PromptMismatchError("paired prompts must be text")
    first = rendered_prompts[0].encode("utf-8")
    if any(prompt.encode("utf-8") != first for prompt in rendered_prompts[1:]):
        digests = [prompt_sha256(prompt) for prompt in rendered_prompts]
        raise PromptMismatchError(f"paired prompts differ (sha256={digests})")
    return hashlib.sha256(first).hexdigest()


# A shorter name is convenient for validation CLIs and preserves compatibility
# with early local prototypes.
assert_paired_prompts_identical = assert_paired_byte_identical


def template_path() -> Path:
    """Return the repository copy of the frozen Jinja template."""

    return Path(__file__).resolve().parents[2] / "configs" / "qwen36_input_role_chat_template.jinja"


__all__ = [
    "ALLOWED_ROLES",
    "PromptContractError",
    "PromptMessage",
    "PromptMismatchError",
    "assert_paired_byte_identical",
    "assert_paired_prompts_identical",
    "build_agent_messages",
    "contains_unescaped_control_token",
    "escape_control_tokens",
    "prompt_sha256",
    "prepare_messages_for_upstream_template",
    "render_agent_prompt",
    "render_qwen36_chat_template",
    "render_qwen36_chat_template_exact",
    "render_qwen36_upstream_template",
    "render_with_upstream_template",
    "serialize_qwen_messages",
    "template_path",
    "validate_agent_prompt",
    "validate_message_sequence",
]
