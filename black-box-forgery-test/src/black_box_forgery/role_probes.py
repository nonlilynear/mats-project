"""Role-probe training and projection helpers for the SecOPD agent study.

This module follows the role-probe construction in Ye, Cui, and
Hadfield-Menell, but uses the Qwen/SecOPD wire format already frozen in this
repository.  In particular, ``input`` is a real probe class here; it is not
silently treated as ``tool``.  Qwen's ``tool`` message is retained as a
separate class because the checked-in template serializes it as a
``<tool_response>`` inside a user block.

Heavy dependencies (PyTorch, Transformers, scikit-learn) are imported lazily
so the offline scaffold and its unit tests remain CPU/dependency free.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
import random
from typing import Any, Callable, Mapping, Sequence

from .rendering import (
    contains_unescaped_control_token,
    render_qwen36_upstream_template,
)


PAPER_ROLES = ("system", "user", "cot", "assistant", "tool")
SECOPD_ROLES = ("system", "user", "cot", "assistant", "input", "tool")
DEFAULT_ROLE_SPACE = SECOPD_ROLES


class RoleProbeError(ValueError):
    """Raised when role-probe inputs or artifacts violate the probe contract."""


@dataclass(frozen=True)
class RoleProbeExample:
    """One role-wrapped copy of a neutral base sequence."""

    base_seq_ix: int
    role: str
    content: str
    rendered_prompt: str
    input_ids: tuple[int, ...]
    target_token_ids: tuple[int, ...]
    target_start: int
    target_end: int


@dataclass(frozen=True)
class ProbeTrainingConfig:
    """Reproducible defaults matching the paper's Qwen probe setting."""

    seed: int = 123
    sequence_length: int = 1024
    test_fraction: float = 0.10
    c: float = 0.1
    max_iter: int = 5000
    activation_source: str = "pre_mlp_post_attention_layernorm"


@dataclass
class TrainedRoleProbe:
    """A fitted multinomial probe and its held-out validation metadata."""

    layer_index: int
    role_space: tuple[str, ...]
    roles_map: dict[str, int]
    classifier: Any
    test_accuracy: float
    test_count: int
    train_count: int


class TorchLinearRoleClassifier:
    """CPU-serializable softmax linear classifier fitted with PyTorch.

    The paper uses multinomial logistic regression.  This class stores only
    the learned CPU weights and implements the small ``predict_proba`` API
    needed by the projection code, so probe artifacts remain loadable without
    a CUDA device.
    """

    def __init__(self, weights: Any, bias: Any, *, chunk_size: int = 65536) -> None:
        self.weights = weights
        self.bias = bias
        self.chunk_size = int(chunk_size)

    def predict_proba(self, features: Any) -> Any:
        import numpy as np  # type: ignore
        import torch  # type: ignore

        values = torch.as_tensor(features, dtype=torch.float32)
        weights = torch.as_tensor(self.weights, dtype=torch.float32)
        bias = torch.as_tensor(self.bias, dtype=torch.float32)
        device = values.device
        weights = weights.to(device)
        bias = bias.to(device)
        probabilities: list[Any] = []
        with torch.inference_mode():
            for offset in range(0, values.shape[0], self.chunk_size):
                logits = values[offset : offset + self.chunk_size] @ weights.T + bias
                probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())
        if not probabilities:
            return np.empty((0, weights.shape[0]), dtype=np.float32)
        return np.concatenate(probabilities, axis=0)


def _require_probe_dependencies() -> tuple[Any, Any]:
    try:
        import torch  # type: ignore
        from sklearn.linear_model import LogisticRegression  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised in GPU envs
        raise RoleProbeError(
            "role probes require torch, numpy, and scikit-learn; "
            "install the 'role-probes' extra"
        ) from exc
    return torch, LogisticRegression


def find_subsequence(sequence: Sequence[int], subsequence: Sequence[int]) -> tuple[int, int]:
    """Return the unique half-open span of ``subsequence`` in ``sequence``."""

    if not subsequence:
        raise RoleProbeError("cannot locate an empty target token sequence")
    matches = [
        start
        for start in range(len(sequence) - len(subsequence) + 1)
        if tuple(sequence[start : start + len(subsequence)]) == tuple(subsequence)
    ]
    if len(matches) != 1:
        raise RoleProbeError(
            f"expected one target-token span, found {len(matches)} matches"
        )
    start = matches[0]
    return start, start + len(subsequence)


def _render_role_variant(
    role: str,
    content: str,
    *,
    partner_text: str = "neutral context",
    position_filler: str = "",
) -> str:
    """Render one target sequence using the frozen Qwen/SecOPD template.

    The assistant variants include a preceding user turn because the Qwen
    template only emits a reasoning block for an assistant turn following a
    user query.  For the final-answer variant, ``partner_text`` occupies the
    reasoning block as the paper's nested-reasoning positional control.
    """

    if position_filler:
        # Keep the padding inside the native role scaffold. A separator avoids
        # merging the final filler subword with the first target subword.
        target_content = position_filler + "\n" + content
    else:
        target_content = content
    if role in {"system", "user", "input", "tool"}:
        messages: list[dict[str, Any]] = [{"role": role, "content": target_content}]
    elif role == "cot":
        messages = [
            {"role": "user", "content": partner_text},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": target_content,
            },
        ]
    elif role == "assistant":
        messages = [
            {"role": "user", "content": partner_text},
            {
                "role": "assistant",
                "content": content,
                "reasoning_content": (
                    position_filler + "\n" + partner_text
                    if position_filler
                    else partner_text
                ),
            },
        ]
    else:
        raise RoleProbeError(f"unsupported probe role: {role!r}")
    return render_qwen36_upstream_template(messages, add_generation_prompt=False)


def _token_ids(tokenizer: Any, text: str) -> tuple[int, ...]:
    """Tokenize text without adding a BOS/EOS token."""

    values = tokenizer(text, add_special_tokens=False)["input_ids"]
    if hasattr(values, "tolist"):
        values = values.tolist()
    return tuple(int(value) for value in values)


def _decode_token_ids(tokenizer: Any, token_ids: Sequence[int]) -> str:
    """Decode a token prefix while preserving its token boundaries."""

    try:
        return tokenizer.decode(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        # Small test tokenizers and older Transformers do not all expose the
        # cleanup keyword, but still provide the required decode operation.
        return tokenizer.decode(list(token_ids), skip_special_tokens=False)


def _stable_filler_token_ids(tokenizer: Any, filler_text: str) -> tuple[int, ...]:
    """Find one ordinary token that remains one token when repeated.

    Repeating an arbitrary BPE sequence and decoding it can change token
    boundaries at each repetition. A single stable token makes prefix length
    predictable and keeps the alignment search cheap. The fallback retains
    the supplied neutral text for tokenizers without a stable candidate.
    """

    candidate_texts = (filler_text, " neutral", " the", " x", "0")
    special_ids = set(getattr(tokenizer, "all_special_ids", ()) or ())
    seen: set[int] = set()
    for candidate_text in candidate_texts:
        for token_id in _token_ids(tokenizer, candidate_text):
            if token_id in seen or token_id in special_ids:
                continue
            seen.add(token_id)
            repeated = (token_id,) * 32
            decoded = _decode_token_ids(tokenizer, repeated)
            if _token_ids(tokenizer, decoded) == repeated:
                return (token_id,)
    return _token_ids(tokenizer, filler_text)


def _position_controlled_variants(
    role_renderers: Mapping[str, Callable[[str], str]],
    target_ids: Sequence[int],
    tokenizer: Any,
    *,
    filler_text: str,
) -> dict[str, tuple[str, tuple[int, ...], int, int]]:
    """Align role targets to one token position using neutral prefix filler.

    Qwen's native reasoning structure necessarily places CoT and final
    Assistant content behind nested context. The matching-filler control
    therefore adds neutral padding inside each role's native message or
    thinking scaffold, so a probe cannot classify role from absolute position
    or from an artificial untagged prefix. We choose the latest unpadded target
    start and add decoded neutral prefixes to earlier variants until every
    target starts there exactly.
    """

    if not role_renderers:
        raise RoleProbeError("at least one role rendering is required")
    raw_renderings = {role: renderer("") for role, renderer in role_renderers.items()}
    tokenized = {role: _token_ids(tokenizer, rendered) for role, rendered in raw_renderings.items()}
    raw_spans = {
        role: find_subsequence(input_ids, target_ids)
        for role, input_ids in tokenized.items()
    }
    filler_ids = _stable_filler_token_ids(tokenizer, filler_text)
    if not filler_ids:
        raise RoleProbeError("positional-control filler tokenizes to an empty sequence")

    # Give every role a small common scaffold filler, then anchor the target
    # position to the latest unpadded role. This avoids an exhaustive search
    # over thousands of BPE prefixes while still checking the actual rendered
    # token span for every role.
    anchor_role = max(raw_spans, key=lambda role: raw_spans[role][0])
    baseline_filler_tokens = 32
    anchor_ids = (filler_ids * ((baseline_filler_tokens // len(filler_ids)) + 1))[
        :baseline_filler_tokens
    ]
    anchor_padding = _decode_token_ids(tokenizer, anchor_ids)
    anchor_rendered = role_renderers[anchor_role](anchor_padding)
    anchor_input_ids = _token_ids(tokenizer, anchor_rendered)
    target_position = find_subsequence(anchor_input_ids, target_ids)[0]

    aligned: dict[str, tuple[str, tuple[int, ...], int, int]] = {}
    for role, renderer in role_renderers.items():
        raw_start, _ = raw_spans[role]
        nominal_count = baseline_filler_tokens + raw_spans[anchor_role][0] - raw_start
        found: tuple[str, tuple[int, ...], int, int] | None = None
        for prefix_count in range(max(0, nominal_count - 16), nominal_count + 17):
            repeated_ids = (filler_ids * ((prefix_count // len(filler_ids)) + 1))[
                :prefix_count
            ]
            prefix = _decode_token_ids(tokenizer, repeated_ids)
            candidate = renderer(prefix)
            candidate_ids = _token_ids(tokenizer, candidate)
            try:
                start, end = find_subsequence(candidate_ids, target_ids)
            except RoleProbeError:
                continue
            if start == target_position:
                found = (candidate, candidate_ids, start, end)
                break
        if found is None:
            raise RoleProbeError(
                f"could not align {role!r} target to anchored token position "
                f"{target_position}; raw_start={raw_start}, "
                f"nominal_prefix_tokens={nominal_count}"
            )
        aligned[role] = found
    return aligned


def build_probe_examples(
    texts: Sequence[str],
    tokenizer: Any,
    *,
    roles: Sequence[str] = DEFAULT_ROLE_SPACE,
    sequence_length: int = 1024,
    seed: int = 123,
) -> list[RoleProbeExample]:
    """Create token-aligned role variants from neutral text.

    Text is first token-truncated and decoded, as in the upstream demo.  The
    target span is then found in the fully rendered prompt, which avoids
    assuming that every role's framing has the same number of tokens.
    """

    if not texts:
        raise RoleProbeError("at least one neutral text sequence is required")
    role_tuple = tuple(roles)
    if not role_tuple or len(set(role_tuple)) != len(role_tuple):
        raise RoleProbeError("roles must be a nonempty sequence of unique names")
    unknown = set(role_tuple) - set(SECOPD_ROLES)
    if unknown:
        raise RoleProbeError(f"unsupported probe roles: {sorted(unknown)}")
    if sequence_length <= 0:
        raise RoleProbeError("sequence_length must be positive")

    normalized: list[str] = []
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue
        if contains_unescaped_control_token(text):
            # Neutral data containing template control tokens would not be a
            # neutral role probe example and could terminate a role wrapper.
            continue
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=sequence_length,
        )["input_ids"]
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        decoded = tokenizer.decode(encoded, skip_special_tokens=False)
        if decoded.strip():
            # The native Qwen template trims rendered content. Normalize the
            # decoded text the same way before computing target token spans so
            # leading/trailing web-corpus whitespace cannot desynchronize the
            # labels from the template's actual content tokens.
            normalized.append(decoded.strip())
    if not normalized:
        raise RoleProbeError("no usable neutral text remained after token filtering")

    rng = random.Random(seed)
    examples: list[RoleProbeExample] = []
    for base_seq_ix, content in enumerate(normalized):
        if len(normalized) > 1:
            partner_index = rng.randrange(len(normalized) - 1)
            if partner_index >= base_seq_ix:
                partner_index += 1
            partner = normalized[partner_index % len(normalized)]
        else:
            partner = "neutral context"
        target_ids = _token_ids(tokenizer, content)
        role_renderers = {
            role: (
                lambda position_filler, role=role, content=content, partner=partner: _render_role_variant(
                    role,
                    content,
                    partner_text=partner,
                    position_filler=position_filler,
                )
            )
            for role in role_tuple
        }
        # Use a distinct neutral sequence as filler where possible. The
        # repeated filler is context, not a role-labeled training target.
        filler = partner if content not in partner else "neutral positional context"
        aligned = _position_controlled_variants(
            role_renderers,
            target_ids,
            tokenizer,
            filler_text=filler,
        )
        for role in role_tuple:
            rendered, input_ids, start, end = aligned[role]
            examples.append(
                RoleProbeExample(
                    base_seq_ix=base_seq_ix,
                    role=role,
                    content=content,
                    rendered_prompt=rendered,
                    input_ids=input_ids,
                    target_token_ids=target_ids,
                    target_start=start,
                    target_end=end,
                )
            )
    return examples


def split_base_sequences(
    examples: Sequence[RoleProbeExample],
    *,
    test_fraction: float = 0.10,
    seed: int = 123,
) -> tuple[set[int], set[int]]:
    """Split by neutral base sequence, preventing role-variant leakage."""

    if not 0 < test_fraction < 1:
        raise RoleProbeError("test_fraction must be between zero and one")
    groups = sorted({example.base_seq_ix for example in examples})
    if len(groups) < 2:
        raise RoleProbeError("at least two base sequences are needed for a split")
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_test = max(1, round(len(groups) * test_fraction))
    n_test = min(n_test, len(groups) - 1)
    test = set(groups[:n_test])
    return set(groups[n_test:]), test


def _model_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:  # pragma: no cover - malformed model
        raise RoleProbeError("model has no parameters") from exc


def _decoder_layers(model: Any) -> Sequence[Any]:
    # Qwen3.6's conditional-generation wrapper exposes the text decoder as
    # ``model.language_model.layers``. Keep the simpler decoder layouts used
    # by GPT-OSS/Qwen decoder-only classes as fallbacks.
    candidates = (
        ("model", "language_model", "layers"),
        ("language_model", "layers"),
        ("model", "layers"),
        ("transformer", "h"),
        ("model", "h"),
        ("layers",),
    )
    for path in candidates:
        parent = model
        for component in path:
            parent = getattr(parent, component, None)
            if parent is None:
                break
        if parent is not None:
            return parent
    raise RoleProbeError("could not find decoder layers for activation hooks")


def forward_pre_mlp_hidden_states(
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    *,
    layers: Sequence[int],
    allow_hidden_state_fallback: bool = False,
) -> dict[int, Any]:
    """Run a forward pass and return selected token activations.

    Qwen decoder layers expose ``post_attention_layernorm`` immediately before
    the MLP; this is the paper-compatible activation.  A fallback to
    ``outputs.hidden_states`` is available only when explicitly requested and
    is recorded by the caller as a different activation source.
    """

    torch, _ = _require_probe_dependencies()
    requested = tuple(sorted(set(int(layer) for layer in layers)))
    if not requested:
        raise RoleProbeError("at least one activation layer is required")
    decoder_layers = _decoder_layers(model)
    if min(requested) < 0 or max(requested) >= len(decoder_layers):
        raise RoleProbeError(
            f"requested layer outside model depth {len(decoder_layers)}: {requested}"
        )
    captured: dict[int, Any] = {}
    handles = []
    hookable = True
    for layer_index in requested:
        layer = decoder_layers[layer_index]
        module = getattr(layer, "post_attention_layernorm", None)
        if module is None:
            hookable = False
            break

        def capture(_module: Any, _inputs: Any, output: Any, *, index: int = layer_index) -> None:
            value = output[0] if isinstance(output, tuple) else output
            captured[index] = value.detach()

        handles.append(module.register_forward_hook(capture))
    if hookable:
        try:
            with torch.inference_mode():
                # Qwen3.6 accepts ``logits_to_keep``. Keeping one position
                # avoids allocating a full vocabulary-logit tensor for a
                # 20--30k-token webpage during projection. Older decoder
                # implementations may not accept it, so retain a fallback.
                try:
                    model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        logits_to_keep=1,
                    )
                except TypeError as exc:
                    if "logits_to_keep" not in str(exc):
                        raise
                    model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        return {index: captured[index] for index in requested}
    for handle in handles:
        handle.remove()
    if not allow_hidden_state_fallback:
        raise RoleProbeError(
            "model does not expose post_attention_layernorm hooks; "
            "rerun with allow_hidden_state_fallback=True only as a documented deviation"
        )
    with torch.inference_mode():
        try:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
                logits_to_keep=1,
            )
        except TypeError as exc:
            if "logits_to_keep" not in str(exc):
                raise
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
    hidden_states = outputs.hidden_states
    return {index: hidden_states[index + 1] for index in requested}


def _pad_examples(examples: Sequence[RoleProbeExample], pad_token_id: int, torch: Any) -> tuple[Any, Any]:
    max_length = max(len(example.input_ids) for example in examples)
    input_ids = torch.full(
        (len(examples), max_length), pad_token_id, dtype=torch.long
    )
    attention_mask = torch.zeros((len(examples), max_length), dtype=torch.long)
    for row, example in enumerate(examples):
        length = len(example.input_ids)
        input_ids[row, :length] = torch.tensor(example.input_ids, dtype=torch.long)
        attention_mask[row, :length] = 1
    return input_ids, attention_mask


def _release_model_storage(model: Any, torch: Any) -> None:
    """Release model parameter storage while retaining the caller's object."""

    # ``to_empty(meta)`` avoids copying the 27B checkpoint back through host
    # RAM before fitting a multi-layer activation sweep. The model is not used
    # again after this point in the torch fitting path.
    try:
        model.to_empty(device="meta")
    except (AttributeError, RuntimeError):  # pragma: no cover - old torch only
        model.to("cpu")
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _fit_torch_probe(
    features: Any,
    labels: Any,
    train_rows: Sequence[int],
    test_rows: Sequence[int],
    role_space: tuple[str, ...],
    *,
    layer_index: int,
    c: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device_name: str,
    seed: int,
) -> TrainedRoleProbe:
    """Fit the same L2-regularized softmax objective on a CUDA device."""

    torch, _ = _require_probe_dependencies()
    if c <= 0:
        raise RoleProbeError("c must be positive")
    if epochs <= 0:
        raise RoleProbeError("fit_epochs must be positive")
    if batch_size <= 0:
        raise RoleProbeError("fit_batch_size must be positive")
    if learning_rate <= 0:
        raise RoleProbeError("fit_learning_rate must be positive")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(device_name)
    except (RuntimeError, ValueError) as exc:
        raise RoleProbeError(f"invalid probe fit device: {device_name!r}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RoleProbeError("GPU probe fitting requested, but CUDA is unavailable")

    import numpy as np  # type: ignore

    features_array = np.asarray(features, dtype=np.float32)
    labels_array = np.asarray(labels, dtype=np.int64)
    train_index = np.asarray(train_rows, dtype=np.int64)
    test_index = np.asarray(test_rows, dtype=np.int64)
    if train_index.size == 0 or test_index.size == 0:
        raise RoleProbeError("both train and test rows are required for probe fitting")

    torch.manual_seed(seed)
    train_features = torch.as_tensor(
        features_array[train_index], dtype=torch.float32, device=device
    )
    train_labels = torch.as_tensor(labels_array[train_index], dtype=torch.long, device=device)
    classifier = torch.nn.Linear(train_features.shape[1], len(role_space), device=device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    sample_count = train_features.shape[0]

    classifier.train()
    for _ in range(epochs):
        order = torch.randperm(sample_count, device=device)
        for offset in range(0, sample_count, batch_size):
            rows = order[offset : offset + batch_size]
            logits = classifier(train_features[rows])
            loss = torch.nn.functional.cross_entropy(logits, train_labels[rows])
            # sklearn's objective is mean log loss + ||W||^2/(2*C*n).
            loss = loss + classifier.weight.square().sum() / (2.0 * c * sample_count)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    classifier.eval()
    correct = 0
    with torch.inference_mode():
        for offset in range(0, len(test_index), batch_size):
            rows = test_index[offset : offset + batch_size]
            test_features = torch.as_tensor(
                features_array[rows], dtype=torch.float32, device=device
            )
            test_labels = torch.as_tensor(labels_array[rows], dtype=torch.long, device=device)
            correct += int((classifier(test_features).argmax(dim=-1) == test_labels).sum())
    accuracy = correct / len(test_index)
    weights = classifier.weight.detach().cpu().numpy()
    bias = classifier.bias.detach().cpu().numpy()
    del train_features, train_labels, classifier, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return TrainedRoleProbe(
        layer_index=layer_index,
        role_space=role_space,
        roles_map={role: index for index, role in enumerate(role_space)},
        classifier=TorchLinearRoleClassifier(weights, bias),
        test_accuracy=float(accuracy),
        test_count=len(test_rows),
        train_count=len(train_rows),
    )


def _fit_probe(
    features: Any,
    labels: Any,
    train_rows: Sequence[int],
    test_rows: Sequence[int],
    role_space: tuple[str, ...],
    *,
    layer_index: int,
    c: float,
    max_iter: int,
    fit_backend: str = "sklearn",
    fit_epochs: int = 10,
    fit_batch_size: int = 8192,
    fit_learning_rate: float = 0.05,
    fit_device: str = "auto",
    seed: int = 123,
) -> TrainedRoleProbe:
    if fit_backend == "torch":
        return _fit_torch_probe(
            features,
            labels,
            train_rows,
            test_rows,
            role_space,
            layer_index=layer_index,
            c=c,
            epochs=fit_epochs,
            batch_size=fit_batch_size,
            learning_rate=fit_learning_rate,
            device_name=fit_device,
            seed=seed,
        )
    if fit_backend != "sklearn":
        raise RoleProbeError(f"unsupported probe fit backend: {fit_backend!r}")
    _, LogisticRegression = _require_probe_dependencies()
    classifier = LogisticRegression(
        C=c,
        penalty="l2",
        fit_intercept=True,
        max_iter=max_iter,
        random_state=seed,
    )
    classifier.fit(features[train_rows], labels[train_rows])
    accuracy = float(classifier.score(features[test_rows], labels[test_rows]))
    return TrainedRoleProbe(
        layer_index=layer_index,
        role_space=role_space,
        roles_map={role: index for index, role in enumerate(role_space)},
        classifier=classifier,
        test_accuracy=accuracy,
        test_count=len(test_rows),
        train_count=len(train_rows),
    )


def train_role_probes(
    model: Any,
    tokenizer: Any,
    examples: Sequence[RoleProbeExample],
    *,
    layers: Sequence[int],
    batch_size: int = 4,
    config: ProbeTrainingConfig = ProbeTrainingConfig(),
    allow_hidden_state_fallback: bool = False,
    fit_backend: str = "sklearn",
    fit_epochs: int = 10,
    fit_batch_size: int = 8192,
    fit_learning_rate: float = 0.05,
    fit_device: str = "auto",
) -> tuple[list[TrainedRoleProbe], dict[str, Any]]:
    """Extract target-token activations and fit one probe per selected layer."""

    torch, _ = _require_probe_dependencies()
    if not examples:
        raise RoleProbeError("cannot train probes without examples")
    layers = tuple(sorted(set(int(layer) for layer in layers)))
    if not layers:
        raise RoleProbeError("at least one activation layer is required")
    if batch_size <= 0:
        raise RoleProbeError("batch_size must be positive")
    if fit_backend not in {"sklearn", "torch"}:
        raise RoleProbeError(f"unsupported probe fit backend: {fit_backend!r}")
    role_space = tuple(dict.fromkeys(example.role for example in examples))
    if len(role_space) < 2:
        raise RoleProbeError("at least two role classes are required")
    train_groups, test_groups = split_base_sequences(
        examples, test_fraction=config.test_fraction, seed=config.seed
    )
    train_rows = [
        index for index, example in enumerate(examples) if example.base_seq_ix in train_groups
    ]
    test_rows = [
        index for index, example in enumerate(examples) if example.base_seq_ix in test_groups
    ]
    labels = [role_space.index(example.role) for example in examples]
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        raise RoleProbeError("tokenizer must define pad_token_id or eos_token_id")

    # Expand labels once. Each later layer reuses the same token-level split.
    expanded_labels: list[int] = []
    expanded_train_rows: list[int] = []
    expanded_test_rows: list[int] = []
    row = 0
    for example, label in zip(examples, labels):
        width = example.target_end - example.target_start
        expanded_labels.extend([label] * width)
        destination = expanded_train_rows if example.base_seq_ix in train_groups else expanded_test_rows
        destination.extend(range(row, row + width))
        row += width

    fitted: list[TrainedRoleProbe] = []
    if fit_backend == "torch" and len(layers) > 1:
        # A one-pass sweep is substantially faster than re-running the 27B
        # model for every layer. The pod has enough host RAM for the coarse
        # every-fourth-layer sweep; fitting still happens one layer at a time
        # on CUDA after the model storage is released.
        import numpy as np  # type: ignore

        feature_matrices: dict[int, Any] = {}
        feature_row = 0
        total_feature_rows = len(expanded_labels)
        for offset in range(0, len(examples), batch_size):
            batch = examples[offset : offset + batch_size]
            input_ids, attention_mask = _pad_examples(batch, int(pad_token_id), torch)
            device = _model_device(model)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            activations = forward_pre_mlp_hidden_states(
                model,
                input_ids,
                attention_mask,
                layers=layers,
                allow_hidden_state_fallback=allow_hidden_state_fallback,
            )
            if not feature_matrices:
                hidden_size = int(next(iter(activations.values())).shape[-1])
                feature_matrices = {
                    layer_index: np.empty(
                        (total_feature_rows, hidden_size), dtype=np.float32
                    )
                    for layer_index in layers
                }
            batch_feature_row = feature_row
            for row_index, example in enumerate(batch):
                width = example.target_end - example.target_start
                for layer_index in layers:
                    values = activations[layer_index]
                    feature_matrices[layer_index][
                        batch_feature_row : batch_feature_row + width
                    ] = (
                        values[row_index, example.target_start : example.target_end]
                        .float()
                        .cpu()
                        .numpy()
                    )
                batch_feature_row += width
            feature_row = batch_feature_row
        del input_ids, attention_mask, activations, values
        if feature_row != total_feature_rows:
            raise RoleProbeError(
                f"activation row count mismatch: wrote {feature_row}, expected {total_feature_rows}"
            )
        _release_model_storage(model, torch)
        label_array = np.asarray(expanded_labels)
        for layer_index in layers:
            features = feature_matrices[layer_index]
            fitted.append(
                _fit_probe(
                    features,
                    label_array,
                    expanded_train_rows,
                    expanded_test_rows,
                    role_space,
                    layer_index=int(layer_index),
                    c=config.c,
                    max_iter=config.max_iter,
                    fit_backend=fit_backend,
                    fit_epochs=fit_epochs,
                    fit_batch_size=fit_batch_size,
                    fit_learning_rate=fit_learning_rate,
                    fit_device=fit_device,
                    seed=config.seed,
                )
            )
            del feature_matrices[layer_index]
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        del feature_matrices, label_array
    else:
        for layer_index in layers:
            layer_features: list[Any] = []
            for offset in range(0, len(examples), batch_size):
                batch = examples[offset : offset + batch_size]
                input_ids, attention_mask = _pad_examples(batch, int(pad_token_id), torch)
                device = _model_device(model)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                activations = forward_pre_mlp_hidden_states(
                    model,
                    input_ids,
                    attention_mask,
                    layers=(layer_index,),
                    allow_hidden_state_fallback=allow_hidden_state_fallback,
                )
                values = activations[layer_index]
                for row_index, example in enumerate(batch):
                    layer_values = values[row_index, example.target_start : example.target_end]
                    layer_features.append(layer_values.float().cpu())
            del input_ids, attention_mask, activations, values
            features = torch.cat(layer_features, dim=0).numpy()
            if fit_backend == "torch":
                _release_model_storage(model, torch)
            fitted.append(
                _fit_probe(
                    features,
                    __import__("numpy").asarray(expanded_labels),
                    expanded_train_rows,
                    expanded_test_rows,
                    role_space,
                    layer_index=int(layer_index),
                    c=config.c,
                    max_iter=config.max_iter,
                    fit_backend=fit_backend,
                    fit_epochs=fit_epochs,
                    fit_batch_size=fit_batch_size,
                    fit_learning_rate=fit_learning_rate,
                    fit_device=fit_device,
                    seed=config.seed,
                )
            )
            del features, layer_features
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
    metadata = {
        "role_space": list(role_space),
        "layers": [int(layer) for layer in layers],
        "base_sequence_count": len({example.base_seq_ix for example in examples}),
        "example_count": len(examples),
        "train_base_sequence_count": len(train_groups),
        "test_base_sequence_count": len(test_groups),
        "train_token_count": sum(
            examples[index].target_end - examples[index].target_start for index in train_rows
        ),
        "test_token_count": sum(
            examples[index].target_end - examples[index].target_start for index in test_rows
        ),
        "seed": config.seed,
        "test_fraction": config.test_fraction,
        "c": config.c,
        "fit_backend": fit_backend,
        "fit_epochs": fit_epochs if fit_backend == "torch" else None,
        "fit_batch_size": fit_batch_size if fit_backend == "torch" else None,
        "fit_learning_rate": fit_learning_rate if fit_backend == "torch" else None,
        "fit_device": fit_device if fit_backend == "torch" else None,
        "activation_source": (
            "pre_mlp_post_attention_layernorm"
            if not allow_hidden_state_fallback
            else "post_block_hidden_states_with_fallback"
        ),
        "position_control": {
            "enabled": True,
            "method": "matching_neutral_scaffold_filler",
            "target_positions_aligned_within_base_sequence": True,
        },
    }
    return fitted, metadata


def project_role_probabilities(probe: TrainedRoleProbe, activations: Any) -> dict[str, Any]:
    """Project a token-by-hidden-state matrix into named role probabilities."""

    probabilities = probe.classifier.predict_proba(activations)
    return {
        role: probabilities[:, index]
        for role, index in probe.roles_map.items()
    }


def aggregate_role_probability(probabilities: Mapping[str, Any], role: str) -> float:
    """Average a role's token probabilities, rejecting empty projections."""

    if role not in probabilities:
        raise RoleProbeError(f"probe does not contain role {role!r}")
    values = probabilities[role]
    if len(values) == 0:
        raise RoleProbeError("cannot average an empty role projection")
    return float(values.mean())


def save_probe_artifact(
    output_dir: str | Path,
    probes: Sequence[TrainedRoleProbe],
    metadata: Mapping[str, Any],
) -> dict[str, str]:
    """Write a pickle artifact plus stable JSON metadata and accuracy table."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    probe_path = output / "probes.pkl"
    metadata_path = output / "metadata.json"
    accuracy_path = output / "accuracy.json"
    with probe_path.open("wb") as handle:
        pickle.dump(list(probes), handle, protocol=pickle.HIGHEST_PROTOCOL)
    metadata_payload = dict(metadata)
    metadata_payload["probe_sha256"] = hashlib.sha256(probe_path.read_bytes()).hexdigest()
    metadata_path.write_text(
        json.dumps(metadata_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    accuracy_path.write_text(
        json.dumps(
            [
                {
                    "layer_index": probe.layer_index,
                    "role_space": list(probe.role_space),
                    "test_accuracy": probe.test_accuracy,
                    "test_count": probe.test_count,
                    "train_count": probe.train_count,
                }
                for probe in probes
            ],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "probes": str(probe_path),
        "metadata": str(metadata_path),
        "accuracy": str(accuracy_path),
    }


__all__ = [
    "DEFAULT_ROLE_SPACE",
    "PAPER_ROLES",
    "ProbeTrainingConfig",
    "RoleProbeError",
    "RoleProbeExample",
    "SECOPD_ROLES",
    "TrainedRoleProbe",
    "TorchLinearRoleClassifier",
    "aggregate_role_probability",
    "build_probe_examples",
    "find_subsequence",
    "forward_pre_mlp_hidden_states",
    "project_role_probabilities",
    "save_probe_artifact",
    "split_base_sequences",
    "train_role_probes",
]
