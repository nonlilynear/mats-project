from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from black_box_forgery.role_probes import (  # noqa: E402
    DEFAULT_ROLE_SPACE,
    RoleProbeError,
    build_probe_examples,
    find_subsequence,
    split_base_sequences,
)


class CharTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, **kwargs):
        values = [ord(char) for char in text]
        max_length = kwargs.get("max_length")
        if kwargs.get("truncation") and max_length is not None:
            values = values[:max_length]
        return {"input_ids": values}

    def decode(self, values, **kwargs):
        return "".join(chr(int(value)) for value in values)


def test_find_subsequence_requires_one_match():
    assert find_subsequence([1, 2, 3, 4], [2, 3]) == (1, 3)
    with pytest.raises(RoleProbeError):
        find_subsequence([1, 1], [1])


def test_secopd_probe_variants_keep_input_and_tool_distinct():
    examples = build_probe_examples(
        ["neutral alpha", "neutral beta"],
        CharTokenizer(),
        roles=DEFAULT_ROLE_SPACE,
        sequence_length=64,
    )
    assert len(examples) == 2 * len(DEFAULT_ROLE_SPACE)
    by_role = {example.role: example for example in examples[: len(DEFAULT_ROLE_SPACE)]}
    assert "<|im_start|>input" in by_role["input"].rendered_prompt
    assert "<tool_response>" in by_role["tool"].rendered_prompt
    assert "<think>" in by_role["cot"].rendered_prompt
    assert "</think>" in by_role["assistant"].rendered_prompt
    for example in examples:
        target = example.input_ids[example.target_start : example.target_end]
        assert tuple(target) == example.target_token_ids


def test_probe_variants_align_target_positions():
    examples = build_probe_examples(
        ["neutral alpha", "neutral beta"],
        CharTokenizer(),
        roles=DEFAULT_ROLE_SPACE,
        sequence_length=64,
    )
    for base_seq_ix in range(2):
        positions = {
            example.role: example.target_start
            for example in examples
            if example.base_seq_ix == base_seq_ix
        }
        assert len(set(positions.values())) == 1


def test_probe_split_is_grouped_by_base_sequence():
    examples = build_probe_examples(
        ["alpha", "beta", "gamma", "delta"],
        CharTokenizer(),
        roles=("user", "input"),
    )
    train, test = split_base_sequences(examples, seed=123)
    assert train.isdisjoint(test)
    assert train | test == {0, 1, 2, 3}
    assert all(example.base_seq_ix in train | test for example in examples)
