"""Caption-budget overflow: the chat template must survive truncation."""

import logging
import re
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from iris3b.config import TextEncoderConfig
from iris3b.text import qwen3_vl as qwen_module
from iris3b.text.qwen3_vl import _PROMPT_PREFIX, _PROMPT_SUFFIX, Qwen3VLTextEncoder

# --- Qwen3-VL stubs -------------------------------------------------------


class _FakeTokenizer:
    """Context-free stub: joint and piecewise tokenization agree by construction."""

    pad_token_id = 0

    def __init__(self):
        self.calls: list[list[str]] = []
        self._vocab: dict[str, int] = {}

    @classmethod
    def from_pretrained(cls, _checkpoint: str) -> "_FakeTokenizer":
        return cls()

    def _ids(self, text: str) -> list[int]:
        pieces = re.findall(r"<\|[^>]+\|>|\w+|[^\s\w]", text, flags=re.UNICODE)
        return [self._vocab.setdefault(piece, len(self._vocab) + 1) for piece in pieces]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return self._ids(text)

    def __call__(self, texts: list[str], *, add_special_tokens: bool) -> dict[str, list[list[int]]]:
        assert add_special_tokens is False
        self.calls.append(list(texts))
        return {"input_ids": [self._ids(text) for text in texts]}


class _FakeDecoder(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, num_hidden_layers=6)
        self.anchor = nn.Parameter(torch.ones(()))
        self.last_kwargs: dict[str, torch.Tensor] = {}

    def forward(self, input_ids, attention_mask, use_cache, return_dict, output_hidden_states=False):
        self.last_kwargs = {
            "input_ids": input_ids.detach().clone(),
            "attention_mask": attention_mask.detach().clone(),
        }
        hidden = input_ids.to(torch.float32).unsqueeze(-1).expand(-1, -1, self.config.hidden_size)
        return SimpleNamespace(last_hidden_state=hidden, hidden_states=None)


class _FakeQwenModel:
    last_decoder: _FakeDecoder | None = None

    def __init__(self):
        self.decoder = _FakeDecoder(4)
        type(self).last_decoder = self.decoder

    @classmethod
    def from_pretrained(cls, _checkpoint: str, **_kwargs) -> "_FakeQwenModel":
        return cls()

    def get_decoder(self) -> _FakeDecoder:
        return self.decoder


def _qwen_encoder(monkeypatch, tmp_path, **kwargs) -> Qwen3VLTextEncoder:
    module = ModuleType("transformers")
    module.AutoTokenizer = _FakeTokenizer
    module.Qwen3VLForConditionalGeneration = _FakeQwenModel
    monkeypatch.setitem(sys.modules, "transformers", module)
    values = {
        "name": "qwen3_vl",
        "pretrained": "Qwen/Qwen3-VL-test",
        "dim": 4,
        "max_length": 8,
        "hidden_layers": [],
        "null_embed_dir": str(tmp_path),
    }
    values.update(kwargs)
    return Qwen3VLTextEncoder(TextEncoderConfig(**values))


# --- Qwen3-VL: the fitting caption is unchanged ---------------------------


def test_caption_within_budget_matches_joint_template_tokenization(monkeypatch, tmp_path):
    encoder = _qwen_encoder(monkeypatch, tmp_path)
    caption = "red panda"
    assert len(encoder.tokenizer.encode(caption)) <= encoder.caption_budget

    encoder.encode([caption])
    fed = _FakeQwenModel.last_decoder.last_kwargs["input_ids"][0]
    joint = encoder.tokenizer.encode(f"{_PROMPT_PREFIX}{caption}{_PROMPT_SUFFIX}")
    assert fed[: len(joint)].tolist() == joint
    assert torch.equal(fed[len(joint) :], torch.zeros(len(fed) - len(joint), dtype=torch.long))
    assert encoder.overflow_rows == 0
    assert encoder.overflow_tokens == 0
    assert encoder.calls == 1


# --- Qwen3-VL: the suffix survives an over-long caption -------------------


def test_over_long_caption_keeps_the_whole_suffix(monkeypatch, tmp_path):
    encoder = _qwen_encoder(monkeypatch, tmp_path)
    budget = encoder.caption_budget
    caption = " ".join(f"w{i}" for i in range(budget + 4))
    caption_ids = encoder.tokenizer.encode(caption)
    assert len(caption_ids) == budget + 4

    result = encoder.encode([caption])
    kwargs = _FakeQwenModel.last_decoder.last_kwargs
    fed = kwargs["input_ids"][0]
    start = encoder._prefix_len
    suffix_len = encoder._suffix_len

    assert torch.equal(fed[:start], encoder._prefix_ids)
    assert fed[start : start + budget].tolist() == caption_ids[:budget]
    assert torch.equal(fed[start + budget : start + budget + suffix_len], encoder._suffix_ids)
    # every position of the returned window is a real token: nothing left to pad
    assert int(result.mask.sum()) == encoder.max_length
    assert int(kwargs["attention_mask"].sum()) == start + encoder.max_length


def test_overflow_counters_accumulate_across_calls(monkeypatch, tmp_path):
    encoder = _qwen_encoder(monkeypatch, tmp_path)
    budget = encoder.caption_budget
    long_one = " ".join(f"w{i}" for i in range(budget + 3))
    long_two = " ".join(f"v{i}" for i in range(budget + 1))

    encoder.encode([long_one, "red panda", long_two])
    assert encoder.calls == 1
    assert encoder.overflow_rows == 2
    assert encoder.overflow_tokens == 4
    assert encoder.max_caption_tokens == budget + 3

    encoder.encode(["red panda"])
    assert encoder.calls == 2
    assert encoder.overflow_rows == 2
    assert encoder.overflow_tokens == 4
    assert encoder.max_caption_tokens == budget + 3


def test_error_mode_refuses_the_batch_with_the_offending_length(monkeypatch, tmp_path):
    encoder = _qwen_encoder(monkeypatch, tmp_path, on_caption_overflow="error")
    caption = " ".join(f"w{i}" for i in range(encoder.caption_budget + 2))
    with pytest.raises(ValueError, match=f"tokenizes to {encoder.caption_budget + 2} tokens"):
        encoder.encode([caption])
    assert encoder.overflow_rows == 1


def test_silent_mode_counts_without_logging(monkeypatch, tmp_path, caplog):
    encoder = _qwen_encoder(monkeypatch, tmp_path, on_caption_overflow="silent")
    caption = " ".join(f"w{i}" for i in range(encoder.caption_budget + 2))
    with caplog.at_level(logging.WARNING, logger="iris3b.text.qwen3_vl"):
        encoder.encode([caption])
    assert caplog.records == []
    assert encoder.overflow_rows == 1
    assert encoder.overflow_tokens == 2


def test_warn_mode_is_rate_limited_and_names_the_longest_caption(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(qwen_module, "_OVERFLOW_LOG_INTERVAL", 2)
    encoder = _qwen_encoder(monkeypatch, tmp_path)
    caption = " ".join(f"w{i}" for i in range(encoder.caption_budget + 2))
    with caplog.at_level(logging.WARNING, logger="iris3b.text.qwen3_vl"):
        for _ in range(3):
            encoder.encode([caption])
    assert len(caplog.records) == 2
    assert f"longest {encoder.caption_budget + 2} tokens" in caplog.records[0].getMessage()


def test_max_length_below_the_template_is_rejected(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="leaves no room for a caption"):
        _qwen_encoder(monkeypatch, tmp_path, max_length=1)


def test_unknown_overflow_policy_is_rejected(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="on_caption_overflow"):
        _qwen_encoder(monkeypatch, tmp_path, on_caption_overflow="truncate")
