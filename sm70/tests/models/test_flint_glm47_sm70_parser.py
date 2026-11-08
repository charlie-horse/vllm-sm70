# SPDX-License-Identifier: Apache-2.0
"""glm47_sm70 reasoning parser (vllm_sm70/reasoning.py), CPU.

What: the parser registers as "glm47_sm70" (on import, and from
  vllm_sm70.models.register()). On a Flint-style template (the generation
  prompt stops at <|assistant|>) it sees that the model opens <think> and
  keeps extraction on even with enable_thinking=False / thinking=False, full
  and streaming. The fork cf9492a cases (tests/reasoning/
  test_glm47_reasoning_parser.py MODEL_OPENS_CASES) give the same
  reasoning/content. A GLM-4.7 template (prompt opens <think>) still honours
  the switch. adjust_request keeps special tokens. Regression: upstream
  v0.33.0 "glm47" leaks "<think>" into content for the same Flint request
  (c4130 Nov 8, conc_bench3.py).
The tokenizer is a byte-level BPE built in memory (fork test helper).
"""

from __future__ import annotations

import functools
import sys
import types

import pytest

pytest.importorskip("vllm.reasoning")
from tokenizers import Tokenizer, decoders, models, pre_tokenizers  # noqa: E402
from transformers import AddedToken, PreTrainedTokenizerFast  # noqa: E402
from vllm.reasoning import ReasoningParserManager  # noqa: E402

from vllm_sm70 import reasoning as sm70_reasoning  # noqa: E402

TURN_TOKENS = [
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|observation|>",
    "<|developer|>",
]
MARKUP_TOKENS = [
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
    "<tool_response>",
    "</tool_response>",
]

_HISTORY = (
    "{%- for m in messages -%}"
    "{%- if m.role == 'user' -%}<|user|>{{ m.content }}"
    "{%- elif m.role == 'assistant' -%}<|assistant|>"
    "{%- if m.reasoning_content %}<think>{{ m.reasoning_content }}</think>"
    "{% endif -%}{{ m.content or '' }}"
    "{%- elif m.role == 'tool' -%}"
    "<|observation|><tool_response>{{ m.content }}</tool_response>"
    "{%- endif -%}"
    "{%- endfor -%}"
)
# The model opens <think> itself: the generation prompt stops at the turn.
MODEL_OPENS_THINK_TEMPLATE = (
    _HISTORY + "{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}"
)
# GLM-4.7 style: the generation prompt opens <think>.
PROMPT_OPENS_THINK_TEMPLATE = _HISTORY + (
    "{%- if add_generation_prompt -%}<|assistant|>"
    "{{- '</think>' if (enable_thinking is defined and not enable_thinking)"
    " else '<think>' -}}{%- endif -%}"
)
# GLM-4.5 style: only an empty block, when thinking is disabled.
EMPTY_THINK_TEMPLATE = _HISTORY + (
    "{%- if add_generation_prompt -%}<|assistant|>"
    "{{- '\\n<think></think>' if (enable_thinking is defined and not"
    " enable_thinking) else '' -}}{%- endif -%}"
)

CALL = (
    "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>"
    "<arg_key>days</arg_key><arg_value>3</arg_value></tool_call>"
)
CALL_ARGS = {"city": "Paris", "days": 3}
NO_ARG_CALL = "<tool_call>list_files</tool_call>"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


@functools.cache
def _tokenizer(chat_template):
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    backend = Tokenizer(
        models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[])
    )
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend)
    tok.add_special_tokens(
        {
            "additional_special_tokens": [
                AddedToken(t, special=True, normalized=False) for t in TURN_TOKENS
            ]
        }
    )
    tok.add_tokens(
        [AddedToken(t, special=False, normalized=False) for t in MARKUP_TOKENS]
    )
    tok.chat_template = chat_template
    return tok


def _parser(template=MODEL_OPENS_THINK_TEMPLATE, name="glm47_sm70", **kwargs):
    return ReasoningParserManager.get_reasoning_parser(name)(
        _tokenizer(template), **kwargs
    )


def _extract(parser, output, streaming):
    """(reasoning, content), as upstream tests/reasoning/utils.py does."""
    if not streaming:
        return parser.extract_reasoning(output, request=None)
    tok = parser.model_tokenizer
    ids = tok.encode(output, add_special_tokens=False)
    reasoning, content = [], []
    prev_text, prev_ids = "", []
    for i in ids:
        delta = tok.decode([i])
        cur_text, cur_ids = prev_text + delta, prev_ids + [i]
        msg = parser.extract_reasoning_streaming(
            prev_text, cur_text, delta, prev_ids, cur_ids, [i]
        )
        if msg is not None:
            reasoning.append(msg.reasoning or "")
            content.append(msg.content or "")
        prev_text, prev_ids = cur_text, cur_ids
    return "".join(reasoning) or None, "".join(content) or None


def _content_of(parser, output, streaming):
    return _extract(parser, output, streaming)[1] or ""


# ------------------------------------------------------------ registration


def test_registered_on_import():
    cls = ReasoningParserManager.get_reasoning_parser("glm47_sm70")
    assert cls is sm70_reasoning.Glm47Sm70ReasoningParser
    # upstream's "glm47" is left alone
    assert ReasoningParserManager.get_reasoning_parser("glm47") is not cls


def test_models_register_imports_the_parser(monkeypatch):
    import vllm_sm70.models as models_area
    from vllm_sm70.models import flint_sm70, qwen

    monkeypatch.setattr(models_area, "_registered", False)
    monkeypatch.setattr(flint_sm70, "register", lambda: False)
    monkeypatch.setattr(qwen, "register", lambda: None)
    monkeypatch.delitem(sys.modules, "vllm_sm70.reasoning")
    monkeypatch.delattr("vllm_sm70.reasoning", raising=False)
    monkeypatch.delitem(ReasoningParserManager.lazy_parsers, "glm47_sm70")
    models_area.register()
    assert "glm47_sm70" in ReasoningParserManager.lazy_parsers


def test_adjust_request_keeps_special_tokens():
    req = types.SimpleNamespace(skip_special_tokens=True)
    assert _parser().adjust_request(req) is req
    assert req.skip_special_tokens is False


# ------------------------------------------------ the Flint <think> leak


@pytest.mark.parametrize("streaming", [False, True], ids=["full", "stream"])
@pytest.mark.parametrize(
    "kwargs",
    [{"enable_thinking": False}, {"thinking": False}, {}],
    ids=["enable_thinking_off", "thinking_off", "default"],
)
def test_flint_template_keeps_reasoning_whatever_the_switch(kwargs, streaming):
    parser = _parser(chat_template_kwargs=kwargs)
    assert parser.model_opens_think and parser.thinking_enabled
    assert _extract(parser, "<think>R</think>A", streaming) == ("R", "A")


@pytest.mark.parametrize("streaming", [False, True], ids=["full", "stream"])
def test_upstream_glm47_leaks_think_into_content(streaming):
    """The bug this module fixes (upstream v0.33.0 behaviour, pinned)."""
    kw = {"chat_template_kwargs": {"enable_thinking": False}}
    up = _parser(name="glm47", **kw)
    ours = _parser(**kw)
    out = "<think>We need answer user.</think>Answer."
    assert "<think>" in _content_of(up, out, streaming)
    assert "<think>" not in _content_of(ours, out, streaming)


def test_prompt_opens_template_still_honours_the_switch():
    parser = _parser(PROMPT_OPENS_THINK_TEMPLATE)
    assert not parser.model_opens_think and parser.thinking_enabled
    off = _parser(
        PROMPT_OPENS_THINK_TEMPLATE, chat_template_kwargs={"enable_thinking": False}
    )
    assert not off.thinking_enabled
    assert off.extract_reasoning("Answer.", request=None) == (None, "Answer.")


# ------------------------------------------------- fork cf9492a cases

# The model opens <think> itself, so output without it is the answer.
# Each case: output, reasoning, content, is_reasoning_end(output).
MODEL_OPENS_CASES = {
    "with_think": (
        "<think>This is a reasoning section</think>This is the rest",
        "This is a reasoning section",
        "This is the rest",
        True,
    ),
    "without_think": ("This is the rest", None, "This is the rest", False),
    "complete_reasoning": (
        "<think>This is a reasoning section</think>",
        "This is a reasoning section",
        None,
        True,
    ),
    "multiline": (
        "<think>This is a reasoning\nsection</think>This is the rest\nThat",
        "This is a reasoning\nsection",
        "This is the rest\nThat",
        True,
    ),
    "empty_think": ("<think></think>The answer is 4.", None, "The answer is 4.", True),
    "trailing_whitespace": (
        "<think>Reason.\n\n</think>\n\nAnswer.",
        "Reason.",
        "\n\nAnswer.",
        True,
    ),
    "only_open_tag": (
        "<think>This is a reasoning section",
        "This is a reasoning section",
        None,
        False,
    ),
    "newline_before_think": (
        "\n<think>Reason.</think>Answer.",
        "Reason.",
        "\nAnswer.",
        True,
    ),
    "stray_think_end": ("Answer.</think> More.", None, "Answer. More.", True),
    "think_then_tool_call": (
        "<think>Need the weather.</think>" + CALL,
        "Need the weather.",
        CALL,
        True,
    ),
    "tool_call_ends_reasoning": (
        "<think>Need the weather." + CALL,
        "Need the weather.",
        CALL,
        True,
    ),
    "tool_call_without_think": (CALL, None, CALL, True),
    "text_then_tool_call": (
        "Let me check." + CALL,
        None,
        "Let me check." + CALL,
        True,
    ),
}


@pytest.mark.parametrize("streaming", [False, True], ids=["full", "stream"])
@pytest.mark.parametrize("case", MODEL_OPENS_CASES.keys())
def test_model_opens_think_matches_fork(case, streaming):
    output, reasoning, content, is_end = MODEL_OPENS_CASES[case]
    parser = _parser()
    assert _extract(parser, output, streaming) == (reasoning, content)
    ids = parser.model_tokenizer.encode(output, add_special_tokens=False)
    assert parser.is_reasoning_end(ids) is is_end


def test_streaming_state_resets_for_new_stream():
    parser = _parser()
    assert _extract(parser, "<think>R</think>A", streaming=True) == ("R", "A")
    assert _extract(parser, "Plain", streaming=True) == (None, "Plain")
