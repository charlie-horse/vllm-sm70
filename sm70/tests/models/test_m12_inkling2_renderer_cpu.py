# SPDX-License-Identifier: Apache-2.0
"""Tests for the M12 Inkling 2 chat renderer (vllm_sm70/models/inkling2_sm70/).

CPU only: upstream's Inkling 2 rendering cases with a fake one-id-per-char
tokenizer, the request fields the renderer reads, and the ``inkling2_sm70``
mode in vLLM's tokenizer and renderer registries at faked cc 8.0.
"""

import types

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

import vllm.platforms
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum
from vllm.renderers.params import ChatParams

from vllm_sm70 import _ext
from vllm_sm70 import platform as sm70_platform
from vllm_sm70.models import inkling2_sm70
from vllm_sm70.models.inkling2_sm70.inkling2_encoding import (
    OPTIONAL_SPECIAL_TOKENS,
    SPECIAL_TOKEN_SPELLINGS,
    render_inkling2_messages,
)
from vllm_sm70.models.inkling2_sm70.renderer import (
    Inkling2Sm70Renderer,
    _HfBackedTmlTokenizer,
    _resolve_reasoning_effort,
)

# Special-token ids of the Inkling 2 Mini tokenizer_config.json.
_INKLING2_VOCAB = {
    "<|endoftext|>": 199999,
    "<|message_user|>": 200000,
    "<|message_model|>": 200001,
    "<|message_system|>": 200002,
    "<|message_tool|>": 200003,
    "<|content_text|>": 200004,
    "<|content_image|>": 200005,
    "<|content_model_end_sampling|>": 200006,
    "<|content_thinking|>": 200008,
    "<|end_message|>": 200010,
    "<|content_tool_error|>": 200022,
    "<|content_xml|>": 200024,
    "<|content_invoke_tool_json|>": 200049,
    "<|content_invoke_tool_text|>": 200057,
}
_INKLING2_ID_TO_SPECIAL = {v: k for k, v in _INKLING2_VOCAB.items()}

_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather information",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


_SEARCH_TOOL = {
    "type": "function",
    "function": {"name": "search", "parameters": {"type": "object"}},
}

_DECLARE_WEATHER_AND_SEARCH = (
    "<|message_system|>tool_declare<|content_xml|>"
    '[{"description":"Get weather information","name":"get_weather",'
    '"parameters":{"properties":{"city":{"type":"string"}},'
    '"required":["city"],"type":"object"},"type":"function"},'
    '{"description":"","name":"search","parameters":{"type":"object"},'
    '"type":"function"}]<|end_message|>'
)
_EFFORT_09 = "<|message_system|><|content_text|>Thinking effort level: 0.9"


_USER_Q = "<|message_user|><|content_text|>q<|end_message|>"


class FakeInkling2HfTokenizer:
    """One id per character. Like a real tokenizer, a special-token string in
    the text becomes that token's id unless ``split_special_tokens`` is set."""

    eos_token_id = None

    def get_vocab(self):
        return dict(_INKLING2_VOCAB)

    def encode(self, text, add_special_tokens=False, split_special_tokens=False):
        assert not add_special_tokens
        ids: list[int] = []
        while text:
            special = next((t for t in _INKLING2_VOCAB if text.startswith(t)), None)
            if special is not None and not split_special_tokens:
                ids.append(_INKLING2_VOCAB[special])
                text = text[len(special) :]
            else:
                ids.append(ord(text[0]))
                text = text[1:]
        return ids


def _decode(token_ids):
    return "".join(_INKLING2_ID_TO_SPECIAL.get(tid, chr(tid)) for tid in token_ids)


def render_inkling2_text(messages, **kwargs):
    tokenizer = _HfBackedTmlTokenizer(FakeInkling2HfTokenizer())
    return _decode(render_inkling2_messages(messages, tokenizer, **kwargs))


def _renderer():
    renderer = Inkling2Sm70Renderer.__new__(Inkling2Sm70Renderer)
    renderer._inkling_tokenizer = _HfBackedTmlTokenizer(FakeInkling2HfTokenizer())
    return renderer


def _render_request(messages, tools=None, **request_fields):
    """Render through the request's own ChatParams, as the server does."""
    request = ChatCompletionRequest(
        model="inkling2", messages=messages, tools=tools, **request_fields
    )
    params = request.build_chat_params(None, "auto")
    if tools:
        params = params.with_defaults({"tools": tools})
    return _renderer()._render(messages, params)


class TestInkling2Rendering:
    def test_system_and_user_without_generation_prefix(self):
        assert render_inkling2_text(
            [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "What is 2 + 2?"},
            ]
        ) == (
            "<|message_system|><|content_text|>You are a concise assistant."
            "<|end_message|>"
            "<|message_system|><|content_text|>Thinking effort level: 0.9"
            "<|end_message|>"
            "<|message_user|><|content_text|>What is 2 + 2?<|end_message|>"
        )

    def test_reasoning_history(self):
        assert render_inkling2_text(
            [
                {"role": "user", "content": "What is 2 + 2?"},
                {
                    "role": "assistant",
                    "reasoning_content": "Add the two numbers.",
                    "content": "4",
                },
                {"role": "user", "content": "Thanks!"},
            ],
            reasoning_effort=0.8,
        ) == (
            "<|message_system|><|content_text|>Thinking effort level: 0.8"
            "<|end_message|>"
            "<|message_user|><|content_text|>What is 2 + 2?<|end_message|>"
            "<|message_model|><|content_thinking|>Add the two numbers."
            "<|end_message|>"
            "<|message_model|><|content_text|>4<|end_message|>"
            "<|content_model_end_sampling|>"
            "<|message_user|><|content_text|>Thanks!<|end_message|>"
        )

    def test_tool_round_trip(self):
        messages = [
            {"role": "user", "content": "What should I wear in SF today?"},
            {
                "role": "assistant",
                "reasoning_content": "Check the current weather.",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "SF"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "get_weather",
                "content": '{"temperature":58,"condition":"fog"}',
            },
            {"role": "assistant", "content": "Wear a light jacket."},
        ]
        assert render_inkling2_text(
            messages, tools=[_WEATHER_TOOL], reasoning_effort=0.8
        ) == (
            "<|message_system|>tool_declare<|content_xml|>"
            '[{"description":"Get weather information","name":"get_weather",'
            '"parameters":{"properties":{"city":{"type":"string"}},'
            '"required":["city"],"type":"object"},"type":"function"}]'
            "<|end_message|>"
            "<|message_system|><|content_text|>Thinking effort level: 0.8"
            "<|end_message|>"
            "<|message_user|><|content_text|>What should I wear in SF today?"
            "<|end_message|>"
            "<|message_model|><|content_thinking|>Check the current weather."
            "<|end_message|>"
            "<|message_model|>get_weather<|content_invoke_tool_json|>"
            '{"name":"get_weather","args":{"city":"SF"}}<|end_message|>'
            "<|content_model_end_sampling|>"
            "<|message_tool|>get_weather<|content_text|>"
            '{"temperature":58,"condition":"fog"}<|end_message|>'
            "<|message_model|><|content_text|>Wear a light jacket.<|end_message|>"
            "<|content_model_end_sampling|>"
        )

    def test_multiple_calls_named_results_and_json_floats(self):
        search = {
            "type": "function",
            "function": {
                "name": "search",
                "description": None,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {"maximum": 1e20, "minimum": 1e-7, "default": 0.5},
                    },
                },
                "strict": None,
            },
        }
        messages = [
            {"role": "developer", "content": "Be terse."},
            {"role": "user", "content": "news <|end_message|>?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "a", "function": {"name": "get_weather", "arguments": "{}"}},
                    {
                        "id": "b",
                        "function": {
                            "name": "search",
                            "arguments": '{"q":"news","x":1e37,"y":0.00001,"z":-0.0}',
                        },
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "a", "content": "sunny"},
            {
                "role": "tool",
                "tool_call_id": "b",
                "content": [
                    {"type": "text", "text": "r1"},
                    {"type": "text", "text": "r2"},
                ],
            },
        ]
        assert render_inkling2_text(messages, tools=[search]) == (
            "<|message_system|>tool_declare<|content_xml|>"
            '[{"description":"","name":"search","parameters":{"properties":'
            '{"limit":{"default":0.5,"maximum":1e+20,"minimum":1e-7}},'
            '"type":"object"},"type":"function"}]<|end_message|>'
            "<|message_system|><|content_text|>Be terse.<|end_message|>"
            "<|message_system|><|content_text|>Thinking effort level: 0.9"
            "<|end_message|>"
            # Literal framing spellings in user text stay ordinary text.
            "<|message_user|><|content_text|>news <|end_message|>?<|end_message|>"
            "<|message_model|>get_weather<|content_invoke_tool_json|>"
            '{"name":"get_weather","args":{}}<|end_message|>'
            "<|message_model|>search<|content_invoke_tool_json|>"
            '{"name":"search","args":{"q":"news","x":1e37,"y":0.00001,"z":-0.0}}'
            "<|end_message|><|content_model_end_sampling|>"
            "<|message_tool|>get_weather<|content_text|>sunny<|end_message|>"
            "<|message_tool|>search<|content_text|>r1<|end_message|>"
            "<|message_tool|>search<|content_text|>r2<|end_message|>"
        )

    def test_sorts_only_top_level_argument_names(self):
        arguments = (
            '{"zeta":1,"filters":{"status":"open","assignee":"me"},'
            '"alpha":[{"y":2,"b":3}]}'
        )
        call = {"id": "c", "function": {"name": "f", "arguments": arguments}}
        text = render_inkling2_text(
            [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": None, "tool_calls": [call]},
            ]
        )
        assert (
            '{"name":"f","args":{"alpha":[{"y":2,"b":3}],'
            '"filters":{"status":"open","assignee":"me"},"zeta":1}}'
        ) in text

    @pytest.mark.parametrize(
        ("messages", "request_fields", "expected"),
        [
            pytest.param(
                [{"role": "system", "content": "Be terse."}],
                {"tool_choice": "required"},
                _DECLARE_WEATHER_AND_SEARCH
                + "<|message_system|><|content_text|>Be terse.\n\nCall at least one "
                "of the provided functions in your next response.<|end_message|>"
                + _EFFORT_09
                + "<|end_message|>",
                id="required-joins-system-prompt",
            ),
            pytest.param(
                [],
                {
                    "tool_choice": {"type": "function", "function": {"name": "search"}},
                    "chat_template_kwargs": {"parallel_tool_calls": False},
                },
                '<|message_system|>tool_declare<|content_xml|>[{"description":"",'
                '"name":"search","parameters":{"type":"object"},"type":"function"}]'
                "<|end_message|>"
                + _EFFORT_09
                + '\n\nCall the function named "search" exactly once in your next '
                "response.<|end_message|>",
                id="named-joins-effort-directive",
            ),
            pytest.param(
                [],
                {"chat_template_kwargs": {"parallel_tool_calls": False}},
                _DECLARE_WEATHER_AND_SEARCH
                + _EFFORT_09
                + "\n\nIf you call a function, call at most one function in your "
                "next response.<|end_message|>",
                id="auto-single-call",
            ),
            pytest.param(
                [],
                {"tool_choice": "none"},
                _EFFORT_09 + "<|end_message|>",
                id="none-declares-no-tools",
            ),
        ],
    )
    def test_tool_choice_renders_as_an_instruction(
        self, messages, request_fields, expected
    ):
        messages = [*messages, {"role": "user", "content": "q"}]
        ids = _render_request(
            messages, tools=[_WEATHER_TOOL, _SEARCH_TOOL], **request_fields
        )
        assert _decode(ids) == expected + _USER_Q

    def test_responses_named_tool_choice_matches_chat_completions(self):
        def render(tool_choice):
            return render_inkling2_text(
                [{"role": "user", "content": "q"}],
                tools=[_WEATHER_TOOL, _SEARCH_TOOL],
                tool_choice=tool_choice,
            )

        assert render({"type": "function", "name": "search"}) == render(
            {"type": "function", "function": {"name": "search"}}
        )

    def test_empty_contents_are_kept(self):
        assert render_inkling2_text(
            [
                {"role": "user", "content": ""},
                {"role": "assistant", "content": ""},
                {"role": "assistant", "content": None},
                {"role": "user", "content": [{"type": "text", "text": ""}]},
            ]
        ) == (
            "<|message_system|><|content_text|>Thinking effort level: 0.9"
            "<|end_message|>"
            "<|message_user|><|content_text|><|end_message|>"
            "<|message_model|><|content_text|><|end_message|>"
            "<|content_model_end_sampling|>"
            "<|message_user|><|content_text|><|end_message|>"
        )

    def test_image_marker_and_typed_parts(self):
        assert render_inkling2_text(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        {"type": "image_url", "image_url": {"url": "x"}},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "hm"},
                        {"type": "tool_error", "text": "no tool"},
                    ],
                },
            ],
            reasoning_effort=0.2,
        ) == (
            "<|message_system|><|content_text|>Thinking effort level: 0.2"
            "<|end_message|>"
            "<|message_user|><|content_text|>look<|end_message|>"
            "<|message_user|><|content_image|><|end_message|>"
            "<|message_model|><|content_thinking|>hm<|end_message|>"
            "<|message_model|><|content_tool_error|>no tool<|end_message|>"
            "<|content_model_end_sampling|>"
        )

    def test_continue_final_message(self):
        messages = [
            {"role": "user", "content": "Finish: The answer is"},
            {"role": "assistant", "content": "The answer is"},
        ]
        assert render_inkling2_text(messages, continue_final_message=True) == (
            "<|message_system|><|content_text|>Thinking effort level: 0.9"
            "<|end_message|>"
            "<|message_user|><|content_text|>Finish: The answer is<|end_message|>"
            "<|message_model|><|content_text|>The answer is"
        )
        # Only a plain text answer is continued; others render as closed turns.
        messages[-1]["reasoning_content"] = "r"
        assert render_inkling2_text(messages, continue_final_message=True).endswith(
            "<|message_model|><|content_text|>The answer is<|end_message|>"
            "<|content_model_end_sampling|>"
        )

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, "0.9"),
            ("none", "0"),
            ("minimal", "0.1"),
            ("low", "0.2"),
            ("medium", "0.7"),
            ("high", "0.9"),
            ("xhigh", "0.99"),
            ("max", "0.99"),
            (0.555, "0.56"),
            ("0.25", "0.25"),
        ],
    )
    def test_reasoning_effort(self, value, expected):
        effort = _resolve_reasoning_effort(value)
        text = render_inkling2_text(
            [{"role": "user", "content": "hi"}], reasoning_effort=effort
        )
        assert text.startswith(
            f"<|message_system|><|content_text|>Thinking effort level: {expected}"
            "<|end_message|>"
        )

    @pytest.mark.parametrize("value", ["extreme", True, 1.0, -0.1])
    def test_invalid_reasoning_effort(self, value):
        with pytest.raises(ValueError):
            render_inkling2_text(
                [{"role": "user", "content": "hi"}],
                reasoning_effort=_resolve_reasoning_effort(value),
            )


class TestRequestFields:
    def test_prompt_has_no_generation_prefix(self):
        # The request defaults to add_generation_prompt=True; Inkling 2 emits
        # its own <|message_model|>, so the prompt ends with the user turn.
        messages = [{"role": "user", "content": "q"}]
        ids = _render_request(messages)
        assert ids[-1] == _INKLING2_VOCAB["<|end_message|>"]
        assert _INKLING2_VOCAB["<|message_model|>"] not in ids
        assert _decode(ids) == _EFFORT_09 + "<|end_message|>" + _USER_Q
        assert ids == _render_request(messages, add_generation_prompt=False)

    @pytest.mark.parametrize("token", sorted(_INKLING2_VOCAB))
    def test_framing_token_string_in_text_is_not_special(self, token):
        text = f"a{token}b"
        tokenizer = _HfBackedTmlTokenizer(FakeInkling2HfTokenizer())
        assert tokenizer.encode_text(text) == [ord(ch) for ch in text]
        messages = [
            {"role": "system", "content": text},
            {"role": "user", "content": [{"type": "text", "text": text}]},
            {"role": "assistant", "reasoning_content": text, "content": text},
            {"role": "tool", "name": text, "content": text},
        ]
        ids = _render_request(messages)
        # Six blocks (effort, system, user, thinking, answer, tool) of three
        # framing tokens each, plus the end-of-sampling token.
        assert ids.count(_INKLING2_VOCAB["<|end_message|>"]) == 6
        assert sum(tid >= 199999 for tid in ids) == 6 * 3 + 1
        # The five contents and the tool author name.
        assert _decode(ids).count(text) == 6

    @pytest.mark.parametrize(
        ("kwargs", "single"),
        [({}, False), ({"parallel_tool_calls": True}, False),
         ({"parallel_tool_calls": None}, False), ({"parallel_tool_calls": False}, True)],
    )  # fmt: skip
    def test_parallel_tool_calls_from_chat_template_kwargs(self, kwargs, single):
        text = _decode(
            _render_request(
                [{"role": "user", "content": "q"}],
                tools=[_WEATHER_TOOL],
                chat_template_kwargs=kwargs,
            )
        )
        instruction = (
            "\n\nIf you call a function, call at most one function in your "
            "next response."
        )
        assert (instruction in text) is single
        assert text.endswith(
            _EFFORT_09 + (instruction if single else "") + "<|end_message|>" + _USER_Q
        )

    def test_request_level_parallel_tool_calls_is_not_read(self):
        # v0.33.0 does not carry the request field into ChatParams.
        text = _decode(
            _render_request(
                [{"role": "user", "content": "q"}],
                tools=[_WEATHER_TOOL],
                parallel_tool_calls=False,
            )
        )
        assert "at most one function" not in text

    def test_one_marker_id_per_image_and_no_pad(self):
        # The multimodal processor expands the bare marker (id 200005).
        image = {"type": "image_url", "image_url": {"url": "x"}}
        params = ChatParams()
        for count in (1, 3):
            content = [{"type": "text", "text": "look"}, *[image] * count]
            ids = _renderer()._render([{"role": "user", "content": content}], params)
            assert ids.count(200005) == count
            assert 200054 not in ids
            marker = ids.index(200005)
            assert ids[marker - 1 : marker + 2] == [200000, 200005, 200010]

    def test_parallel_tool_calls_must_be_boolean(self):
        params = ChatParams(chat_template_kwargs={"parallel_tool_calls": "no"})
        with pytest.raises(ValueError, match="parallel_tool_calls"):
            _renderer()._render([{"role": "user", "content": "q"}], params)

    @pytest.mark.parametrize(
        ("effort", "expected"), [("low", "0.2"), ("max", "0.99"), ("none", "0")]
    )
    def test_request_reasoning_effort(self, effort, expected):
        text = _decode(
            _render_request([{"role": "user", "content": "q"}], reasoning_effort=effort)
        )
        assert text.startswith(
            f"<|message_system|><|content_text|>Thinking effort level: {expected}"
            "<|end_message|>"
        )

    def test_request_continue_final_message(self):
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "The answer is"},
        ]
        text = _decode(
            _render_request(
                messages, add_generation_prompt=False, continue_final_message=True
            )
        )
        assert text.endswith(_USER_Q + "<|message_model|><|content_text|>The answer is")

    def test_malformed_content_is_a_request_error(self):
        params = ChatParams()
        with pytest.raises(ValueError, match="unsupported Inkling message role"):
            _renderer()._render([{"role": "nobody", "content": "q"}], params)
        with pytest.raises(ValueError, match="must be a string or a sequence"):
            _renderer()._render([{"role": "user", "content": 7}], params)


class TestTokenizerAdapter:
    def test_inkling2_vocab_needs_no_audio_tokens(self):
        tokenizer = _HfBackedTmlTokenizer(FakeInkling2HfTokenizer())
        assert tokenizer.encode_special("<|content_xml|>") == 200024
        assert tokenizer.encode_special("<|content_tool_error|>") == 200022
        with pytest.raises(ValueError, match="does not define"):
            tokenizer.encode_special("<|audio_end|>")
        assert {"<|content_audio_input|>", "<|audio_end|>"} <= OPTIONAL_SPECIAL_TOKENS

    def test_missing_required_token_raises(self):
        class Incomplete(FakeInkling2HfTokenizer):
            def get_vocab(self):
                vocab = dict(_INKLING2_VOCAB)
                del vocab["<|end_message|>"]
                return vocab

        with pytest.raises(ValueError, match=r"missing special tokens.*end_message"):
            _HfBackedTmlTokenizer(Incomplete())

    def test_unused_slot_spelling_resolves(self):
        class Unused(FakeInkling2HfTokenizer):
            def get_vocab(self):
                vocab = dict(_INKLING2_VOCAB)
                vocab["<|unused_200024|>"] = vocab.pop("<|content_xml|>")
                return vocab

        assert (
            _HfBackedTmlTokenizer(Unused()).encode_special("<|content_xml|>") == 200024
        )
        assert all(SPECIAL_TOKEN_SPELLINGS.values())


def test_renderer_constructs_on_inkling2_vocab():
    config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            renderer_num_workers=1, supports_multimodal_inputs=False
        ),
        parallel_config=types.SimpleNamespace(_api_process_rank=0),
    )
    renderer = Inkling2Sm70Renderer(config, FakeInkling2HfTokenizer())
    try:
        ids = renderer._render([{"role": "user", "content": "q"}], ChatParams())
        assert _decode(ids) == _EFFORT_09 + "<|end_message|>" + _USER_Q
        # No eos_token in the tokenizer: the end-of-sampling token stands in.
        assert renderer.get_eos_token_id() == 200006
        renderer.tokenizer.eos_token_id = 7
        assert renderer.get_eos_token_id() == 7
    finally:
        renderer.shutdown()


class _FakeCuda(Platform):
    _enum = PlatformEnum.CUDA
    device_name = "cuda"
    device_type = "cuda"
    capability: DeviceCapability | None = None

    @classmethod
    def get_device_capability(cls, device_id: int = 0):
        return cls.capability


@pytest.fixture
def cc80(monkeypatch):
    cls = type("Dev", (_FakeCuda,), {})
    cls.capability = DeviceCapability(8, 0)
    monkeypatch.setattr(vllm.platforms, "_current_platform", cls())
    monkeypatch.setattr(sm70_platform, "all_visible_gpus_are_sm70", lambda: False)
    monkeypatch.setattr(_ext, "is_loaded", lambda: True)
    sm70_platform.is_sm70_ready.cache_clear()
    yield
    sm70_platform.is_sm70_ready.cache_clear()


@pytest.fixture
def registries(monkeypatch):
    """Both registries without the mode; restored afterwards."""
    from vllm.renderers.registry import RENDERER_REGISTRY
    from vllm.tokenizers.registry import TokenizerRegistry

    mode = inkling2_sm70.RENDERER_MODE
    monkeypatch.setattr(
        RENDERER_REGISTRY, "renderers", dict(RENDERER_REGISTRY.renderers)
    )
    monkeypatch.setattr(
        TokenizerRegistry, "tokenizers", dict(TokenizerRegistry.tokenizers)
    )
    RENDERER_REGISTRY.renderers.pop(mode, None)
    TokenizerRegistry.tokenizers.pop(mode, None)
    monkeypatch.setattr(inkling2_sm70, "_registered", False)
    return RENDERER_REGISTRY, TokenizerRegistry


def test_mode_registered_at_cc80(cc80, registries, caplog):
    from vllm.model_executor.models import ModelRegistry
    from vllm.tokenizers.hf import CachedHfTokenizer

    renderers, tokenizers = registries
    mode = inkling2_sm70.RENDERER_MODE
    assert mode == "inkling2_sm70"
    before_r, before_t = dict(renderers.renderers), dict(tokenizers.tokenizers)
    had_arch = inkling2_sm70.ARCH in ModelRegistry.models
    with caplog.at_level("WARNING"):
        assert inkling2_sm70.register() is False  # the model class is gated
        assert inkling2_sm70.register() is False  # idempotent, no overwrite warning
    assert "already registered" not in caplog.text
    assert (inkling2_sm70.ARCH in ModelRegistry.models) is had_arch
    assert renderers.load_renderer_cls(mode) is Inkling2Sm70Renderer
    assert tokenizers.load_tokenizer_cls(mode) is CachedHfTokenizer
    # Nothing else moved; upstream's Inkling 1 mode keeps its own renderer.
    assert {k: v for k, v in renderers.renderers.items() if k != mode} == before_r
    assert {k: v for k, v in tokenizers.tokenizers.items() if k != mode} == before_t
    assert renderers.renderers["inkling"] == (
        "vllm.renderers.inkling",
        "InklingRenderer",
    )


def test_mode_is_a_valid_tokenizer_mode():
    from vllm.config.model import ModelConfig

    field = ModelConfig.__dataclass_fields__["tokenizer_mode"]
    assert "str" in str(field.type)
