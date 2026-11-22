# SPDX-License-Identifier: Apache-2.0
"""Tests for the M12 Inkling 2 parsers (vllm_sm70/models/inkling2_sm70/parsers.py).

CPU only: upstream's Inkling 2 parser cases with a mock one-id-per-char
tokenizer, in streaming and non-streaming, the JSON response-format
structural tag compiled with the installed xgrammar, and the
``inkling2_sm70`` names in vLLM's reasoning and tool parser managers.
"""

import json
import os
import subprocess
import sys
from unittest.mock import MagicMock

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
xgrammar = pytest.importorskip("xgrammar")

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.parser.abstract_parser import DelegatingParser
from vllm.parser.parser_manager import ParserManager
from vllm.reasoning import ReasoningParserManager
from vllm.sampling_params import SamplingParams
from vllm.tool_parsers import ToolParserManager
from vllm.v1.structured_output.backend_xgrammar import validate_xgrammar_grammar

from vllm_sm70.models import inkling2_sm70
from vllm_sm70.models.inkling2_sm70 import parsers
from vllm_sm70.models.inkling2_sm70.parsers import (
    Inkling2Sm70EngineToolParser,
    Inkling2Sm70Parser,
    Inkling2Sm70ParserReasoningAdapter,
)

MSG_USER = "<|message_user|>"
MSG_MODEL = "<|message_model|>"
MSG_SYSTEM = "<|message_system|>"
MSG_TOOL = "<|message_tool|>"
TEXT_START = "<|content_text|>"
THINK_START = "<|content_thinking|>"
TOOL_JSON = "<|content_invoke_tool_json|>"
END_MESSAGE = "<|end_message|>"
END_SAMPLING = "<|content_model_end_sampling|>"

# Special-token ids of the Inkling 2 Mini tokenizer_config.json.
_INKLING2_VOCAB = {
    "<|endoftext|>": 199999,
    MSG_USER: 200000,
    MSG_MODEL: 200001,
    MSG_SYSTEM: 200002,
    MSG_TOOL: 200003,
    TEXT_START: 200004,
    "<|content_image|>": 200005,
    END_SAMPLING: 200006,
    THINK_START: 200008,
    END_MESSAGE: 200010,
    "<|content_tool_error|>": 200022,
    "<|content_xml|>": 200024,
    TOOL_JSON: 200049,
    "<|content_invoke_tool_text|>": 200057,
}
# Inkling (v1) has audio tokens and no native <|content_xml|>.
_INKLING1_VOCAB = {
    **{k: v for k, v in _INKLING2_VOCAB.items() if k != "<|content_xml|>"},
    "<|content_audio_input|>": 200020,
    "<|audio_end|>": 200021,
}
_MARKERS = sorted(_INKLING2_VOCAB, key=len, reverse=True)

PROMPT = [
    _INKLING2_VOCAB[MSG_SYSTEM],
    _INKLING2_VOCAB[TEXT_START],
    ord("e"),
    _INKLING2_VOCAB[END_MESSAGE],
    _INKLING2_VOCAB[MSG_USER],
    _INKLING2_VOCAB[TEXT_START],
    ord("q"),
    _INKLING2_VOCAB[END_MESSAGE],
]


def _make_mock_tokenizer(vocab):
    id_to_text = {v: k for k, v in vocab.items()}
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3]
    tokenizer.get_vocab.return_value = dict(vocab)
    tokenizer.decode.side_effect = lambda ids: "".join(
        id_to_text.get(i, chr(i) if i < 128 else f"<{i}>") for i in ids
    )
    tokenizer.all_special_tokens = list(vocab)
    tokenizer.all_special_ids = list(vocab.values())
    return tokenizer


@pytest.fixture
def mock_tokenizer():
    return _make_mock_tokenizer(_INKLING2_VOCAB)


@pytest.fixture
def parser(mock_tokenizer):
    return Inkling2Sm70Parser(mock_tokenizer)


@pytest.fixture
def mock_request():
    req = MagicMock(spec=ChatCompletionRequest)
    req.tools = []
    req.tool_choice = "auto"
    req.include_reasoning = True
    return req


def _tool_block(name, args):
    return f'{TOOL_JSON}{{"name":"{name}","args":{args}}}{END_MESSAGE}'


def _tokenize(text):
    """Markers are atomic special tokens; other text is one token per char."""
    tokens = []
    i = 0
    while i < len(text):
        for marker in _MARKERS:
            if text.startswith(marker, i):
                tokens.append((_INKLING2_VOCAB[marker], marker))
                i += len(marker)
                break
        else:
            tokens.append((ord(text[i]), text[i]))
            i += 1
    return tokens


def _function_tool(name="get_weather"):
    return ChatCompletionToolsParam(
        function=FunctionDefinition(
            name=name,
            parameters={"type": "object", "properties": {}},
        ),
    )


def _delegating(mock_tokenizer, tools=None):
    """The served reasoning+tool DelegatingParser, resolved by name."""
    parser_cls = ParserManager.get_parser(
        tool_parser_name=parsers.PARSER_NAME,
        reasoning_parser_name=parsers.PARSER_NAME,
        enable_auto_tools=True,
        tokenizer=mock_tokenizer,
    )
    assert parser_cls is not None
    assert parser_cls.reasoning_parser_cls is Inkling2Sm70ParserReasoningAdapter
    assert parser_cls.tool_parser_cls is Inkling2Sm70EngineToolParser
    return parser_cls(mock_tokenizer, tools or [])


def _stream_delegating(parser, request, text, chunk_size, prompt_token_ids):
    """Stream through ``parse_delta``, ``chunk_size`` tokens per delta; return
    (content, reasoning, tool names, tool arguments), arguments joined per
    tool index."""
    tokens = _tokenize(text)
    content, reasoning = "", ""
    tools = {}
    args = {}
    for start in range(0, len(tokens), chunk_size):
        batch = tokens[start : start + chunk_size]
        delta = parser.parse_delta(
            "".join(t for _, t in batch),
            [tid for tid, _ in batch],
            request,
            prompt_token_ids=prompt_token_ids if start == 0 else None,
            finished=(start + chunk_size >= len(tokens)),
        )
        if delta and delta.content:
            content += delta.content
        if delta and delta.reasoning:
            reasoning += delta.reasoning
        if delta and delta.tool_calls:
            for tc in delta.tool_calls:
                if tc.function and tc.function.name:
                    tools[tc.index] = tc.function.name
                if tc.function and tc.function.arguments:
                    args[tc.index] = args.get(tc.index, "") + tc.function.arguments
    return (
        content,
        reasoning,
        [tools[k] for k in sorted(tools)],
        [args.get(k, "") for k in sorted(tools)],
    )


class _ToolOnlyParser(DelegatingParser):
    tool_parser_cls = Inkling2Sm70EngineToolParser


OUTPUT = (
    f"{MSG_MODEL}{THINK_START}plan{END_MESSAGE}"
    f"{MSG_MODEL}{TEXT_START}Checking.{END_MESSAGE}"
    + MSG_MODEL
    + _tool_block("f", '{"a":1}')
    + MSG_MODEL
    + _tool_block("g", '{"b":[2]}')
)
# Thinking, text and one tool call, closed by the end-of-sampling token.
OUTPUT_END_SAMPLING = (
    f"{MSG_MODEL}{THINK_START}plan{END_MESSAGE}"
    f"{MSG_MODEL}{TEXT_START}Checking.{END_MESSAGE}"
    + MSG_MODEL
    + _tool_block("f", '{"a":1}')
    + END_SAMPLING
)


def test_vocabulary_selects_inkling2(parser):
    assert parser._is_inkling2
    assert not Inkling2Sm70Parser(_make_mock_tokenizer(_INKLING1_VOCAB))._is_inkling2


def test_prompt_does_not_end_reasoning(parser):
    model, think, text = (
        _INKLING2_VOCAB[t] for t in (MSG_MODEL, THINK_START, TEXT_START)
    )
    assert not parser.is_reasoning_end(PROMPT)
    assert not parser.is_reasoning_end(PROMPT + [model])
    assert not parser.is_reasoning_end(PROMPT + [model, think])
    assert parser.is_reasoning_end(PROMPT + [model, text, ord("x")])
    assert parser.is_reasoning_end(tuple(PROMPT + [model, text]))


def test_inkling1_vocabulary_keeps_base_reasoning_end():
    """With an Inkling (v1) vocabulary the base rule applies: the prompt's
    last text block ends reasoning."""
    v1 = Inkling2Sm70Parser(_make_mock_tokenizer(_INKLING1_VOCAB))
    assert v1.is_reasoning_end(PROMPT)


def test_non_streaming(mock_tokenizer, mock_request):
    tools = [_function_tool("f"), _function_tool("g")]
    mock_request.tools = tools
    reasoning, content, calls = _delegating(mock_tokenizer, tools).parse(
        OUTPUT, mock_request, enable_auto_tools=True
    )
    assert reasoning == "plan"
    assert content == "Checking."
    assert [(c.name, json.loads(c.arguments)) for c in calls] == [
        ("f", {"a": 1}),
        ("g", {"b": [2]}),
    ]


def test_non_streaming_end_sampling(mock_tokenizer, mock_request):
    tools = [_function_tool("f")]
    mock_request.tools = tools
    reasoning, content, calls = _delegating(mock_tokenizer, tools).parse(
        OUTPUT_END_SAMPLING, mock_request, enable_auto_tools=True
    )
    assert (reasoning, content) == ("plan", "Checking.")
    assert [(c.name, json.loads(c.arguments)) for c in calls] == [("f", {"a": 1})]


@pytest.mark.parametrize("chunk_size", [1, 3, 64])
def test_streaming(mock_tokenizer, mock_request, chunk_size):
    tools = [_function_tool("f"), _function_tool("g")]
    mock_request.tools = tools
    content, reasoning, names, args = _stream_delegating(
        _delegating(mock_tokenizer, tools), mock_request, OUTPUT, chunk_size, PROMPT
    )
    assert reasoning == "plan"
    assert content == "Checking."
    assert names == ["f", "g"]
    assert [json.loads(a) for a in args] == [{"a": 1}, {"b": [2]}]


@pytest.mark.parametrize("chunk_size", [1, 3, 64])
def test_streaming_end_sampling(mock_tokenizer, mock_request, chunk_size):
    tools = [_function_tool("f")]
    mock_request.tools = tools
    content, reasoning, names, args = _stream_delegating(
        _delegating(mock_tokenizer, tools),
        mock_request,
        OUTPUT_END_SAMPLING,
        chunk_size,
        PROMPT,
    )
    assert (reasoning, content, names) == ("plan", "Checking.", ["f"])
    assert [json.loads(a) for a in args] == [{"a": 1}]


@pytest.mark.parametrize("chunk_size", [1, 3, 64])
def test_streaming_thinking_after_text(mock_tokenizer, mock_request, chunk_size):
    # Blocks after the first text block keep their framing out of the content.
    output = (
        f"{MSG_MODEL}{THINK_START}aa{END_MESSAGE}{MSG_MODEL}{TEXT_START}bb{END_MESSAGE}"
        f"{MSG_MODEL}{THINK_START}cc{END_MESSAGE}{MSG_MODEL}{TEXT_START}dd{END_MESSAGE}"
        f"{END_SAMPLING}"
    )
    content, reasoning, names, _ = _stream_delegating(
        _delegating(mock_tokenizer), mock_request, output, chunk_size, PROMPT
    )
    assert (content, reasoning, names) == ("bbdd", "aacc", [])


@pytest.mark.parametrize(
    "tool_parser",
    [
        {},
        {"tool_parser_name": parsers.PARSER_NAME},
        {"tool_parser_name": "hermes", "enable_auto_tools": True},
    ],
    ids=["none", "auto_tools_off", "other"],
)
def test_reasoning_parser_needs_the_tool_parser(mock_tokenizer, tool_parser):
    # v0.33.0 streams everything after the first text block as raw content
    # unless this tool parser parses it, so the composition fails at startup.
    with pytest.raises(TypeError, match="requires --tool-call-parser inkling2_sm70"):
        ParserManager.get_parser(
            reasoning_parser_name=parsers.PARSER_NAME,
            tokenizer=mock_tokenizer,
            **tool_parser,
        )


def test_reasoning_adapter_alone_still_constructs(mock_tokenizer):
    # The structured-output manager builds the reasoning parser on its own.
    assert parsers.Inkling2Sm70Parser.reasoning_parser_cls is (
        Inkling2Sm70ParserReasoningAdapter
    )
    adapter = Inkling2Sm70ParserReasoningAdapter(mock_tokenizer)
    assert adapter.is_reasoning_end(PROMPT) is False


@pytest.mark.parametrize("chunk_size", [1, 64])
def test_streaming_text_only(mock_tokenizer, mock_request, chunk_size):
    content, reasoning, names, _ = _stream_delegating(
        _delegating(mock_tokenizer),
        mock_request,
        f"{MSG_MODEL}{THINK_START}hm{END_MESSAGE}"
        f"{MSG_MODEL}{TEXT_START}4{END_SAMPLING}",
        chunk_size,
        PROMPT,
    )
    assert (reasoning, content, names) == ("hm", "4", [])


def test_non_streaming_text_only(mock_tokenizer, mock_request):
    reasoning, content, calls = _delegating(mock_tokenizer).parse(
        f"{MSG_MODEL}{THINK_START}hm{END_MESSAGE}"
        f"{MSG_MODEL}{TEXT_START}4{END_SAMPLING}",
        mock_request,
        enable_auto_tools=True,
    )
    assert (reasoning, content) == ("hm", "4")
    assert not calls


def _json_request(tools=None, assistant=None, tool_choice=None):
    messages = [{"role": "user", "content": "q"}]
    if assistant is not None:
        messages.append({"role": "assistant", "content": assistant})
    kwargs = {} if tool_choice is None else {"tool_choice": tool_choice}
    return ChatCompletionRequest(
        model="inkling2",
        messages=messages,
        tools=tools,
        continue_final_message=assistant is not None,
        add_generation_prompt=assistant is None,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {"type": "object", "properties": {"a": {}}},
            },
        },
        **kwargs,
    )


def _json_grammar_matcher(parser, **request_kwargs):
    """Adjust a JSON-schema request and compile its grammar over a
    one-token-per-character vocab that holds the Inkling markers."""
    request = parser.adjust_request(_json_request(**request_kwargs))
    assert request.response_format is None
    assert request.resolve_reasoning_ended(None, PROMPT) is True
    validate_xgrammar_grammar(
        SamplingParams(structured_outputs=request.structured_outputs)
    )

    vocab = [chr(i) if i < 128 else f"<unused_{i}>" for i in range(200058)]
    for marker, token_id in _INKLING2_VOCAB.items():
        vocab[token_id] = marker
    compiler = xgrammar.GrammarCompiler(
        xgrammar.TokenizerInfo(vocab, stop_token_ids=[_INKLING2_VOCAB[END_SAMPLING]])
    )
    compiled = compiler.compile_structural_tag(
        request.structured_outputs.structural_tag
    )
    return xgrammar.GrammarMatcher(compiled)


def _accepts(matcher, output):
    token_ids = [tid for tid, _ in _tokenize(output + END_SAMPLING)]
    return all(matcher.accept_token(tid) for tid in token_ids) and (
        matcher.is_terminated()
    )


def test_json_response_format_frames_final_text(parser):
    matcher = _json_grammar_matcher(parser)
    assert _accepts(
        matcher,
        f"{MSG_MODEL}{THINK_START}any {{text}}{END_MESSAGE}"
        f'{MSG_MODEL}{TEXT_START}{{"a": 1}}{END_MESSAGE}',
    )


def test_json_response_format_continues_empty_answer(parser):
    matcher = _json_grammar_matcher(parser, assistant="")
    assert _accepts(matcher, f"{{}}{END_MESSAGE}")


def test_json_response_format_rejects_nonempty_continuation(parser):
    with pytest.raises(ValueError, match="nonempty assistant message"):
        parser.adjust_request(_json_request(assistant="{"))


@pytest.mark.parametrize(
    "output",
    [
        # JSON outside a text block.
        '{"a": 1}',
        # A final text block that is not the schema.
        f"{MSG_MODEL}{TEXT_START}plain{END_MESSAGE}",
        # Framing inside reasoning.
        f"{MSG_MODEL}{THINK_START}a{TOOL_JSON}{END_MESSAGE}",
        # Tool calls in place of the JSON answer, even under auto.
        f"{MSG_MODEL}{_tool_block('f', '{}')}",
    ],
)
def test_json_response_format_rejects(parser, output):
    matcher = _json_grammar_matcher(parser, tools=[_function_tool("f")])
    assert not _accepts(matcher, output)


def test_json_response_format_with_tool_parser_only(mock_tokenizer):
    """The tool parser alone frames the grammar; it never sees the prompt."""
    matcher = _json_grammar_matcher(
        _ToolOnlyParser(mock_tokenizer, [_function_tool("f")]),
        tools=[_function_tool("f")],
    )
    assert _accepts(matcher, f'{MSG_MODEL}{TEXT_START}{{"a":1}}{END_MESSAGE}')


def test_json_response_format_with_required_tool_choice_is_unframed(parser):
    request = _json_request(tools=[_function_tool("f")], tool_choice="required")
    response_format = request.response_format
    adjusted = parser.adjust_request(request)
    assert adjusted.response_format == response_format
    assert adjusted._grammar_from_parser is False


def test_inkling1_vocabulary_json_response_format_is_unframed():
    v1 = Inkling2Sm70Parser(_make_mock_tokenizer(_INKLING1_VOCAB))
    request = _json_request()
    response_format = request.response_format
    assert v1.adjust_request(request).response_format == response_format


def test_request_without_response_format_keeps_special_tokens(parser):
    request = ChatCompletionRequest(
        model="inkling2", messages=[{"role": "user", "content": "q"}]
    )
    adjusted = parser.adjust_request(request)
    assert adjusted.skip_special_tokens is False
    assert adjusted.structured_outputs is None


def test_tool_parser_has_no_structural_tag():
    assert Inkling2Sm70EngineToolParser.structural_tag_model is None
    assert Inkling2Sm70EngineToolParser.supports_required_and_named is False


def test_names_resolve_in_both_managers():
    name = inkling2_sm70.PARSER_NAME
    assert name == parsers.PARSER_NAME == "inkling2_sm70"
    assert (
        ReasoningParserManager.get_reasoning_parser(name)
        is Inkling2Sm70ParserReasoningAdapter
    )
    assert ToolParserManager.get_tool_parser(name) is Inkling2Sm70EngineToolParser
    assert name in ReasoningParserManager.list_registered()
    assert name in ToolParserManager.list_registered()


_FRESH_PROCESS_CASES = {
    # The plugin's lazy entries resolve without a parser-plugin flag.
    "lazy": (
        "from vllm_sm70.models import inkling2_sm70\n"
        "inkling2_sm70.register_parsers()\n"
        "inkling2_sm70.register_parsers()\n"
        "assert inkling2_sm70.PARSER_MODULE not in sys.modules\n"
    ),
    # --reasoning-parser-plugin / --tool-parser-plugin import the module.
    "plugin_flags": (
        "import argparse\n"
        "from vllm.entrypoints.common.cli_args import init_parser_plugin\n"
        "m = 'vllm_sm70.models.inkling2_sm70.parsers'\n"
        "init_parser_plugin(argparse.Namespace(\n"
        "    tool_parser_plugin=m, reasoning_parser_plugin=m))\n"
        "assert m in sys.modules\n"
    ),
}


@pytest.mark.parametrize("case", sorted(_FRESH_PROCESS_CASES))
def test_names_register_in_a_fresh_process(case):
    code = (
        "import sys\n"
        "from vllm.reasoning import ReasoningParserManager as R\n"
        "from vllm.tool_parsers import ToolParserManager as T\n"
        "n = 'inkling2_sm70'\n"
        "assert n not in R.list_registered() and n not in T.list_registered()\n"
        + _FRESH_PROCESS_CASES[case]
        + "assert n in R.list_registered() and n in T.list_registered()\n"
        "r, t = R.get_reasoning_parser(n), T.get_tool_parser(n)\n"
        "assert r.__name__ == 'Inkling2Sm70ParserReasoningAdapter', r\n"
        "assert t.__name__ == 'Inkling2Sm70EngineToolParser', t\n"
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    subprocess.run([sys.executable, "-c", code], check=True, timeout=600, env=env)
