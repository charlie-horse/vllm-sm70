# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``inkling2_sm70``: reasoning and tool parsers for Inkling 2 output.

vLLM v0.33.0's ``InklingParser`` already parses the block grammar. This
subclass adds the Inkling 2 parts of upstream's `vllm/parser/inkling.py`
(see PROVENANCE.md): the vocabulary check, ``is_reasoning_end`` for prompts
that end between messages, and the JSON response-format structural tag.

Use: ``--reasoning-parser-plugin
vllm_sm70.models.inkling2_sm70.parsers --reasoning-parser inkling2_sm70
--tool-parser-plugin vllm_sm70.models.inkling2_sm70.parsers
--tool-call-parser inkling2_sm70 --enable-auto-tool-choice``. The plugin
also registers both names lazily (``inkling2_sm70.register_parsers``).
The reasoning parser is served only together with the tool parser; any other
pairing fails at server start.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.abstract_parser import structured_outputs_to_format
from vllm.parser.engine.adapters import make_adapters
from vllm.parser.inkling import INKLING_SPECIAL_TOKENS, InklingParser
from vllm.reasoning import ReasoningParserManager
from vllm.sampling_params import StructuredOutputsParams
from vllm.tool_parsers import ToolParserManager
from xgrammar import StructuralTag
from xgrammar.structural_tag import (
    AnyTextFormat,
    Format,
    SequenceFormat,
    StarFormat,
    TokenFormat,
)

from .inkling2_encoding import (
    AUDIO_END,
    CONTENT_AUDIO_INPUT,
    CONTENT_IMAGE,
    CONTENT_MODEL_END_SAMPLING,
    CONTENT_TEXT,
    CONTENT_THINKING,
    CONTENT_XML,
    END_MESSAGE,
    END_OF_TEXT,
    MESSAGE_MODEL,
    MESSAGE_SYSTEM,
    MESSAGE_TOOL,
    MESSAGE_USER,
    continuation_prefix,
)

if TYPE_CHECKING:
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

PARSER_NAME = "inkling2_sm70"

# Every framing token; free text in a grammar never contains one.
INKLING_CONTROL_TOKENS = (
    END_OF_TEXT,
    MESSAGE_USER,
    MESSAGE_SYSTEM,
    MESSAGE_TOOL,
    CONTENT_IMAGE,
    CONTENT_XML,
    CONTENT_AUDIO_INPUT,
    AUDIO_END,
    *INKLING_SPECIAL_TOKENS,
)


class Inkling2Sm70Parser(InklingParser):
    CONFIG_NAME = PARSER_NAME

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(tokenizer, tools, **kwargs)
        # Inkling2 spells <|content_xml|> natively and has no audio tokens.
        # Its prompts end between messages: the model opens its own turn.
        vocab = self.vocab
        self._is_inkling2 = CONTENT_XML in vocab and CONTENT_AUDIO_INPUT not in vocab

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        """Frame an Inkling2 JSON response format with the block grammar.

        As in the reference, any number of free thinking blocks precede the
        final text block, whose payload alone must match the JSON constraint;
        the API then returns that payload as the content.
        """
        request = super().adjust_request(request)
        if not self._is_inkling2:
            return request
        structured_outputs = request.extract_structured_outputs()
        if structured_outputs is None or (
            structured_outputs.json is None and not structured_outputs.json_object
        ):
            return request
        if request.tools and request.tool_choice not in ("auto", "none", None):
            # Required and named tool choices keep the default handling.
            return request
        json_format = structured_outputs_to_format(structured_outputs)

        vocab = self.vocab
        model, text, thinking, end = (
            TokenFormat(token=vocab[token])
            for token in (MESSAGE_MODEL, CONTENT_TEXT, CONTENT_THINKING, END_MESSAGE)
        )
        prefix = None
        if isinstance(request, ChatCompletionRequest) and (
            request.continue_final_message and request.messages
        ):
            prefix = continuation_prefix(request.messages[-1])
        if prefix:
            raise ValueError(
                "Inkling JSON response formats cannot continue a nonempty "
                "assistant message."
            )
        output_format: Format
        if prefix is None:
            excludes = [token for token in INKLING_CONTROL_TOKENS if token in vocab]
            thinking_block = SequenceFormat(
                elements=[model, thinking, AnyTextFormat(excludes=excludes), end]
            )
            output_format = SequenceFormat(
                elements=[
                    StarFormat(content=thinking_block),
                    model,
                    text,
                    json_format,
                    end,
                ]
            )
        else:
            output_format = SequenceFormat(elements=[json_format, end])
        tag = StructuralTag(format=output_format)
        grammar = StructuredOutputsParams(structural_tag=json.dumps(tag.model_dump()))
        if isinstance(request, ChatCompletionRequest):
            request.structured_outputs = grammar
            request.response_format = None
        else:
            request._parser_structured_outputs = grammar
        request._grammar_from_parser = True
        return request

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        if self._is_inkling2:
            return self._inkling2_is_reasoning_end(input_ids)
        return super().is_reasoning_end(list(input_ids))

    def _inkling2_is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """Whether the latest block is a text block of a model message.

        A prompt that ends between messages has not ended reasoning: the
        model may open a thinking block.
        """
        vocab = self.vocab
        text_id = vocab[CONTENT_TEXT]
        block_ids = {
            text_id,
            vocab[CONTENT_THINKING],
            vocab[CONTENT_MODEL_END_SAMPLING],
        }
        role_ids = {
            vocab[token]
            for token in (MESSAGE_MODEL, MESSAGE_USER, MESSAGE_SYSTEM, MESSAGE_TOOL)
        }
        latest_block = None
        for token_id in reversed(input_ids):
            if token_id in role_ids:
                return latest_block == text_id and token_id == vocab[MESSAGE_MODEL]
            if latest_block is None and token_id in block_ids:
                latest_block = token_id
        return latest_block == text_id


_ReasoningAdapter, Inkling2Sm70ParserToolAdapter = make_adapters(Inkling2Sm70Parser)


class Inkling2Sm70EngineToolParser(Inkling2Sm70ParserToolAdapter):  # type: ignore[valid-type, misc]
    # No tool-call structural tag: named and required tool choice parse as auto.
    structural_tag_model = None
    supports_required_and_named = False


class _NeedsToolParser(type):
    """Rejects a composed parser that pairs the reasoning parser with no or
    another tool parser.

    v0.33.0's DelegatingParser stops reasoning parsing at the first text
    block; only this tool parser parses the later blocks of a stream.
    ``ParserManager`` composes the parser class at server start, which calls
    this hook.
    """

    def __set_name__(cls, owner: type, name: str) -> None:
        if name != "reasoning_parser_cls":
            return
        tool_parser_cls = getattr(owner, "tool_parser_cls", None)
        if tool_parser_cls is None or not issubclass(
            tool_parser_cls, Inkling2Sm70EngineToolParser
        ):
            raise TypeError(
                f"--reasoning-parser {PARSER_NAME} requires "
                f"--tool-call-parser {PARSER_NAME} --enable-auto-tool-choice "
                f"(got tool parser {getattr(tool_parser_cls, '__name__', None)}): "
                "without it, streamed content carries the framing tokens."
            )


class Inkling2Sm70ParserReasoningAdapter(_ReasoningAdapter, metaclass=_NeedsToolParser):  # type: ignore[valid-type, misc]
    pass


Inkling2Sm70Parser.reasoning_parser_cls = Inkling2Sm70ParserReasoningAdapter  # type: ignore[attr-defined]

ReasoningParserManager.register_module(
    PARSER_NAME, module=Inkling2Sm70ParserReasoningAdapter
)
ToolParserManager.register_module(PARSER_NAME, module=Inkling2Sm70EngineToolParser)
