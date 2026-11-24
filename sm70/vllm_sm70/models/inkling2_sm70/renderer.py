# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native Inkling 2 chat rendering to token ids.

What: renderer mode ``inkling2_sm70`` (``--tokenizer-mode inkling2_sm70``).
  Chat messages render straight to token ids through inkling2_encoding.py;
  the tokenizer is the plain cached HF one.
Why: vLLM v0.33.0's ``inkling`` renderer is Inkling 1 only: it needs the
  audio tokens, adds a generation prefix and lets payload text encode to
  framing tokens. The tokenizer adapter and the request handling are ported
  from upstream's `vllm/renderers/inkling.py` (see PROVENANCE.md).
Request fields: ``reasoning_effort``, ``continue_final_message``, ``tools``
  and ``parallel_tool_calls`` are read from ``chat_template_kwargs``
  (v0.33.0's ChatParams has no ``parallel_tool_calls`` field; default true).
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING

from vllm.renderers.base import BaseRenderer
from vllm.renderers.inkling import InklingRenderer
from vllm.utils.async_utils import make_async

from .inkling2_encoding import (
    CONTENT_MODEL_END_SAMPLING,
    DEFAULT_REASONING_EFFORT,
    OPTIONAL_SPECIAL_TOKENS,
    SPECIAL_TOKEN_SPELLINGS,
    render_inkling2_messages,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.renderers.chat_utils import ChatCompletionMessageParam
    from vllm.renderers.params import ChatParams
    from vllm.tokenizers.hf import HfTokenizer

logger = logging.getLogger("vllm.sm70")

_NAMED_REASONING_EFFORT = {
    "none": 0.0,
    "minimal": 0.1,
    "low": 0.2,
    "medium": 0.7,
    "high": 0.9,
    "xhigh": 0.99,
    "max": 0.99,
}


def _resolve_reasoning_effort(value: object) -> float:
    """A name, a number or a numeric string; None is the default effort."""
    effort = value
    if effort is None:
        effort = DEFAULT_REASONING_EFFORT
    elif isinstance(effort, str):
        effort = _NAMED_REASONING_EFFORT.get(effort)
        if effort is None:
            with contextlib.suppress(ValueError):
                effort = float(value)
    if isinstance(effort, bool) or not isinstance(effort, (int, float)):
        # ValueError, not TypeError: a bad request value, as upstream raises.
        raise ValueError(f"invalid Inkling reasoning_effort: {value!r}")  # noqa: TRY004
    return float(effort)


def _resolve_parallel_tool_calls(value: object) -> bool:
    if value is None:
        return True
    if not isinstance(value, bool):
        raise TypeError(f"parallel_tool_calls must be a boolean, got {value!r}")
    return value


class _HfBackedTmlTokenizer:
    """Adapts an HF tokenizer to the encoding core's tokenizer protocol.

    Special-token ids are resolved from the tokenizer vocab at construction,
    trying each known spelling. Text is encoded with
    ``split_special_tokens=True``, so a framing-token string inside payload
    text stays ordinary text.
    """

    def __init__(self, tokenizer: HfTokenizer) -> None:
        self._tokenizer = tokenizer

        vocab = tokenizer.get_vocab()
        special_ids: dict[str, int] = {}
        missing: list[str] = []
        for token, spellings in SPECIAL_TOKEN_SPELLINGS.items():
            for spelling in spellings:
                token_id = vocab.get(spelling)
                if token_id is not None:
                    special_ids[token] = token_id
                    break
            else:
                if token not in OPTIONAL_SPECIAL_TOKENS:
                    missing.append(token)
        if missing:
            raise ValueError(f"Inkling tokenizer is missing special tokens: {missing}")
        self._special_ids = special_ids

    def encode_text(self, text: str) -> list[int]:
        return self._tokenizer.encode(
            text, add_special_tokens=False, split_special_tokens=True
        )

    def encode_special(self, token: str) -> int:
        try:
            return self._special_ids[token]
        except KeyError:
            raise ValueError(f"Inkling tokenizer does not define {token}") from None


class Inkling2Sm70Renderer(InklingRenderer):
    """v0.33.0's InklingRenderer message plumbing with the Inkling 2 encoding."""

    def __init__(self, config: VllmConfig, tokenizer: HfTokenizer | None) -> None:
        # Not InklingRenderer.__init__: its adapter requires the audio tokens.
        BaseRenderer.__init__(self, config, tokenizer)
        self._inkling_tokenizer = _HfBackedTmlTokenizer(self.get_tokenizer())
        self._render_async = make_async(self._render, executor=self._executor)

    def _render(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> list[int]:
        kwargs = params.chat_template_kwargs or {}
        try:
            return render_inkling2_messages(
                messages,
                self._inkling_tokenizer,
                tools=kwargs.get("tools"),
                tool_choice=params.tool_choice,
                parallel_tool_calls=_resolve_parallel_tool_calls(
                    kwargs.get("parallel_tool_calls")
                ),
                reasoning_effort=_resolve_reasoning_effort(
                    kwargs.get("reasoning_effort")
                ),
                continue_final_message=bool(kwargs.get("continue_final_message")),
            )
        except ValueError:
            raise
        except (TypeError, KeyError) as e:
            # Malformed request content; surface as a request error.
            raise ValueError(str(e)) from e
        except Exception as e:
            logger.exception("Error while rendering Inkling 2 chat messages")
            raise ValueError(str(e)) from e

    def get_eos_token_id(self) -> int | None:
        eos_token_id = super().get_eos_token_id()
        if eos_token_id is None:
            return self._inkling_tokenizer.encode_special(CONTENT_MODEL_END_SAMPLING)
        return eos_token_id
