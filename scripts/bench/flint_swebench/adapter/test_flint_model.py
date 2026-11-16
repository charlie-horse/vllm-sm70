"""reasoning must round-trip into the next request's messages.

Run: cd /home/user/swe && .venv/bin/python -m pytest adapter -q
"""

import os
import sys
from pathlib import Path
from unittest import mock

import litellm
import pytest

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "1")

from flint_model import FlintModel  # noqa: E402
from wire_probe import run  # noqa: E402

THINK = "THINK-1: inspect repo first"


def _fake_response(field: str) -> litellm.ModelResponse:
    msg = {
        "role": "assistant",
        "content": "Let me look.",
        field: THINK,
        "tool_calls": [{"id": "call_1", "type": "function",
                        "function": {"name": "bash",
                                     "arguments": '{"command": "ls"}'}}],
    }
    return litellm.ModelResponse(
        model="flint",
        choices=[{"index": 0, "message": msg, "finish_reason": "tool_calls"}],
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
def test_fake_litellm_roundtrip(field):
    model = FlintModel(model_name="hosted_vllm/flint",
                       cost_tracking="ignore_errors")
    sent = []

    def fake_completion(**kw):
        sent.append(kw["messages"])
        return _fake_response(field)

    with mock.patch.object(litellm, "completion", side_effect=fake_completion):
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        a1 = model.query(msgs)
        obs = model.format_observation_messages(
            a1, [{"output": "f", "returncode": 0, "exception_info": ""}])
        model.query(msgs + [a1] + obs)
    asst = [m for m in sent[1] if m["role"] == "assistant"]
    assert asst[0]["reasoning_content"] == THINK
    assert "reasoning" not in asst[0] and "extra" not in asst[0]


@pytest.mark.parametrize("prefix", ["openai", "hosted_vllm"])
@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
def test_wire_roundtrip_through_litellm(prefix, field):
    """Real litellm HTTP path against a fake server: checks litellm doesn't strip."""
    sent = run(prefix, field, "flint_model.FlintModel")
    assert sent["reasoning_content"] == THINK
