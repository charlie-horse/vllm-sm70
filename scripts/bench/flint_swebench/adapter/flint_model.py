"""LitellmModel that guarantees the assistant's reasoning is kept in history.

Stock mini-swe-agent 2.4.6 + litellm 1.106.6 already round-trip it (litellm maps
the server's `message.reasoning` to `reasoning_content`; mini keeps
`message.model_dump()`). This adapter pins that behaviour so a litellm/mini
upgrade can't silently drop it: whichever of `reasoning_content` / `reasoning`
the server returns is stored as `reasoning_content` on the assistant message,
which the flint vLLM chat template renders (verified via prompt_tokens).

Use: model.model_class: flint_model.FlintModel  (with adapter/ on PYTHONPATH)
"""

from minisweagent.models.litellm_model import LitellmModel


def _raw_reasoning(message: dict) -> str | None:
    raw = message.get("extra", {}).get("response")
    if not isinstance(raw, dict):
        return None
    try:
        m = raw["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return None
    for key in ("reasoning_content", "reasoning"):
        val = m.get(key) or (m.get("provider_specific_fields") or {}).get(key)
        if isinstance(val, str) and val:
            return val
    return None


class FlintModel(LitellmModel):
    def query(self, messages: list[dict], **kwargs) -> dict:
        message = super().query(messages, **kwargs)
        if not isinstance(message.get("reasoning_content"), str):
            reasoning = message.get("reasoning") or _raw_reasoning(message)
            if isinstance(reasoning, str) and reasoning:
                message["reasoning_content"] = reasoning
        message.pop("reasoning", None)  # send one key only
        return message

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        prepared = super()._prepare_messages_for_api(messages)
        for msg in prepared:
            # litellm's hosted_vllm transform pops non-str reasoning_content;
            # drop None ourselves so openai/ provider doesn't send null either.
            if msg.get("role") == "assistant" and msg.get("reasoning_content") is None:
                msg.pop("reasoning_content", None)
        return prepared
