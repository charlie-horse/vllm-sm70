# SPDX-License-Identifier: Apache-2.0
"""The Inkling 2 DSpark draft: its staged directory, engine check and
speculator class.

Staged directory: ``config.json`` (model_type ``inkling2_dspark``,
architecture ``DRAFT_ARCH``, built by ``Inkling2DSparkConfig.from_target``
from the target's ``mtp_config``) and a symlink to the target's
``mtp.safetensors``. Stage it with

    python -m vllm_sm70.models.inkling2_sm70.draft <target dir> <draft dir>

and pass the draft dir as ``"model"`` with ``"method": "dspark"`` in
``--speculative-config``. The draft's tokenizer is the target's.

``speculative_rejection`` is the engine-config hook's check (config.py).
``speculator_cls`` answers the U6 platform hook (patches/0006,
``Sm70CudaPlatform.get_speculator_cls``) for the V2 runner.
``VLLM_SM70_INKLING_DSPARK=0`` turns both off.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
from typing import Any

DRAFT_MODEL_TYPE = "inkling2_dspark"
DRAFT_ARCH = "Inkling2DSparkSm70Model"
DRAFT_LAZY_TARGET = "vllm_sm70.models.inkling2_sm70.dspark:Inkling2DSparkSm70Model"
SPECULATOR_TARGET = (
    "vllm_sm70.models.inkling2_sm70.speculator.Inkling2DSparkSm70Speculator"
)
DRAFT_WEIGHTS = "mtp.safetensors"
METHOD = "dspark"
STAGE_COMMAND = "python -m vllm_sm70.models.inkling2_sm70.draft"

__all__ = [
    "DRAFT_ARCH",
    "DRAFT_LAZY_TARGET",
    "DRAFT_MODEL_TYPE",
    "SPECULATOR_TARGET",
    "draft_config",
    "enabled",
    "speculative_rejection",
    "speculator_cls",
    "stage_draft_dir",
]


def enabled() -> bool:
    from vllm_sm70 import envs

    return bool(envs.VLLM_SM70_INKLING_DSPARK)


def draft_config(target_dir: str | os.PathLike) -> Any:
    """The draft's ``Inkling2DSparkConfig`` from the target's config.json."""
    from .sm70_configs import CONFIG_CLASSES

    raw = json.loads((pathlib.Path(target_dir) / "config.json").read_text())
    target = CONFIG_CLASSES[raw["model_type"]](**raw)
    config = CONFIG_CLASSES[DRAFT_MODEL_TYPE].from_target(target)
    config.architectures = [DRAFT_ARCH]
    return config


def stage_draft_dir(
    target_dir: str | os.PathLike, draft_dir: str | os.PathLike
) -> pathlib.Path:
    """Write ``draft_dir``/config.json and link ``draft_dir``/mtp.safetensors
    to the target's. Rewrites both when they differ; returns ``draft_dir``."""
    target = pathlib.Path(target_dir).resolve()
    weights = target / DRAFT_WEIGHTS
    if not weights.is_file():
        raise FileNotFoundError(f"vllm_sm70: no {DRAFT_WEIGHTS} in {target}")
    out = pathlib.Path(draft_dir)
    out.mkdir(parents=True, exist_ok=True)
    fields = draft_config(target).to_diff_dict()
    # The class's __init__ sets it and refuses it as a keyword.
    fields.pop("tie_word_embeddings", None)
    text = json.dumps(fields, indent=2, sort_keys=True) + "\n"
    config = out / "config.json"
    if not config.is_file() or config.read_text() != text:
        tmp = out / "config.json.tmp"
        tmp.write_text(text)
        tmp.replace(config)
    link = out / DRAFT_WEIGHTS
    if link.is_symlink() and pathlib.Path(os.readlink(link)) == weights:
        return out
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(weights)
    return out


def _is_inkling_draft(spec: Any) -> bool:
    draft = getattr(spec, "draft_model_config", None)
    hf_config = getattr(draft, "hf_config", None)
    return getattr(hf_config, "model_type", None) == DRAFT_MODEL_TYPE


def speculative_rejection(
    spec: Any, target_hf_config: Any, v2_runner: bool = True
) -> str | None:
    """Why the target can't serve ``spec``, or None (no spec, or method
    dspark with a staged draft whose block and visits fit the target, on the
    V2 runner)."""
    if spec is None:
        return None
    if not enabled():
        return (
            "has speculative decoding off (VLLM_SM70_INKLING_DSPARK=0); drop "
            "--speculative-config"
        )
    method = getattr(spec, "method", None)
    if method != METHOD or not _is_inkling_draft(spec):
        return (
            f"supports speculative decoding only with method {METHOD} and a "
            f"{DRAFT_MODEL_TYPE} draft dir ({STAGE_COMMAND}), got method "
            f"{method}"
        )
    if not v2_runner:
        return (
            f"runs a {DRAFT_MODEL_TYPE} draft only on the V2 model runner; set "
            "VLLM_USE_V2_MODEL_RUNNER=1"
        )
    draft = spec.draft_model_config.hf_config
    k = getattr(spec, "num_speculative_tokens", None)
    if k != draft.dspark_block_size:
        return (
            f"drafts {draft.dspark_block_size} tokens per step with "
            f"{DRAFT_MODEL_TYPE}, got num_speculative_tokens {k}; "
            "sm70.inkling2.dspark_spec_tokens verifies fewer of them"
        )
    get_text = getattr(target_hf_config, "get_text_config", None)
    text = get_text() if callable(get_text) else target_hf_config
    visits = getattr(text, "num_layer_visits", None)
    if visits is not None and draft.target_layer_ids[-1:] != [visits]:
        return (
            f"has {visits} layer visits, but the draft reads visits "
            f"{draft.target_layer_ids}; restage it from this checkpoint"
        )
    return None


def speculator_cls(vllm_config: Any) -> type | None:
    """The speculator class of an Inkling 2 DSpark draft, or None."""
    spec = getattr(vllm_config, "speculative_config", None)
    if spec is None or not enabled():
        return None
    if getattr(spec, "method", None) != METHOD or not _is_inkling_draft(spec):
        return None
    from vllm.utils.import_utils import resolve_obj_by_qualname

    return resolve_obj_by_qualname(SPECULATOR_TARGET)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {STAGE_COMMAND} <target dir> <draft dir>", file=sys.stderr)
        return 2
    print(stage_draft_dir(*argv))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
