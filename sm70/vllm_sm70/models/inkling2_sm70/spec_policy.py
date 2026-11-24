# SPDX-License-Identifier: Apache-2.0
"""When an Inkling 2 DSpark engine speculates: a batch-size schedule.

The engine-config hook (config.py) sets upstream's dynamic speculative
decoding schedule, ``SpeculativeConfig.num_speculative_tokens_per_batch_size``
(UP config/speculative.py:524), from ``sm70.inkling2``:

    [(1, T, K), (T + 1, max_num_seqs, 0)]

with T = ``dspark_spec_max_reqs`` and K = ``dspark_spec_tokens``. The V2
scheduler looks up K by the number of scheduled requests of a step (UP
v1/core/sched/scheduler.py:1710-1714). The drafts proposed after that step
are cut to K: K = 0 stores the draft's context K/V only and skips its block
pass (UP v1/worker/gpu/spec_decode/dflash/speculator.py:546-553), so the
draft's caches hold every committed position whichever tier a step takes.
The next step then verifies K drafts, or decodes one token per request.
A step's tier depends on that step alone.

No schedule is set when K = num_speculative_tokens and T >= max_num_seqs
(every step speculates), when ``--speculative-config`` already holds one,
or with ``VLLM_SM70_INKLING_DSPARK_AUTO=0``.
"""

from __future__ import annotations

from typing import Any

SWITCH = "VLLM_SM70_INKLING_DSPARK_AUTO"

__all__ = ["SWITCH", "apply", "enabled", "schedule", "schedule_of"]

Schedule = list[tuple[int, int, int]]


def enabled() -> bool:
    from vllm_sm70 import envs

    return bool(envs.VLLM_SM70_INKLING_DSPARK_AUTO)


def schedule_of(spec: Any) -> Schedule | None:
    """The batch-size schedule `spec` holds, or None."""
    return getattr(spec, "num_speculative_tokens_per_batch_size", None) or None


def schedule(
    num_speculative_tokens: int,
    max_num_seqs: int | None,
    max_reqs: int,
    spec_tokens: int,
) -> Schedule | None:
    """The schedule for a draft of `num_speculative_tokens` (module
    docstring), or None when every step speculates with all of them."""
    if not 1 <= spec_tokens <= num_speculative_tokens:
        raise ValueError(
            f"sm70.inkling2.dspark_spec_tokens must be 1 to "
            f"num_speculative_tokens ({num_speculative_tokens}), got {spec_tokens}"
        )
    if max_num_seqs is not None and max_reqs >= max_num_seqs:
        if spec_tokens == num_speculative_tokens:
            return None
        return [(1, max_num_seqs, spec_tokens)]
    end = max_reqs + 1 if max_num_seqs is None else max_num_seqs
    return [(1, max_reqs, spec_tokens), (max_reqs + 1, end, 0)]


def apply(vllm_config: Any, inkling2: Any) -> Schedule | None:
    """Set the schedule on the engine's speculative config (module
    docstring); returns the schedule it holds afterwards."""
    spec = getattr(vllm_config, "speculative_config", None)
    if spec is None:
        return None
    if schedule_of(spec) is not None or not enabled():
        return schedule_of(spec)
    max_num_seqs = getattr(vllm_config.scheduler_config, "max_num_seqs", None)
    value = schedule(
        int(spec.num_speculative_tokens),
        max_num_seqs if isinstance(max_num_seqs, int) else None,
        inkling2.dspark_spec_max_reqs,
        inkling2.dspark_spec_tokens,
    )
    if value is not None:
        spec.num_speculative_tokens_per_batch_size = value
    return value
