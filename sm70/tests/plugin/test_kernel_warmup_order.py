# SPDX-License-Identifier: Apache-2.0
"""Sm70CudaPlatform.kernel_warmup (U1) runs its steps in a fixed order.

(1) #78 TurboMind pre-capture warmup  (vllm_sm70.warmup.run)
(2) K13 fused router admission       (vllm_sm70.router.prepare_model)
(3) K16 top-k sampler prepare        (vllm_sm70.sample_v1.prepare_worker)

No step returns early past a later one; a worker without get_model skips
only (2). A failure in (2) or (3) raises and stops startup (no later step
runs). A step whose module is not in this build is not asserted.
"""

import importlib
import importlib.util

import pytest
from conftest import requires_vllm

pytestmark = requires_vllm

STEPS = (
    ("warmup", "vllm_sm70.warmup", "run"),
    ("router", "vllm_sm70.router", "prepare_model"),
    ("sampler", "vllm_sm70.sample_v1", "prepare_worker"),
)


def _present(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


@pytest.fixture
def recorded(stub_stable_libtorch, monkeypatch):
    """(platform class, calls list); every present entry point records."""
    from vllm_sm70 import platform

    monkeypatch.delenv("VLLM_SM70_FLINT_SAMPLER", raising=False)
    calls: list[tuple[str, object]] = []
    for step, module, attr in STEPS:
        if not _present(module):
            continue
        mod = importlib.import_module(module)

        def record(arg, *a, _step=step, **k):
            calls.append((_step, arg))

        monkeypatch.setattr(mod, attr, record)
    return platform.Sm70CudaPlatform, calls


def _expected(*, with_model: bool, worker, model) -> list[tuple[str, object]]:
    out = []
    if _present("vllm_sm70.warmup"):
        out.append(("warmup", worker))
    if with_model and _present("vllm_sm70.router"):
        out.append(("router", model))
    out.append(("sampler", worker))  # K16 is on main
    return out


def test_steps_run_in_order(recorded):
    platform_cls, calls = recorded
    model = object()

    class Worker:
        def get_model(self):
            return model

    worker = Worker()
    assert platform_cls.kernel_warmup(worker) is None
    assert calls == _expected(with_model=True, worker=worker, model=model)


def test_missing_model_skips_only_the_router_step(recorded):
    platform_cls, calls = recorded
    worker = object()  # no get_model
    assert platform_cls.kernel_warmup(worker) is None
    assert calls == _expected(with_model=False, worker=worker, model=None)
    assert ("sampler", worker) in calls  # step 3 still ran


def test_router_failure_raises_and_stops_before_the_sampler(recorded, monkeypatch):
    if not _present("vllm_sm70.router"):
        pytest.skip("K13 (vllm_sm70.router) not in this build")
    from vllm_sm70 import router

    platform_cls, calls = recorded

    def boom(model):
        calls.append(("router", model))
        raise ValueError("no workspace")

    monkeypatch.setattr(router, "prepare_model", boom)
    model = object()

    class Worker:
        def get_model(self):
            return model

    worker = Worker()
    with pytest.raises(RuntimeError, match=f"{router.KILL_SWITCH}=0") as ei:
        platform_cls.kernel_warmup(worker)
    assert router.KILL_SWITCH == "VLLM_SM70_FLINT_FUSED_ROUTER"
    assert "K13 fused router" in str(ei.value)
    assert isinstance(ei.value.__cause__, ValueError)
    # Step 1 ran, step 3 did not: startup stops at the failed step.
    assert [step for step, _ in calls] == [
        s for s in ("warmup", "router") if s != "warmup" or _present("vllm_sm70.warmup")
    ]
