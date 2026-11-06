# SPDX-License-Identifier: Apache-2.0
"""Sm70CompositePass: installs into the custom-post-pass slot, stable uuid."""

import subprocess
import sys

import pytest
from conftest import requires_vllm

pytestmark = requires_vllm


@pytest.fixture
def passes(monkeypatch):
    from vllm_sm70 import passes as p

    monkeypatch.setattr(p, "_PATTERNS", [])
    return p


def _config():
    from vllm.config import CompilationConfig

    return CompilationConfig()


def test_install_into_empty_slot(passes):
    cc = _config()
    assert passes.PASS_KEY not in cc.inductor_compile_config
    composite = passes.install_composite_pass(cc)
    assert cc.inductor_compile_config[passes.PASS_KEY] is composite
    assert passes.install_composite_pass(cc) is composite  # idempotent


def test_existing_pass_is_kept(passes):
    from vllm.compilation.passes.inductor_pass import CallableInductorPass

    seen = []
    existing = CallableInductorPass(lambda g: seen.append("existing"), uuid="x1")
    cc = _config()
    cc.inductor_compile_config[passes.PASS_KEY] = existing
    composite = passes.install_composite_pass(cc)
    assert composite.inner == [existing]
    composite(None)
    assert seen == ["existing"]
    assert composite.uuid() != passes.Sm70CompositePass().uuid()


def test_non_pass_rejected(passes):
    cc = _config()
    cc.inductor_compile_config[passes.PASS_KEY] = object()
    with pytest.raises(TypeError):
        passes.install_composite_pass(cc)


def test_uuid_stable(passes):
    a, b = passes.Sm70CompositePass(), passes.Sm70CompositePass()
    assert a.uuid() == b.uuid()
    assert len(a.uuid()) == 64


def test_uuid_stable_across_processes(passes):
    code = (
        "from vllm_sm70.passes import Sm70CompositePass; "
        "print(Sm70CompositePass().uuid())"
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        ).stdout.strip()
        for _ in range(2)
    }
    assert len(runs) == 1


def test_uuid_tracks_patterns_and_extension(passes, monkeypatch):
    from vllm_sm70 import _ext

    base = passes.Sm70CompositePass().uuid()
    passes.register_pattern("demo", 1, lambda g: None)
    passes.register_pattern("demo", 1, lambda g: None)  # idempotent
    assert passes.patterns() == [("demo", 1)]
    with_pattern = passes.Sm70CompositePass().uuid()
    assert with_pattern != base
    with pytest.raises(ValueError):
        passes.register_pattern("demo", 2, lambda g: None)

    monkeypatch.setattr(_ext, "fingerprint", lambda: "abi=1;so=deadbeef")
    assert passes.Sm70CompositePass().uuid() != with_pattern


def test_survives_deepcopy_and_pickle(passes):
    """UP backends.py deep-copies inductor_compile_config; VllmConfig is
    pickled to the engine-core process."""
    import copy
    import pickle

    p = passes.Sm70CompositePass()
    assert copy.deepcopy(p).uuid() == p.uuid()
    assert pickle.loads(pickle.dumps(p)).uuid() == p.uuid()
