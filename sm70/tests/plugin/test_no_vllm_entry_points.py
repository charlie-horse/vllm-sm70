# SPDX-License-Identifier: Apache-2.0
"""Entry points: nix-entry_points.txt is the source of truth and resolves."""

import configparser
import importlib
import importlib.metadata as md
from pathlib import Path

import pytest
import tomllib

SM70_ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {
    "vllm.general_plugins": {"sm70": "vllm_sm70:register"},
    "vllm.platform_plugins": {"sm70": "vllm_sm70.platform:activate"},
}


def _read_nix_entry_points():
    parser = configparser.ConfigParser(delimiters=("=",))
    parser.optionxform = str
    parser.read(SM70_ROOT / "nix-entry_points.txt", encoding="utf-8")
    return {g: dict(parser.items(g)) for g in parser.sections()}


def test_nix_entry_points_file():
    assert _read_nix_entry_points() == EXPECTED


def test_pyproject_defers_to_nix_file():
    meta = tomllib.loads((SM70_ROOT / "pyproject.toml").read_text())["project"]
    assert meta["name"] == "vllm-sm70"
    assert meta["version"] == "0.1.0"
    assert "entry-points" in meta.get("dynamic", [])
    assert "entry-points" not in meta and "scripts" not in meta


@pytest.mark.parametrize(
    "group,name", [(g, n) for g, eps in EXPECTED.items() for n in eps]
)
def test_entry_point_targets_resolve(group, name):
    module_name, attr = EXPECTED[group][name].split(":")
    target = getattr(importlib.import_module(module_name), attr)
    assert callable(target)


def test_installed_metadata_matches():
    try:
        dist = md.distribution("vllm-sm70")
    except md.PackageNotFoundError:
        pytest.skip("vllm-sm70 is not installed")
    got = {}
    for ep in dist.entry_points:
        got.setdefault(ep.group, {})[ep.name] = ep.value
    assert got == EXPECTED
    for ep in dist.entry_points:
        assert callable(ep.load())
