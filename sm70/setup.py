# SPDX-License-Identifier: Apache-2.0
"""Build vllm-sm70: the Python plugin plus the `vllm_sm70._sm70_C` extension.

Offline by design (the Nix sandbox has no network): this file never fetches
or pip-installs anything. CMake, a CUDA 12.6 toolkit and torch 2.14 must
already be on the path.

Environment:
  VLLM_SM70_SKIP_EXT=1      build a pure-Python package (no _sm70_C); for CPU CI.
  VLLM_SM70_CUTLASS_DIR     optional CUTLASS source tree (headers only, read-only).
  TORCH_CUDA_ARCH_LIST      arch list; intersected with 7.0 (default "7.0").
  MAX_JOBS                  parallel compile jobs, capped at 8.
  CMAKE_BUILD_TYPE          default Release.
"""

import configparser
import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

ROOT = Path(__file__).resolve().parent
MAX_JOBS_CAP = 8


def read_entry_points() -> dict[str, list[str]]:
    """nix-entry_points.txt is the single source of truth for entry points."""
    parser = configparser.ConfigParser(delimiters=("=",))
    parser.optionxform = str  # keep case
    parser.read(ROOT / "nix-entry_points.txt", encoding="utf-8")
    return {
        group: [f"{name} = {value}" for name, value in parser.items(group)]
        for group in parser.sections()
    }


def max_jobs() -> int:
    try:
        jobs = int(os.environ.get("MAX_JOBS", MAX_JOBS_CAP))
    except ValueError:
        jobs = MAX_JOBS_CAP
    return max(1, min(jobs, MAX_JOBS_CAP))


class CMakeExtension(Extension):
    def __init__(self, name: str) -> None:
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def build_extension(self, ext: Extension) -> None:
        import torch.utils  # torch must already be installed; never fetched

        extdir = Path(self.get_ext_fullpath(ext.name)).parent.resolve()
        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)
        cfg = os.environ.get("CMAKE_BUILD_TYPE", "Release")

        args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
            "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
        ]
        cutlass = os.environ.get("VLLM_SM70_CUTLASS_DIR")
        if cutlass:
            args.append(f"-DVLLM_SM70_CUTLASS_DIR={cutlass}")
        if os.environ.get("VLLM_SM70_ALLOW_NO_CUDA") == "1":
            # Test-only: lets the CPU-only binding build against CPU torch.
            args.append("-DVLLM_SM70_REQUIRE_CUDA=OFF")
        generator = ["-G", "Ninja"] if _has_ninja() else []

        subprocess.check_call(["cmake", str(ROOT), *generator, *args], cwd=build_temp)
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", "_sm70_C", "-j", str(max_jobs())],
            cwd=build_temp,
        )


def _has_ninja() -> bool:
    from shutil import which

    return which("ninja") is not None


skip_ext = os.environ.get("VLLM_SM70_SKIP_EXT") == "1"

setup(
    entry_points=read_entry_points(),
    ext_modules=[] if skip_ext else [CMakeExtension("vllm_sm70._sm70_C")],
    cmdclass={} if skip_ext else {"build_ext": CMakeBuild},
)
