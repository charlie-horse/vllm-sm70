# `.#gpu-tests`: the `vllm` wrapper's python plus pytest, for `-m gpu` runs.
# pytest and its pure-python deps come from the pinned nixpkgs for the SAME
# python as the vllm env (`vllm.python`) and go on PYTHONPATH behind the vllm
# wrapper's own entries (one interpreter, one site-packages). `packaging` is
# left out: the venv has it, and a copy on PYTHONPATH would shadow it.
# Nothing is added to the serving closure (`.#vllm`).
{
  pkgs,
  lib,
  vllm,
}:
let
  python = vllm.python;
  mods = lib.filter (m: (m.pname or "") != "packaging") (
    python.pkgs.requiredPythonModules [ python.pkgs.pytest ]
  );
  pypath = lib.concatMapStringsSep ":" (m: "${m}/${python.sitePackages}") mods;
in
pkgs.runCommand "vllm-sm70-gpu-tests"
  {
    nativeBuildInputs = [ pkgs.makeWrapper ];
    meta.mainProgram = "pytest";
  }
  ''
    mkdir -p $out/bin
    makeWrapper ${vllm}/bin/python $out/bin/python --prefix PYTHONPATH : ${pypath}
    makeWrapper ${vllm}/bin/python $out/bin/pytest \
      --add-flags "-m pytest -p no:cacheprovider" --prefix PYTHONPATH : ${pypath}
  ''
