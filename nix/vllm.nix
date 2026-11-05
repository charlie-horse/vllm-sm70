# The packages the modules run, on top of the sm_70 build (./package.nix).
#
# Contract the modules rely on, called as
#   import ./vllm.nix { inherit pkgs lib inputs; }
# with the host's `pkgs`; returns at least:
#
#   vllm       bin/vllm (`vllm serve`), bin/python, bin/hf (huggingface-hub
#              from the same venv, for `hf download`) and bin/lmcache
#              (`lmcache server`).
#   lmcacheUi  wrapper with only bin/python, able to import
#              lmcache.lmcache_frontend.app (kept off the system PATH).
#   tree       the sm_70-built vLLM tree.
#
# vllm/python/tree come from ./package.nix (Phase 1 build). LMCache is not in
# the venv yet, so bin/lmcache and lmcacheUi are still stubs that fail at
# start; the LMCache sdist build replaces them, applying
# `import ./patches/lmcache` (a list of patch files, in order) with
# patchFlags = [ "-p1" "--fuzz=0" "--no-backup-if-mismatch" ].
{
  pkgs,
  lib,
  inputs,
  ...
}:
let
  src = import ./source.nix { inherit pkgs lib; };
  build = import ./package.nix { inherit pkgs lib inputs src; };

  stub =
    name: exes:
    pkgs.runCommand name { } ''
      mkdir -p $out/bin
      for exe in ${lib.escapeShellArgs exes}; do
        cat > $out/bin/$exe <<SH
      #!${pkgs.runtimeShell}
      echo "vllm-sm70: '$exe' is a stub; LMCache is not built into this flake yet" >&2
      exit 1
      SH
        chmod +x $out/bin/$exe
      done
    '';

  # `hf` from the venv (huggingface-hub[hf-xet]); it needs no CUDA setup.
  hf = pkgs.runCommand "vllm-sm70-hf" { nativeBuildInputs = [ pkgs.makeWrapper ]; } ''
    mkdir -p $out/bin
    makeWrapper ${build.venv}/bin/hf $out/bin/hf
  '';
in
{
  vllm = pkgs.symlinkJoin {
    name = "vllm-sm70-service-${src.version}";
    paths = [
      build.vllm
      hf
      (stub "lmcache-sm70-stub" [ "lmcache" ])
    ];
    passthru = { inherit (build) venv tree; };
  };
  lmcacheUi = stub "lmcache-ui-sm70-stub" [ "python" ];
  inherit (build) tree;
}
