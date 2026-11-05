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
# vllm/python/lmcache/tree come from ./package.nix. LMCache 0.5.6 is built
# from the sdist for sm_70 (./lmcache-build.nix, with ./patches/lmcache) and
# sits on the wrapper's PYTHONPATH, so bin/lmcache and lmcacheUi's bin/python
# are the same wrapper as bin/vllm.
{
  pkgs,
  lib,
  inputs,
  ...
}:
let
  src = import ./source.nix { inherit pkgs lib; };
  build = import ./package.nix { inherit pkgs lib inputs src; };

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
    ];
    passthru = { inherit (build) venv tree lmcacheBuild; };
  };
  # Only bin/python (kept off the system PATH); it imports
  # lmcache.lmcache_frontend.app like the vllm wrapper does.
  lmcacheUi = pkgs.runCommand "lmcache-ui-sm70-${src.version}" { } ''
    mkdir -p $out/bin
    ln -s ${build.vllm}/bin/python $out/bin/python
  '';
  inherit (build) tree;
}
