# Flint (RecursiveSI/flint) NVFP4 checkpoint: the GGUF converted to a ModelOpt
# W4A16_NVFP4 checkpoint, served from the Nix store.
#
# Fixed-output: the store path depends only on the name, the hash mode and the
# hash, so it is the same path the old flake (charlie-horse/vLLM, from 647bceb)
# produced: /nix/store/0qxvf6z208m6bff62acgwfg6ajhbinpp-flint-nvfp4-modelopt.
# Existing copies (c4130, r440's cache) are reused and nothing re-converts.
# `checks.<system>.flint-checkpoint-outpath` fails evaluation if it drifts.
#
# Upstream vLLM has no converter (vllm.models.flint.convert_gguf is fork-only),
# so the build recipe comes from the old flake (input `vllm-legacy`, pinned to
# cf9492a): its converter env, GGUF fetch and HF config files, evaluated with
# that flake's own locked inputs (nixpkgs, uv2nix, ...) so a host nixpkgs bump
# can't break it. They only build when the path is missing from every store
# and cache (~25 min CUDA tree + ~15 min / ~25 GB RAM conversion).
#
# Change checkpointHash only together with the converter, GGUF or config pin
# (i.e. a new checkpoint format); then update expectedOutPath too.
{ inputs }:
let
  legacy = inputs.vllm-legacy;
  pkgs = import legacy.inputs.nixpkgs {
    system = "x86_64-linux";
    config.allowUnfree = true;
  };
  inherit (pkgs) lib;

  legacyPkgs = import "${legacy}/nix/package.nix" {
    inherit pkgs lib;
    inputs = legacy.inputs;
    src = import "${legacy}/nix/source.nix" { inherit lib; };
  };
  inherit (legacyPkgs) flintConverter gguf hfConfig;

  checkpointHash = "sha256-HyBUnISsTQXSKmmOft+/fNzrhFLwBJ6imMCa7YRAwoQ=";
  expectedOutPath = "/nix/store/0qxvf6z208m6bff62acgwfg6ajhbinpp-flint-nvfp4-modelopt";

  # ~23 GB output, ~25 GB peak RAM. The output must not reference store paths
  # (a fixed-output rule).
  flintCheckpoint =
    pkgs.runCommand "flint-nvfp4-modelopt"
      {
        passthru = { inherit gguf hfConfig flintConverter; };
        outputHashMode = "recursive";
        outputHashAlgo = "sha256";
        outputHash = checkpointHash;
      }
      ''
        export HOME=$TMPDIR HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1
        ${flintConverter}/bin/python -m vllm.models.flint.convert_gguf \
          ${gguf} $out --hf-config ${hfConfig} --device cpu
        # Run metadata (duration, absolute paths): not reproducible.
        rm -f $out/conversion_report.json
      '';
in
{
  inherit flintCheckpoint expectedOutPath checkpointHash;
}
