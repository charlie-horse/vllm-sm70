# Flint (RecursiveSI/flint) NVFP4 checkpoint: the GGUF converted to a ModelOpt
# W4A16_NVFP4 checkpoint, served from the Nix store.
#
# Fixed-output: the store path depends only on the name, the hash mode and the
# hash, so it is the same path the old flake (charlie-horse/vLLM, from 647bceb)
# produced: /nix/store/0qxvf6z208m6bff62acgwfg6ajhbinpp-flint-nvfp4-modelopt.
# Existing copies (c4130, r440's cache) are reused and nothing re-converts.
# `checks.<system>.flint-checkpoint-outpath` fails evaluation if it drifts.
#
# There is no build recipe here: upstream vLLM has no Flint converter; the
# fork's converter is copied to sm70/tools/flint/ (from cf9492a) but is not
# wired into a derivation (its env needs gguf, which .#vllm lacks). requireFile only
# substitutes; if the path is in no store or cache, the build fails with the
# message below. Recovery procedure: docs/nix.md, "Flint checkpoint store path".
#
# Change checkpointHash only together with a new checkpoint format; then update
# expectedOutPath too.
{ inputs }:
let
  pkgs = import inputs.nixpkgs { system = "x86_64-linux"; };

  checkpointHash = "sha256-HyBUnISsTQXSKmmOft+/fNzrhFLwBJ6imMCa7YRAwoQ=";
  expectedOutPath = "/nix/store/0qxvf6z208m6bff62acgwfg6ajhbinpp-flint-nvfp4-modelopt";

  flintCheckpoint = pkgs.requireFile {
    name = "flint-nvfp4-modelopt";
    hashMode = "recursive";
    hash = checkpointHash;
    message = ''
      The Flint checkpoint ${expectedOutPath}
      is in no store or cache, and this flake cannot build it.
      Copy it from a host that has it (c4130, r440):
        nix copy --from ssh://<host> ${expectedOutPath}
      or regenerate it with the in-repo copy of the fork's converter,
      sm70/tools/flint/convert_gguf.py (from charlie-horse/vLLM cf9492a;
      inputs and env: sm70/tools/flint/PROVENANCE.md), and add it:
        nix-store --add-fixed --recursive sha256 <dir>
      where <dir> is named flint-nvfp4-modelopt. Steps: docs/nix.md,
      "Flint checkpoint store path".
    '';
  };
in
{
  inherit flintCheckpoint expectedOutPath checkpointHash;
}
