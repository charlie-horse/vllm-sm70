# The package set the modules use (`vllmSm70Packages`):
#
#   vllm             the wrapped env: bin/vllm, bin/hf, bin/lmcache (./vllm.nix)
#   lmcacheUi        a wrapped bin/python that can import lmcache (./vllm.nix)
#   flintCheckpoint  Flint's converted weights, fixed-output (./checkpoint.nix)
#
# `pkgs` is the host's (via ./packages-module.nix) or this flake's own.
{ pkgs, lib, inputs }:
let
  serving = import ./vllm.nix { inherit pkgs lib inputs; };
  checkpoint = import ./checkpoint.nix { inherit inputs; };
in
serving
// {
  inherit (checkpoint) flintCheckpoint;
}
