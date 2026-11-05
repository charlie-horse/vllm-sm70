# Shared by the vllm and lmcache modules: builds the package set once with
# the host's `pkgs` and hands it to both as the `vllmSm70Packages` module
# argument. The `key` makes the module system import it only once.
{ inputs }:
{ lib, pkgs, ... }:
{
  key = "vllm-sm70/packages";
  _module.args.vllmSm70Packages = import ./packages.nix {
    inherit pkgs lib inputs;
  };
}
