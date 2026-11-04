{
  description = "vLLM for Volta (sm_70): upstream vLLM + a patch series, with NixOS modules";

  inputs = {
    # docs/pins.md. Hosts should set `inputs.nixpkgs.follows = "nixpkgs"`.
    nixpkgs.url = "github:NixOS/nixpkgs/7dcdd87373d72e7a2173e4db02e20d3d7f495cfd";

    # The former fork's flake, kept only for the Flint checkpoint recipe
    # (converter, GGUF and config pins; ./nix/checkpoint.nix). Its own lock
    # pins its nixpkgs/uv2nix; nothing from it reaches the serving closure.
    vllm-legacy.url = "github:charlie-horse/vLLM/cf9492a01eeca0cac731c0fa62015e2c85aaf272";
  };

  outputs =
    inputs@{ self, nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      p = import ./nix/packages.nix {
        inherit pkgs inputs;
        inherit (nixpkgs) lib;
      };
      vllm = import ./nix/module.nix { inherit inputs; };
      # Upstream vLLM at patches/BASE with patches/series applied (the same
      # file as on phase1-spike).
      source = import ./nix/source.nix {
        inherit pkgs;
        inherit (nixpkgs) lib;
      };
    in
    {
      packages.${system} = {
        default = p.vllm;
        vllm = p.vllm;
        flint-checkpoint = p.flintCheckpoint;
        inherit source;
      };

      # `vllm` (services.vllm-sm70) imports `lmcache` (services.lmcache-sm70);
      # use `lmcache` alone to run only the cache server. Both build their
      # packages with the host's `pkgs`.
      nixosModules = {
        default = vllm;
        inherit vllm;
        lmcache = import ./nix/lmcache.nix { inherit inputs; };
      };

      checks.${system} = import ./nix/checks.nix { inherit self nixpkgs pkgs; };
    };
}
