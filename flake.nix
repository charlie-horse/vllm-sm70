{
  description = "Upstream vLLM for V100 (sm_70): pinned tag + ordered patch series, built with Nix, with NixOS modules";

  # nixpkgs is locked to nixos-26.05 @ 7dcdd87 (the Phase 1 pin sheet).
  # Hosts should set `inputs.nixpkgs.follows = "nixpkgs"`.
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";

    pyproject-nix.url = "github:pyproject-nix/pyproject.nix";
    pyproject-nix.inputs.nixpkgs.follows = "nixpkgs";
    uv2nix.url = "github:pyproject-nix/uv2nix";
    uv2nix.inputs.nixpkgs.follows = "nixpkgs";
    uv2nix.inputs.pyproject-nix.follows = "pyproject-nix";
    pyproject-build-systems.url = "github:pyproject-nix/build-system-pkgs";
    pyproject-build-systems.inputs.nixpkgs.follows = "nixpkgs";
    pyproject-build-systems.inputs.pyproject-nix.follows = "pyproject-nix";
    pyproject-build-systems.inputs.uv2nix.follows = "uv2nix";

    # The former fork's flake, kept only for the Flint checkpoint recipe
    # (converter, GGUF and config pins; ./nix/checkpoint.nix). Its own lock
    # pins its nixpkgs/uv2nix; nothing from it reaches the serving closure.
    vllm-legacy.url = "github:charlie-horse/vLLM/cf9492a01eeca0cac731c0fa62015e2c85aaf272";
  };

  outputs =
    inputs@{ self, nixpkgs, ... }:
    let
      system = "x86_64-linux";
      inherit (nixpkgs) lib;
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
        # Only the toolkit comes from nixpkgs (the CUDA libs ship in the
        # wheels); target V100 only.
        config.cudaCapabilities = [ "7.0" ];
      };
      src = import ./nix/source.nix { inherit pkgs lib; };
      p = import ./nix/package.nix { inherit pkgs lib inputs src; };
      checkpoint = import ./nix/checkpoint.nix { inherit inputs; };
      vllmModule = import ./nix/module.nix { inherit inputs; };
    in
    {
      packages.${system} = {
        default = p.vllm;
        inherit (p) vllm tree venv;
        gpu-tests = import ./nix/gpu-tests.nix { inherit pkgs lib; vllm = p.vllm; };
        sm70-ext = p.sm70ext;
        lmcache = p.lmcacheBuild;
        source = src;
        flint-checkpoint = checkpoint.flintCheckpoint;
      };

      # `vllm` (services.vllm-sm70) imports `lmcache` (services.lmcache-sm70);
      # use `lmcache` alone to run only the cache server. Both build their
      # packages with the host's `pkgs` (./nix/packages-module.nix).
      nixosModules = {
        default = vllmModule;
        vllm = vllmModule;
        lmcache = import ./nix/lmcache.nix { inherit inputs; };
      };

      checks.${system} = import ./nix/checks.nix { inherit self nixpkgs pkgs; };
    };
}
