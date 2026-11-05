{
  description = "Upstream vLLM for V100 (sm_70): pinned tag + ordered patch series, built with Nix";

  # nixpkgs is locked to nixos-26.05 @ 7dcdd87 (the Phase 1 pin sheet).
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
  };

  outputs =
    inputs@{ nixpkgs, ... }:
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
    in
    {
      packages.${system} = {
        default = p.vllm;
        inherit (p) vllm tree venv;
        sm70-ext = p.sm70ext;
        source = src;
      };
    };
}
