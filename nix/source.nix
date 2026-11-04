# Upstream vLLM at the pinned tag with ../patches/*.patch applied in name
# order (NNNN-*.patch). Phase 1 has no patches: this is stock upstream.
{ pkgs, lib }:
let
  version = "0.33.0";
  upstream = pkgs.fetchFromGitHub {
    owner = "vllm-project";
    repo = "vllm";
    rev = "v${version}"; # 4ba787ec9959e566d3b587a796dff07673324da5
    hash = "sha256-I72HXJjBOCmNSwbuK/D3KB5/YGGJ0f1OY5BhWDevI/w=";
  };
  patchDir = ../patches;
  patches = map (n: patchDir + "/${n}") (
    lib.sort lib.lessThan (
      lib.filter (lib.hasSuffix ".patch") (lib.attrNames (builtins.readDir patchDir))
    )
  );
in
if patches == [ ] then
  upstream // { inherit version; }
else
  (pkgs.applyPatches {
    name = "vllm-${version}-sm70-source";
    src = upstream;
    inherit patches;
  })
  // {
    inherit version;
  }
