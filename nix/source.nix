# Upstream vLLM at the tag in ../patches/BASE, with the patches listed in
# ../patches/series applied in that order (the files scripts/regen-patches.sh
# writes). With an empty series this is the stock upstream source, with the
# same store path as the plain fetch.
{ pkgs, lib }:
let
  version = "0.33.0";
  base = lib.trim (builtins.readFile ../patches/BASE);

  upstream = pkgs.fetchFromGitHub {
    owner = "vllm-project";
    repo = "vllm";
    rev = "v${version}"; # 4ba787ec9959e566d3b587a796dff07673324da5
    hash = "sha256-I72HXJjBOCmNSwbuK/D3KB5/YGGJ0f1OY5BhWDevI/w=";
  };

  # One file name per line; blank lines and '#' comments are skipped.
  series = lib.filter (l: l != "" && !lib.hasPrefix "#" l) (
    map lib.trim (lib.splitString "\n" (builtins.readFile ../patches/series))
  );
  patches = map (n: ../patches + "/${n}") series;
in
assert lib.assertMsg (base == "v${version}")
  "patches/BASE is ${base}, but nix/source.nix fetches v${version}";
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
