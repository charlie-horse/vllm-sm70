# Eval-only checks (`nix flake check --no-build` covers them; building them is
# cheap too: nothing here depends on the vLLM build or the 23 GB checkpoint).
{
  self,
  nixpkgs,
  pkgs,
}:
let
  inherit (pkgs) lib;
  ckpt = import ./checkpoint.nix { inputs = self.inputs; };

  # A minimal NixOS system importing nixosModules.vllm, one per built-in model.
  evalHost =
    activeModel:
    nixpkgs.lib.nixosSystem {
      modules = [
        self.nixosModules.vllm
        {
          nixpkgs.hostPlatform = "x86_64-linux";
          nixpkgs.config.allowUnfree = true;
          boot.loader.grub.enable = false;
          fileSystems."/" = {
            device = "none";
            fsType = "tmpfs";
          };
          system.stateVersion = "26.05";
          services.vllm-sm70 = {
            enable = true;
            inherit activeModel;
            firewallInterfaces = [ "eno1" ];
          };
          services.lmcache-sm70.ui.firewallInterfaces = [ "eno1" ];
          # Exercises extraArgs (appended to the server's ExecStart).
          services.lmcache-sm70.extraArgs = [
            "--worker-registration-grace-seconds"
            "7200"
          ];
        }
      ];
    };

  # The rendered units, without their store-path context, so building the
  # check never builds vLLM or the checkpoint. Evaluating it still forces
  # every ExecStart, environment and the whole toplevel.
  unitsOf =
    activeModel:
    let
      cfg = (evalHost activeModel).config;
      units = lib.filterAttrs (n: _: lib.hasPrefix "vllm" n) cfg.systemd.units;
      text = lib.concatStringsSep "\n" (
        [ "toplevel: ${cfg.system.build.toplevel.drvPath}" ]
        ++ lib.mapAttrsToList (n: u: "### ${n}\n${u.text}") units
      );
    in
    pkgs.writeText "vllm-sm70-units-${activeModel}" (builtins.unsafeDiscardStringContext text);

  # The LMCache sdist the build uses (docs/pins.md) with our patches applied.
  # Building this check fetches the sdist (~MBs); no compiling.
  lmcacheSdist = pkgs.fetchurl {
    url = "https://files.pythonhosted.org/packages/84/25/3ccd46844c94e08897aa8d72716a538c07e4583410d84dbc6cd12cb60dc1/lmcache-0.5.6.tar.gz";
    sha256 = "d6951e60ce8818a76aa42d0866a4ba85f1c95cd919355fff30ff24dcc3243f1e";
  };
in
{
  lmcache-patches-apply = pkgs.runCommand "lmcache-patches-apply" { } ''
    tar xzf ${lmcacheSdist}
    cd lmcache-0.5.6
    for p in ${lib.concatMapStringsSep " " (p: "${p}") (import ./patches/lmcache)}; do
      echo "applying $p"
      patch -p1 --batch --forward < "$p"
    done
    ${pkgs.python3}/bin/python3 -I -c 'import ast, sys; ast.parse(open(sys.argv[1]).read())' \
      lmcache/integration/vllm/vllm_multi_process_adapter.py
    touch $out
  '';

  flint-checkpoint-outpath =
    assert lib.assertMsg (ckpt.flintCheckpoint.outPath == ckpt.expectedOutPath) ''
      flintCheckpoint is ${ckpt.flintCheckpoint.outPath}, expected ${ckpt.expectedOutPath}.
      Its name or hash changed: that re-converts the 23 GB checkpoint.'';
    pkgs.writeText "flint-checkpoint-outpath" ckpt.expectedOutPath;
}
// lib.mapAttrs' (m: _: lib.nameValuePair "module-eval-${lib.replaceStrings [ "." ] [ "_" ] m}" (unitsOf m)) {
  flint = null;
  "qwen3.6-35b-a3b" = null;
  "qwen3.8-27b" = null;
}
