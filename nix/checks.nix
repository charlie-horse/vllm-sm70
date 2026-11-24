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
    activeModel: extra:
    nixpkgs.lib.nixosSystem {
      modules = [
        self.nixosModules.vllm
        extra
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
    activeModel: extra:
    let
      cfg = (evalHost activeModel extra).config;
      units = lib.filterAttrs (n: _: lib.hasPrefix "vllm" n) cfg.systemd.units;
      text = lib.concatStringsSep "\n" (
        [ "toplevel: ${cfg.system.build.toplevel.drvPath}" ]
        ++ lib.mapAttrsToList (n: u: "### ${n}\n${u.text}") units
      );
    in
    pkgs.writeText "vllm-sm70-units-${activeModel}" (builtins.unsafeDiscardStringContext text);

  # The turbomind options (tuned + LUT) reach the vllm unit's environment,
  # and the defaults set no VLLM_SM70_* variable.
  tmLut = "/var/lib/vllm/tm-lut-flint-{arch}.bin";
  tmOptions = {
    services.vllm-sm70.turbomind = {
      tuneSmallShapes = true;
      gemmLutPath = tmLut;
      gemmLutStrict = true;
    };
  };
  tmUnit =
    let
      env = (evalHost "flint" tmOptions).config.systemd.services.vllm.environment;
      defaults = (evalHost "flint" { }).config.systemd.services.vllm.environment;
      want = {
        VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES = "1";
        VLLM_SM70_FP8_TUNE_SMALL_SHAPES = "1";
        VLLM_SM70_GEMM_LUT_PATH = tmLut;
        VLLM_SM70_GEMM_LUT_STRICT = "1";
      };
    in
    assert lib.assertMsg (lib.all (n: (env.${n} or null) == want.${n}) (lib.attrNames want))
      "turbomind options did not reach vllm.service: ${builtins.toJSON (lib.attrNames env)}";
    assert lib.assertMsg (
      !(lib.any (lib.hasPrefix "VLLM_SM70_") (lib.attrNames defaults))
    ) "the default flint unit sets a VLLM_SM70_* variable";
    unitsOf "flint" tmOptions;

  # inkling-2-mini: the serve flags, the V2 runner, the draft staging and no
  # LMCache reach the rendered units.
  inklingUnit =
    let
      cfg = (evalHost "inkling-2-mini" { }).config;
      svc = cfg.systemd.services.vllm;
      exec = svc.serviceConfig.ExecStart;
      m = cfg.services.vllm-sm70.models."inkling-2-mini";
      want = [
        [ "--tensor-parallel-size" "2" ]
        [ "--dtype" "half" ]
        [ "--max-model-len" (toString m.maxModelLen) ]
        [ "--max-num-seqs" (toString m.maxNumSeqs) ]
        [ "--max-num-batched-tokens" (toString m.maxNumBatchedTokens) ]
        [ "--gpu-memory-utilization" (toString m.gpuMemoryUtilization) ]
        [ "--enable-prefix-caching" "--enable-auto-tool-choice" ]
        [
          "--hf-overrides"
          (builtins.toJSON { architectures = [ "Inkling2LoopSm70ForConditionalGeneration" ]; })
        ]
        [ "--tokenizer-mode" "inkling2_sm70" ]
        [ "--reasoning-parser" "inkling2_sm70" ]
        [ "--tool-call-parser" "inkling2_sm70" ]
        [ "--block-size" "32" ]
        [ "--limit-mm-per-prompt" (builtins.toJSON { image = 2; }) ]
        [ "--enable-prompt-tokens-details" ]
        [
          "--speculative-config"
          (builtins.toJSON {
            method = "dspark";
            model = "/var/lib/vllm/models/Inkling-2-Mini-DSpark";
            num_speculative_tokens = 7;
          })
        ]
      ];
      missing = lib.filter (a: !(lib.hasInfix (lib.escapeShellArgs a) exec)) want;
    in
    assert lib.assertMsg (missing == [ ]) "inkling-2-mini ExecStart lacks: ${builtins.toJSON missing}";
    assert lib.assertMsg (
      lib.hasInfix "--no-async-scheduling" exec == !m.asyncScheduling
    ) "inkling-2-mini async scheduling does not reach ExecStart";
    assert lib.assertMsg (!(lib.hasInfix "--kv-transfer-config" exec)) "inkling-2-mini ExecStart has LMCache";
    assert lib.assertMsg (!cfg.services.lmcache-sm70.enable) "inkling-2-mini enables vllm-lmcache";
    assert lib.assertMsg (
      (svc.environment.VLLM_USE_V2_MODEL_RUNNER or null) == "1"
    ) "inkling-2-mini does not run the V2 model runner";
    assert lib.assertMsg (lib.hasInfix "vllm_sm70.models.inkling2_sm70.draft" svc.preStart)
      "inkling-2-mini does not stage its DSpark draft";
    unitsOf "inkling-2-mini" { };

  # The LMCache sdist the build itself uses (nix/lmcache-build.nix), so the
  # check tests exactly what ships. Building this check fetches the sdist
  # (~MBs); no compiling.
  lmcacheSdist = self.packages.${pkgs.stdenv.hostPlatform.system}.lmcache.sdist;
in
{
  lmcache-patches-apply = pkgs.runCommand "lmcache-patches-apply" { } ''
    tar xzf ${lmcacheSdist}
    cd lmcache-0.5.6
    for p in ${lib.concatMapStringsSep " " (p: "${p}") (import ./patches/lmcache)}; do
      echo "applying $p"
      patch -p1 --fuzz=0 --no-backup-if-mismatch --batch --forward < "$p"
    done
    ${pkgs.python3}/bin/python3 -I -c 'import ast, sys; ast.parse(open(sys.argv[1]).read())' \
      lmcache/integration/vllm/vllm_multi_process_adapter.py
    touch $out
  '';

  module-eval-flint-turbomind-lut = tmUnit;
  module-eval-inkling-2-mini = inklingUnit;

  flint-checkpoint-outpath =
    assert lib.assertMsg (ckpt.flintCheckpoint.outPath == ckpt.expectedOutPath) ''
      flintCheckpoint is ${ckpt.flintCheckpoint.outPath}, expected ${ckpt.expectedOutPath}.
      Its name or hash changed: that re-converts the 23 GB checkpoint.'';
    pkgs.writeText "flint-checkpoint-outpath" ckpt.expectedOutPath;
}
// lib.mapAttrs' (m: _: lib.nameValuePair "module-eval-${lib.replaceStrings [ "." ] [ "_" ] m}" (unitsOf m { })) {
  flint = null;
  "qwen3.6-35b-a3b" = null;
  "qwen3.8-27b" = null;
}
