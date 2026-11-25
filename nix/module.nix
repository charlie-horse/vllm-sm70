# vLLM inference service on two Volta GPUs (sm_70, TP=2), one model at a time.
# Host-agnostic: the host enables it and sets its own values, e.g.
#
#   services.vllm-sm70 = {
#     enable = true;
#     activeModel = "flint";
#     firewallInterfaces = [ "eth0" ];
#   };
#   services.lmcache-sm70.ui.firewallInterfaces = [ "eth0" ];
# LMCache is its own module (./lmcache.nix); this one imports it, enables it
# when the active model has `lmcache = true` and passes the chunk size.
#
# Switch models by setting `activeModel` and running nixos-rebuild switch.
# `inputs` are this flake's inputs, bound by flake.nix; packages are built
# with the host's `pkgs` (./packages-module.nix).
{ inputs }:
{
  config,
  lib,
  vllmSm70Packages,
  ...
}:
let
  cfg = config.services.vllm-sm70;
  lcfg = config.services.lmcache-sm70;

  inherit (vllmSm70Packages)
    vllm
    flintCheckpoint
    ;

  inherit (cfg) port;
  # systemd StateDirectory of the vllm user (the units say "vllm").
  stateDir = "/var/lib/vllm";
  modelsDir = "${stateDir}/models";

  # DFlash2 speculative decoding with a local draft checkpoint (upstream
  # `method = "dflash"`; draft KV follows the target's FP16 KV).
  # No `attention_backend`: with none chosen, the sm70 platform picks
  # Flash-V100 (registered as AttentionBackendEnum.CUSTOM) for the target's
  # and the draft's fp16 head-256 layers. Leaving it unchosen keeps VLLM_SM70_FLASH_V100=0
  # (fall back to TRITON_ATTN) working; an explicit CUSTOM would bypass it.
  dflashConfig =
    draftDir:
    builtins.toJSON {
      method = "dflash";
      model = draftDir;
      num_speculative_tokens = 7;
      kv_cache_dtype = "auto";
      draft_sample_method = "probabilistic";
      enforce_eager = false;
    };

  # Environment of the Qwen DFlash2 and Inkling 2 entries: Flint's (see
  # there). The V2 model runner is the separate `v2ModelRunner` field, so a
  # host that replaces `env` (e.g. to set a kill switch) keeps it.
  qwenEnv = {
    CUDA_DEVICE_ORDER = "PCI_BUS_ID";
    VLLM_SKIP_P2P_CHECK = "0";
  };

  # Inkling 2 Mini serve settings for a checkpoint `repo` at `revision`,
  # stored in `modelsDir/<dir>` with its DSpark draft in `<dir>-DSpark`.
  inkling =
    { repo, revision, dir }:
    let
      targetDir = "${modelsDir}/${dir}";
      draftDir = "${modelsDir}/${dir}-DSpark";
    in
    {
      downloads = [
        {
          inherit repo revision;
          dir = targetDir;
        }
      ];
      maxModelLen = 262144;
      maxNumSeqs = 8;
      maxNumBatchedTokens = 512;
      gpuMemoryUtilization = 0.88;
      asyncScheduling = true;
      lmcache = true;
      lmcacheChunkSize = 256;
      # DSpark drafts need the V2 model runner.
      v2ModelRunner = true;
      env = qwenEnv // { VLLM_USE_BREAKABLE_CUDAGRAPH = "1"; };
      preStart = ''
        python -m vllm_sm70.models.inkling2_sm70.draft ${targetDir} ${draftDir}
      '';
      args = [
        "--hf-overrides" (builtins.toJSON { architectures = [ "Inkling2LoopSm70ForConditionalGeneration" ]; })
        "--tokenizer-mode" "inkling2_sm70"
        "--reasoning-parser" "inkling2_sm70"
        "--tool-call-parser" "inkling2_sm70"
        "--block-size" "32"
        "--limit-mm-per-prompt" (builtins.toJSON { image = 2; })
        "--enable-prompt-tokens-details"
        "--speculative-config"
        (builtins.toJSON {
          method = "dspark";
          model = draftDir;
          num_speculative_tokens = 7;
        })
      ];
    };

  # Built-in models (see the `models` option for the fields). The Qwen entries
  # run async scheduling, the V2 runner and Flash-V100 attention. Each
  # built-in field is a mkDefault, so a host overrides one field, e.g.
  #   services.vllm-sm70.models."qwen3.8-27b".asyncScheduling = false;
  builtinModels = {
    # Flint NVFP4.
    # Weights: the fixed-output checkpoint converted from the GGUF
    # (./checkpoint.nix); its store path never changes with this flake.
    # LMCache's MP connector supports the hybrid (sliding-window + full
    # attention) KV cache manager; its ~2.2 GiB per GPU fits under the 0.90
    # utilization below.
    flint = {
      path = flintCheckpoint;
      lmcache = true;
      lmcacheChunkSize = 256;
      asyncScheduling = true;
      maxNumSeqs = 2;
      gpuMemoryUtilization = 0.90;
      env = {
        CUDA_DEVICE_ORDER = "PCI_BUS_ID";
        # Run the real CUDA IPC peer-write test; PCIe pairs that report P2P
        # but drop peer stores otherwise hang at the first decode.
        VLLM_SKIP_P2P_CHECK = "0";
        # No PYTORCH_ALLOC_CONF=expandable_segments: its CUDA IPC goes through
        # pidfd_getfd, which kernel.yama.ptrace_scope=1 denies to the LMCache
        # server (not an ancestor of vLLM), so register_kv_caches times out.
      };
      args = [
        "--kv-cache-dtype" "auto"
        # No `--kernel-config.sm70_skinny_moe`: upstream's KernelConfig
        # rejects unknown fields at startup.
        "--enable-prompt-tokens-details"
        "--tool-call-parser" "glm47"
        # glm47_sm70 is vllm_sm70/reasoning.py's glm47 parser. Upstream's glm47 leaks Flint's <think> into content when a request
        # sends enable_thinking=false. "--reasoning-parser glm47" restores upstream.
        "--reasoning-parser-plugin" "vllm_sm70.reasoning"
        "--reasoning-parser" "glm47_sm70"
        "--exclude-tools-when-tool-choice-none"
        "--override-generation-config" ''{"temperature":1.0,"top_p":0.95,"top_k":20}''
      ];
    };

    "qwen3.6-35b-a3b" =
      let
        draftDir = "${modelsDir}/Qwen3.6-35B-A3B-DFlash2";
      in
      {
        downloads = [
          {
            repo = "Qwen/Qwen3.6-35B-A3B-FP8";
            revision = "95a723d08a9490559dae23d0cff1d9466213d989";
            dir = "${modelsDir}/Qwen3.6-35B-A3B-FP8";
          }
          {
            repo = "incoai/Qwen3.6-35B-A3B-DFlash2";
            revision = "51ef7b6923ad6c14cb1bb41c37a9041446496ab6";
            dir = draftDir;
          }
        ];
        # Must be a multiple of vLLM's "kv lcm block sizes". The block is
        # pinned below so it doesn't depend on the attention backend (vLLM only
        # ever raises a pinned block). With these flags and the DFlash draft,
        # Flash-V100 picks 1104 on its own, with KV groups of 1104 and 2208
        # (LCM 2208). The VLLM_SM70_FLASH_V100=0 start check establishes
        # TRITON_ATTN's LCM; if a start logs an LCM N that doesn't divide this
        # chunk, set the chunk to N.
        lmcacheChunkSize = 2208;
        asyncScheduling = true;
        # Mamba cache blocks must be >= max-num-seqs.
        maxNumSeqs = 20;
        # DFlash candidate-head drafts need the V2 model runner ("Model Runner
        # V1 does not support: DFlash candidate-head drafts"); the wrapper
        # defaults it to 0 for Flint.
        v2ModelRunner = true;
        env = qwenEnv;
        args = [
          "--tool-call-parser" "qwen3_coder"
          "--reasoning-parser" "qwen3"
          "--language-model-only"
          # Kept for LMCache (see lmcacheChunkSize).
          "--block-size" "1104"
          "--mamba-block-size" "1104"
          "--mamba-cache-mode" "align"
          "--speculative-config" (dflashConfig draftDir)
        ];
      };

    "qwen3.8-27b" =
      let
        draftDir = "${modelsDir}/Qwen3.8-27B-DFlash2";
      in
      {
        downloads = [
          {
            repo = "Qwen/Qwen3.8-27B-FP8";
            revision = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a";
            dir = "${modelsDir}/Qwen3.8-27B-FP8";
          }
          {
            repo = "incoai/Qwen3.8-27B-DFlash2";
            revision = "dedf8df68adfb1afeaf7b7480c0a0243108177b4";
            dir = draftDir;
          }
        ];
        # >= 102K, so a 100K-token prompt fits.
        maxModelLen = 150000;
        # gpuMemoryUtilization: the module default 0.88. 0.91 runs out of
        # memory at startup next to vllm-lmcache (~1.7 GiB on GPU0 at chunk 8192).
        # Must be a multiple of vLLM's block size, which --mamba-block-size sets.
        lmcacheChunkSize = 8192;
        asyncScheduling = true;
        # The DFlash2 verifier fast paths (incl. the QPN8 rerank) take one
        # drafting request (M=8 rows). More requests fall back to the batched
        # kernels (more total throughput, slower per request).
        maxNumSeqs = 8;
        # DFlash candidate-head drafts need the V2 model runner ("Model Runner
        # V1 does not support: DFlash candidate-head drafts"); the wrapper
        # defaults it to 0 for Flint.
        v2ModelRunner = true;
        env = qwenEnv;
        args = [
          "--tool-call-parser" "qwen3_coder"
          "--reasoning-parser" "qwen3"
          "--language-model-only"
          # Kept for LMCache: the 8192 chunk must be a multiple of the block size.
          "--block-size" "2048"
          "--mamba-block-size" "8192"
          "--mamba-cache-mode" "align"
          "--seed" "0"
          "--trust-remote-code"
          "--speculative-config" (dflashConfig draftDir)
        ];
      };

    # Inkling 2 Mini (plugin arch, docs/nix.md#inkling-2) with its DSpark
    # draft, staged from the target's mtp.safetensors at every start.
    "inkling-2-mini" = inkling {
      repo = "thinkingmachines/Inkling-2-Mini";
      revision = "ed40e7da625f594fc9c4990f5f66be24873c22d7";
      dir = "Inkling-2-Mini";
    };

    # Inkling 2 Mini with a rank-2 edit to the attention output projections
    # of layers 4 to 23 that removes refusals. Same serve settings.
    "inkling-2-mini-uncensored" = inkling {
      repo = "OpenUncensored/Inkling-2-Mini-Uncensored";
      revision = "4e5f06d72b5a70c9dcd5e8046aea576eb3ee8487";
      dir = "Inkling-2-Mini-Uncensored";
    };
  };

  active = cfg.activeModel;
  model = cfg.models.${active};
  marker = d: "${d.dir}/.complete-${d.revision}";
  downloads = model.downloads or [ ];
  modelPath = model.path or (lib.head downloads).dir;
  # One package serves every model; `package` stays as a per-model override.
  package = model.package or vllm;
  lmcache = model.lmcache or true;

  serveArgs = [
    modelPath
    "--served-model-name" active
    "--host" "0.0.0.0"
    "--port" (toString port)
    "--tensor-parallel-size" "2"
    "--dtype" "half"
    "--max-model-len" (toString (model.maxModelLen or 262144))
    "--max-num-seqs" (toString (model.maxNumSeqs or 4))
    "--max-num-batched-tokens" (toString (model.maxNumBatchedTokens or 8192))
    # Leaves ~2.2 GiB per GPU outside vLLM's budget for the LMCache server's
    # CUDA context and its KV transfer staging during stores.
    "--gpu-memory-utilization" (toString (model.gpuMemoryUtilization or 0.88))
  ]
  ++ lib.optional (!(model.asyncScheduling or false)) "--no-async-scheduling"
  ++ [
    "--enable-prefix-caching"
    "--enable-auto-tool-choice"
  ]
  ++ lib.optionals lmcache [
    # LMCache's own MP connector (lmcache.integration.vllm), not the
    # LMCacheMPConnectorUpstream that vLLM v0.33.0 bundles: hybrid KV cache
    # models need SupportsHMA, which only LMCache's class has. Never set
    # LMCACHE_USE_UPSTREAM_MP. Failed loads fall back to recomputing the
    # prefix (upstream's default policy is "fail").
    "--kv-transfer-config"
    (builtins.toJSON {
      kv_connector = "LMCacheMPConnector";
      kv_connector_module_path = "lmcache.integration.vllm.lmcache_mp_connector";
      kv_role = "kv_both";
      kv_load_failure_policy = "recompute";
      kv_connector_extra_config = {
        "lmcache.mp.host" = "tcp://127.0.0.1";
        "lmcache.mp.port" = lcfg.port;
      };
    })
  ]
  ++ model.args;

  env = {
    HOME = stateDir;
    HF_HOME = "${stateDir}/hf";
    XDG_CACHE_HOME = "${stateDir}/cache";
    VLLM_CACHE_ROOT = "${stateDir}/cache/vllm";
    TRITON_CACHE_DIR = "${stateDir}/cache/triton";
    TILELANG_CACHE_DIR = "${stateDir}/cache/tilelang";
    VLLM_NO_USAGE_STATS = "1";
    DO_NOT_TRACK = "1";
  };

  # TurboMind small-shape autotune + persisted GEMM LUT (see the `turbomind`
  # options). Nothing is set while the options are at their defaults, so the
  # C++ defaults apply.
  tm = cfg.turbomind;
  tmEnv =
    lib.optionalAttrs tm.tuneSmallShapes {
      VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES = "1";
      VLLM_SM70_FP8_TUNE_SMALL_SHAPES = "1";
    }
    // lib.optionalAttrs (tm.gemmLutPath != null) { VLLM_SM70_GEMM_LUT_PATH = tm.gemmLutPath; }
    // lib.optionalAttrs (tm.gemmLutExportPath != null) {
      VLLM_SM70_GEMM_LUT_EXPORT_PATH = tm.gemmLutExportPath;
    }
    // lib.optionalAttrs tm.gemmLutStrict { VLLM_SM70_GEMM_LUT_STRICT = "1"; };
in
{
  imports = [
    (import ./packages-module.nix { inherit inputs; })
    (import ./lmcache.nix { inherit inputs; })
    # Alias for the old option path.
    (lib.mkRenamedOptionModule
      [ "c4130" "vllm" "activeModel" ]
      [ "services" "vllm-sm70" "activeModel" ]
    )
  ];

  options.services.vllm-sm70 = {
    enable = lib.mkEnableOption "the vLLM server on two V100s (TP=2)";

    activeModel = lib.mkOption {
      type = lib.types.enum (lib.attrNames cfg.models);
      default = "flint";
      description = "Model served by vllm.service. Only one fits on the GPUs at a time.";
    };

    # Served model name -> weights, either pinned HF `downloads` (a list of
    # { repo, revision, dir }; the first is the served checkpoint) or a Nix
    # store `path`; optional maxModelLen / gpuMemoryUtilization / maxNumSeqs /
    # maxNumBatchedTokens overrides (--max-num-batched-tokens, default 8192);
    # `lmcache` (default true) and its `lmcacheChunkSize`; `package`
    # (the wrapped vllm to run, default this flake's `vllm`); extra service `env`
    # (a host value replaces the built-in one as a whole); `v2ModelRunner`
    # (default false: sets VLLM_USE_V2_MODEL_RUNNER=1, kept when `env` is
    # replaced; an explicit `env` value still wins); `asyncScheduling`
    # (default false); `preStart` (shell commands run as the vllm user before
    # every start, with the model's `package` first on PATH); and
    # model-specific `vllm serve` `args`.
    models = lib.mkOption {
      type = lib.types.attrsOf (lib.types.attrsOf lib.types.raw);
      description = ''
        Models vllm.service can serve. Built-in: flint, qwen3.6-35b-a3b,
        qwen3.8-27b and inkling-2-mini. Add entries, or override single
        fields of a built-in one.
      '';
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 8000;
      description = "Port of the OpenAI-compatible API (no authentication).";
    };

    firewallInterfaces = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      example = [ "eno1" "tailscale0" ];
      description = "Interfaces on which the API port is opened (no authentication).";
    };

    # TurboMind small-shape tuning and the GEMM LUT; export and use the LUT
    # as in docs/nix.md, "TurboMind GEMM LUT".
    turbomind = {
      tuneSmallShapes = lib.mkOption {
        type = lib.types.bool;
        default = false;
        description = ''
          Time TurboMind NVFP4 and FP8 kernels for small shapes (M <= 64) at
          startup, as the fork did (VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES=1,
          VLLM_SM70_FP8_TUNE_SMALL_SHAPES=1). Without `gemmLutPath` each
          start tunes on its own and may pick different kernels than the
          last one (not bitwise reproducible across restarts).
        '';
      };

      gemmLutPath = lib.mkOption {
        type = lib.types.nullOr lib.types.str;
        default = null;
        example = "/var/lib/vllm/tm-lut-flint-{arch}.bin";
        description = ''
          TurboMind GEMM LUT to import on every TP rank
          (VLLM_SM70_GEMM_LUT_PATH; "{arch}", "{device}" and "{rank}" are
          substituted). Calls its manifest covers are not timed. A LUT
          written by another _sm70_C build, GPU or knob setting is refused
          with a WARNING and the warmup tunes as without it. Read only when
          `tuneSmallShapes` is on. Re-export after a change to the build,
          model, TP size, max-num-seqs or capture sizes.
        '';
      };

      gemmLutExportPath = lib.mkOption {
        type = lib.types.nullOr lib.types.str;
        default = null;
        example = "/var/lib/vllm/tm-lut-flint-{arch}.bin";
        description = ''
          Write the tuned TurboMind dispatch cache here after the pre-capture
          warmup (VLLM_SM70_GEMM_LUT_EXPORT_PATH; atomic, plus
          `<path>.plan.json`). For the one-off export start; unset it again
          afterwards. The directory must be writable by the vllm user.
        '';
      };

      gemmLutStrict = lib.mkOption {
        type = lib.types.bool;
        default = false;
        description = ''
          Fail startup instead of re-tuning when the LUT is missing, stale
          or mismatched (VLLM_SM70_GEMM_LUT_STRICT=1). For gate runs: in
          production a rebuild would otherwise stop vllm until the LUT is
          re-exported.
        '';
      };
    };
  };

  config = lib.mkIf cfg.enable {
    # mkDefault per field, so a host can override one field of a built-in
    # model (e.g. models.flint.maxNumSeqs) without replacing the whole entry.
    services.vllm-sm70.models = lib.mapAttrs (
      _: lib.mapAttrs (_: lib.mkDefault)
    ) builtinModels;

    # The model's `lmcache` field is the switch; set it there, not here.
    services.lmcache-sm70 = {
      enable = lmcache;
      user = "vllm";
      homeStateDirectory = "vllm";
      environment = env;
      chunkSize = lib.mkIf lmcache model.lmcacheChunkSize;
    };

    environment.systemPackages = [ vllm ];

    assertions = [
      {
        assertion = tm.gemmLutStrict -> tm.gemmLutPath != null;
        message = "services.vllm-sm70.turbomind.gemmLutStrict needs gemmLutPath.";
      }
    ];
    # With tuning off every TurboMind call takes the default kernel, and the
    # warmup neither reads nor writes a LUT.
    warnings =
      lib.optional
        (!tm.tuneSmallShapes && (tm.gemmLutPath != null || tm.gemmLutExportPath != null))
        "services.vllm-sm70.turbomind: gemmLutPath/gemmLutExportPath do nothing while tuneSmallShapes is false.";

    hardware.nvidia.nvidiaPersistenced = true;

    users.users.vllm = {
      isSystemUser = true;
      group = "vllm";
      extraGroups = [ "video" ];
      home = stateDir;
    };
    users.groups.vllm = { };

    systemd.services.vllm-model-fetch = {
      description =
        if downloads == [ ] then "Check Nix-store weights for ${active}" else "Download weights for ${active}";
      wants = [ "network-online.target" ];
      after = [ "network-online.target" ];
      environment = env;
      # Runs only while some pinned revision is missing (triggering "|" conditions
      # are OR-ed), so boots with everything on disk don't depend on Hugging Face.
      # A model served from the Nix store has no downloads: its weights are in
      # the system closure, so the unit only checks they are there.
      unitConfig.ConditionPathExists =
        if downloads == [ ] then [ "!${modelPath}" ] else map (d: "|!${marker d}") downloads;
      script =
        if downloads == [ ] then
          ''
            echo "${modelPath} is missing" >&2
            exit 1
          ''
        else
          lib.concatMapStrings (d: ''
            if [ ! -e ${marker d} ]; then
              ${vllm}/bin/hf download ${d.repo} --revision ${d.revision} --local-dir ${d.dir}
              touch ${marker d}
            fi
          '') downloads;
      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        User = "vllm";
        Group = "vllm";
        StateDirectory = "vllm";
        TimeoutStartSec = "6h";
      };
    };

    systemd.services.vllm = {
      description = "vLLM OpenAI-compatible server (${active})";
      wantedBy = [ "multi-user.target" ];
      requires = [ "vllm-model-fetch.service" ] ++ lib.optional lmcache "vllm-lmcache.service";
      after = [
        "vllm-model-fetch.service"
        "nvidia-persistenced.service"
      ] ++ lib.optional lmcache "vllm-lmcache.service";
      environment =
        env
        // { HF_HUB_OFFLINE = "1"; }
        // lib.optionalAttrs (model.v2ModelRunner or false) { VLLM_USE_V2_MODEL_RUNNER = "1"; }
        // tmEnv
        // (model.env or { });
      path = lib.mkIf (model ? preStart) [ package ];
      preStart = lib.mkIf (model ? preStart) model.preStart;
      serviceConfig = {
        User = "vllm";
        Group = "vllm";
        SupplementaryGroups = [ "video" ];
        StateDirectory = "vllm";
        ExecStart = "${package}/bin/vllm serve ${lib.escapeShellArgs serveArgs}";
        Restart = "on-failure";
        RestartSec = 30;
        TimeoutStopSec = 60;
        LimitMEMLOCK = "infinity";
      };
    };

    # No authentication; the host relies on network policy.
    networking.firewall.interfaces = lib.genAttrs cfg.firewallInterfaces (_: {
      allowedTCPPorts = [ port ];
    });
  };
}
