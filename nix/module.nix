# vLLM inference service on 2x V100 (sm_70, TP=2), one model at a time.
#
# Host-agnostic: the host enables it and sets its own values, e.g.
#
#   services.vllm-sm70 = {
#     enable = true;
#     activeModel = "flint";
#     firewallInterfaces = [ "eno1" "tailscale0" ];
#   };
#   services.lmcache-sm70.ui.firewallInterfaces = [ "eno1" "tailscale0" ];
#
# LMCache is its own module (./lmcache.nix); this one imports it, enables it
# when the active model has `lmcache = true` and passes the chunk size.
#
# Switch models by setting `activeModel` and running nixos-rebuild switch.
# `inputs` are this flake's inputs, bound by flake.nix; packages are built
# with the host's `pkgs` (./packages-module.nix).
#
# Ported from charlie-horse/vLLM (cf9492a, nix/module.nix) with the option
# surface unchanged, so this flake is a drop-in for the host's `vllm-fork`.
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
  # Flash-V100 (K06, registered as AttentionBackendEnum.CUSTOM) for the
  # target's and the draft's fp16 head-256 layers, the backend the fork ran
  # as FLASH_ATTN_V100. Leaving it unchosen keeps VLLM_SM70_FLASH_V100=0
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

  # Environment of the Qwen DFlash2 entries: Flint's (see there). The V2
  # model runner is the separate `v2ModelRunner` field, so a host that
  # replaces `env` (e.g. to set a kill switch) keeps it.
  qwenEnv = {
    CUDA_DEVICE_ORDER = "PCI_BUS_ID";
    VLLM_SKIP_P2P_CHECK = "0";
  };

  # Built-in models (see the `models` option for the fields).
  #
  # The Qwen entries carry the serve flags measured on c4130 (Phase 3 Block F,
  # 2026-11-09): async scheduling on, V2 runner, Flash-V100 attention. The
  # fork/1Cat serve flags measured 12-27% slower per step on this stack
  # (--no-async-scheduling: rank skew), so they are an opt-in for A/B runs
  # only. A host sets them per field (each built-in field is a mkDefault):
  #
  #   services.vllm-sm70.models."qwen3.8-27b" = {
  #     asyncScheduling = false; # 1Cat: --no-async-scheduling
  #     maxNumSeqs = 4;
  #     gpuMemoryUtilization = 0.91; # 1Cat row 4; OOMs with LMCache on
  #   };
  #   services.vllm-sm70.models."qwen3.6-35b-a3b" = {
  #     asyncScheduling = false;
  #     maxNumSeqs = 4;
  #   };
  #
  # The other 1Cat values (max-model-len, block sizes) are unchanged from the
  # fork's entries below.
  builtinModels = {
    # Flint NVFP4, per the fork's scripts/serve_flint_nvfp4_v100_pcie.sh.
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
        # TODO(sm70-moe): the fork passed `--kernel-config.sm70_skinny_moe false`
        # (TurboMind NVFP4 MoE route, ~40% faster decode than the skinny route).
        # Upstream's KernelConfig rejects unknown fields (startup error), so it
        # stays out until the patch series adds the field (Mara/Kai's MoE
        # patches); then restore it here.
        "--enable-prompt-tokens-details"
        "--tool-call-parser" "glm47"
        # glm47_sm70 = the fork's glm47 parser, ported in vllm_sm70/reasoning.py.
        # Upstream's glm47 leaks Flint's <think> into content when a request
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
        # vLLM's attention block size for this model (aligned to the GDN page).
        lmcacheChunkSize = 1056;
        asyncScheduling = true;
        # Not measured above batch 1 yet (Bea's gate serve checks it starts:
        # Mamba cache blocks must be >= max-num-seqs).
        maxNumSeqs = 20;
        # DFlash candidate-head drafts need the V2 model runner ("Model Runner
        # V1 does not support: DFlash candidate-head drafts"); the wrapper
        # defaults it to 0 for Flint (D2).
        v2ModelRunner = true;
        env = qwenEnv;
        args = [
          "--tool-call-parser" "qwen3_coder"
          "--reasoning-parser" "qwen3"
          "--language-model-only"
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
        # >= 102K for the cold 100K-token TTFT exit (Phase 3).
        maxModelLen = 150000;
        # gpuMemoryUtilization: the module default 0.88. The fork's 0.91 OOMs
        # at startup next to vllm-lmcache, which holds ~1.7 GiB on GPU0 at
        # chunk 8192 (q3.8 run, 2026-11-12); 0.88 starts.
        # Must be a multiple of vLLM's block size, which --mamba-block-size sets.
        lmcacheChunkSize = 8192;
        asyncScheduling = true;
        # The DFlash2 verifier fast paths (K18, M9a, the QPN8 rerank) are M8
        # paths: one drafting request. More requests fall back to the batched
        # kernels (more total throughput, less tok/s per request).
        maxNumSeqs = 8;
        # DFlash candidate-head drafts need the V2 model runner ("Model Runner
        # V1 does not support: DFlash candidate-head drafts"); the wrapper
        # defaults it to 0 for Flint (D2).
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
    "--max-num-batched-tokens" "8192"
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
in
{
  imports = [
    (import ./packages-module.nix { inherit inputs; })
    (import ./lmcache.nix { inherit inputs; })
    # The options lived under c4130.vllm while this module was host config.
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
    # store `path`; optional maxModelLen / gpuMemoryUtilization / maxNumSeqs
    # overrides; `lmcache` (default true) and its `lmcacheChunkSize`; `package`
    # (the wrapped vllm to run, default this flake's `vllm`); extra service `env`
    # (a host value replaces the built-in one as a whole); `v2ModelRunner`
    # (default false: sets VLLM_USE_V2_MODEL_RUNNER=1, kept when `env` is
    # replaced; an explicit `env` value still wins); `asyncScheduling`
    # (default false); and model-specific `vllm serve` `args`.
    models = lib.mkOption {
      type = lib.types.attrsOf (lib.types.attrsOf lib.types.raw);
      description = ''
        Models vllm.service can serve. Built-in: flint, qwen3.6-35b-a3b and
        qwen3.8-27b. Add entries, or override single fields of a built-in one.
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
        // (model.env or { });
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
