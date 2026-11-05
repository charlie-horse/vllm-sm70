# LMCache multiprocess server (KV cache tiers below the GPU: L1 in pinned
# CPU RAM, L2 on local disk), its dashboard and the L2 pruner.
#
# Usable on its own, but normally driven by ./module.nix, which enables it
# when the active model has `lmcache = true`, sets the chunk size and runs it
# as the vllm user (CUDA IPC needs the same user). The unit names keep the
# `vllm-` prefix they had when this was part of the vllm module.
{ inputs }:
{
  config,
  lib,
  pkgs,
  utils,
  vllmSm70Packages,
  ...
}:
let
  cfg = config.services.lmcache-sm70;

  l2Dir = "/var/lib/lmcache/l2";

  lmcacheArgs = [
    "server"
    "--host" "127.0.0.1"
    "--port" (toString cfg.port)
    "--http-host" "127.0.0.1"
    # Already the default, but vllm-lmcache-ui points at this port.
    "--http-port" (toString cfg.httpPort)
    # Metrics are on (Prometheus pull); the port is LMCache's default, made
    # explicit here. --prometheus-host comes from patch 0002.
    "--prometheus-port" "9090"
    "--prometheus-host" cfg.metricsHost
    # One object group per attention type, required for hybrid GDN models.
    "--separate-object-groups"
    # Must be a multiple of the served model's vLLM block size.
    "--chunk-size" (toString cfg.chunkSize)
    "--l1-size-gb" (toString cfg.l1SizeGb)
    "--eviction-policy" "LRU"
    "--l2-adapter"
    (builtins.toJSON {
      type = "fs_native";
      base_path = l2Dir;
      num_workers = 8;
      max_capacity_gb = cfg.l2SizeGb;
    })
  ]
  ++ cfg.extraArgs;

  # LMCache's dashboard is a read-only proxy in front of the MP server's admin
  # API: `--nodes` is both the node list it renders and the allowlist its
  # /proxy/{host}/{port}/... route checks against.
  uiArgs = [
    "--host" "0.0.0.0"
    "--port" (toString cfg.ui.port)
    "--nodes"
    (builtins.toJSON [
      {
        name = "lmcache_server";
        host = "127.0.0.1";
        port = toString cfg.httpPort;
      }
    ])
    "--log-level" "warning"
  ];
in
{
  imports = [
    (import ./packages-module.nix { inherit inputs; })
    (lib.mkRenamedOptionModule
      [ "c4130" "vllm" "kvTiers" "l1SizeGb" ]
      [ "services" "lmcache-sm70" "l1SizeGb" ]
    )
    (lib.mkRenamedOptionModule
      [ "c4130" "vllm" "kvTiers" "l2SizeGb" ]
      [ "services" "lmcache-sm70" "l2SizeGb" ]
    )
  ];

  options.services.lmcache-sm70 = {
    enable = lib.mkEnableOption "the LMCache MP server (CUDA 12, sm_70 build)";

    package = lib.mkOption {
      type = lib.types.package;
      default = vllmSm70Packages.vllm;
      defaultText = lib.literalMD "this flake's `vllm` wrapper (provides `bin/lmcache`)";
      description = "Package providing `bin/lmcache`.";
    };

    uiPackage = lib.mkOption {
      type = lib.types.package;
      default = vllmSm70Packages.lmcacheUi;
      defaultText = lib.literalMD "a wrapped `python` with lmcache importable";
      description = "Python that runs `lmcache.lmcache_frontend.app`.";
    };

    user = lib.mkOption {
      type = lib.types.str;
      default = "lmcache";
      description = ''
        User (and group) the server runs as. Must be the vLLM user: KV moves
        over CUDA IPC. The user is created only when left at "lmcache".
      '';
    };

    homeStateDirectory = lib.mkOption {
      type = lib.types.str;
      default = "lmcache";
      description = "StateDirectory (under /var/lib) holding the user's HOME.";
    };

    environment = lib.mkOption {
      type = lib.types.attrsOf lib.types.str;
      default = {
        HOME = "/var/lib/${cfg.homeStateDirectory}";
      };
      defaultText = lib.literalExpression ''{ HOME = "/var/lib/''${homeStateDirectory}"; }'';
      description = "Environment of the server and the dashboard.";
    };

    chunkSize = lib.mkOption {
      type = lib.types.ints.positive;
      description = "Chunk size in tokens; a multiple of the model's vLLM block size.";
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 5555;
      description = "ZMQ port vLLM's LMCacheMPConnector talks to (loopback).";
    };

    httpPort = lib.mkOption {
      type = lib.types.port;
      default = 8080;
      description = "Admin API port (loopback only; no authentication).";
    };

    metricsHost = lib.mkOption {
      type = lib.types.str;
      default = "127.0.0.1";
      example = "0.0.0.0";
      description = ''
        Address LMCache's Prometheus /metrics endpoint (port 9090, in
        lmcacheArgs) binds to. Loopback by
        default; set it only if a scraper on another host needs it, and open
        the port on that interface yourself.
      '';
    };

    # Lower both to run LMCache alongside something else that wants the RAM or
    # the disk; L1 is pinned and allocated in full at startup.
    l1SizeGb = lib.mkOption {
      type = lib.types.ints.positive;
      default = 64;
      description = "L1 size in GB (pinned CPU RAM).";
    };

    l2SizeGb = lib.mkOption {
      type = lib.types.ints.positive;
      default = 180;
      description = "L2 cap in GB (local disk, enforced by vllm-lmcache-l2-prune).";
    };

    extraArgs = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      example = [
        "--worker-registration-grace-seconds"
        "604800"
      ];
      description = ''
        Extra `lmcache server` arguments, appended after the module's own
        (a later flag overrides an earlier one).
      '';
    };

    ui = {
      enable = lib.mkOption {
        type = lib.types.bool;
        default = true;
        description = "Whether to run LMCache's dashboard (vllm-lmcache-ui).";
      };
      port = lib.mkOption {
        type = lib.types.port;
        default = 8081;
        description = "Dashboard port (no authentication).";
      };
      firewallInterfaces = lib.mkOption {
        type = lib.types.listOf lib.types.str;
        default = [ ];
        example = [ "tailscale0" ];
        description = "Interfaces on which the dashboard port is opened.";
      };
    };
  };

  config = lib.mkIf cfg.enable (
    lib.mkMerge [
      (lib.mkIf (cfg.user == "lmcache") {
        users.users.lmcache = {
          isSystemUser = true;
          group = "lmcache";
          extraGroups = [ "video" ];
          home = "/var/lib/${cfg.homeStateDirectory}";
        };
        users.groups.lmcache = { };
      })
      {
        # Holds L1/L2 and copies KV to/from vLLM over CUDA IPC, so it runs as
        # vLLM's user with GPU access.
        systemd.services.vllm-lmcache = {
          description = "LMCache KV cache server (L1 ${toString cfg.l1SizeGb} GB RAM, L2 ${toString cfg.l2SizeGb} GB disk)";
          after = [ "nvidia-persistenced.service" ];
          inherit (cfg) environment;
          serviceConfig = {
            User = cfg.user;
            Group = cfg.user;
            SupplementaryGroups = [ "video" ];
            StateDirectory = lib.unique [
              cfg.homeStateDirectory
              "lmcache"
              "lmcache/l2"
            ];
            # systemd quoting (escapes `%` and `$` too), since extraArgs is free-form.
            ExecStart = "${cfg.package}/bin/lmcache ${utils.escapeSystemdExecArgs lmcacheArgs}";
            Restart = "on-failure";
            RestartSec = 10;
            LimitMEMLOCK = "infinity";
          };
        };

        # LMCache doesn't evict L2 on its own (that needs its fleet
        # coordinator). Above 90% of the cap, delete least recently accessed
        # files down to 80%; fs_native checks files on every lookup, so pruned
        # chunks are just misses.
        systemd.services.vllm-lmcache-l2-prune = {
          description = "Keep LMCache L2 under ${toString cfg.l2SizeGb} GB";
          path = [ pkgs.coreutils pkgs.findutils pkgs.gawk ];
          serviceConfig = {
            Type = "oneshot";
            User = cfg.user;
            Group = cfg.user;
          };
          script = ''
            dir=${l2Dir}
            [ -d "$dir" ] || exit 0
            cap=$(( ${toString cfg.l2SizeGb} * 1024 * 1024 * 1024 ))
            used=$(du -sb "$dir" | cut -f1)
            [ "$used" -gt $(( cap * 9 / 10 )) ] || exit 0
            excess=$(( used - cap * 8 / 10 ))
            find "$dir" -type f -printf '%A@ %s %p\n' | sort -n \
              | awk -v excess="$excess" 'freed >= excess { exit } { print $3; freed += $2 }' \
              | xargs -r -d '\n' rm -f --
          '';
        };

        systemd.timers.vllm-lmcache-l2-prune = {
          wantedBy = [ "timers.target" ];
          timerConfig = {
            OnBootSec = "2min";
            OnUnitActiveSec = "1min";
          };
        };
      }
      (lib.mkIf cfg.ui.enable {
        # LMCache's own dashboard (lmcache.lmcache_frontend). It only talks
        # HTTP to the admin API, so it needs no GPU and no cache state.
        systemd.services.vllm-lmcache-ui = {
          description = "LMCache dashboard (port ${toString cfg.ui.port})";
          wantedBy = [ "multi-user.target" ];
          requires = [ "vllm-lmcache.service" ];
          after = [ "vllm-lmcache.service" ];
          inherit (cfg) environment;
          serviceConfig = {
            User = cfg.user;
            Group = cfg.user;
            # Importing lmcache probes for a compute backend.
            SupplementaryGroups = [ "video" ];
            StateDirectory = cfg.homeStateDirectory;
            ExecStart = "${cfg.uiPackage}/bin/python -m lmcache.lmcache_frontend.app ${lib.escapeShellArgs uiArgs}";
            Restart = "on-failure";
            RestartSec = 10;
          };
        };

        # After other modules' ports on the same interface (stable rule order).
        networking.firewall.interfaces = lib.genAttrs cfg.ui.firewallInterfaces (_: {
          allowedTCPPorts = lib.mkAfter [ cfg.ui.port ];
        });
      })
    ]
  );
}
