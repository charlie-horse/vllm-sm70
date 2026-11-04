# Patches for the LMCache 0.5.6 sdist (docs/pins.md), in order. The LMCache
# build in ./vllm.nix applies them:
#   lmcache = prev.lmcache.overrideAttrs (old: {
#     patches = (old.patches or [ ]) ++ import ./patches/lmcache;
#     patchFlags = [ "-p1" "--fuzz=0" "--no-backup-if-mismatch" ];
#   });
# (paths relative to nix/: this is nix/patches/lmcache, not the top-level
# patches/ series, which is vLLM-only)
# `checks.<system>.lmcache-patches-apply` checks they apply to the pinned sdist.
[
  # Start the MP worker's heartbeat at KV-cache registration, so a worker that
  # serves no full-chunk request within the registration grace (1 h) is not
  # reaped while alive. docs/upstream/lmcache-heartbeat-issue.md.
  ./0001-mp-worker-heartbeat-from-registration.patch
]
