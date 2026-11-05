#!/usr/bin/env bash
# CPU-only checks of the LMCache build (nix/lmcache-build.nix) inside a built
# vllm-sm70 package. No GPU, no sudo, no services: it sets CUDA_VISIBLE_DEVICES
# empty and only imports modules and runs `--help`.
#
# Usage (on the serving host, after `nix-store --realise $OUT`):
#   scripts/check-lmcache-cpu.sh /nix/store/<hash>-vllm-sm70-0.33.0
# Exit status: 0 when every check passes, 1 otherwise. docs/checks.md.
set -u
OUT=${1:?usage: $0 <outPath of .#vllm>}
PY=$OUT/bin/python
export CUDA_VISIBLE_DEVICES=
cd /tmp || exit 1
fails=0
ok() { printf 'PASS  %s\n' "$1"; }
bad() { printf 'FAIL  %s\n' "$1"; fails=$((fails + 1)); }
# Run the wrapped python and keep only the last line (LMCache and torch log
# on import).
py() { "$PY" -c "$1" 2>&1 | tail -1; }

LM=$(py 'import lmcache, os; print(os.path.dirname(lmcache.__file__))')
[ -d "$LM" ] || { bad "import lmcache ($LM)"; exit 1; }
echo "lmcache: $LM"

# 1. Native extensions (`import lmcache` swallows a missing one, so import each).
r=$(py 'import lmcache.cuda_ops, lmcache.lmcache_native, lmcache.lmcache_redis, lmcache.lmcache_fs; print("ok")')
[ "$r" = ok ] && ok "extensions cuda_ops lmcache_native lmcache_redis lmcache_fs" || bad "extensions: $r"

# 2. cuda_ops links the CUDA 12 runtime, never 13.
so=$(ls "$LM"/cuda_ops*.so)
if ldd "$so" | grep -q 'libcudart\.so\.12' && ! ldd "$so" | grep -q 'libcudart\.so\.13'; then
  ok "cuda_ops links libcudart.so.12, not .so.13"
else
  bad "cuda_ops libcudart: $(ldd "$so" | grep cudart)"
fi

# 3. The service's entry point (bin/lmcache, what services.lmcache-sm70 runs).
if "$OUT/bin/lmcache" server --help >/dev/null 2>&1; then
  ok "bin/lmcache server --help"
else
  bad "bin/lmcache server --help (exit $?)"
fi

# 4. vLLM's LMCache MP connector, from both sides.
r=$(py 'import lmcache.integration.vllm.lmcache_mp_connector, vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector; print("ok")')
[ "$r" = ok ] && ok "MP connector imports (lmcache and vllm side)" || bad "MP connector: $r"

# 5. Patch markers in the installed source (a dropped patch must fail here).
adapter=$LM/integration/vllm/vllm_multi_process_adapter.py
if grep -q 'Ping from registration on' "$adapter" && grep -q 'def _ensure_heartbeat_started' "$adapter"; then
  ok "patch 0001 (heartbeat from registration) present"
else
  bad "patch 0001 markers missing in $adapter"
fi

# 6. Distribution metadata: vLLM's dependency check reads the version.
r=$(py 'import importlib.metadata as m; print(m.version("lmcache"))')
[ "$r" = 0.5.6 ] && ok "importlib.metadata version 0.5.6" || bad "version: $r"

# 7. Dashboard: package data and the module lmcacheUi runs.
r=$(py 'import lmcache.lmcache_frontend.app; print("ok")')
if [ -f "$LM/lmcache_frontend/static/index.html" ] && [ "$r" = ok ]; then
  ok "UI: static/index.html, import lmcache.lmcache_frontend.app"
else
  bad "UI: $r"
fi

# 8. cupy (register_kv_caches): its own CUDA runtime and NVRTC are 12.x.
r=$(py 'import cupy.cuda.runtime as rt, cupy_backends.cuda.libs.nvrtc as nv; print(rt.runtimeGetVersion() // 1000, nv.getVersion()[0])')
[ "$r" = "12 12" ] && ok "cupy runtime and NVRTC are CUDA 12" || bad "cupy runtime/NVRTC major: $r"

# 9. No CUDA 13 runtime anywhere in the closure.
if nix-store -qR "$OUT" | xargs -I{} find {} -name 'libcudart.so.13*' 2>/dev/null | grep -q .; then
  bad "closure contains libcudart.so.13"
else
  ok "closure has no libcudart.so.13"
fi

echo "$fails failed"
[ "$fails" -eq 0 ]
