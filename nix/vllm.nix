# STUB until the sm_70 build lands (Felix owns this file; replace it whole).
#
# Contract the modules rely on, called as
#   import ./vllm.nix { inherit pkgs lib inputs; }
# with the host's `pkgs`; returns at least:
#
#   vllm       wrapped env with bin/vllm (`vllm serve`), bin/hf
#              (huggingface-hub[hf-xet], for `hf download`) and bin/lmcache
#              (LMCache 0.5.6 from the same env: `lmcache server`).
#   lmcacheUi  wrapper with only bin/python, able to import
#              lmcache.lmcache_frontend.app (kept off the system PATH).
#
# Both wrappers set CUDA_HOME/CUDA_PATH (CUDA 12.6), NVCC_PREPEND_FLAGS,
# TRITON_LIBCUDA_PATH=/run/opengl-driver/lib, CC/CXX (gcc 13 backendStdenv)
# and LD_LIBRARY_PATH (/run/opengl-driver/lib, the nvidia-*/torch lib dirs,
# libstdc++/zlib/libGL/glib/numactl).
#
# The stub's executables only fail, so the modules evaluate (and CI can check
# them) before the real build exists.
{ pkgs, lib, ... }:
let
  stub =
    name: exes:
    pkgs.runCommand name { } ''
      mkdir -p $out/bin
      for exe in ${lib.escapeShellArgs exes}; do
        cat > $out/bin/$exe <<SH
      #!${pkgs.runtimeShell}
      echo "vllm-sm70: '$exe' is a stub; the sm_70 build is not wired in yet" >&2
      exit 1
      SH
        chmod +x $out/bin/$exe
      done
    '';
in
{
  vllm = stub "vllm-sm70-stub" [ "vllm" "hf" "lmcache" ];
  lmcacheUi = stub "lmcache-ui-sm70-stub" [ "python" ];
}
