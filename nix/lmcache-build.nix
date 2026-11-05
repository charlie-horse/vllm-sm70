# LMCache 0.5.6 built from the PyPI sdist for sm_70 (CUDA 12.6) against the
# same torch as vLLM's tree. The PyPI wheels are a CUDA 13 build (archs 7.5+),
# so they can't serve c4130 (docs/pins.md, LMCache).
#
# Output layout (put it on PYTHONPATH after `tree` and `sm70ext`):
#   $out/lmcache/                    the package, incl. cuda_ops, lmcache_native,
#                                    lmcache_redis and lmcache_fs (*.so)
#   $out/lmcache-0.5.6.dist-info/    METADATA (version 0.5.6, which vLLM reads)
#                                    and entry_points.txt (`lmcache` & co.)
#
# Patches: ./patches/lmcache (Theo's list), applied with --fuzz=0.
# Runtime deps not in vLLM's set are the `lmcache` group in ./pyenv; the four
# locked versions above LMCache's declared caps (numpy, protobuf,
# opentelemetry-api, prometheus_client) are not import problems on our paths.
#
# Not built:
# - the gRPC stubs (`build_py` runs grpc_tools.protoc; grpcio-tools 1.78 needs
#   protobuf<7, the lock has 7.x). Only `--transport grpc` needs them; the
#   services use the default zmq transport.
# - rust/raw_block (a separate maturin package; only the raw-block L2 adapter
#   uses it, and it fails lazily).
# - NIXL / Mooncake / Aerospike / HF3FS: their headers and libs are not inputs.
{
  pkgs,
  lib,
  cuda,
  buildVenv,
  site,
}:
let
  version = "0.5.6";

  sdist = pkgs.fetchurl {
    url = "https://files.pythonhosted.org/packages/84/25/3ccd46844c94e08897aa8d72716a538c07e4583410d84dbc6cd12cb60dc1/lmcache-${version}.tar.gz";
    sha256 = "d6951e60ce8818a76aa42d0866a4ba85f1c95cd919355fff30ff24dcc3243f1e";
  };

  src = pkgs.applyPatches {
    name = "lmcache-${version}-sm70-source";
    src = sdist;
    patches = import ./patches/lmcache;
    patchFlags = [
      "-p1"
      "--fuzz=0"
      "--no-backup-if-mismatch"
    ];
  };
in
cuda.backendStdenv.mkDerivation {
  pname = "lmcache-sm70";
  inherit version src;

  nativeBuildInputs = [ cuda.cudatoolkit ];
  dontConfigure = true;
  dontStrip = true;
  dontPatchELF = true;

  env = {
    # Pick the CUDA profile outright (else ascend/maca are probed first).
    BUILD_WITH_CUDA = "1";
    TORCH_CUDA_ARCH_LIST = "7.0";
    # Only selects requirements/cuda12_core.txt; no effect on compiled code.
    LMCACHE_CUDA_MAJOR = "12";
    # No .git in the sdist.
    SETUPTOOLS_SCM_PRETEND_VERSION = version;
    NVCC_THREADS = "1";
    CUDA_HOME = "${cuda.cudatoolkit}";
    CUDA_PATH = "${cuda.cudatoolkit}";
    NVCC_PREPEND_FLAGS = "-I${cuda.cudatoolkit}/include";
  };

  buildPhase = ''
    runHook preBuild
    export HOME=$TMPDIR
    # setup.py imports torch, whose CUDA libs ship in the nvidia-* wheels.
    export LD_LIBRARY_PATH="$(find -L ${site}/nvidia ${site}/torch/lib \
      -name '*.so*' -printf '%h\n' 2>/dev/null | sort -u | paste -sd:)"
    # nvcc needs a few GB per job; never more than 8 (r440 limit).
    export MAX_JOBS=$(( NIX_BUILD_CORES > 8 ? 8 : NIX_BUILD_CORES ))
    ${buildVenv}/bin/python setup.py build_ext --inplace
    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
    mkdir -p $out
    cp -r lmcache $out/
    # The extensions must exist: `import lmcache` swallows a missing one.
    for m in cuda_ops lmcache_native lmcache_redis lmcache_fs; do
      ls $out/lmcache/$m*.so >/dev/null
    done
    test -f $out/lmcache/lmcache_frontend/static/index.html
    di=$out/lmcache-${version}.dist-info
    mkdir -p $di
    cp PKG-INFO $di/METADATA
    cp lmcache.egg-info/entry_points.txt $di/entry_points.txt
    grep -qx 'Version: ${version}' $di/METADATA
    ${buildVenv}/bin/python -m compileall -q --invalidation-mode unchecked-hash \
      $out/lmcache || true
    runHook postInstall
  '';

  passthru = { inherit src sdist; };
}
