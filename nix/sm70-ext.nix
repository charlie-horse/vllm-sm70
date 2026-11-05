# The vllm-sm70 plugin (sm70/): the `vllm_sm70` Python package plus its
# `_sm70_C` CUDA extension, built for sm_70 only against the same torch as
# vLLM's tree. It is a separate derivation, so a kernel edit rebuilds only
# this, not vLLM.
#
# Output layout (put it on PYTHONPATH after `tree`):
#   $out/vllm_sm70/                       package incl. _sm70_C*.so
#   $out/vllm_sm70-<version>.dist-info/   METADATA + entry_points.txt, so vLLM
#                                         finds the plugins via importlib.metadata
#
# Inputs (agreed with Felix): buildVenv (torch: any change rebuilds this),
# cuda, the CUTLASS source from deps.nix, the sm70/ fileset, and vLLM's
# patched source (vllmSrc, ./source.nix: BASE + series) for its csrc headers.
# sm70/csrc/custom_ar (K17) includes custom_all_reduce.cuh and
# custom_collective_common.cuh from it, so the bytes match what vLLM's _C was
# built from; any change to patches/ therefore rebuilds this too.
{
  pkgs,
  lib,
  cuda,
  buildVenv,
  site,
  deps,
  vllmSrc,
  version ? "0.1.0",
}:
let
  root = ../sm70;
  src = lib.fileset.toSource {
    inherit root;
    fileset = lib.fileset.difference root (
      lib.fileset.unions [
        (lib.fileset.maybeMissing (root + "/tests"))
        (lib.fileset.maybeMissing (root + "/tools"))
      ]
    );
  };
in
cuda.backendStdenv.mkDerivation {
  pname = "vllm-sm70-ext";
  inherit version src;

  nativeBuildInputs = [
    cuda.cudatoolkit
    pkgs.cmake
    pkgs.ninja
  ];
  # setup.py drives CMake itself.
  dontUseCmakeConfigure = true;
  dontUseNinjaBuild = true;
  dontUseNinjaInstall = true;
  dontUseNinjaCheck = true;
  dontConfigure = true;
  dontStrip = true;
  dontPatchELF = true;

  env = {
    TORCH_CUDA_ARCH_LIST = "7.0";
    CMAKE_CUDA_ARCHITECTURES = "70";
    CMAKE_BUILD_TYPE = "Release";
    CUDA_HOME = "${cuda.cudatoolkit}";
    CUDA_PATH = "${cuda.cudatoolkit}";
    NVCC_PREPEND_FLAGS = "-I${cuda.cudatoolkit}/include";
    # Headers only; CMake never writes into it, so the store path is fine.
    VLLM_SM70_CUTLASS_DIR = "${deps.VLLM_CUTLASS_SRC_DIR}";
    # Headers only, read by sm70/CMakeLists for the custom_ar target.
    VLLM_SM70_VLLM_CSRC_DIR = "${vllmSrc}/csrc";
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
    cp -r vllm_sm70 $out/
    ls $out/vllm_sm70/_sm70_C*.so >/dev/null  # the extension must exist
    di=$out/vllm_sm70-${version}.dist-info
    mkdir -p $di
    printf 'Metadata-Version: 2.1\nName: vllm-sm70\nVersion: %s\n' ${version} > $di/METADATA
    cp nix-entry_points.txt $di/entry_points.txt
    ${buildVenv}/bin/python -m compileall -q --invalidation-mode unchecked-hash \
      $out/vllm_sm70 || true
    runHook postInstall
  '';

  passthru = { inherit src; };
}
