# Upstream vLLM built for sm_70 (CUDA 12.6) on a uv2nix wheel env.
#
# Structure follows charlie-horse/vLLM nix/package.nix + flint.nix @ cf9492a:
#   venv  Python deps from ./pyenv/uv.lock (torch 2.14.0+cu126, triton 3.8.0, cp313),
#         wheels used as-is; native libs come from the wrapper's LD_LIBRARY_PATH.
#         vLLM itself is not in the venv.
#   tree  the patched source built in place (`setup.py build_ext --inplace`)
#         for arch 7.0 only, against the venv's torch.
#   vllm  the venv with `tree` first on PYTHONPATH, wrapped (`vllm`, `python`).
{ pkgs, lib, inputs, src }:
let
  inherit (inputs) pyproject-nix uv2nix pyproject-build-systems;

  python = pkgs.python313;
  # CUDA 13 can't target sm_70 and vLLM's CMake drops 7.0 for nvcc >= 12.8;
  # torch 2.14.0+cu126 was built with 12.6 too.
  cuda = pkgs.cudaPackages_12_6;

  workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./pyenv; };

  # Wheels reference optional/system libs (libcuda, MPI/UCX in nvshmem, ...).
  lockedNames = map (p: p.name) (lib.importTOML ./pyenv/uv.lock).package;
  ignoreMissingLibs =
    final: prev:
    lib.genAttrs (lib.filter (n: n != "vllm-sm70-env" && prev ? ${n}) lockedNames) (
      n: prev.${n}.overrideAttrs (_: { autoPatchelfIgnoreMissingDeps = true; })
    );

  pythonSet =
    (pkgs.callPackage pyproject-nix.build.packages { inherit python; }).overrideScope
      (lib.composeManyExtensions [
        pyproject-build-systems.overlays.wheel
        (workspace.mkPyprojectOverlay { sourcePreference = "wheel"; })
        ignoreMissingLibs
      ]);

  # Runtime deps + the build group (cmake/ninja/setuptools-scm/... for setup.py).
  venv = pythonSet.mkVirtualEnv "vllm-sm70-env" workspace.deps.default;
  buildVenv = pythonSet.mkVirtualEnv "vllm-sm70-build-env" (
    workspace.deps.default // { vllm-sm70-env = [ "build" ]; }
  );

  deps = import ./deps.nix { inherit pkgs; };
  depPath = v: if v ? subdir then "${v.src}/${v.subdir}" else "${v}";

  site = "${buildVenv}/${python.sitePackages}";

  tree = cuda.backendStdenv.mkDerivation {
    pname = "vllm-sm70";
    inherit (src) version;
    inherit src;

    nativeBuildInputs = [
      cuda.cudatoolkit
      pkgs.cmake
      pkgs.ninja
      pkgs.git
      pkgs.which
    ];
    dontUseCmakeConfigure = true;
    dontUseNinjaBuild = true;
    dontUseNinjaInstall = true;
    dontUseNinjaCheck = true;
    dontConfigure = true;
    dontStrip = true;
    dontPatchELF = true;
    dontPatchShebangs = true;

    env = {
      TORCH_CUDA_ARCH_LIST = "7.0";
      CMAKE_CUDA_ARCHITECTURES = "70";
      VLLM_TARGET_DEVICE = "cuda";
      VLLM_USE_PRECOMPILED = "0";
      # patches/0001: skip the external projects that build nothing for
      # sm_70 (FA2/FA3 kernels, FlashMLA, DeepGEMM, ...). See nix/deps.nix.
      VLLM_SM70_TRIM_EXTERNALS = "1";
      VLLM_USE_PRECOMPILED_RUST = "0";
      # No git metadata in a fetched source: skip setuptools-scm's git describe.
      VLLM_VERSION_OVERRIDE = src.version;
      VLLM_RS_BUILD_VERSION = src.version;
      CMAKE_BUILD_TYPE = "Release";
      NVCC_THREADS = "1";
      CUDA_HOME = "${cuda.cudatoolkit}";
      CUDA_PATH = "${cuda.cudatoolkit}";
      NVCC_PREPEND_FLAGS = "-I${cuda.cudatoolkit}/include";
      # The Nix ld wrapper adds its own RPATHs; CMake's install-time rewrite fails.
      CMAKE_ARGS = lib.concatStringsSep " " [
        "-DPython3_EXECUTABLE=${buildVenv}/bin/python3"
        "-DPython_EXECUTABLE=${buildVenv}/bin/python3"
        "-DCMAKE_SKIP_RPATH=ON"
      ];
    };

    buildPhase = ''
      runHook preBuild
      export HOME=$TMPDIR
      # setup.py imports torch, whose CUDA libs ship in the nvidia-* wheels.
      export LD_LIBRARY_PATH="$(find -L ${site}/nvidia ${site}/torch/lib \
        -name '*.so*' -printf '%h\n' 2>/dev/null | sort -u | paste -sd:)"

      # Writable copies of every FetchContent source (some get patched/written).
      d=$TMPDIR/deps; mkdir -p $d
      ${lib.concatStringsSep "\n" (
        lib.mapAttrsToList (n: v: ''
          cp -r --no-preserve=mode ${depPath v} $d/${n}
          export ${n}=$d/${n}
        '') deps
      )}

      # nvcc needs a few GB per job: one job per builder core, never more
      # than 8 (two 16-job CUDA trees hung r440 on 2026-11-04).
      export MAX_JOBS=$(( NIX_BUILD_CORES > 8 ? 8 : NIX_BUILD_CORES ))
      ${buildVenv}/bin/python setup.py build_ext --inplace
      runHook postBuild
    '';

    installPhase = ''
      runHook preInstall
      rm -rf build .deps
      mkdir -p $out
      cp -r . $out/
      # Not pip-installed: give importlib.metadata a vllm dist (the CLI reads
      # its version from it).
      mkdir -p $out/vllm-${src.version}.dist-info
      printf 'Metadata-Version: 2.1\nName: vllm\nVersion: %s\n' ${src.version} \
        > $out/vllm-${src.version}.dist-info/METADATA
      ${buildVenv}/bin/python -m compileall -q -j $NIX_BUILD_CORES \
        --invalidation-mode unchecked-hash $out/vllm || true
      runHook postInstall
    '';

    passthru = { inherit src deps; };
  };

  # Triton JIT-compiles kernels at runtime; it needs ptxas from a 12.x toolkit.
  jitTools = [
    cuda.cudatoolkit
    cuda.backendStdenv.cc
    pkgs.ninja
    pkgs.coreutils
    pkgs.which
  ];
  hostLibs = lib.makeLibraryPath [
    pkgs.stdenv.cc.cc.lib
    pkgs.zlib
    pkgs.libGL
    pkgs.glib
    pkgs.numactl
  ];

  vllm =
    pkgs.runCommand "vllm-sm70-${src.version}"
      {
        nativeBuildInputs = [ pkgs.makeWrapper ];
        passthru = { inherit venv tree python pythonSet; };
      }
      ''
        site=${venv}/${python.sitePackages}
        libdirs=$(find -L "$site/nvidia" "$site/torch/lib" \
          -name '*.so*' -printf '%h\n' 2>/dev/null | sort -u | paste -sd:)
        mkdir -p $out/bin
        # `vllm` is the venv python running vllm's CLI from the tree.
        # TRITON_PTXAS_PATH is only a default (12.6 ptxas): set it to Triton's
        # bundled ptxas (12.9) to compare.
        for exe in python vllm; do
          args=(--argv0 ${venv}/bin/python)
          [ $exe = vllm ] && args+=(--add-flags "-m vllm.entrypoints.cli.main")
          makeWrapper ${venv}/bin/python $out/bin/$exe "''${args[@]}" \
            --prefix PYTHONPATH : ${tree} \
            --set CUDA_HOME ${cuda.cudatoolkit} \
            --set CUDA_PATH ${cuda.cudatoolkit} \
            --set-default TRITON_PTXAS_PATH ${cuda.cudatoolkit}/bin/ptxas \
            --set TRITON_LIBCUDA_PATH /run/opengl-driver/lib \
            --set CC ${cuda.backendStdenv.cc}/bin/cc \
            --set CXX ${cuda.backendStdenv.cc}/bin/c++ \
            --prefix PATH : ${lib.makeBinPath jitTools} \
            --prefix LD_LIBRARY_PATH : "/run/opengl-driver/lib:$libdirs:${hostLibs}"
        done
      '';
in
{
  inherit vllm tree venv buildVenv;
}
