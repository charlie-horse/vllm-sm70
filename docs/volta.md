# The chip: Volta (sm_70)

Volta is the NVIDIA GPU generation in the Tesla V100. NVIDIA's tools call it
compute capability 7.0, or `sm_70`. This page lists what the chip can and
can't do, and what the code in this repository does about each gap.

## What it has

- **Tensor cores, first generation.** Small matrix-multiply units that make
  model maths fast. On Volta they work only on 16-bit floating point
  (`fp16`) numbers.
- **Fast on-chip memory (HBM2)** next to the GPU. Large language models are
  limited mostly by how fast weights can be read from this memory, not by
  arithmetic.
- **NVLink or PCIe** to talk to other GPUs when a model is split across
  several of them (tensor parallelism, "TP").

## What it lacks, and what the code does instead

| Missing on Volta | What this repo does |
| --- | --- |
| `bf16` numbers (the format most new models ship in) | Runs every model in `fp16` (`--dtype half`) and checks values stay in `fp16`'s range |
| Hardware support for 8-bit and 4-bit numbers (`FP8`, `NVFP4`) | Keeps weights compressed in memory and unpacks them to `fp16` inside its own kernels (TurboMind, QPN8, NVFP4 GEMV) |
| FlashAttention 2, FlashInfer and other fast attention libraries (they need newer GPUs) | Its own attention kernels, Flash-V100, with Triton kernels as fallback |
| Support in CUDA 13 | Builds with CUDA 12.x and the last PyTorch release that still ships Volta code |
| Tensor cores in the Triton compiler's code generation | Writes the hot paths in CUDA instead |

## Rules that follow

- Every GPU kernel is compiled for `sm_70` only.
- Models run in `fp16`. A model whose values overflow `fp16` needs a fix in
  its plugin code, not a different number format.
- Upstream vLLM paths that require a newer GPU stay unused; the plugin
  provides its own.
