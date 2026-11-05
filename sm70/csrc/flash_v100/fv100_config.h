// SPDX-License-Identifier: Apache-2.0
//
// Flash-V100 compile-time configuration (patch K05).
//
// Replaces the fork's include/flash_v100_policy.h (cf9492a, 466 lines). The
// fork read 66 VLLM_FLASH_V100_* environment fields through a per-engine
// PreparedPolicy bound by pybind `*_configured` entry points. The lean
// subset reaches only the six fields below, and each takes the value the
// fork's parse() returned with the variable unset, so the kept routes are
// the fork defaults. No environment variable is read: every knob that
// selected a non-default route was deleted together with that route.
//
// The interface (flash_v100::policy::Field, value()) is kept
// so the ported kernel bodies compile unchanged.

#pragma once

namespace flash_v100::policy {

enum class Field {
  // fork VLLM_FLASH_V100_XQA_PADDED_SMEM (unset -> 1): G6/P256 XQA uses the
  // padded shared-memory layout.
  xqa_padded_smem_enabled,
  // fork VLLM_FLASH_V100_PREFILL_D256_LOW_SMEM (unset -> 1).
  prefill_d256_low_smem_default_on,
  // fork VLLM_FLASH_V100_PREFILL_D256_BM32_PHASE (unset -> 1).
  prefill_d256_bm32_phase_default_on,
  // fork VLLM_FLASH_V100_PREFILL_CONTIG_FAST (unset -> 0; page 16 still on).
  prefill_contig_fast_off,
  // fork VLLM_FLASH_V100_PREFILL_D256_OUTPUT_STRIDE_268 (unset -> 1 for
  // page 784, 0 otherwise); both readings of the one variable.
  prefill_d256_output_stride_268_default_on,
  prefill_d256_output_stride_268_off,
  count
};

constexpr int value(Field field) {
  switch (field) {
    case Field::xqa_padded_smem_enabled:
      return 1;
    case Field::prefill_d256_low_smem_default_on:
      return 1;
    case Field::prefill_d256_bm32_phase_default_on:
      return 1;
    case Field::prefill_contig_fast_off:
      return 0;
    case Field::prefill_d256_output_stride_268_default_on:
      return 1;
    case Field::prefill_d256_output_stride_268_off:
      return 0;
    default:
      return 0;
  }
}

}  // namespace flash_v100::policy
