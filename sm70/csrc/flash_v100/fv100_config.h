// SPDX-License-Identifier: Apache-2.0
//
// Flash-V100 compile-time configuration: the policy fields the kernels read,
// as constants. No environment variable is read. The interface
// (flash_v100::policy::Field, value()) lets the vendored kernel bodies
// compile unchanged. Source: PROVENANCE.md.

#pragma once

namespace flash_v100::policy {

enum class Field {
  // G6/P256 XQA uses the padded shared-memory layout.
  xqa_padded_smem_enabled,
  // D256 low-SMEM prefill.
  prefill_d256_low_smem_default_on,
  // D256 BM32 phase prefill.
  prefill_d256_bm32_phase_default_on,
  // Contiguous fast prefill (page 16 always takes it).
  prefill_contig_fast_off,
  // D256 output stride 268: on for page 784, off otherwise.
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
