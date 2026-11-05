// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Compatibility shim. The vendored lmdeploy files (gemm.cu, dispatch_cache.cu,
// sm70_dflash_context.h) and gemm_runtime.cpp `#include "sm70_policy.h"`, as
// they did in fork cf9492a. The fork header was the per-engine native-policy
// ABI. Here it only forwards to tm_config.h (env knobs read once per process).
#pragma once
#include "tm_config.h"
