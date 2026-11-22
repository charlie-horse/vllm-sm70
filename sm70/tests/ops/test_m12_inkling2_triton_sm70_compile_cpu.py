# SPDX-License-Identifier: Apache-2.0
"""M12 Inkling 2 Triton kernels compile for sm_70, on CPU.

Every kernel of inkling2_sm70/ops (paged_conv.py, paged_attention.py,
norm.py, attn_res.py, loop.py, activation.py, grouped_gemm.py) goes through
Triton's compiler and ptxas for GPUTarget("cuda", 70, 32) at Inkling 2 Mini
sizes (hidden 2304, head dim 128, 12 query and 4 KV heads per rank at TP2,
KV block 32, 4 conv taps, expert intermediate 576 per rank), with the
constexprs and launch options of its launcher. The PTX holds no tanh.approx, mma or bf16.
This catches what the interpreter tests cannot: Triton's constexpr and
static control-flow rules. Skipped without Triton's CUDA backend or ptxas.
"""

from __future__ import annotations

import importlib
import importlib.util
import pathlib
import sys
import types

import pytest

pytest.importorskip("torch")

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _sm70_compile as aot


def _have_vllm() -> bool:
    try:
        return importlib.util.find_spec("vllm.envs") is not None
    except (ImportError, ValueError):
        return False


pytestmark = [
    pytest.mark.skipif(not _have_vllm(), reason="vLLM not importable"),
    pytest.mark.skipif(aot.skip_reason() is not None, reason=str(aot.skip_reason())),
]

HIDDEN = 2304
HEAD_DIM = 128
KV_HEADS = 4  # per rank at TP2
Q_HEADS = 12  # per rank at TP2
KV_BLOCK = 32
TAPS = 4


@pytest.fixture(autouse=True)
def _stub_stable_libtorch(monkeypatch):
    try:
        importlib.import_module("vllm._C_stable_libtorch")
    except ImportError:
        monkeypatch.setitem(
            sys.modules,
            "vllm._C_stable_libtorch",
            types.ModuleType("vllm._C_stable_libtorch"),
        )


def _ops(name: str):
    return importlib.import_module(f"vllm_sm70.models.inkling2_sm70.ops.{name}")


def _check(kernel, args, **options) -> None:
    ptx = aot.compile_ptx(kernel, args, **options)
    assert ".target sm_70" in ptx
    assert aot.banned_in(ptx) == []


def _i32(*names: str) -> dict[str, str]:
    return dict.fromkeys(names, "i32")


# --- paged_conv.py --------------------------------------------------------------


@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
@pytest.mark.parametrize(("block_t", "block_c"), [(1, 1024), (4, 256)])
def test_short_conv_kernel(dynamic, block_t, block_c):
    args = {
        **dict.fromkeys(("x_ptr", "w_ptr", "dyn_ptr", "cache_ptr", "out_ptr"), "*fp16"),
        "pos_ptr": "*i64",
        "seq_ptr": "*i32",
        "qstart_ptr": "*i32",
        "slot_ptr": "*i64",
        "table_ptr": "*i32",
        **_i32(
            "T",
            "num_tokens",
            "stride_x",
            "stride_out",
            "stride_w0",
            "stride_w1",
            "stride_dyn",
            "stride_cb",
            "stride_ch",
            "stride_cn",
            "cache_block",
            "offset",
            "stride_bt",
            "max_blocks",
        ),
        "divisor": "fp32",
        "C": HIDDEN,
        "WS": HIDDEN // KV_HEADS,
        "H": KV_HEADS,
        "W": TAPS,
        "DYNAMIC": dynamic,
        "BT": block_t,
        "BLOCK_C": block_c,
    }
    _check(_ops("paged_conv")._short_conv_kernel, args, enable_fp_fusion=False)


@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
@pytest.mark.parametrize("block_t", [1, 4])
def test_key_value_conv_kernel(dynamic, block_t):
    args = {
        **dict.fromkeys(
            (
                "k_ptr",
                "v_ptr",
                "wk_ptr",
                "wv_ptr",
                "dk_ptr",
                "dv_ptr",
                "cache_ptr",
                "kv_ptr",
            ),
            "*fp16",
        ),
        "kv_slot_ptr": "*i64",
        "pos_ptr": "*i64",
        "seq_ptr": "*i32",
        "qstart_ptr": "*i32",
        "slot_ptr": "*i64",
        "table_ptr": "*i32",
        **_i32(
            "num_tokens",
            "stride_k",
            "stride_v",
            "stride_wk0",
            "stride_wk1",
            "stride_wv0",
            "stride_wv1",
            "stride_dk",
            "stride_dv",
            "stride_cb",
            "stride_ch",
            "stride_cn",
            "cache_block",
            "offset_k",
            "offset_v",
            "stride_bt",
            "max_blocks",
            "stride_kvb",
            "stride_kvh",
            "stride_kvn",
            "kv_block",
        ),
        "divisor": "fp32",
        "clip": "fp32",
        "eps": "fp32",
        "D": HEAD_DIM,
        "H": KV_HEADS,
        "W": TAPS,
        "DYNAMIC": dynamic,
        "BT": block_t,
        "BLOCK_D": HEAD_DIM,
    }
    _check(_ops("paged_conv")._key_value_conv_kernel, args, enable_fp_fusion=False)


# --- paged_attention.py ----------------------------------------------------------


@pytest.mark.parametrize("indexed", [False, True], ids=["first_rows", "indexed"])
@pytest.mark.parametrize("slots", [False, True], ids=["eager", "padded"])
def test_decode_attention_kernel(indexed, slots):
    args = {
        **dict.fromkeys(
            ("q_ptr", "rel_ptr", "sink_ptr", "cache_ptr", "out_ptr"), "*fp16"
        ),
        "rows_ptr": "*i64",
        "pos_ptr": "*i64",
        "seq_ptr": "*i32",
        "slot_ptr": "*i64",
        "table_ptr": "*i32",
        **_i32(
            "stride_q0",
            "stride_q1",
            "stride_r0",
            "stride_r1",
            "stride_s0",
            "stride_o0",
            "stride_o1",
            "stride_cb",
            "stride_ch",
            "stride_cn",
            "stride_bt",
            "extent",
            "window",
        ),
        "scale": "fp32",
        "D": HEAD_DIM,
        "GROUPS": Q_HEADS // KV_HEADS,
        "BLOCK_N": KV_BLOCK,
        "BLOCK_D": HEAD_DIM,
        "INDEXED": indexed,
        "SLOTS": slots,
    }
    _check(
        _ops("paged_attention")._decode_attention_kernel,
        args,
        num_warps=4,
        enable_fp_fusion=False,
    )


# --- norm.py, attn_res.py, loop.py ------------------------------------------------


@pytest.mark.parametrize("out", ["*fp16", "*fp32"])
def test_rms_norm_kernel(out):
    args = {
        "X": "*fp16",
        "W": "*fp32",
        "Y": out,
        "stride_x": "i32",
        "N": HIDDEN,
        "eps": "fp32",
        "BLOCK": 4096,
    }
    _check(_ops("norm")._rms_norm_kernel, args, num_warps=8)


def _attn_res_args(addend: bool, norm: bool) -> dict:
    return {
        "Stack": "*fp32",
        "Last": "*fp32",
        "Addend": "*fp32" if addend else None,
        "Queries": "*fp32",
        "NormW": "*fp32",
        "Out": "*fp16" if norm else "*fp32",
        **_i32("stride_s", "stride_t", "T"),
        "eps": "fp32",
        "norm_eps": "fp32",
        "q_scale": "fp32",
        "D": HIDDEN,
        "N_STACK": 4,
        "HAS_ADDEND": addend,
        "NORMALIZE": True,
        "APPLY_NORM": norm,
        "STACK_NORMED": True,
        "USE_GDC": False,
        "BLOCK_N": 8,
    }


@pytest.mark.parametrize("addend", [False, True], ids=["last", "last_addend"])
@pytest.mark.parametrize("norm", [False, True], ids=["mix", "mix_norm"])
def test_attn_res_kernels(addend, norm):
    attn_res = _ops("attn_res")
    args = _attn_res_args(addend, norm)
    _check(
        attn_res._attn_res_kernel,
        {**args, "BLOCK_D": 1024},
        enable_fp_fusion=False,
    )
    _check(
        attn_res._attn_res_row_kernel,
        {**args, "BLOCK_A": 2048, "BLOCK_B": 256},
        num_warps=8,
        enable_fp_fusion=False,
    )


def test_loop_kernels():
    loop = _ops("loop")
    _check(
        loop._pair_rms_norm_kernel,
        {
            "Dest": "*fp32",
            "Message": "*fp32",
            "Out": "*fp16",
            **_i32("stride_dest_s", "T"),
            "eps": "fp32",
            "D": HIDDEN,
            "BLOCK": 4096,
        },
        num_warps=8,
    )
    _check(
        loop._blend_rms_norm_kernel,
        {
            "Dest": "*fp32",
            "Message": "*fp32",
            "GainForget": "*fp32",
            "NormW": "*fp16",
            "Out": "*fp16",
            **_i32("stride_dest_s", "stride_w_s", "T"),
            "norm_eps": "fp32",
            "D": HIDDEN,
            "BLOCK": 4096,
        },
        num_warps=8,
        enable_fp_fusion=False,
    )


# --- activation.py, grouped_gemm.py -------------------------------------------------

LOCAL_INTERMEDIATE = 576  # per rank at TP2


def test_softcapped_swiglu_kernel():
    _check(
        _ops("activation")._softcapped_swiglu_kernel,
        {
            "gate_up_ptr": "*fp16",
            "out_ptr": "*fp16",
            **_i32("stride_g", "stride_o", "units"),
            "gate_cap": "fp32",
            "up_cap": "fp32",
            "BLOCK": 1024,
        },
        enable_fp_fusion=False,
    )


@pytest.mark.parametrize(
    ("row_div", "k_in"), [(8, HIDDEN), (1, LOCAL_INTERMEDIATE)], ids=["w13", "w2"]
)
def test_slot_gemv_kernel(row_div, k_in):
    grouped_gemm = _ops("grouped_gemm")
    _check(
        grouped_gemm._slot_gemv_kernel,
        {
            "in_ptr": "*fp16",
            "w_ptr": "*fp16",
            "out_ptr": "*fp16",
            "expert_ptr": "*i32",
            **_i32(
                "stride_in", "stride_we", "stride_wn", "stride_out", "n_out", "k_in"
            ),
            "ROW_DIV": row_div,
            "BLOCK_N": grouped_gemm.BLOCK_N,
            "BLOCK_K": grouped_gemm.BLOCK_K,
        },
        enable_fp_fusion=False,
    )
