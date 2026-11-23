# SPDX-License-Identifier: Apache-2.0
"""M12 Inkling 2 Triton kernels compile for sm_70, on CPU.

Every kernel of inkling2_sm70/ops (paged_conv.py, paged_attention.py,
norm.py, attn_res.py, loop.py, fused.py, activation.py, grouped_gemm.py,
routing.py, dspark.py, small_ops.py, top1.py) goes through Triton's compiler and ptxas
for GPUTarget("cuda", 70, 32) at Inkling 2 Mini sizes (hidden 2304, head dim
128, 12 query and 4 KV heads per rank at TP2, KV block 32, 4 conv taps, expert
intermediate 576 per rank, 128 experts, top-8, decode batches of 1 to 8
tokens, message-gate rank 64, d_rel 16, rel extent 1024; draft: 8-query
blocks, 4 feature taps, Markov rank 256), with the
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


# --- fused.py ------------------------------------------------------------------------

GATE_RANK = 64


@pytest.mark.parametrize("weight", ["*fp16", "*fp32"])
def test_gate_kernels(weight):
    fused = _ops("fused")
    _check(
        fused._gate_down_kernel,
        {
            "Dest": "*fp32",
            "Message": "*fp32",
            **dict.fromkeys(("Down0", "Down1", "Down2"), weight),
            "Low": "*fp32",
            **_i32("stride_dest_s", "T"),
            "eps": "fp32",
            "D": HIDDEN,
            "R": GATE_RANK,
            "BLOCK_R": fused.BLOCK_R,
            "BLOCK_K": fused.BLOCK_K,
        },
        num_warps=4,
    )
    _check(
        fused._gate_up_kernel,
        {
            "Low": "*fp32",
            **dict.fromkeys(
                ("Gain0", "Gain1", "Gain2", "Forget0", "Forget1", "Forget2"), weight
            ),
            "Out": "*fp32",
            "T": "i32",
            "D": HIDDEN,
            "R": GATE_RANK,
            "BLOCK_D": fused.BLOCK_D,
        },
        num_warps=4,
    )


@pytest.mark.parametrize("base", [False, True], ids=["update", "base_update"])
def test_residual_add_kernel(base):
    fused = _ops("fused")
    _check(
        fused._residual_add_kernel,
        {
            **dict.fromkeys(("Base", "Update", "Addend", "Out"), "*fp16"),
            "N": "i32",
            "HAS_BASE": base,
            "BLOCK": fused.BLOCK_ADD,
        },
        num_warps=4,
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


@pytest.mark.parametrize(
    ("row_div", "k_in"), [(8, HIDDEN), (1, LOCAL_INTERMEDIATE)], ids=["w13", "w2"]
)
def test_slot_gemv_mx_kernel(row_div, k_in):
    grouped_gemm = _ops("grouped_gemm")
    _check(
        grouped_gemm._slot_gemv_mx_kernel,
        {
            "in_ptr": "*fp16",
            "w_ptr": "*u8",
            "s_ptr": "*u8",
            "out_ptr": "*fp16",
            "expert_ptr": "*i32",
            **_i32(
                "stride_in", "stride_we", "stride_wn", "stride_se", "stride_sn",
                "stride_out", "n_out", "k_in", "offset",
            ),
            "ROW_DIV": row_div,
            "BLOCK_N": grouped_gemm.BLOCK_N,
            "BLOCK_K": grouped_gemm.BLOCK_K,
            "MX_BLOCK": grouped_gemm.MX_BLOCK,
        },
        enable_fp_fusion=False,
    )  # fmt: skip


def test_mx_dequant_kernel():
    grouped_gemm = _ops("grouped_gemm")
    _check(
        grouped_gemm._mx_dequant_kernel,
        {
            "w_ptr": "*u8",
            "s_ptr": "*u8",
            "out_ptr": "*fp16",
            **_i32("k_in", "stride_w", "stride_s", "stride_out", "offset"),
            "BLOCK": grouped_gemm.DEQUANT_BLOCK,
            "MX_BLOCK": grouped_gemm.MX_BLOCK,
        },
    )


# --- routing.py ---------------------------------------------------------------------

EXPERTS = 128
TOP_K = 8


@pytest.mark.parametrize("x", ["*fp16", "*fp32"])
def test_router_logits_kernel(x):
    routing = _ops("routing")
    _check(
        routing._router_logits_kernel,
        {
            "x_ptr": x,
            "w_ptr": "*fp32",
            "out_ptr": "*fp32",
            **_i32("stride_x", "stride_w", "stride_out", "hidden", "experts"),
            "BLOCK_E": routing.GEMV_BLOCK_E,
            "BLOCK_K": routing.GEMV_BLOCK_K,
        },
        enable_fp_fusion=False,
    )


@pytest.mark.parametrize("tokens", [1, 2, 8])
@pytest.mark.parametrize("has_pad", [False, True], ids=["no_pad", "pad"])
@pytest.mark.parametrize("dispatch", [False, True], ids=["route", "dispatch"])
def test_route_kernel(tokens, has_pad, dispatch):
    routing = _ops("routing")
    _check(
        routing._route_kernel,
        {
            "logits_ptr": "*fp32",
            "bias_ptr": "*fp32",
            "pad_ptr": "*i1" if has_pad else "*i32",
            "ids_ptr": "*i32",
            "weights_ptr": "*fp32",
            "pos_ptr": "*i32",
            "sorted_ptr": "*i32",
            "offsets_ptr": "*i32",
            "src_ptr": "*fp16" if dispatch else "*i32",
            "rows_ptr": "*fp16" if dispatch else "*i32",
            **_i32(
                "tokens",
                "experts",
                "hidden",
                "stride_l",
                "stride_src",
                "stride_rows",
                "pad_id",
            ),
            "route_scale": "fp32",
            "TOP_K": TOP_K,
            "BLOCK_M": max(2, tokens),
            "BLOCK_E": EXPERTS,
            "BLOCK_TOP": TOP_K,
            "BLOCK_H": routing.BLOCK_H,
            "HAS_PAD": has_pad,
            "DISPATCH": dispatch,
        },
        enable_fp_fusion=False,
    )


@pytest.mark.parametrize("has_pos", [False, True], ids=["by_token", "sorted"])
def test_combine_kernel(has_pos):
    routing = _ops("routing")
    _check(
        routing._combine_kernel,
        {
            "y_ptr": "*fp16",
            "ids_ptr": "*i32",
            "weights_ptr": "*fp32",
            "pos_ptr": "*i32",
            "out_ptr": "*fp16",
            **_i32("stride_y", "stride_out", "hidden", "experts"),
            "TOP_K": TOP_K,
            "BLOCK_H": routing.BLOCK_H,
            "HAS_POS": has_pos,
        },
        enable_fp_fusion=False,
    )


# --- dspark.py -----------------------------------------------------------------

DRAFT_BLOCK = 8
DRAFT_TAPS = 4
MARKOV_RANK = 256


def test_mix_rmsnorm_kernel():
    _check(
        _ops("dspark")._mix_rmsnorm_kernel,
        {
            **dict.fromkeys(("x_ptr", "out_ptr", "weight_ptr"), "*fp16"),
            "alpha_ptr": "*fp32",
            **_i32("stride_row", "stride_tap", "stride_h"),
            "eps": "fp32",
            "TAPS": DRAFT_TAPS,
            "HIDDEN": HIDDEN,
            "BLOCK": 4096,
        },
        num_warps=8,
        enable_fp_fusion=False,
    )


@pytest.mark.parametrize("norm", [False, True], ids=["rope", "norm_rope"])
def test_qk_norm_rope_kernel(norm):
    _check(
        _ops("dspark")._qk_norm_rope_kernel,
        {
            **dict.fromkeys(("x_ptr", "q_weight_ptr", "k_weight_ptr"), "*fp16"),
            "cos_sin_ptr": "*fp32",
            "pos_ptr": "*i64",
            **_i32("stride_row", "q_size"),
            "eps": "fp32",
            "NHQ": Q_HEADS if norm else 0,
            "D": HEAD_DIM,
            "NORM": norm,
        },
        num_warps=1,
        enable_fp_fusion=False,
    )


def test_silu_and_mul_kernel():
    _check(
        _ops("dspark")._silu_and_mul_kernel,
        {"x_ptr": "*fp16", "out_ptr": "*fp16", **_i32("stride_x", "n"), "BLOCK": 1024},
        num_warps=4,
        enable_fp_fusion=False,
    )


def test_add_softcap_kernel():
    _check(
        _ops("dspark")._add_softcap_kernel,
        {
            **dict.fromkeys(("a_ptr", "b_ptr"), "*fp16"),
            "out_ptr": "*fp32",
            **_i32("stride_a", "stride_b", "stride_o", "n"),
            "cap": "fp32",
            "BLOCK": 1024,
        },
        num_warps=4,
        enable_fp_fusion=False,
    )


def test_markov_gate_kernel():
    _check(
        _ops("dspark")._markov_gate_kernel,
        {
            "raw_ptr": "*fp32",
            **dict.fromkeys(("state_ptr", "out_ptr", "latent_ptr"), "*fp16"),
            "RANK": MARKOV_RANK,
        },
        num_warps=1,
        enable_fp_fusion=False,
    )


def test_block_attention_kernel():
    _check(
        _ops("dspark")._block_attention_kernel,
        {
            **dict.fromkeys(("q_ptr", "key_ptr", "value_ptr", "out_ptr"), "*fp16"),
            "table_ptr": "*i32",
            "seq_ptr": "*i32",
            **_i32(
                "stride_qt",
                "stride_qh",
                "stride_kb",
                "stride_kn",
                "stride_kh",
                "stride_ot",
                "stride_bt",
                "window",
            ),
            "scale": "fp32",
            "BLOCK_Q": DRAFT_BLOCK,
            "GROUPS": Q_HEADS // KV_HEADS,
            "D": HEAD_DIM,
            "BLOCK_N": KV_BLOCK,
        },
        num_warps=4,
        enable_fp_fusion=False,
    )


# --- small_ops.py -------------------------------------------------------------------

D_REL = 16
REL_EXTENT = 1024


@pytest.mark.parametrize("scale", ["*fp16", "*fp32"])
@pytest.mark.parametrize("log_scale", [False, True], ids=["sliding", "global"])
def test_query_rel_kernel(scale, log_scale):
    small_ops = _ops("small_ops")
    _check(
        small_ops._query_rel_kernel,
        {
            **dict.fromkeys(("q_ptr", "r_ptr", "proj_ptr"), "*fp16"),
            "q_scale_ptr": scale,
            "r_scale_ptr": scale,
            "slot_ptr": "*i64",
            "pos_ptr": "*i64",
            "q_out_ptr": "*fp16",
            "rel_out_ptr": "*fp16",
            **_i32(
                "num_tokens",
                "stride_q",
                "stride_r",
                "stride_qo0",
                "stride_qo1",
                "stride_ro0",
                "stride_ro1",
                "stride_p",
                "extent",
            ),
            **dict.fromkeys(("q_clip", "r_clip", "alpha", "n_floor"), "fp32"),
            "D": HEAD_DIM,
            "DR": D_REL,
            "BLOCK_D": HEAD_DIM,
            "BLOCK_R": D_REL,
            "BLOCK_E": small_ops.BLOCK_E,
            "LOG_SCALE": log_scale,
        },
        enable_fp_fusion=False,
    )


def test_gate_kernel():
    small_ops = _ops("small_ops")
    _check(
        small_ops._gate_kernel,
        {
            "attn_ptr": "*fp16",
            "gate_ptr": "*fp16",
            "slot_ptr": "*i64",
            "out_ptr": "*fp16",
            **_i32("num_tokens", "stride_a", "stride_g", "stride_o"),
            "C": Q_HEADS * HEAD_DIM,
            "BLOCK": small_ops.BLOCK_GATE,
        },
        enable_fp_fusion=False,
    )


# --- top1.py ---------------------------------------------------------------------


@pytest.mark.parametrize("soft_cap", [True, False], ids=["cap", "no_cap"])
def test_softcap_top1_kernel(soft_cap):
    top1 = _ops("top1")
    args = {
        "logits_ptr": "*fp16",
        "out_ptr": "*fp32",
        **_i32("stride_row", "n_cols", "vocab_start"),
        "cap": "fp32",
        "SOFT_CAP": soft_cap,
        "SPLIT": top1.SPLIT_COLS,
        "BLOCK": top1.BLOCK,
    }
    _check(top1._softcap_top1_kernel, args)


@pytest.mark.parametrize("block_n", [16, 32])
def test_pick_kernel(block_n):
    args = {"pairs_ptr": "*fp32", "out_ptr": "*i64", "n": "i32", "BLOCK_N": block_n}
    _check(_ops("top1")._pick_kernel, args, num_warps=1)
