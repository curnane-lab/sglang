from types import SimpleNamespace

import sgl_kernel_npu
import torch
import torch_npu

from sglang.srt.hardware_backend.npu.attention.ascend_gdn_backend import (
    AscendGDNAttnBackend,
)
from sglang.srt.hardware_backend.npu.attention.ascend_hybrid_linear_attn_backend import (
    AscendHybridLinearAttnBackend,
)
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout


class _Pool:
    def __init__(self, state, all_state):
        self.state = state
        self.all_state = all_state

    def mamba2_layer_cache(self, layer_id):
        assert layer_id == 0
        return self.state

    def get_speculative_mamba2_params_all_layers(self):
        return self.all_state


def _run_forward(lengths, accepted_steps, initial_conv, initial_ssm, mixed_qkv, a, b, packed=False):
    device = mixed_qkv.device
    torch.manual_seed(12345)
    total = mixed_qkv.shape[0]
    bs = len(lengths)
    w = 8
    nk, nv, dk, dv = 4, 8, 128, 128
    dim = 2 * nk * dk + nv * dv
    cache_indices = torch.tensor([2, 0, 1][:bs], dtype=torch.int32, device=device)
    layout = (
        RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=lengths,
            device=device,
            grid=[sum(lengths)],
        )
        if packed
        else None
    )
    spec_info = SimpleNamespace(draft_token_num=w, ragged_verify_layout=layout)
    query_start_loc = (
        torch.arange(0, bs * w + 1, w, dtype=torch.int32, device=device)
        if layout is None
        else layout.qo_indptr_device
    )
    metadata = SimpleNamespace(
        query_start_loc=query_start_loc,
        mamba_cache_indices=cache_indices,
        retrieve_next_token=None,
        retrieve_next_sibling=None,
        retrieve_parent_token=None,
        has_mamba_track_mask=False,
    )
    intermediate = torch.full((1, bs, w, nv, dv, dk), torch.nan, dtype=torch.bfloat16, device=device)
    state = MambaPool.SpeculativeState(
        conv=[initial_conv.clone()],
        temporal=initial_ssm.clone(),
        intermediate_ssm=intermediate,
        intermediate_conv_window=[torch.empty(0, device=device)],
    )
    all_state = MambaPool.SpeculativeState(
        conv=[initial_conv.clone().unsqueeze(0)],
        temporal=initial_ssm.clone().unsqueeze(0),
        intermediate_ssm=intermediate,
        intermediate_conv_window=[torch.empty(0, device=device)],
    )
    backend = object.__new__(AscendGDNAttnBackend)
    backend.graph_mode = False
    backend.req_to_token_pool = _Pool(state, all_state)
    backend.forward_metadata = metadata
    backend.prepare_gdn_inputs(bs, ForwardMode.TARGET_VERIFY, spec_info)
    layer = SimpleNamespace(
        layer_id=0,
        q_dim=nk * dk,
        k_dim=nk * dk,
        v_dim=nv * dv,
        num_q_heads=nk,
        num_k_heads=nk,
        num_v_heads=nv,
        head_q_dim=dk,
        head_k_dim=dk,
        head_v_dim=dv,
        conv_weights=torch.rand(dim, 4, dtype=torch.bfloat16, device=device),
        bias=torch.rand(dim, dtype=torch.bfloat16, device=device),
        A_log=torch.rand(nv, dtype=torch.float32, device=device),
        dt_bias=torch.rand(nv, dtype=torch.float32, device=device),
    )
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        spec_info=spec_info,
        input_ids=torch.zeros(total, dtype=torch.int32, device=device),
        global_num_token_non_padded_cpu=total,
        batch_size=bs,
    )
    output = backend.forward_extend(layer, forward_batch, mixed_qkv.clone(), a.clone(), b.clone())

    hybrid = object.__new__(AscendHybridLinearAttnBackend)
    hybrid.linear_attn_backend = backend
    hybrid.update_mamba_state_after_mtp_verify(
        torch.tensor(accepted_steps, dtype=torch.int32, device=device),
        None,
        None,
        None,
    )
    return output, intermediate, all_state.temporal, all_state.conv[0]


def _inputs(lengths):
    device = "npu:0"
    torch.npu.set_device(device)
    torch.manual_seed(20261027 + sum(lengths))
    total = 8 * len(lengths)
    nk, nv, dk, dv = 4, 8, 128, 128
    dim = 2 * nk * dk + nv * dv
    return (
        torch.rand(3, 3, dim, dtype=torch.bfloat16, device=device),
        torch.rand(3, nv, dv, dk, dtype=torch.bfloat16, device=device),
        torch.rand(total, dim, dtype=torch.bfloat16, device=device),
        torch.rand(total, nv, dtype=torch.bfloat16, device=device),
        torch.rand(total, nv, dtype=torch.bfloat16, device=device),
    )


def _packed_tokens(tensor, lengths):
    pieces = []
    for i, length in enumerate(lengths):
        pieces.append(tensor[i * 8 : i * 8 + length])
    return torch.cat(pieces)


def test_b1_full_forward_and_commit_bitwise():
    lengths = [8, 8]
    args = _inputs(lengths)
    native = _run_forward(lengths, [7, 7], *args)
    packed = _run_forward(lengths, [7, 7], *args, packed=True)
    assert packed[0].shape == native[0].shape
    for actual, expected in zip(packed, native):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_b2_ragged_forward_and_commit_matches_native_prefix():
    lengths = [2, 6]
    args = _inputs(lengths)
    native = _run_forward([8, 8], [1, 5], *args)
    packed_args = (
        args[0],
        args[1],
        _packed_tokens(args[2], lengths),
        _packed_tokens(args[3], lengths),
        _packed_tokens(args[4], lengths),
    )
    packed = _run_forward(lengths, [1, 5], *packed_args, packed=True)
    torch.testing.assert_close(packed[0], _packed_tokens(native[0], lengths), rtol=0, atol=0)
    for i, length in enumerate(lengths):
        torch.testing.assert_close(
            packed[1][0, i, :length], native[1][0, i, :length], rtol=0, atol=0
        )
        assert torch.isnan(packed[1][0, i, length:]).all()
    torch.testing.assert_close(packed[2], native[2], rtol=0, atol=0)
    torch.testing.assert_close(packed[3], native[3], rtol=0, atol=0)
