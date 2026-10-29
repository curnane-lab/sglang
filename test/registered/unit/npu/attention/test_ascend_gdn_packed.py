from types import SimpleNamespace
from unittest import mock

import sgl_kernel_npu
import torch
import torch_npu

from sglang.srt.hardware_backend.npu.attention.ascend_gdn_backend import (
    AscendGDNAttnBackend,
)


def _backend(qo_indptr, lengths, ssm_indices, cache_indices):
    backend = object.__new__(AscendGDNAttnBackend)
    backend.graph_mode = False
    backend.num_accept_tokens = torch.ones(
        len(lengths), dtype=torch.int32, device=qo_indptr.device
    )
    backend.actual_seq_lengths = torch.tensor(
        lengths, dtype=torch.int32, device=qo_indptr.device
    )
    backend.ssm_state_indices = ssm_indices
    backend.forward_metadata = SimpleNamespace(
        query_start_loc=qo_indptr,
        mamba_cache_indices_gdn=ssm_indices.reshape(-1),
    )
    return backend


def _call(backend, mix_qkv, recurrent, beta, g, cache_indices, intermediate):
    return backend.fused_recurrent_gated_delta_rule_update(
        mix_qkv,
        num_heads=4,
        num_value_heads=8,
        head_k_dim=128,
        head_v_dim=128,
        recurrent_state=recurrent,
        beta=beta,
        g=g,
        cache_indices=cache_indices,
        intermediate_state=intermediate,
    )


def _run_backend_packed_vs_dense(lengths, cache_slots):
    torch.manual_seed(20261027 + sum(lengths))
    device = "npu:0"
    torch.npu.set_device(device)
    b, w = len(lengths), 8
    nk, nv, dk, dv = 4, 8, 128, 128
    total = sum(lengths)
    q = torch.rand(total, nk, dk, dtype=torch.bfloat16, device=device)
    k = torch.rand_like(q)
    v = torch.rand(total, nv, dv, dtype=torch.bfloat16, device=device)
    packed_mix = torch.cat([q.flatten(1), k.flatten(1), v.flatten(1)], dim=-1).contiguous()
    beta = torch.rand(total, nv, dtype=torch.bfloat16, device=device)
    g = -torch.rand(total, nv, dtype=torch.float32, device=device)
    recurrent = torch.rand(max(cache_slots) + 1, nv, dv, dk, dtype=torch.bfloat16, device=device)
    recurrent_reference = recurrent.clone()
    cache_indices = torch.tensor(cache_slots, dtype=torch.int32, device=device)
    qo_indptr = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)
    ssm_indices = torch.arange(b * w, dtype=torch.int32, device=device).view(b, w)

    def make_intermediate():
        intermediate = torch.full((b, w, nv, dv, dk), torch.nan, dtype=torch.bfloat16, device=device)
        for i, slot in enumerate(cache_slots):
            intermediate[i, 0] = recurrent[slot]
        return intermediate

    packed_intermediate = make_intermediate()
    packed_backend = _backend(qo_indptr, lengths, ssm_indices, cache_indices)
    packed_output = _call(
        packed_backend,
        packed_mix,
        recurrent,
        beta,
        g,
        cache_indices,
        packed_intermediate,
    )

    reference_intermediate = make_intermediate()
    reference_outputs = []
    token_base = 0
    for i, length in enumerate(lengths):
        token_end = token_base + length
        request_cache = cache_indices[i : i + 1]
        request_indices = (i * w + torch.arange(length, dtype=torch.int32, device=device)).view(1, length)
        reference_outputs.append(
            torch.ops.npu.recurrent_gated_delta_rule(
                packed_mix[token_base:token_end].unsqueeze(0),
                recurrent_reference,
                beta=beta[token_base:token_end].unsqueeze(0),
                scale=dk**-0.5,
                actual_seq_lengths=torch.tensor([length], dtype=torch.int32, device=device),
                ssm_state_indices=request_indices,
                nk=nk,
                nv=nv,
                intermediate_state=reference_intermediate,
                cache_indices=request_cache,
                num_accepted_tokens=torch.ones(1, dtype=torch.int32, device=device),
                g=g[token_base:token_end].unsqueeze(0),
            )[0]
        )
        token_base = token_end

    torch.testing.assert_close(packed_output, torch.cat(reference_outputs), rtol=0, atol=0)
    torch.testing.assert_close(packed_intermediate, reference_intermediate, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(recurrent, recurrent_reference, rtol=0, atol=0)


def test_b1_uniform_packed_matches_native_dense():
    _run_backend_packed_vs_dense([8, 8], [2, 0])


def test_b2_ragged_packed_matches_native_dense_per_request():
    _run_backend_packed_vs_dense([2, 6], [2, 0])
    _run_backend_packed_vs_dense([1, 4, 8], [2, 0, 1])


def test_packed_graph_uses_staged_metadata():
    device = "npu:0"
    torch.npu.set_device(device)
    backend = _backend(
        torch.tensor([0, 1], dtype=torch.int32, device=device),
        [1],
        torch.zeros(1, 8, dtype=torch.int32, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
    )
    backend.graph_mode = True
    output = _call(
        backend,
        torch.zeros(1, 4 * 128 * 2 + 8 * 128, dtype=torch.bfloat16, device=device),
        torch.zeros(1, 8, 128, 128, dtype=torch.bfloat16, device=device),
        torch.zeros(1, 8, dtype=torch.bfloat16, device=device),
        torch.zeros(1, 8, dtype=torch.float32, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.zeros(1, 8, 8, 128, 128, dtype=torch.bfloat16, device=device),
    )
    assert output.shape == (1, 8, 128)
    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)


def test_packed_graph_reuses_persistent_accept_tensor():
    device = "npu:0"
    torch.npu.set_device(device)
    backend = _backend(
        torch.tensor([0, 1], dtype=torch.int32, device=device),
        [1],
        torch.zeros(1, 8, dtype=torch.int32, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
    )
    backend.graph_mode = True
    with mock.patch(
        "sglang.srt.hardware_backend.npu.attention.ascend_gdn_backend.torch.ops.npu.recurrent_gated_delta_rule_packed",
        return_value=torch.zeros(1, 8, 128, device=device),
    ) as packed:
        _call(
            backend,
            torch.zeros(1, 4 * 128 * 2 + 8 * 128, dtype=torch.bfloat16, device=device),
            torch.zeros(1, 8, 128, 128, dtype=torch.bfloat16, device=device),
            torch.zeros(1, 8, dtype=torch.bfloat16, device=device),
            torch.zeros(1, 8, dtype=torch.float32, device=device),
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.zeros(1, 8, 8, 128, 128, dtype=torch.bfloat16, device=device),
        )
    assert packed.call_args.kwargs["num_accepted_tokens"] is backend.num_accept_tokens
    assert (
        packed.call_args.kwargs["ssm_state_indices"].data_ptr()
        == backend.forward_metadata.mamba_cache_indices_gdn.data_ptr()
    )


def test_prepare_reuses_accept_tensor_across_graph_replays():
    device = "npu:0"
    backend = object.__new__(AscendGDNAttnBackend)
    backend._num_accept_tokens_by_bs = {}
    backend.forward_metadata = SimpleNamespace(
        mamba_cache_indices=torch.zeros(2, dtype=torch.int32, device=device),
        query_start_loc=torch.tensor([0, 8, 16], dtype=torch.int32, device=device),
    )
    forward_mode = mock.Mock()
    forward_mode.is_target_verify.return_value = True
    forward_mode.is_prefill.return_value = False
    spec_info = SimpleNamespace(draft_token_num=8, ragged_verify_layout=None)

    backend.prepare_gdn_inputs(2, forward_mode, spec_info)
    captured = backend.num_accept_tokens
    backend.prepare_gdn_inputs(2, forward_mode, spec_info)

    assert backend.num_accept_tokens is captured
