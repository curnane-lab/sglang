from types import SimpleNamespace

import torch

from sglang.srt.hardware_backend.npu.attention.ascend_hybrid_linear_attn_backend import (
    AscendMambaAttnBackendBase,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout


class _Pool:
    def get_mamba_indices(self, req_pool_indices):
        return req_pool_indices.clone()


def _backend(bs):
    backend = object.__new__(AscendMambaAttnBackendBase)
    backend.device = torch.device("cpu")
    backend.pad_slot_id = -1
    backend.req_to_token_pool = _Pool()
    backend.state_indices_list = [torch.full((i + 1,), -1, dtype=torch.int32) for i in range(bs)]
    backend.state_indices_list_gdn = [torch.full(((i + 1) * 8,), -1, dtype=torch.int32) for i in range(bs)]
    backend.query_start_loc_list = [torch.zeros(i + 2, dtype=torch.int32) for i in range(bs)]
    backend.cached_cuda_graph_verify_query_start_loc = torch.arange(
        0, bs * 8 + 1, 8, dtype=torch.int32
    )
    backend.cached_cuda_graph_decode_query_start_loc = torch.arange(bs + 1, dtype=torch.int32)
    return backend


def test_capture_uses_ragged_qo_indptr():
    backend = _backend(2)
    layout = RaggedVerifyLayout.from_verify_lens(
        verify_lens_cpu=[2, 6], device=torch.device("cpu"), grid=[8]
    )
    spec_info = SimpleNamespace(draft_token_num=8, ragged_verify_layout=layout, topk=1)
    metadata = backend._capture_metadata(
        2, torch.tensor([3, 1], dtype=torch.int32), ForwardMode.TARGET_VERIFY, spec_info
    )
    torch.testing.assert_close(metadata.query_start_loc, layout.qo_indptr_device)
    torch.testing.assert_close(metadata.mamba_cache_indices, torch.tensor([3, 1], dtype=torch.int32))


def test_replay_pads_ragged_layout_and_uses_negative_state_sentinel():
    backend = _backend(4)
    layout = RaggedVerifyLayout.from_verify_lens(
        verify_lens_cpu=[2, 6], device=torch.device("cpu"), grid=[8]
    )
    spec_info = SimpleNamespace(draft_token_num=8, ragged_verify_layout=layout, topk=1)
    metadata = backend._replay_metadata(
        4,
        torch.tensor([3, 1, 0, 0], dtype=torch.int32),
        ForwardMode.TARGET_VERIFY,
        spec_info,
        torch.tensor([10, 10, 0, 0], dtype=torch.int32),
    )
    expected = layout.padded_to_bucket(padded_bs=4, cap=8)
    torch.testing.assert_close(metadata.query_start_loc, expected.qo_indptr_device)
    torch.testing.assert_close(
        metadata.mamba_cache_indices,
        torch.tensor([3, 1, -1, -1], dtype=torch.int32),
    )
    torch.testing.assert_close(
        metadata.mamba_cache_indices_gdn,
        torch.arange(32, dtype=torch.int32),
    )


def test_replay_zeroes_lengths_for_negative_cache_padding():
    backend = _backend(4)
    layout = RaggedVerifyLayout.from_verify_lens(
        verify_lens_cpu=[8, 8, 8, 8], device=torch.device("cpu"), grid=[32]
    )
    spec_info = SimpleNamespace(draft_token_num=8, ragged_verify_layout=layout, topk=1)
    metadata = backend._replay_metadata(
        4,
        torch.tensor([3, 1, 0, 0], dtype=torch.int32),
        ForwardMode.TARGET_VERIFY,
        spec_info,
        torch.tensor([10, 10, 0, 0], dtype=torch.int32),
    )
    torch.testing.assert_close(
        metadata.query_start_loc,
        torch.tensor([0, 8, 16, 16, 16], dtype=torch.int32),
    )
    torch.testing.assert_close(
        metadata.mamba_cache_indices,
        torch.tensor([3, 1, -1, -1], dtype=torch.int32),
    )


def test_replay_uniform_keeps_static_metadata():
    backend = _backend(2)
    spec_info = SimpleNamespace(draft_token_num=8, ragged_verify_layout=None, topk=1)
    metadata = backend._replay_metadata(
        2,
        torch.tensor([3, 1], dtype=torch.int32),
        ForwardMode.TARGET_VERIFY,
        spec_info,
        torch.tensor([10, 10], dtype=torch.int32),
    )
    torch.testing.assert_close(
        metadata.query_start_loc,
        torch.tensor([0, 8, 16], dtype=torch.int32),
    )
    torch.testing.assert_close(metadata.mamba_cache_indices, torch.tensor([3, 1], dtype=torch.int32))
