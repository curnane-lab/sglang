from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
    AscendAttnBackend,
    ForwardMetadata,
    _bounded_full_coverage,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.srt.speculative.spec_info import SpecInputType


def test_graph_apply_updates_q_boundaries_and_dflash_kv_lengths():
    backend = object.__new__(AscendAttnBackend)
    metadata = ForwardMetadata(
        block_tables=torch.zeros(2, 4, dtype=torch.int32),
        seq_lens=torch.zeros(2, dtype=torch.int32),
        actual_seq_lengths_q=torch.tensor([8, 16], dtype=torch.int32),
    )
    backend.graph_metadata = {2: metadata}
    backend.use_sliding_window_kv_pool = False
    backend.is_hybrid_swa = False
    backend.use_mla = False
    backend.req_to_token = torch.zeros(4, 512, dtype=torch.int32)
    backend.page_size = 128
    backend.speculative_num_draft_tokens = 8
    layout = RaggedVerifyLayout.from_verify_lens(
        verify_lens_cpu=[2, 6], device=torch.device("cpu"), grid=[16]
    )
    spec_info = SimpleNamespace(
        draft_token_num=8,
        ragged_verify_layout=layout,
        spec_input_type=SpecInputType.DFLASH_VERIFY,
    )
    with mock.patch(
        "sglang.srt.hardware_backend.npu.attention.ascend_backend.get_parallel",
        return_value=SimpleNamespace(dcp_enabled=False),
    ):
        backend._apply_cuda_graph_metadata(
            2,
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([10, 12], dtype=torch.int32),
            torch.tensor([12, 18], dtype=torch.int32),
            ForwardMode.TARGET_VERIFY,
            spec_info,
        )
    torch.testing.assert_close(
        metadata.actual_seq_lengths_q,
        torch.tensor([2, 16], dtype=torch.int32),
    )
    assert metadata.seq_lens_cpu_list == [12, 26]


def test_full_coverage_uses_host_lengths_preserved_before_bucket_padding():
    layout = RaggedVerifyLayout.from_verify_lens(
        verify_lens_cpu=[4] * 16, device=torch.device("cpu"), grid=[64]
    ).padded_to_bucket(padded_bs=64, cap=8)
    assert layout.verify_lens_cpu == [4] * 16 + [0] * 48
    coverage, extras = _bounded_full_coverage(layout, 64, 8)
    assert len(coverage) == 64
    assert sum(coverage) == 64
    assert coverage[:16] == [4] * 16
    assert coverage[16:] == [0] * 48
    assert extras == [0] * 64


def test_full_coverage_assigns_tier_slack_to_padding_not_real_requests():
    layout = RaggedVerifyLayout.from_verify_lens(
        verify_lens_cpu=[2, 6], device=torch.device("cpu"), grid=[16]
    )
    coverage, extras = _bounded_full_coverage(layout, 4, 8)
    assert coverage == [2, 6, 4, 4]
    assert extras == [0, 0, 4, 4]
    assert torch.cumsum(torch.tensor(coverage), dim=0).tolist() == [2, 8, 12, 16]
