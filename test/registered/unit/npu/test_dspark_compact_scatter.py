import torch
import torch_npu

from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
    scatter_compact_to_strided,
    scatter_compact_to_strided_into,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=2, suite="stage-a-unit-test-npu")


def test_scatter_compact_to_strided_into_matches_reference_with_ghost_slots():
    torch.manual_seed(20261028)
    device = "npu:0"
    layout = RaggedVerifyLayout.from_verify_lens(
        verify_lens_cpu=[4] * 16, device=torch.device(device), grid=[64]
    ).padded_to_bucket(padded_bs=64, cap=8)
    compact = torch.randn(64, 32, dtype=torch.bfloat16, device=device)
    out = torch.empty(64 * 8, 32, dtype=torch.bfloat16, device=device)

    scatter_compact_to_strided_into(
        compact=compact,
        verify_lens=layout.verify_lens,
        out=out,
        stride=8,
        fill_value=0.0,
    )
    expected = scatter_compact_to_strided(
        compact=compact,
        layout=layout,
        fill_value=0.0,
        verify_num_draft_tokens=8,
    )

    torch.testing.assert_close(out, expected, rtol=0, atol=0)
