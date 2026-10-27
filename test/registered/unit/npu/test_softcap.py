import torch

from sglang.kernels.ops.activation.softcap import (
    softcap_inplace_logits,
    softcap_to_float32_logits,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=20, suite="base-a-test-npu")


def test_softcap_matches_torch_for_one_dim_and_contiguous_rank_three():
    for shape in ((4096,), (2, 3, 1024)):
        logits = torch.randn(shape, device="npu", dtype=torch.bfloat16)
        expected = (20.0 * torch.tanh(logits.float() / 20.0)).to(logits.dtype)
        softcap_inplace_logits(logits, 20.0)
        torch.testing.assert_close(logits, expected)


def test_large_verify_logits_respect_npu_grid_limit():
    logits = torch.randn((512, 248320), device="npu", dtype=torch.bfloat16)
    output = torch.empty_like(logits, dtype=torch.float32)
    softcap_to_float32_logits(logits, 30.0, output)
    for row, col in ((0, 0), (1, 1023), (17, 1024), (255, 123456), (511, 248319)):
        expected = 30.0 * torch.tanh(logits[row, col].float() / 30.0)
        torch.testing.assert_close(output[row, col], expected)
