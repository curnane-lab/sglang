"""Unit tests for DFlash mRoPE (VLM) position handling on the draft path.

Covers the pure-tensor helpers added for VL-input DFlash serving:
``_request_mrope_delta_tensor`` (per-request mRoPE deltas) and
``_maybe_build_mrope_ctx_positions`` (true mRoPE rows for prefill context
materialization). Text drafts must observe no behavior change.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


def _fake_self(*, uses_mrope: bool):
    return SimpleNamespace(
        _draft_uses_mrope=uses_mrope,
        device=torch.device("cpu"),
    )


def _mm(mrope_positions=None, delta=None):
    return SimpleNamespace(
        mrope_positions=mrope_positions,
        mrope_position_delta=delta,
    )


class RequestMropeDeltaTensorTest(CustomTestCase):
    def test_text_only_batch_yields_zero_deltas(self):
        batch = SimpleNamespace(multimodal_inputs=[None, None])
        out = DFlashWorkerV2._request_mrope_delta_tensor(
            _fake_self(uses_mrope=True), batch, 2, torch.device("cpu")
        )
        self.assertEqual(out.shape, (2, 1))
        self.assertEqual(out.dtype, torch.int64)
        self.assertTrue((out == 0).all())

    def test_request_delta_is_used_and_broadcastable(self):
        batch = SimpleNamespace(
            multimodal_inputs=[
                None,
                _mm(delta=torch.tensor([[7]], dtype=torch.int64)),
            ]
        )
        out = DFlashWorkerV2._request_mrope_delta_tensor(
            _fake_self(uses_mrope=True), batch, 2, torch.device("cpu")
        )
        self.assertEqual(out.view(-1).tolist(), [0, 7])


class MaybeBuildMropeCtxPositionsTest(CustomTestCase):
    def test_non_mrope_draft_returns_fallback_unchanged(self):
        fallback = torch.arange(6, dtype=torch.int64)
        batch = SimpleNamespace(multimodal_inputs=[None])
        out = DFlashWorkerV2._maybe_build_mrope_ctx_positions(
            _fake_self(uses_mrope=False),
            batch=batch,
            prefix_lens=torch.tensor([0], dtype=torch.int32),
            extend_lens=torch.tensor([6], dtype=torch.int32),
            fallback_positions=fallback,
        )
        self.assertIs(out, fallback)

    def test_text_only_batch_expands_to_three_identical_rows(self):
        fallback = torch.arange(5, dtype=torch.int64) + 3
        batch = SimpleNamespace(multimodal_inputs=[None, None])
        out = DFlashWorkerV2._maybe_build_mrope_ctx_positions(
            _fake_self(uses_mrope=True),
            batch=batch,
            prefix_lens=torch.tensor([0, 0], dtype=torch.int32),
            extend_lens=torch.tensor([2, 3], dtype=torch.int32),
            fallback_positions=fallback,
        )
        self.assertEqual(out.shape, (3, 5))
        for row in range(3):
            self.assertEqual(out[row].tolist(), fallback.tolist())

    def test_image_request_contributes_sliced_true_positions(self):
        true_rows = torch.arange(30, dtype=torch.int64).reshape(3, 10) + 100
        batch = SimpleNamespace(
            multimodal_inputs=[
                _mm(mrope_positions=true_rows),
                None,
            ]
        )
        fallback = torch.cat([torch.arange(2, 6), torch.arange(0, 3)]).to(torch.int64)
        out = DFlashWorkerV2._maybe_build_mrope_ctx_positions(
            _fake_self(uses_mrope=True),
            batch=batch,
            prefix_lens=torch.tensor([2, 0], dtype=torch.int32),
            extend_lens=torch.tensor([4, 3], dtype=torch.int32),
            fallback_positions=fallback,
        )
        self.assertEqual(out.shape, (3, 7))
        self.assertEqual(out[:, 0].tolist(), true_rows[:, 2].tolist())
        self.assertEqual(out[:, 3].tolist(), true_rows[:, 5].tolist())
        # Text-only second request gets flat arange on all three rows.
        for row in range(3):
            self.assertEqual(out[row, 4:].tolist(), [0, 1, 2])

    def test_short_mrope_positions_raise_loudly(self):
        batch = SimpleNamespace(
            multimodal_inputs=[
                _mm(mrope_positions=torch.zeros(3, 4, dtype=torch.int64))
            ]
        )
        with self.assertRaises(RuntimeError):
            DFlashWorkerV2._maybe_build_mrope_ctx_positions(
                _fake_self(uses_mrope=True),
                batch=batch,
                prefix_lens=torch.tensor([2], dtype=torch.int32),
                extend_lens=torch.tensor([4], dtype=torch.int32),
                fallback_positions=torch.arange(4, dtype=torch.int64),
            )


if __name__ == "__main__":
    unittest.main()
