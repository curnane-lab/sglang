import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.mtp_anchor_ascend import AnchorAscendForward
from sglang.srt.speculative.mtp_append_pack import pack_commit, scatter_commit_mapping


class AscendAnchorTests(unittest.TestCase):
    def test_visible_kv_uses_last_4096_and_current_block(self):
        class Backend:
            page_size = 1
            speculative_num_draft_tokens = 8

            def __init__(self):
                self.req_to_token = torch.arange(10000).reshape(2, 5000)
                self.forward_metadata = SimpleNamespace(
                    block_tables=torch.tensor([[91]]),
                    seq_lens_cpu_int=torch.tensor([92]),
                    seq_lens_cpu_list=[92],
                )

            def forward_mtp(self, *args, **kwargs):
                metadata = self.forward_metadata
                self.seen = (
                    metadata.block_tables.clone(),
                    metadata.seq_lens_cpu_int.clone(),
                    list(metadata.seq_lens_cpu_list),
                )
                return self.seen

        backend = Backend()
        adapter = AnchorAscendForward(backend)
        old_metadata = (
            backend.forward_metadata.block_tables,
            backend.forward_metadata.seq_lens_cpu_int,
            backend.forward_metadata.seq_lens_cpu_list,
        )
        blocks = torch.tensor([[8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007],
                               [9000, 9001, 9002, 9003, 9004, 9005, 9006, 9007]])
        batch = SimpleNamespace(
            batch_size=2,
            seq_lens=torch.tensor([5000, 3]),
            req_pool_indices=torch.tensor([1, 0]),
            out_cache_loc=blocks,
            forward_mode=SimpleNamespace(is_target_verify=lambda: True),
        )
        table, device_lengths, host_lengths = adapter(
            torch.zeros(16, 1), None, None, None, batch, q_descale=torch.ones(1)
        )
        self.assertEqual(table.shape, (2, 4104))
        self.assertEqual(device_lengths.tolist(), [4104, 11])
        self.assertEqual(host_lengths, [4104, 11])
        torch.testing.assert_close(table[0, :4096], backend.req_to_token[1, 904:].int())
        torch.testing.assert_close(table[0, 4096:], blocks[0].int())
        torch.testing.assert_close(table[1, :3], backend.req_to_token[0, :3].int())
        torch.testing.assert_close(table[1, 3:11], blocks[1].int())
        self.assertEqual(torch.count_nonzero(table[1, 11:]).item(), 0)
        self.assertIs(backend.forward_metadata.block_tables, old_metadata[0])
        self.assertIs(backend.forward_metadata.seq_lens_cpu_int, old_metadata[1])
        self.assertIs(backend.forward_metadata.seq_lens_cpu_list, old_metadata[2])

        backend.page_size = 16
        adapter = AnchorAscendForward(backend)
        table, device_lengths, host_lengths = adapter(
            torch.zeros(16, 1), None, None, None, batch, q_descale=torch.ones(1)
        )
        self.assertEqual(table.shape, (2, 258))
        self.assertEqual(device_lengths.tolist(), [4104, 11])
        self.assertEqual(host_lengths, [4104, 11])
        torch.testing.assert_close(table[0, :256], torch.arange(369, 625).int())
        self.assertEqual(table[0, 256].item(), 500)
        self.assertEqual(table[1, :2].tolist(), [0, 562])
        self.assertEqual(torch.count_nonzero(table[0, 257:]).item(), 0)
        self.assertEqual(torch.count_nonzero(table[1, 2:]).item(), 0)

    def test_page16_unaligned_anchor_window_has_capacity(self):
        backend = SimpleNamespace(
            page_size=16,
            speculative_num_draft_tokens=8,
            forward_mtp=None,
            req_to_token=torch.arange(5000).reshape(1, 5000),
        )
        adapter = AnchorAscendForward(backend)
        batch = SimpleNamespace(
            batch_size=1,
            seq_lens=torch.tensor([5000]),
            req_pool_indices=torch.tensor([0]),
            out_cache_loc=torch.arange(8000, 8008),
        )
        table, lengths = adapter._visible_kv(batch)
        self.assertEqual(table.shape, (1, 258))
        self.assertEqual(lengths, [4104])
        torch.testing.assert_close(table[0, :257], torch.arange(56, 313).int())
        self.assertEqual(table[0, 257].item(), 500)

    def test_controller_does_not_materialize_dense_mask(self):
        from sglang.srt.speculative.mtp_first_controller import MTPFirstController

        backend = SimpleNamespace(mtp_anchor_ascend=object())
        controller = MTPFirstController.__new__(MTPFirstController)
        controller.worker = SimpleNamespace(
            draft_model_runner=SimpleNamespace(attn_backend=backend)
        )
        self.assertIsNone(controller.mask(object()))


class EagerCommitPackTests(unittest.TestCase):
    def test_ragged_acceptance_and_private_mapping(self):
        candidates = torch.arange(16).reshape(2, 8)
        bonus = torch.tensor([90, 91])
        hidden = torch.arange(64).reshape(2, 8, 4)
        locations = torch.tensor(
            [[10, 11, 12, 13, 14, 15, 16, 17], [20, 21, 22, 23, 24, 25, 26, 27]]
        )
        reqs = torch.tensor([1, 0])
        packed = pack_commit(
            candidates, bonus, hidden, locations, reqs, [1, 3], [2, 5], 100
        )
        self.assertEqual(packed.tokens.tolist(), [90, 9, 10, 91])
        self.assertEqual(packed.cache_locations.tolist(), [10, 20, 21, 22])
        self.assertEqual(packed.native_locations.tolist(), [101, 21, 22, 100])
        self.assertEqual(packed.positions.tolist(), [3, 6, 7, 8])
        self.assertEqual(packed.last_rows.tolist(), [0, 3])
        self.assertEqual(packed.anchor_locations.tolist(), [10, 20])
        self.assertEqual(packed.pending_locations.tolist(), [101, 100])
        torch.testing.assert_close(
            packed.hidden, torch.cat((hidden[0, :1], hidden[1, :3]))
        )
        mapping = torch.full((2, 16), -1)
        scatter_commit_mapping(reqs, packed, mapping)
        self.assertEqual(mapping[1, 1].item(), 10)
        self.assertEqual(mapping[1, 2].item(), 101)
        self.assertEqual(mapping[0, 4].item(), 20)
        self.assertEqual(mapping[0, 5:8].tolist(), [21, 22, 100])


class EagerRunnerTests(unittest.TestCase):
    def test_worker_accepts_eager_runner(self):
        from contextlib import nullcontext
        from unittest.mock import patch

        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
        from sglang.srt.speculative.dmtp_worker import DMTPWorkerV2

        calls = []
        controller = SimpleNamespace(
            two_graphs=False,
            draft_slots=8,
            native=SimpleNamespace(
                init_cuda_graphs=lambda: calls.append("native")
            ),
            init_graphs=lambda: calls.append("selectors"),
        )
        worker = DMTPWorkerV2.__new__(DMTPWorkerV2)
        worker._mtp_first = controller
        worker.draft_model_runner = SimpleNamespace(
            decode_cuda_graph_runner=object(), attn_backend=SimpleNamespace()
        )
        worker._draft_block_spec_info = SimpleNamespace(
            draft_token_num=8,
            num_tokens_per_req=8,
            num_tokens_for_logprob_per_req=8,
        )
        graph_config = SimpleNamespace(decode=SimpleNamespace(bs=[]))
        with (
            patch(
                "sglang.srt.speculative.dmtp_worker.get_exec",
                return_value=SimpleNamespace(
                    graph=SimpleNamespace(cuda_graph_config=graph_config)
                ),
            ),
            patch(
                "sglang.srt.speculative.dmtp_worker.get_schedule",
                return_value=SimpleNamespace(max_running_requests=1),
            ),
            patch(
                "sglang.srt.speculative.dmtp_worker.scoped_config",
                return_value=nullcontext(),
            ),
            patch.object(DFlashWorkerV2, "init_cuda_graphs", return_value=None),
        ):
            worker.init_cuda_graphs()
        self.assertEqual(calls, ["native", "selectors"])

    def test_eager_runner_does_not_capture_selectors(self):
        from unittest.mock import patch

        from sglang.srt.speculative.mtp_first_controller import MTPFirstController

        enabled = []

        class Selector:
            def __init__(self, controller, *, t1, enable_graph):
                enabled.append(enable_graph)

        controller = MTPFirstController.__new__(MTPFirstController)
        controller.native = SimpleNamespace(bridge=object())
        controller.two_graphs = False
        controller.worker = SimpleNamespace(
            draft_model_runner=SimpleNamespace(decode_cuda_graph_runner=object())
        )
        with patch(
            "sglang.srt.speculative.dmtp_selector_graph.DMTPSelectorGraphs", Selector
        ):
            controller.init_graphs()
        self.assertEqual(enabled, [False, False])


class NativeKVLayoutTests(unittest.TestCase):
    def test_page16_token_rows_and_hidden_cache_capacity(self):
        from sglang.srt.speculative.mtp_first_controller import MTPFirstController
        from sglang.srt.speculative.native_sglang_mtp import _kv_token_rows

        keys = torch.arange(65568).reshape(4098, 16, 1)
        values = keys + 100000
        pool = SimpleNamespace(
            start_layer=0,
            get_kv_buffer=lambda layer: (keys, values),
            get_key_buffer=lambda layer: keys,
        )
        flat_keys, flat_values = _kv_token_rows(pool, 16)
        self.assertEqual(flat_keys.shape, (65568, 1))
        self.assertEqual(flat_values.shape, (65568, 1))
        pending = torch.tensor([65553])
        anchors = torch.tensor([20])
        flat_keys.index_copy_(0, anchors, flat_keys[pending])
        self.assertEqual(keys.view(-1)[20].item(), 65553)

        controller = MTPFirstController.__new__(MTPFirstController)
        controller.final_buffer = torch.empty(4, 1)
        controller.native = SimpleNamespace(
            worker=SimpleNamespace(
                model_runner=SimpleNamespace(token_to_kv_pool=pool)
            )
        )
        controller.worker = SimpleNamespace(
            server_args=SimpleNamespace(page_size=16)
        )
        controller.alloc_hidden_cache()
        self.assertEqual(controller.target_hidden_cache.shape, (65568, 1))


class PageRecipeTests(unittest.TestCase):
    def test_ascend_uses_page16_and_fa3_keeps_page1(self):
        from sglang.srt.runtime_context import get_context
        from sglang.srt.speculative.dmtp_runtime import scoped_config, validate_recipe

        fields = dict(
            page_size=16,
            quantization=None,
            speculative_num_draft_tokens=8,
            speculative_draft_attention_backend="ascend",
            max_running_requests=1,
            enable_lora=False,
            enable_two_batch_overlap=False,
            enable_unified_memory=False,
            speculative_dflash_draft_kv_ring=False,
        )
        with get_context().override_server_args(**fields):
            validate_recipe()
            with scoped_config(page_size=1):
                with self.assertRaises(ValueError):
                    validate_recipe()
            with scoped_config(speculative_draft_attention_backend="fa3", page_size=1):
                validate_recipe()
            with scoped_config(speculative_draft_attention_backend="fa3", page_size=16):
                with self.assertRaises(ValueError):
                    validate_recipe()


if __name__ == "__main__":
    unittest.main()
