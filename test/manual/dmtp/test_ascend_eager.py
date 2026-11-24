import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.mtp_anchor_ascend import (
    AnchorAscendForward,
    AscendDMTPAttention,
)
from sglang.srt.speculative.mtp_append_pack import pack_commit, scatter_commit_mapping


def run_gathered_attention(*, draft_block, batch, mapping, values, num_queries):
    values = values.reshape(-1, 1, 1)
    pool = SimpleNamespace(
        get_key_buffer=lambda _: torch.zeros_like(values),
        get_value_buffer=lambda _: values,
    )
    backend = SimpleNamespace(
        token_to_kv_pool=pool,
        req_to_token_pool=SimpleNamespace(req_to_token=mapping),
        forward_metadata=SimpleNamespace(swa_out_cache_loc=None),
        runner_ids=None,
    )
    layer = SimpleNamespace(
        layer_id=0,
        tp_q_head_num=2,
        tp_k_head_num=1,
        tp_v_head_num=1,
        qk_head_dim=1,
        v_head_dim=1,
        scaling=1.0,
    )
    attention = AscendDMTPAttention(backend, draft_block=draft_block)
    return attention(
        torch.zeros(num_queries, 2), None, None, layer, batch, save_kv_cache=False
    )


class AscendAnchorTests(unittest.TestCase):
    def test_visible_slots_keep_unaligned_window_and_all_block_addresses(self):
        mapping = torch.arange(24000).reshape(4, 6000) + 10000
        backend = SimpleNamespace(
            page_size=16,
            speculative_num_draft_tokens=8,
            forward_mtp=None,
            req_to_token_pool=SimpleNamespace(req_to_token=mapping),
        )
        adapter = AnchorAscendForward(backend)
        prefixes = [4097, 4105, 5000, 3]
        blocks = torch.tensor(
            [
                [8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007],
                [9000, 9001, 9002, 9003, 9004, 9005, 9006, 9007],
                [10000, 10001, 10002, 10003, 10004, 10005, 10006, 10007],
                [11000, 11001, 11002, 11003, 11004, 11005, 11006, 11007],
            ]
        )
        batch = SimpleNamespace(
            batch_size=4,
            seq_lens=torch.tensor(prefixes),
            req_pool_indices=torch.tensor([0, 1, 2, 3]),
            out_cache_loc=blocks,
        )
        visible = adapter._visible_slots(batch)
        for req, prefix in enumerate(prefixes):
            expected = torch.cat(
                (mapping[req, max(0, prefix - 4096) : prefix], blocks[req])
            )
            torch.testing.assert_close(visible[req], expected)
            torch.testing.assert_close(visible[req][-8:], blocks[req])

    def test_draft_gather_reads_all_current_block_rows(self):
        prefix = 4105
        mapping = torch.arange(4113).reshape(1, -1)
        blocks = torch.arange(prefix, prefix + 8)
        batch = SimpleNamespace(
            batch_size=1,
            seq_lens=torch.tensor([prefix]),
            req_pool_indices=torch.tensor([0]),
            out_cache_loc=blocks,
        )
        values = torch.full((4113,), 50000.0)
        values[torch.cat((mapping[0, prefix - 4096 : prefix], blocks))] = 1.0
        out = run_gathered_attention(
            draft_block=True,
            batch=batch,
            mapping=mapping,
            values=values,
            num_queries=8,
        )
        torch.testing.assert_close(out, torch.ones(8, 2))

    def test_native_mtp_reads_shifted_mapping_with_causal_mask(self):
        batch = SimpleNamespace(
            req_pool_indices=torch.tensor([1, 0]),
            extend_prefix_lens_cpu=[2, 1],
            extend_seq_lens_cpu=[2, 1],
        )
        out = run_gathered_attention(
            draft_block=False,
            batch=batch,
            mapping=torch.tensor([[8, 2, 0, 0], [6, 0, 3, 7]]),
            values=torch.arange(10, dtype=torch.float32),
            num_queries=3,
        )
        torch.testing.assert_close(
            out, torch.tensor([3.0, 4.0, 5.0])[:, None].expand(-1, 2)
        )

    def test_ascend_rejects_non_greedy_requests(self):
        from sglang.srt.speculative.mtp_first_controller import MTPFirstController

        controller = MTPFirstController.__new__(MTPFirstController)
        controller.worker = SimpleNamespace(
            server_args=SimpleNamespace(speculative_draft_attention_backend="ascend")
        )
        with self.assertRaisesRegex(ValueError, "temperature=0"):
            controller.validate_sampling(
                SimpleNamespace(sampling_info=SimpleNamespace(is_all_greedy=False))
            )

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

    def test_fia_token_rows_do_not_multiply_hidden_cache_capacity(self):
        from sglang.srt.speculative.mtp_first_controller import MTPFirstController

        keys = torch.empty(64, 1, 1, 1)
        pool = SimpleNamespace(
            start_layer=0,
            get_kv_buffer=lambda layer: (keys, torch.empty_like(keys)),
        )
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
        self.assertEqual(controller.target_hidden_cache.shape, (64, 1))

    def test_gathered_write_preserves_physical_window_locations(self):
        values = torch.zeros(16, 1, 1)
        writes = []
        pool = SimpleNamespace(
            get_key_buffer=lambda _: torch.zeros_like(values),
            get_value_buffer=lambda _: values,
            set_kv_buffer=lambda *args: writes.append(args),
        )
        backend = SimpleNamespace(
            token_to_kv_pool=pool,
            req_to_token_pool=SimpleNamespace(req_to_token=torch.tensor([[3]])),
            forward_metadata=SimpleNamespace(
                swa_out_cache_loc=torch.tensor([3])
            ),
            runner_ids=SimpleNamespace(window_storage="window"),
        )
        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=2,
            tp_k_head_num=1,
            tp_v_head_num=1,
            qk_head_dim=1,
            v_head_dim=1,
            scaling=1.0,
        )
        batch = SimpleNamespace(
            req_pool_indices=torch.tensor([0]),
            extend_prefix_lens_cpu=[0],
            extend_seq_lens_cpu=[1],
            out_cache_loc=torch.tensor([3]),
            out_cache_loc_is_physical=True,
        )
        attention = AscendDMTPAttention(backend, draft_block=False)
        attention(
            torch.zeros(1, 2),
            torch.zeros(1, 1),
            torch.zeros(1, 1),
            layer,
            batch,
        )
        _, write, _, _ = writes[0]
        self.assertTrue(write.physical)
        self.assertEqual(write.loc.tolist(), [3])
        self.assertEqual(write.window_locs["window"].tolist(), [3])


class RequestAdmissionTests(unittest.TestCase):
    @staticmethod
    def make_request():
        return SimpleNamespace(
            return_hidden_states=False,
            return_logprob=False,
            grammar=None,
            sampling_params=SimpleNamespace(
                top_k=1,
                min_p=0.0,
                logit_bias=None,
                json_schema=None,
                regex=None,
                ebnf=None,
                structural_tag=None,
            ),
        )

    def test_non_greedy_is_rejected_before_worker_and_greedy_is_allowed(self):
        from unittest.mock import patch

        from sglang.srt.speculative.dflash_utils import validate_dflash_request

        request = self.make_request()
        request.sampling_params.top_k = 1 << 30
        spec = SimpleNamespace(speculative_draft_attention_backend="ascend")
        with (
            patch.dict("os.environ", {"SGLANG_DMTP_T1_RESIDUAL": "1"}),
            patch("sglang.srt.speculative.dflash_utils.get_spec", return_value=spec),
        ):
            error = validate_dflash_request(request, enable_overlap=False)
            self.assertIn("top_k=1", error)
            request.sampling_params.top_k = 1
            self.assertIsNone(validate_dflash_request(request, enable_overlap=False))
            with patch.dict("os.environ", {"SGLANG_DMTP_T1_RESIDUAL": "0"}):
                request.sampling_params.top_k = 1 << 30
                self.assertIsNone(
                    validate_dflash_request(request, enable_overlap=False)
                )
            spec.speculative_draft_attention_backend = "fa3"
            self.assertIsNone(validate_dflash_request(request, enable_overlap=False))

    def test_dmtp_rejects_unsupported_sampling_options_before_worker(self):
        from unittest.mock import patch

        from sglang.srt.speculative.dflash_utils import validate_dflash_request

        cases = (
            ("logprobs", "return_logprob", True),
            ("grammar", "grammar", object()),
            ("grammar", "json_schema", "{}"),
            ("grammar", "regex", ".*"),
            ("grammar", "ebnf", "root ::= 'x'"),
            ("grammar", "structural_tag", "{}"),
            ("logit_bias", "logit_bias", {"1": 1.0}),
            ("min_p", "min_p", 0.1),
        )
        spec = SimpleNamespace(speculative_draft_attention_backend="ascend")
        with (
            patch.dict("os.environ", {"SGLANG_DMTP_T1_RESIDUAL": "1"}),
            patch("sglang.srt.speculative.dflash_utils.get_spec", return_value=spec),
        ):
            for expected, field, value in cases:
                with self.subTest(field=field):
                    request = self.make_request()
                    target = (
                        request
                        if field in ("return_logprob", "grammar")
                        else request.sampling_params
                    )
                    setattr(target, field, value)
                    error = validate_dflash_request(request, enable_overlap=False)
                    self.assertIn(expected, error)
            self.assertIsNone(
                validate_dflash_request(self.make_request(), enable_overlap=False)
            )

    def test_rejected_request_is_retired_without_entering_scheduler_queue(self):
        from http import HTTPStatus
        from unittest.mock import Mock, patch

        from sglang.srt.managers.scheduler import Scheduler

        scheduler = SimpleNamespace(
            request_retirer=Mock(), _add_request_to_queue=Mock()
        )
        request = SimpleNamespace(rid="non-greedy")
        with patch("sglang.srt.managers.scheduler.prepare_abort") as prepare:
            Scheduler._reject_dflash_request(scheduler, request, "greedy only")
        prepare.assert_called_once_with(
            request, "greedy only", status_code=HTTPStatus.BAD_REQUEST
        )
        scheduler.request_retirer.retire_unadmitted.assert_called_once_with(request)
        scheduler._add_request_to_queue.assert_not_called()


class PageRecipeTests(unittest.TestCase):
    def test_ascend_uses_page16_and_fa3_keeps_page1(self):
        from unittest.mock import patch

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
            disable_overlap_schedule=True,
            enable_hierarchical_cache=False,
        )
        with patch(
            "sglang.srt.speculative.dmtp_runtime.cuda_graph_fully_disabled",
            return_value=True,
        ):
            with get_context().override_server_args(**fields):
                validate_recipe()
                with scoped_config(enable_hierarchical_cache=True):
                    with self.assertRaisesRegex(ValueError, "HiCache"):
                        validate_recipe()
                with scoped_config(page_size=1):
                    with self.assertRaises(ValueError):
                        validate_recipe()
                with scoped_config(
                    speculative_draft_attention_backend="fa3", page_size=1
                ):
                    validate_recipe()
                with scoped_config(
                    speculative_draft_attention_backend="fa3", page_size=16
                ):
                    with self.assertRaises(ValueError):
                        validate_recipe()

    def test_ascend_rejects_graphs_overlap_and_graph_flags(self):
        from unittest.mock import patch

        from sglang.srt.runtime_context import get_context
        from sglang.srt.speculative.dmtp_runtime import validate_recipe

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
            disable_overlap_schedule=True,
            enable_hierarchical_cache=False,
        )
        with get_context().override_server_args(**fields):
            with patch(
                "sglang.srt.speculative.dmtp_runtime.cuda_graph_fully_disabled",
                return_value=False,
            ):
                with self.assertRaisesRegex(ValueError, "graphs"):
                    validate_recipe()
            with (
                patch(
                    "sglang.srt.speculative.dmtp_runtime.cuda_graph_fully_disabled",
                    return_value=True,
                ),
                patch.dict("os.environ", {"SGLANG_MTP_PREFIX_MTP_GRAPH": "1"}),
            ):
                with self.assertRaisesRegex(ValueError, "eager"):
                    validate_recipe()


if __name__ == "__main__":
    unittest.main()
