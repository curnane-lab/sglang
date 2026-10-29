import numpy as np
import torch
import torch_npu

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1, suite="stage-a-unit-test-npu")


def _attention(query, key_cache, value_cache, block_table, mask, q_lens, kv_lens):
    return torch_npu.npu_fused_infer_attention_score_v2(
        query,
        key_cache,
        value_cache,
        block_table=block_table,
        block_size=128,
        num_query_heads=8,
        num_key_value_heads=1,
        input_layout="TND",
        atten_mask=mask,
        softmax_scale=256**-0.5,
        actual_seq_qlen=np.cumsum(q_lens).tolist(),
        actual_seq_kvlen=kv_lens,
        sparse_mode=3,
        pre_tokens=65536,
        next_tokens=65536,
    )


def test_fia_ragged_full_coverage_graph_with_zero_length_slots():
    torch.manual_seed(20261027)
    device = "npu:0"
    query = torch.randn(64, 8, 256, dtype=torch.bfloat16, device=device)
    key_cache = torch.randn(256, 128, 256, dtype=torch.bfloat16, device=device)
    value_cache = torch.randn_like(key_cache)
    block_table = torch.tensor([[0, 1]] * 64, dtype=torch.int32, device=device)
    mask = torch.triu(torch.ones(2048, 2048, dtype=torch.bool, device=device), diagonal=1)
    capture_q = [4] * 16 + [0] * 48
    capture_kv = [100] * 16 + [0] * 48
    replay_q = [8] * 5 + [4] + [2] * 10 + [0] * 48
    replay_kv = [106] * 5 + [102] + [100] * 10 + [0] * 48

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        _attention(query, key_cache, value_cache, block_table, mask, capture_q, capture_kv)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
        output, _ = _attention(query, key_cache, value_cache, block_table, mask, capture_q, capture_kv)
    torch.npu.synchronize()

    graph.update(
        cpu_update_input=[
            {
                "actual_seq_qlen": np.cumsum(replay_q).tolist(),
                "actual_seq_kvlen": replay_kv,
            }
        ]
    )
    graph.replay()
    torch.npu.synchronize()
    expected, _ = _attention(query, key_cache, value_cache, block_table, mask, replay_q, replay_kv)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
