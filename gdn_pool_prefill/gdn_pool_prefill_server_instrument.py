import atexit
import inspect
import json
import os
from pathlib import Path

import torch

MODE = os.environ["GDN_DIAG_MODE"]
REPORT_DIR = Path(os.environ["GDN_DIAG_REPORT_DIR"])
EVENTS = []
CURRENT = None


def elapsed(start, end):
    return start.elapsed_time(end)


def report():
    if not EVENTS:
        return
    torch.npu.synchronize()
    records = []
    for event in EVENTS:
        records.append(
            {
                "forward_pass_id": event["forward_pass_id"],
                "mode": event["mode"],
                "rids": event["rids"],
                "batch_size": event["batch_size"],
                "input_tokens": event["input_tokens"],
                "extend_seq_lens": event["extend_seq_lens"],
                "extend_prefix_lens": event["extend_prefix_lens"],
                "forward_stream_ms": elapsed(event["start"], event["end"]),
                "gdn": [
                    {"use_pool": use_pool, "stream_ms": elapsed(start, end)}
                    for use_pool, start, end in event["gdn"]
                ],
                "state_copies": [
                    {
                        "copy_temporal": copy_temporal,
                        "stream_ms": elapsed(start, end),
                    }
                    for copy_temporal, start, end in event["state_copies"]
                ],
            }
        )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    destination = REPORT_DIR / f"stream-{MODE}-{os.getpid()}.json"
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(
            {"mode": MODE, "pid": os.getpid(), "events": records}, stream, indent=2
        )
        stream.write("\n")


def install():
    from sglang.srt.hardware_backend.npu.attention.ascend_gdn_backend import (
        AscendGDNAttnBackend,
    )
    from sglang.srt.mem_cache.kv.state_storage import StateStorage
    from sglang.srt.model_executor.model_runner import ModelRunner

    original_init = AscendGDNAttnBackend.__init__
    original_chunk_gdn = AscendGDNAttnBackend.chunk_gdn
    original_copy = StateStorage.copy
    original_forward = ModelRunner.forward
    signature = inspect.signature(original_chunk_gdn)

    def init(self, model_runner):
        original_init(self, model_runner)
        if MODE == "legacy":
            model_runner.req_to_token_pool.gdn_prefill_pool_io = False
        elif not model_runner.req_to_token_pool.gdn_prefill_pool_io:
            raise RuntimeError("pool operator is unavailable")

    def chunk_gdn(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        use_pool = bound.arguments["use_pool"]
        active_mode = CURRENT["mode"] if CURRENT is not None else MODE
        if active_mode == "legacy":
            bound.arguments["has_initial_state"] = None
        elif not use_pool:
            raise RuntimeError("pool request did not select the pool operator")
        if CURRENT is None:
            return original_chunk_gdn(*bound.args, **bound.kwargs)
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        result = original_chunk_gdn(*bound.args, **bound.kwargs)
        end.record()
        CURRENT["gdn"].append((use_pool, start, end))
        return result

    def copy(self, src_index, dst_index, *, copy_temporal=True):
        if CURRENT is None:
            return original_copy(
                self, src_index, dst_index, copy_temporal=copy_temporal
            )
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        result = original_copy(self, src_index, dst_index, copy_temporal=copy_temporal)
        end.record()
        CURRENT["state_copies"].append((copy_temporal, start, end))
        return result

    def forward(self, forward_batch, *args, **kwargs):
        global CURRENT
        if not forward_batch.forward_mode.is_extend():
            return original_forward(self, forward_batch, *args, **kwargs)
        mode = MODE
        rids = forward_batch.rids
        if os.environ.get("GDN_DIAG_SELECT_BY_RID") == "1":
            assert rids
            legacy = rids[0].startswith("legacy-")
            assert all(rid.startswith("legacy-") == legacy for rid in rids)
            mode = "legacy" if legacy else "pool"
            self.req_to_token_pool.gdn_prefill_pool_io = mode == "pool"
        CURRENT = {
            "forward_pass_id": self.forward_pass_id,
            "mode": mode,
            "rids": rids,
            "batch_size": forward_batch.batch_size,
            "input_tokens": forward_batch.input_ids.shape[0],
            "extend_seq_lens": forward_batch.extend_seq_lens_cpu,
            "extend_prefix_lens": forward_batch.extend_prefix_lens_cpu,
            "gdn": [],
            "state_copies": [],
        }
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        CURRENT["start"] = start
        CURRENT["end"] = end
        try:
            return original_forward(self, forward_batch, *args, **kwargs)
        finally:
            end.record()
            EVENTS.append(CURRENT)
            CURRENT = None

    AscendGDNAttnBackend.__init__ = init
    AscendGDNAttnBackend.chunk_gdn = chunk_gdn
    StateStorage.copy = copy
    ModelRunner.forward = forward
    atexit.register(report)
