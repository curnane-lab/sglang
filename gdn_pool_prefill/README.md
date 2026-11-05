# GDN pool-prefill server A/B reproduction

The backend microbenchmark remains in `bench_gdn_pool_prefill.py`. One-time server diagnostic scripts and raw results are kept out of the source tree in fixed evidence revision [`499cf181`](https://github.com/curnane-lab/sglang/commit/499cf1812587b7f0db8ac01e0c8acf09c2e4fec6):

- [Server client](https://github.com/curnane-lab/sglang/blob/499cf1812587b7f0db8ac01e0c8acf09c2e4fec6/gdn_pool_prefill/bench_gdn_pool_prefill_server.py) and [NPU Event instrumentation](https://github.com/curnane-lab/sglang/blob/499cf1812587b7f0db8ac01e0c8acf09c2e4fec6/gdn_pool_prefill/gdn_pool_prefill_server_instrument.py).
- Raw results: [24-layer backend](https://github.com/curnane-lab/sglang/blob/499cf1812587b7f0db8ac01e0c8acf09c2e4fec6/gdn_pool_prefill/24layers.json), [CoW scaling](https://github.com/curnane-lab/sglang/blob/499cf1812587b7f0db8ac01e0c8acf09c2e4fec6/gdn_pool_prefill/cow_scaling.json), and [server A/B](https://github.com/curnane-lab/sglang/blob/499cf1812587b7f0db8ac01e0c8acf09c2e4fec6/gdn_pool_prefill/server_ab.json).

The benchmark is specific to cached/chunked GDN prefill; it is not a general model throughput benchmark.

## Tested revisions and environment

| Item | Value |
|---|---|
| Hardware | One Ascend 910B3, TP=1 |
| Model | Qwen3.5-4B, BF16, 24 GDN layers |
| SGLang version | 0.5.19 |
| Engine commit for both arms | `0b42377248bcf010560afa3b4bcb55541538a3bb` |
| Kernel commit for both arms | `a967b3cfb804dc5e9da88814d50483b7a1963401` |
| PyTorch / torch-npu | 2.10.0+cpu / 2.10.0 |
| Temporal state pool | BF16, 384 slots |
| Attention backend | `ascend` |
| Prefill/decode graphs | Disabled |
| Client workload | 64 concurrent requests, 384 cached tokens + 128 continuation tokens, 16 output tokens per request |

Both arms use the same engine and the same kernel build containing sgl-project/sgl-kernel-npu#970. There is no separate legacy engine or kernel checkout.

`gdn_pool_prefill_server_instrument.py` selects the arm from the request ID before deferred CoW and metadata construction:

- `legacy-*`: set `gdn_prefill_pool_io=False`, retain temporal CoW, and call the legacy operator with `has_initial_state=None` after the destination state has been restored. This reproduces the pre-pool-I/O path.
- `pool-*`: require the pool operator to be available, assert that every GDN layer selects it, and omit eligible temporal CoW.

The instrumentation also records actual extend shapes, 24 GDN layer calls per forward, state-copy behavior, and NPU Event intervals. The reported server measurements include this instrumentation on both arms.

## Start the server

Install the matching engine and kernel first. Run from the SGLang repository root and replace `MODEL_PATH` with the local Qwen3.5-4B path. Download the linked client and instrumentation from the fixed evidence revision into one local directory, then set `EVIDENCE_DIR` to that directory.

```bash
export MODEL_PATH=/path/to/Qwen3.5-4B
export EVIDENCE_DIR=/path/to/gdn-pool-prefill-evidence
export PYTHONPATH="$PWD/python:${EVIDENCE_DIR}:${PYTHONPATH}"
export GDN_DIAG_MODE=pool
export GDN_DIAG_SELECT_BY_RID=1
export GDN_DIAG_REPORT_DIR="$PWD/server_ab_stream"
export CHUNK=1024

python -c 'import runpy; from gdn_pool_prefill_server_instrument import install; install(); runpy.run_module("sglang.launch_server", run_name="__main__")' \
  --model-path "$MODEL_PATH" \
  --served-model-name gdn-pool-test \
  --host 127.0.0.1 \
  --port 31234 \
  --device npu \
  --attention-backend ascend \
  --mamba-ssm-dtype bfloat16 \
  --chunked-prefill-size "$CHUNK" \
  --max-prefill-tokens "$CHUNK" \
  --max-running-requests 64 \
  --max-mamba-cache-size 384 \
  --mem-fraction-static 0.6 \
  --disable-cuda-graph \
  --trust-remote-code \
  --random-seed 20261105 \
  --log-level info
```

`--disable-cuda-graph` was the option used for the recorded runs. Newer revisions may prefer the equivalent decode/prefill graph-backend options; keep graphs disabled on both arms.

## Run the client

```bash
python "${EVIDENCE_DIR}/bench_gdn_pool_prefill_server.py" \
  --chunk-size "$CHUNK" \
  --batch-size 64 \
  --cycles 2 \
  --output-len 16 \
  --output "chunk${CHUNK}_client.json"
```

One trial executes eight batches in `legacy, pool, pool, legacy, legacy, pool, pool, legacy` order. The client verifies a nonzero prefix-cache hit for each arm before measurement and asserts `cached_tokens=384` for every measured request. Each output includes per-request absolute TTFT, end-to-end latency, cache-hit metadata, output token IDs, and text hashes.

Stop the server normally after the client completes so the instrumentation writes `stream-*.json` under `GDN_DIAG_REPORT_DIR`. Repeat with `CHUNK=128`, `512`, and `1024`, restarting the server for each configuration.

## Recorded evidence

The externally hosted [`server_ab.json`](https://github.com/curnane-lab/sglang/blob/499cf1812587b7f0db8ac01e0c8acf09c2e4fec6/gdn_pool_prefill/server_ab.json) contains:

- Every measured batch's wall time, output throughput, request count, and cache-hit counts.
- Every request's absolute TTFT sample, grouped by batch.
- Per-mode and per-trial aggregate values.
- Actual forward shapes, GDN layer-call counts, and temporal-CoW path validation from the stream records.
- Engine/kernel SHAs and aggregation rules.

Output token IDs and text hashes are omitted from this compact evidence file; all timing values and cache-hit counts are preserved. No outlier round is removed. In particular, `chunk1024_trial1` retains the legacy round with `wall_ms=6261.85963`; the PR uses medians rather than presenting the resulting mean wall speedup as representative.

Wall and throughput speedups use per-mode medians across four batches per trial. TTFT speedups use the median across all 64 requests per batch. Throughput is `64 * 16 * 1000 / wall_ms` output tokens/s.

### Absolute TTFT by trial

Values are median milliseconds across 256 requests per arm and trial.

| Chunk size | Trial | Legacy TTFT | Pool TTFT | Speedup |
|---:|---:|---:|---:|---:|
| 128 | 1 | 5083.220 ms | 4751.113 ms | **1.0699×** |
| 128 | 2 | 5121.010 ms | 4923.542 ms | **1.0401×** |
| 512 | 1 | 1771.227 ms | 1705.973 ms | **1.0383×** |
| 1024 | 1 | 1319.251 ms | 1173.353 ms | **1.1243×** |
| 1024 | 2 | 1367.377 ms | 1153.226 ms | **1.1857×** |

Chunk 128 and 1024 have two independent trials; chunk 512 has one. The combined chunk-1024 result is `1321.820 -> 1164.392 ms` across 512 requests per arm, or **1.1352×**. This is a workload-specific TTFT result, not a claim of 1.10× wall-clock or throughput improvement.
