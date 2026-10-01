# PR: H-Spec (mamba_attn_hybrid) speculative decoding with NPU support

## Title

```
[Spec Decode] H-Spec (mamba_attn_hybrid) speculative decoding with NPU (Ascend) support
```

Alternate (shorter) title:

```
feat(spec_decode): add H-Spec (mamba_attn_hybrid) speculative decoding with NPU support
```

---

## Body

### Motivation

Port the H-Spec drafter (arXiv:2609.24197 — *H-Spec: Parallel Speculative Decoding Without a Drafter-Side KV Cache*) onto SGLang, including full Ascend NPU support. H-Spec is a hybrid Mamba/attention parallel drafter that needs **no drafter-side KV cache**: attention sub-layers reuse the target model's KV pool in place, while Mamba modules are seeded from last-position target hidden states. Internally the method is named `mamba_attn_hybrid`.

This brings SGLang to parity with the reference vLLM implementation (vllm-ascend PR #17805) for serving H-Spec checkpoints.

### What's included

Single commit `9e7ae0d593` (squashed after review; replaces the earlier 3-commit series):

**Feature**

- `SpeculativeAlgorithm.MAMBA_ATTN_HYBRID`, registered in the dflash **family** (`is_dflash_family`) — this also makes the NPU TARGET_VERIFY path skip the verify-block seq_len increment that non-dflash algorithms need (the dflash worker already publishes `seq_lens_cpu` as `prefix + block_size`; incrementing again made FIA attend `block_size` stale slots past the real KV and corrupted row-0 predictions).
- Server-args hook `_handle_mamba_attn_hybrid`: verify block size inferred from the speculators draft config (`speculative_tokens + 1`), resolution writes via `declare_resolution`; PP/DP-attention/CP and `--speculative-draft-window-size` rejected.
- Draft model `MambaAttnHybridDraftModel` (`models/hspec_draft.py`): block-pattern Mamba-2 / attention / MLP hybrid stack.
  - Draft attention sub-layers reuse the *target* paged KV pool at `attn_kv_layer_ids` (post-RoPE), with a GQA/TP layout fail-fast check on the copied K/V.
  - The Mamba mixer uses a reference SSD scan in fp32 (no `mamba_ssm` dependency) — also the correct kernel on Ascend NPU.
  - Draft vocab head (`draft_vocab_size=32000`) with target-vocab offset mapping and the optional Markov bias head; latent-seed plumbing (`fc` → `hidden_norm` → per-sublayer seed projections).
- Worker `HSpecWorkerV2`: draft sampling from the infill rows (`1..K-1`) through the draft-vocab head; greedy verify against the target.

**Enforced restrictions**

- Draft CUDA graph is disabled for `MAMBA_ATTN_HYBRID`: the dflash folded sampler samples the *target* lm_head over captured state, while H-Spec must sample its draft-vocab head over per-step latent seeds that are not captured (warning logged). The draft runs eager.
- Greedy drafter sampling only; lossless serving requires `temperature=0`.

### Validation (Ascend 910, single card, eager, bf16, greedy, batch-1 sequential)

Target: Qwen3-4B + [weifanjiang/qwen3-4b.speculators.hspec](https://huggingface.co/weifanjiang/qwen3-4b.speculators.hspec), 8 draft tokens/step.

**Correctness**: greedy output parity vs no-speculative-decoding baseline, token-by-token: **5/5 prompts OK** (Fibonacci chain, QA, code-completion style prompts).

**Accept length & speed** (Spec-Bench tasks × first 15 prompts + HumanEval × first 15, max 512 new tokens, Qwen3 chat template, thinking on; τ = accepted tokens per verify step; baseline = same build, no spec):

| Task | τ (this PR) | τ (H-Spec paper, Table 2, Qwen3-4B) | Spec tok/s | Base tok/s | Speedup |
|---|---|---|---|---|---|
| math_reasoning | 4.278 | 4.17 | 74.4 | 34.9 | 2.13x |
| qa | 2.967 | 3.04 | 52.1 | 34.2 | 1.52x |
| rag | 3.191 | 3.12 | 53.7 | 34.5 | 1.56x |
| summarization | 2.622 | 2.62 | 41.9 | 34.4 | 1.22x |
| translation | 2.656 | 2.74 | 44.8 | 34.8 | 1.29x |
| code (HumanEval) | 3.584 | 3.83 | 61.7 | 34.3 | 1.80x |
| **avg (6 tasks)** | **3.216** | **3.253** | | | **1.59x** |

Paper protocol differs (max 4,096 tokens, GPU/vLLM, 8 tasks incl. chat/tool, avg τ=3.24, Spd. 2.50x); numbers are for directional comparison.

### Usage

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-4B \
  --speculative-algorithm MAMBA_ATTN_HYBRID \
  --speculative-draft-model-path weifanjiang/qwen3-4b.speculators.hspec \
  --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 8 \
  --device npu --tp-size 1 --disable-cuda-graph --mem-fraction-static 0.8
```

Notes:

- The current implementation samples the drafter greedily; lossless behavior is under `temperature=0`.
- Debug probes are env-gated and default-off: `SGLANG_HSPEC_DEBUG`, `HSPEC_DEBUG_TOPK`, `HSPEC_DECODE_CHECK`, `HSPEC_ATTN_PROBE`, `HSPEC_SAMPLE_ROW_SHIFT`.

### Checklist

- [ ] Format / lint
- [ ] CUDA smoke run (feature was brought up and validated on NPU; CUDA path shares the same code but was not re-validated here)
- [ ] Add spec-decode unit test for the verify block-seqlen convention on NPU

---
*Local artifacts (analysis, eval scripts, raw results): `/mnt/workspace/hspec-eval/` — `ANALYSIS-row0-kvend.md`, `REPORT.md`, `eval_spec_bench.py`, `parity_test.py`, `results/`.*
