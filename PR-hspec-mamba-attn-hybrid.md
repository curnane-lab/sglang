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

**Feature (`8d19016e`)**

- `SpeculativeAlgorithm.MAMBA_ATTN_HYBRID` + server-args hook (`_handle_mamba_attn_hybrid`): PP!=1 / DP-attention / CP rejected, verify block size inferred from the speculators draft config (`speculative_tokens + 1`) unless set explicitly, overlap/`max_running_requests` defaults aligned with the dflash family.
- Draft model `MambaAttnHybridDraftModel` (`models/hspec_draft.py`): block-pattern-driven Mamba-2 / attention / MLP hybrid stack.
  - Draft attention sub-layers are `DFlashAttention` (RadixAttention over the draft KV pool); the context prefix K/V are materialized from the *target* paged KV pool at `attn_kv_layer_ids` (post-RoPE) — the semantics the drafter was trained with.
  - The Mamba mixer uses a reference SSD scan in fp32 (no `mamba_ssm` dependency), which also makes it the correct kernel on Ascend NPU.
  - Draft vocab head (`draft_vocab_size=32000`) with target-vocab offset mapping and the optional Markov bias head.
- Worker `HSpecWorkerV2` (`speculative/hspec_worker_v2.py`): draft sampling from the infill mask positions (rows `1..K-1`) through the draft vocab head, and greedy verify against the target.
- Latent-seed plumbing: target last-position hidden states → `fc` → `hidden_norm` → per-sublayer seed projections (config: `latent_fusion_layer_ids`, `attn_kv_layer_ids`, `fc_norm=false`, `mask_token_id`).

**Bring-up fixes (`0fccb532`)**

- Register `MAMBA_ATTN_HYBRID` in `SpeculativeAlgorithm` (previously `from_string` failed).
- `candidate_selector` is optional — the H-Spec drafter has none.
- `set_block_size` override (the hybrid stack has no DFLASH conv sublayers) and checkpoint compatibility (`fc_norm=false`).
- Align `_sample_draft_next` signature with the base class and sample draft ids from the **mask positions (rows `1..K-1`, DFLASH infill convention)**; the previous rows `0..K-2` convention broke first-position hits (first-hit rate went from ~0% to ~100% after the fix).
- Refresh rotary phases in every draft attention sublayer: each sublayer owns its `rotary_emb` instance and the `layer_id == 0` guard left later sublayers with stale cos/sin buffers.
- **NPU eager-path fix: TARGET_VERIFY KV-length double count.** DFLASH verify pre-expands `batch.seq_lens_cpu` to `committed_prefix + block_size`, and the eager backend added `spec_tokens_per_req` again, so FIA received `kv_end = prefix + 2*block_size`. With bottom-right aligned causal, row 0 of each verify group then attended the whole verify block's own KV (corrupting its prediction), while later rows only saw zero-filled stale slots past the real KV — which masked the bug. The graph path already had the `_is_dflash_verify` guard; this adds the same treatment to the eager path.

**Rebase adaptations (`012b26e1`, onto current main)**

- `_handle_mamba_attn_hybrid` no longer assigns `server_args` directly during resolution; uses `resolving_view` reads + `declare_resolution` writes, matching `_handle_dflash`.
- Draft-worker constructor drops the removed `ps` argument.
- `MambaAttnHybridDraftModel` defines the attributes the shared dflash-family worker reads on any drafter (`candidate_selector` / `lilicorr` / `is_nemotron_35_draft` / `embed_tokens` / `prefix_gru` / `embed_proj` / `shift_label` / `lm_head`) and implements `project_target_hidden` / `prepare_context_hidden_for_kv` for the shared target-hidden materialization path.

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
