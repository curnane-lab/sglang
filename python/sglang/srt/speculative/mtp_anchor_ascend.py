"""Token-addressed attention for the private DMTP runners on Ascend."""

import torch
from torch.nn.functional import scaled_dot_product_attention

from sglang.srt.mem_cache.memory_pool import KVWriteLoc, window_write_locs


def _attention(query, keys, values, *, scale, causal):
    query = query.transpose(0, 1).unsqueeze(0)
    keys = keys.transpose(0, 1).unsqueeze(0)
    values = values.transpose(0, 1).unsqueeze(0)
    mask = None
    if causal:
        kv_len, q_len = keys.shape[-2], query.shape[-2]
        rows = torch.arange(kv_len - q_len, kv_len, device=query.device)
        cols = torch.arange(kv_len, device=query.device)
        mask = cols[None, :] <= rows[:, None]
    output = scaled_dot_product_attention(
        query,
        keys,
        values,
        attn_mask=mask,
        enable_gqa=query.shape[1] != keys.shape[1],
        scale=scale,
    )
    return output.squeeze(0).transpose(0, 1).flatten(1)


class AscendDMTPAttention:
    def __init__(self, backend, *, draft_block):
        if getattr(backend, "use_mla", False):
            raise ValueError("Ascend DMTP requires dense, unquantized attention")
        self.backend = backend
        self.draft_block = draft_block
        self.slots = getattr(backend, "speculative_num_draft_tokens", 8)

    def _visible_slots(self, batch):
        if self.draft_block:
            prefixes = batch.seq_lens.tolist()
            query_lens = [self.slots] * batch.batch_size
        else:
            prefixes = batch.extend_prefix_lens_cpu
            query_lens = batch.extend_seq_lens_cpu
        mapping = self.backend.req_to_token_pool.req_to_token
        blocks = (
            batch.out_cache_loc.reshape(batch.batch_size, self.slots)
            if self.draft_block
            else None
        )
        result = []
        for index, (req, prefix, width) in enumerate(
            zip(
                batch.req_pool_indices.tolist(),
                prefixes,
                query_lens,
                strict=True,
            )
        ):
            if self.draft_block:
                cached = mapping[req, max(0, prefix - 4096) : prefix]
                slots = torch.cat((cached, blocks[index]))
            else:
                slots = mapping[req, : prefix + width]
            result.append(slots.long())
        return result

    def __call__(self, q, k, v, layer, batch, save_kv_cache=True, **kwargs):
        backend = self.backend
        pool = backend.token_to_kv_pool
        if save_kv_cache:
            metadata = backend.forward_metadata
            pool.set_kv_buffer(
                layer,
                KVWriteLoc.for_batch(
                    batch,
                    window_locs=window_write_locs(
                        getattr(backend, "runner_ids", None),
                        getattr(metadata, "swa_out_cache_loc", None),
                    ),
                ),
                k,
                v,
            )
        keys = pool.get_key_buffer(layer.layer_id).reshape(
            -1, layer.tp_k_head_num, layer.qk_head_dim
        )
        values = pool.get_value_buffer(layer.layer_id).reshape(
            -1, layer.tp_v_head_num, layer.v_head_dim
        )
        q = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
        outputs = []
        start = 0
        for index, slots in enumerate(self._visible_slots(batch)):
            width = self.slots if self.draft_block else batch.extend_seq_lens_cpu[index]
            end = start + width
            outputs.append(
                _attention(
                    q[start:end],
                    keys.index_select(0, slots),
                    values.index_select(0, slots),
                    scale=layer.scaling,
                    causal=not self.draft_block,
                )
            )
            start = end
        return torch.cat(outputs, dim=0)


class AnchorAscendForward:
    def __init__(self, backend):
        assert backend.page_size in (1, 16)
        self.backend = backend
        self.original = backend.forward_mtp
        self.page_size = backend.page_size
        self.slots = backend.speculative_num_draft_tokens
        assert self.slots == 8
        self.attention = AscendDMTPAttention(backend, draft_block=True)

    def _visible_slots(self, batch):
        return self.attention._visible_slots(batch)

    def __call__(self, q, k, v, layer, batch, save_kv_cache=True, **kwargs):
        if not batch.forward_mode.is_target_verify():
            return self.original(q, k, v, layer, batch, save_kv_cache, **kwargs)
        assert 1 <= batch.batch_size <= 32 and q.shape[0] == batch.batch_size * self.slots
        return self.attention(q, k, v, layer, batch, save_kv_cache, **kwargs)


def install_native_ascend_attention(backend):
    full_attention = getattr(backend, "full_attn_backend", backend)
    if type(full_attention).__name__ != "AscendAttnBackend":
        raise ValueError("Ascend DMTP requires the dense Ascend attention backend")
    full_attention.forward_extend = AscendDMTPAttention(
        full_attention, draft_block=False
    )
