"""Anchor-relative visibility for Ascend paged attention."""

import torch


class AnchorAscendForward:
    def __init__(self, backend):
        assert backend.page_size in (1, 16)
        self.backend = backend
        self.original = backend.forward_mtp
        self.page_size = backend.page_size
        self.slots = backend.speculative_num_draft_tokens
        assert self.slots == 8
        if self.page_size == 1:
            self.width = 4096 + self.slots
        else:
            prefix_pages = 4096 // self.page_size + 1
            block_pages = (self.slots + self.page_size - 1) // self.page_size
            self.width = prefix_pages + block_pages

    def _visible_kv(self, batch):
        bs = batch.batch_size
        prefixes = batch.seq_lens.tolist()
        blocks = batch.out_cache_loc.reshape(bs, self.slots)
        table = torch.zeros(
            (bs, self.width), dtype=torch.int32, device=blocks.device
        )
        lengths = []
        for i, prefix in enumerate(prefixes):
            kept = min(prefix, 4096)
            req = int(batch.req_pool_indices[i])
            pages = blocks[i] // self.page_size
            if kept:
                prefix_pages = self.backend.req_to_token[
                    req, prefix - kept : prefix
                ] // self.page_size
                pages = torch.cat((prefix_pages, pages))
            pages = pages.unique_consecutive()
            table[i, : pages.numel()] = pages.to(torch.int32)
            lengths.append(kept + self.slots)
        return table, lengths

    def __call__(
        self,
        q,
        k,
        v,
        layer,
        forward_batch,
        save_kv_cache=True,
        q_rope=None,
        k_rope=None,
        sinks=None,
        q_descale=None,
    ):
        kwargs = dict(
            q_rope=q_rope, k_rope=k_rope, sinks=sinks, q_descale=q_descale
        )
        if not forward_batch.forward_mode.is_target_verify():
            return self.original(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        bs = forward_batch.batch_size
        assert 1 <= bs <= 32 and q.shape[0] == bs * self.slots
        table, lengths = self._visible_kv(forward_batch)
        metadata = self.backend.forward_metadata
        old = (
            metadata.block_tables,
            metadata.seq_lens_cpu_int,
            metadata.seq_lens_cpu_list,
        )
        metadata.block_tables = table
        metadata.seq_lens_cpu_int = torch.tensor(lengths, dtype=torch.int32)
        metadata.seq_lens_cpu_list = lengths
        try:
            return self.original(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        finally:
            (
                metadata.block_tables,
                metadata.seq_lens_cpu_int,
                metadata.seq_lens_cpu_list,
            ) = old
