# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("vllm.models.hy_v4_flash")

from vllm.models.hy_v4_flash.reference import INDEX_DIM  # noqa: E402

from vllm_ascend.models.hy_v4_flash import engram  # noqa: E402
from vllm_ascend.models.hy_v4_flash.attention import (  # noqa: E402
    POSITION_LANES,
    AscendHYV4FlashAttention,
    FlashAttentionMetadata,
    FlashIndexPool,
    FlashMetadataBuilder,
)


def _index_pool():
    indexer = FlashIndexPool.__new__(FlashIndexPool)
    nn.Module.__init__(indexer)
    indexer.ratio, indexer.budget, indexer.num_heads = 2, 1, 1
    indexer.rope_dim, indexer.theta = 4, 10000.0
    indexer.sections, indexer.interleaved = [1, 1, 0], True
    indexer.eps = 1e-5
    indexer.token_key_projection = lambda x: (x,)
    indexer.query_projection = lambda x: (x,)
    indexer.head_weight_projection = lambda x: (x[:, :1].abs() + 1,)
    indexer.token_key_norm = nn.RMSNorm(INDEX_DIM, dtype=torch.bfloat16)
    indexer.key_value_projection = nn.Linear(INDEX_DIM, INDEX_DIM, bias=False, dtype=torch.float32)
    indexer.key_gate_projection = nn.Linear(INDEX_DIM, INDEX_DIM, bias=False, dtype=torch.float32)
    indexer.key_norm = nn.RMSNorm(INDEX_DIM, dtype=torch.float32)
    indexer.intra_group_ape = nn.Parameter(torch.zeros(indexer.ratio, INDEX_DIM))
    return indexer


@pytest.mark.parametrize("use_indexpool", [False, True], ids=["swa", "indexpool"])
@torch.inference_mode()
def test_paged_prefill_and_decode_match_uninterrupted_attention(use_indexpool):
    torch.manual_seed(9)
    layer = AscendHYV4FlashAttention.__new__(AscendHYV4FlashAttention)
    nn.Module.__init__(layer)
    layer.indexer = _index_pool() if use_indexpool else None
    layer.window = 3
    layer.latent_dim, layer.kv_width = 2, 3
    layer.scale = 0.5
    layer.learnable_sink_param = nn.Parameter(torch.zeros(1))
    layer.cache_width = layer.kv_width + INDEX_DIM + POSITION_LANES
    cache_shape = (7, 1, 2, layer.cache_width)
    table = torch.tensor([[3, 1, 5], [6, 2, 4]], dtype=torch.int32)
    hidden = torch.randn(2, 6, INDEX_DIM).bfloat16()
    kv = torch.randn(2, 6, layer.kv_width).bfloat16()
    query = torch.randn(2, 6, 1, layer.kv_width).bfloat16()
    wv = torch.eye(layer.latent_dim, dtype=torch.bfloat16).unsqueeze(0)
    # Consecutive coordinates above the BF16 integer precision test cache bitcasts.
    coordinates = torch.arange(6) + torch.tensor([[32768], [65536], [131072]])

    def run(starts, stops):
        actual = [torch.arange(start, stop) for start, stop in zip(starts, stops)]
        x = torch.cat([hidden[i, rows] for i, rows in enumerate(actual)])
        k = torch.cat([kv[i, rows] for i, rows in enumerate(actual)])
        q = torch.cat([query[i, rows] for i, rows in enumerate(actual)])
        positions = torch.cat([coordinates[:, rows] for rows in actual], dim=-1)
        slots = torch.cat([table[i, rows // 2] * 2 + rows % 2 for i, rows in enumerate(actual)])
        boundaries = [0, actual[0].numel(), q.shape[0]]
        meta = FlashAttentionMetadata(boundaries, stops, table, slots)
        out = torch.empty(q.shape[0], layer.latent_dim, dtype=torch.bfloat16)
        layer._write_cache_rows(meta, x, k[:, : layer.latent_dim], k[:, layer.latent_dim :], positions, q.shape[0])
        layer._attend(meta, x, x, positions, q, wv, out)
        return out[: boundaries[1]], out[boundaries[1] :]

    layer.kv_cache = torch.zeros(cache_shape, dtype=torch.bfloat16)
    full = run([0, 0], [6, 4])
    layer.kv_cache.zero_()
    chunked = [run([0, 0], [3, 2]), run([3, 2], [5, 3]), run([5, 3], [6, 4])]
    for request, expected in enumerate(full):
        torch.testing.assert_close(torch.cat([chunk[request] for chunk in chunked]), expected)
        if not use_indexpool:
            for pos, actual in enumerate(expected):
                rows = kv[request, max(0, pos - layer.window + 1) : pos + 1].float()
                logits = rows @ query[request, pos, 0].float() * layer.scale
                probs = torch.cat([logits, torch.zeros(1)]).softmax(0)[:-1]
                torch.testing.assert_close(actual, (probs @ rows[:, : layer.latent_dim]).bfloat16())
        else:
            rows = layer._gather(table[request], torch.arange(len(expected)))
            stored = rows[:, -POSITION_LANES:].contiguous().view(torch.int64).T
            torch.testing.assert_close(stored, coordinates[:, : len(expected)])
    torch.testing.assert_close(layer.kv_cache[0], torch.zeros_like(layer.kv_cache[0]))


def test_metadata_builder_excludes_padding():
    builder = FlashMetadataBuilder(None, ["layer.attn"], None, torch.device("cpu"))
    table = torch.tensor([[2], [3], [0]])
    slots = torch.tensor([4, 5, 6, -1])
    meta = builder.build(
        0,
        SimpleNamespace(
            num_reqs=2,
            query_start_loc_cpu=torch.tensor([0, 2, 3, 4]),
            seq_lens_cpu=torch.tensor([2, 1, 0]),
            block_table_tensor=table,
            slot_mapping=slots,
        ),
    )
    assert meta.query_start_loc == [0, 2, 3]
    assert meta.seq_lens == [2, 1]
    assert meta.block_table is table
    assert meta.slot_mapping is slots


def test_engram_hash_matches_chunked_requests(monkeypatch):
    config = SimpleNamespace(
        engram_layer_ids=[1],
        engram_max_ngram_size=3,
        engram_n_embed_per_ngram=4,
        engram_n_head_per_ngram=2,
        engram_pad_id=0,
        vocab_size=12,
        engram_seed=12,
        engram_layer_head_vocab_sizes={"1": [[41, 43], [47, 53]]},
    )
    state = engram.FlashNgramHashState(config)
    state.load_lookup(torch.arange(config.vocab_size))
    state.visual_token_ids = (10,)

    def hash_batch(tokens, lookback, starts):
        context = SimpleNamespace(attn_metadata={"attn": SimpleNamespace(query_start_loc=starts)})
        monkeypatch.setattr(engram, "get_forward_context", lambda: context)
        hashes = torch.empty((1, len(tokens), 4), dtype=torch.int32)
        state.prepare_hashes_with_output(torch.tensor(tokens), torch.tensor(lookback), "attn", hashes)
        return hashes

    full = hash_batch([1, 2, 3, 4, 10, 5], [[-1, -1], [-1, -1]], [0, 3, 6])
    chunked = hash_batch([3, 5], [[2, 1], [10, 4]], [0, 1, 2])
    torch.testing.assert_close(chunked, full[:, [2, 5]])
    torch.testing.assert_close(full[:, :3], hash_batch([1, 2, 3], [[-1, -1]], [0, 3]))
    assert (full[:, 4] == -1).all()
