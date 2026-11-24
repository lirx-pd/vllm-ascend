# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("vllm.models.hy_v4_flash")

from vllm.model_executor.models.interfaces import SupportsMRoPE  # noqa: E402
from vllm.models.hy_v4_flash.model import HYV4FlashModel  # noqa: E402
from vllm.models.hy_v4_flash.reference import INDEX_DIM, sparse_candidates  # noqa: E402

from vllm_ascend.models.hy_v4_flash import engram  # noqa: E402
from vllm_ascend.models.hy_v4_flash.attention import (  # noqa: E402
    POSITION_LANES,
    AscendHYV4FlashAttention,
    FlashAttentionMetadata,
    FlashIndexPool,
    FlashMetadataBuilder,
)
from vllm_ascend.models.hy_v4_flash.model import AscendHYV4FlashForCausalLM  # noqa: E402


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
    layer.max_model_len = 8
    layer.latent_dim, layer.kv_width = 2, 3
    layer.scale = 0.5
    layer.learnable_sink_param = nn.Parameter(torch.zeros(1))
    layer.cache_width = layer.kv_width + INDEX_DIM + POSITION_LANES
    cache_shape = (7, 1, 2, layer.cache_width)
    table = torch.tensor([[3, 1, 5, 0], [6, 2, 4, 0]], dtype=torch.int32)
    hidden = torch.randn(2, 6, INDEX_DIM).bfloat16()
    kv = torch.randn(2, 6, layer.kv_width).bfloat16()
    query = torch.randn(2, 6, 1, layer.kv_width).bfloat16()
    wv = torch.eye(layer.latent_dim, dtype=torch.bfloat16).unsqueeze(0)
    # Consecutive coordinates above the BF16 integer precision test cache bitcasts.
    coordinates = torch.arange(6) + torch.tensor([[32768], [65536], [131072]])

    def run(starts, stops, *, decode_seq_lens=None, pad_last_request=False):
        actual = [torch.arange(start, stop) for start, stop in zip(starts, stops)]
        x = torch.cat([hidden[i, rows] for i, rows in enumerate(actual)])
        k = torch.cat([kv[i, rows] for i, rows in enumerate(actual)])
        q = torch.cat([query[i, rows] for i, rows in enumerate(actual)])
        positions = torch.cat([coordinates[:, rows] for rows in actual], dim=-1)
        slots = torch.cat([table[i, rows // 2] * 2 + rows % 2 for i, rows in enumerate(actual)])
        if pad_last_request:
            slots[-1] = -1
        boundaries = [0, actual[0].numel(), q.shape[0]]
        meta = FlashAttentionMetadata(boundaries, stops, table, slots, decode_seq_lens)
        out = torch.empty(q.shape[0], layer.latent_dim, dtype=torch.bfloat16)
        layer._write_cache_rows(meta, x, k[:, : layer.latent_dim], k[:, layer.latent_dim :], positions, q.shape[0])
        if decode_seq_lens is None:
            layer._attend(meta, x, x, positions, q, wv, out)
        else:
            layer._attend_decode_graph(meta, x, x, positions, q, wv, out)
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

    graph_lengths = torch.tensor([6, 4])
    graph = run([5, 3], [6, 4], decode_seq_lens=graph_lengths)
    for graph_row, eager_row in zip(graph, chunked[-1]):
        torch.testing.assert_close(graph_row, eager_row)
    layer.kv_cache[0].fill_(17)
    cache_before = layer.kv_cache.clone()
    graph_lengths[0] = 5
    padded = run([4, 3], [5, 4], decode_seq_lens=graph_lengths, pad_last_request=True)
    torch.testing.assert_close(padded[0], full[0][4:5])
    torch.testing.assert_close(padded[1], torch.zeros_like(padded[1]))
    torch.testing.assert_close(layer.kv_cache, cache_before)


def test_metadata_builder_excludes_padding():
    builder = FlashMetadataBuilder(None, ["layer.attn"], None, torch.device("cpu"))
    table = torch.tensor([[2], [3], [0]])
    slots = torch.tensor([4, 5, 6, -1])
    common = SimpleNamespace(
        num_reqs=2,
        query_start_loc_cpu=torch.tensor([0, 2, 3, 4]),
        seq_lens_cpu=torch.tensor([2, 1, 0]),
        seq_lens=torch.tensor([2, 1]),
        block_table_tensor=table,
        slot_mapping=slots,
    )
    meta = builder.build(0, common)
    assert meta.query_start_loc == [0, 2, 3]
    assert meta.seq_lens == [2, 1]
    assert meta.block_table is table
    assert meta.slot_mapping is slots
    captured = builder.build_for_cudagraph_capture(common)
    assert captured.decode_seq_lens is common.seq_lens
    common.seq_lens[0] = 3
    assert captured.decode_seq_lens.tolist() == [3, 1]


def test_text_model_provides_mrope_positions_for_mrv2():
    model = AscendHYV4FlashForCausalLM.__new__(AscendHYV4FlashForCausalLM)
    assert isinstance(model, SupportsMRoPE)
    positions, delta = model.get_mrope_input_positions([5, 6, 7], [])
    torch.testing.assert_close(positions, torch.arange(3).repeat(3, 1))
    assert delta == 0


def test_indexpool_ties_are_independent_of_graph_padding():
    query = torch.ones(1, 1, 2)
    weights = torch.ones(1, 1)
    keys = torch.ones(4, 2)
    position = torch.tensor([4])
    eager = sparse_candidates(query, weights, keys[:2], position, 2, 1)
    graph = sparse_candidates(query, weights, keys, position, 2, 1)
    torch.testing.assert_close(eager, torch.tensor([[0, 1, 4, -1]]))
    torch.testing.assert_close(graph, eager)


def test_quantized_kv_b_weight_restores_out_in_layout_and_keeps_address():
    layer = AscendHYV4FlashAttention.__new__(AscendHYV4FlashAttention)
    nn.Module.__init__(layer)
    layer.register_buffer("_kv_b_dense_weight", None, persistent=False)
    layer.kv_b_proj = SimpleNamespace(
        weight=torch.tensor([[1, 3, 5], [-2, 4, -6]], dtype=torch.int8),
        weight_scale=torch.tensor([0.5, 2.0, 0.25], dtype=torch.bfloat16),
        params_dtype=torch.bfloat16,
    )
    layer.prepare_kv_b_weight()
    weight = layer._kv_b_dense_weight
    expected = torch.tensor([[0.5, -1], [6, 8], [1.25, -1.5]], dtype=torch.bfloat16)
    torch.testing.assert_close(weight, expected)
    layer.kv_b_proj.weight_scale.mul_(2)
    layer.prepare_kv_b_weight()
    assert layer._kv_b_dense_weight is weight
    torch.testing.assert_close(weight, expected * 2)


def test_checkpoint_loader_splits_packed_int8_expert_weights_and_channel_parameters():
    model = HYV4FlashModel.__new__(HYV4FlashModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(num_experts=2)
    model.loaded_parts = {}
    routed = nn.Module()
    experts = nn.Module()
    experts.routed_experts = routed
    mlp = nn.Module()
    mlp.experts = experts
    decoder = nn.Module()
    decoder.mlp = mlp
    model.layers = nn.ModuleList([decoder])

    def load_shard(param, value, name, shard_id, expert_id, return_success):
        shard = 1 if shard_id == "w3" else 0
        param.data[expert_id].chunk(2)[shard].copy_(value)
        return True

    prefix = "layers.0.mlp.experts"
    weights = torch.arange(24, dtype=torch.int8).reshape(2, 4, 3)
    scales = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    for suffix, value in (("", weights), ("_scale", scales[..., None]), ("_offset", scales[..., None])):
        param = nn.Parameter(torch.zeros_like(value), requires_grad=False)
        param.weight_loader = load_shard
        routed.register_parameter(f"w13_weight{suffix}", param)

    model.load_weights(
        [
            (f"{prefix}.gate_up_proj.weight", weights),
            (f"{prefix}.gate_up_proj.weight_scale", scales),
            (f"{prefix}.gate_up_proj.weight_offset", -scales.to(torch.int8)),
        ]
    )
    torch.testing.assert_close(routed.w13_weight, weights)
    torch.testing.assert_close(routed.w13_weight_scale[..., 0], scales)
    torch.testing.assert_close(routed.w13_weight_offset[..., 0], -scales)
    model.loaded_parts.clear()
    with pytest.raises(ValueError, match="precision mismatch"):
        model.load_weights([(f"{prefix}.gate_up_proj.weight", weights.bfloat16())])


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
    padded = torch.empty((1, 4, 4), dtype=torch.int32)
    state.prepare_batch_hashes(torch.tensor([3, 5, 9, 9]), torch.tensor([[2, 1], [10, 4]]), [0, 1, 2], padded)
    torch.testing.assert_close(padded[:, :2], chunked)
    assert (padded[:, 2:] == -1).all()
