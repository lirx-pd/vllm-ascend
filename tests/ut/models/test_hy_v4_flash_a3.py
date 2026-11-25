# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("vllm.models.hy_v4_flash")

from vllm.config import LoadConfig  # noqa: E402
from vllm.model_executor.model_loader.reload import record_metadata_for_reloading  # noqa: E402
from vllm.model_executor.model_loader.reload.inplace import reload_weights  # noqa: E402
from vllm.model_executor.models.interfaces import SupportsMRoPE  # noqa: E402
from vllm.models.hy_v4_flash.model import HYV4FlashModel  # noqa: E402
from vllm.models.hy_v4_flash.reference import INDEX_DIM, mla_attention_probs, sparse_candidates  # noqa: E402

from vllm_ascend.models.hy_v4_flash import attention, engram  # noqa: E402
from vllm_ascend.models.hy_v4_flash.attention import (  # noqa: E402
    POSITION_LANES,
    AscendHYV4FlashAttention,
    FlashAttentionMetadata,
    FlashIndexPool,
    FlashMetadataBuilder,
)
from vllm_ascend.models.hy_v4_flash.model import AscendHYV4FlashForCausalLM  # noqa: E402
from vllm_ascend.quantization.method_adapters import AscendLinearMethod  # noqa: E402
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import AscendW8A8DynamicLinearMethod  # noqa: E402


def _index_pool(ratio):
    indexer = FlashIndexPool.__new__(FlashIndexPool)
    nn.Module.__init__(indexer)
    indexer.ratio, indexer.budget, indexer.num_heads = ratio, 1, 1
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


@pytest.mark.parametrize("ratio", [None, 2, 3], ids=["swa", "indexpool", "indexpool-cross-page"])
@pytest.mark.parametrize("batched_prefill", [False, True], ids=["small-tiles", "shared-kv"])
@torch.inference_mode()
def test_paged_prefill_and_decode_match_uninterrupted_attention(ratio, batched_prefill, monkeypatch):
    torch.manual_seed(9)
    if batched_prefill:
        monkeypatch.setattr(attention, "QUERY_CHUNK", 1)
        monkeypatch.setattr(attention, "SWA_PREFILL_QUERY_CHUNK", 2)
        monkeypatch.setattr(attention, "INDEXPOOL_PREFILL_QUERY_CHUNK", 2)
    layer = AscendHYV4FlashAttention.__new__(AscendHYV4FlashAttention)
    nn.Module.__init__(layer)
    layer.indexer = _index_pool(ratio) if ratio is not None else None
    layer.window = 3
    layer.max_model_len = 8
    layer.latent_dim, layer.kv_width = 2, 3
    layer.scale = 0.5
    layer.learnable_sink_param = nn.Parameter(torch.zeros(1))
    layer.cache_width = layer.kv_width + INDEX_DIM + POSITION_LANES
    cache_shape = (8, 1, 2, layer.cache_width)
    table = torch.tensor([[3, 1, 5, 7], [6, 2, 4, 0]], dtype=torch.int32)
    hidden = torch.randn(2, 7, INDEX_DIM).bfloat16()
    kv = torch.randn(2, 7, layer.kv_width).bfloat16()
    query = torch.randn(2, 7, 1, layer.kv_width).bfloat16()
    wv = torch.eye(layer.latent_dim, dtype=torch.bfloat16).unsqueeze(0)
    # Consecutive coordinates above the BF16 integer precision test cache bitcasts.
    coordinates = torch.arange(7) + torch.tensor([[32768], [65536], [131072]])

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
    with monkeypatch.context() as context:
        gathers = []
        gather = layer._gather

        def record_gather(block_table, positions):
            gathers.append(positions.shape)
            return gather(block_table, positions)

        context.setattr(layer, "_gather", record_gather)
        full = run([0, 0], [7, 4])
        if batched_prefill:
            assert gathers == [(7,), (4,)]
    layer.kv_cache.zero_()
    chunked = [run([0, 0], [3, 2]), run([3, 2], [6, 3]), run([6, 3], [7, 4])]
    for request, expected in enumerate(full):
        torch.testing.assert_close(torch.cat([chunk[request] for chunk in chunked]), expected)
        if ratio is None:
            for pos, actual in enumerate(expected):
                rows = kv[request, max(0, pos - layer.window + 1) : pos + 1].float()
                logits = rows @ query[request, pos, 0].float() * layer.scale
                probs = torch.cat([logits, torch.zeros(1)]).softmax(0)[:-1]
                torch.testing.assert_close(actual, (probs @ rows[:, : layer.latent_dim]).bfloat16())
        else:
            rows = layer._gather(table[request], torch.arange(len(expected)))
            stored = rows[:, -POSITION_LANES:].contiguous().view(torch.int64).T
            torch.testing.assert_close(stored, coordinates[:, : len(expected)])
            # Compare with pooling directly from the original token keys.
            indexer = layer.indexer
            complete = len(expected) // ratio * ratio
            raw = indexer.token_key_norm(hidden[request, :complete])
            keys = indexer.encode_keys(raw, coordinates[:, :complete])
            group_ends = torch.arange(ratio - 1, complete, ratio)
            torch.testing.assert_close(layer._read_pool(table[request], group_ends), keys, rtol=0, atol=0)
            for pos, actual in enumerate(expected):
                logical = torch.tensor([pos])
                if pos // ratio > indexer.budget:
                    x = hidden[request, pos : pos + 1]
                    q, w = indexer.encode_queries(x, x, coordinates[:, pos : pos + 1])
                    candidates = sparse_candidates(q, w, keys, logical, ratio, indexer.budget)
                else:
                    candidates = torch.arange(pos + 1).unsqueeze(0)
                selected = kv[request, candidates.clamp_min(0)].masked_fill(candidates[..., None] < 0, 0)
                probs = mla_attention_probs(
                    query[request, pos : pos + 1], selected, candidates, layer.learnable_sink_param, layer.scale
                )
                value = torch.bmm(probs, selected[..., : layer.latent_dim].float()).bfloat16()
                torch.testing.assert_close(actual, value.flatten())
    torch.testing.assert_close(layer.kv_cache[0], torch.zeros_like(layer.kv_cache[0]))

    layer.kv_cache.zero_()
    run([0, 0], [5, 2])
    pool_batches = []
    if ratio is not None:
        encode_keys = layer.indexer.encode_keys

        def encode_current_group(raw, positions):
            pool_batches.append(raw.shape[0])
            return encode_keys(raw, positions)

        monkeypatch.setattr(layer.indexer, "encode_keys", encode_current_group)

    graph_lengths = torch.tensor([6, 3])
    graph = run([5, 2], [6, 3], decode_seq_lens=graph_lengths)
    torch.testing.assert_close(graph[0], full[0][5:6])
    torch.testing.assert_close(graph[1], full[1][2:3])
    graph_lengths.copy_(torch.tensor([7, 4]))
    graph = run([6, 3], [7, 4], decode_seq_lens=graph_lengths)
    for graph_row, eager_row in zip(graph, chunked[-1]):
        torch.testing.assert_close(graph_row, eager_row)
    layer.kv_cache[0].fill_(17)
    cache_before = layer.kv_cache.clone()
    graph_lengths[0] = 5
    padded = run([4, 3], [5, 4], decode_seq_lens=graph_lengths, pad_last_request=True)
    torch.testing.assert_close(padded[0], full[0][4:5])
    torch.testing.assert_close(padded[1], torch.zeros_like(padded[1]))
    torch.testing.assert_close(layer.kv_cache, cache_before)
    if ratio is not None:
        assert pool_batches == [2 * ratio] * 3

    if ratio == 3:
        # Two requests share the first page and diverge before this group closes.
        layer.kv_cache.zero_()
        table[1, 0] = table[0, 0]
        run([0, 2], [3, 2])
        cache_before = layer.kv_cache.clone()
        run([3, 2], [3, 3])
        torch.testing.assert_close(layer.kv_cache[table[0, :2]], cache_before[table[0, :2]], rtol=0, atol=0)
        raw = layer.indexer.token_key_norm(torch.cat((hidden[0, :2], hidden[1, 2:3])))
        expected = encode_keys(raw, coordinates[:, :3])
        torch.testing.assert_close(layer._read_pool(table[1], torch.tensor([2])), expected, rtol=0, atol=0)


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


@pytest.mark.parametrize("budget", [0, 1, 2, 5])
@pytest.mark.parametrize("pool_groups", [4, 6])
def test_prefill_pool_mask_preserves_stable_recall_and_partial_tail(budget, pool_groups):
    logical = torch.tensor([0, 2, 3, 5, 8, 10, 11, 13])
    query = torch.ones(len(logical), 1, 2)
    weights = torch.tensor([[1.0], [-1.0]]).repeat(4, 1)
    pool = torch.tensor([[2.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 0.0], [2.0, 0.0], [0.0, 0.0]])[:pool_groups]
    ratio, visible_len = 3, 14
    candidates = sparse_candidates(query, weights, pool, logical, ratio, budget)
    expected = torch.ones(len(logical), visible_len + 1, dtype=torch.bool)
    expected.scatter_(1, candidates.masked_fill(candidates < 0, visible_len), False)

    actual = attention._prefill_pool_mask(query, weights, pool, logical, ratio, budget, visible_len)
    torch.testing.assert_close(actual, expected[:, :visible_len], atol=0, rtol=0)


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


def _flash_model(decoder, num_experts=0):
    config = SimpleNamespace(
        engram_layer_ids=[],
        num_hidden_layers=1,
        num_nextn_predict_layers=0,
        tie_word_embeddings=False,
        num_experts=num_experts,
    )
    model = HYV4FlashModel.__new__(HYV4FlashModel)
    nn.Module.__init__(model)
    model.config = config
    model.loaded_parts = {}
    model.layers = nn.ModuleList([decoder])
    root = AscendHYV4FlashForCausalLM.__new__(AscendHYV4FlashForCausalLM)
    nn.Module.__init__(root)
    root.model = model
    root.config = config
    root._uses_symmetric_w8a8 = True
    root._load_config = LoadConfig()
    root._weight_loading_depth = 0
    root._seen_weights = set()
    root._loaded_weights = set()
    return root


@pytest.mark.parametrize("scale_suffix", [".weight_scale", "_scale"])
@pytest.mark.parametrize("with_vision", [False, True], ids=["text", "vl"])
def test_checkpoint_loader_splits_packed_int8_expert_weights_and_channel_parameters(scale_suffix, with_vision):
    routed = nn.Module()
    routed.expert_map_manager = SimpleNamespace(expert_map=None)
    experts = nn.Module()
    experts.routed_experts = routed
    mlp = nn.Module()
    mlp.experts = experts
    decoder = nn.Module()
    decoder.mlp = mlp
    model = _flash_model(decoder, num_experts=2)
    visual_weights = []
    if with_vision:
        from vllm_ascend.models.hy_v4_flash.vl_model import HYV4FlashVLForConditionalGeneration

        wrapper = HYV4FlashVLForConditionalGeneration.__new__(HYV4FlashVLForConditionalGeneration)
        nn.Module.__init__(wrapper)
        wrapper.language_model = model
        wrapper.visual = nn.Module()
        wrapper.visual.proj = nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
        model = wrapper
        visual_weights = [("vit.vit.proj.weight", torch.ones((2, 2), dtype=torch.bfloat16))]

    def load_shard(param, value, name, shard_id, expert_id, return_success):
        shard = 1 if shard_id == "w3" else 0
        param.data[expert_id].chunk(2)[shard].copy_(value)
        return True

    prefix = "model.layers.0.mlp.experts.gate_up_proj"
    weights = torch.arange(24, dtype=torch.int8).reshape(2, 4, 3)
    scales = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    for suffix, value in (("", weights), ("_scale", scales[..., None]), ("_offset", scales[..., None])):
        param = nn.Parameter(torch.full_like(value, -1), requires_grad=False)
        param.weight_loader = load_shard
        routed.register_parameter(f"w13_weight{suffix}", param)

    checkpoint = [(f"{prefix}.weight", weights), (f"{prefix}{scale_suffix}", scales)] + visual_weights
    model.load_weights(checkpoint)
    torch.testing.assert_close(routed.w13_weight, weights)
    torch.testing.assert_close(routed.w13_weight_scale[..., 0], scales)
    torch.testing.assert_close(routed.w13_weight_offset[..., 0], torch.zeros_like(scales))
    model.load_weights(checkpoint + [(f"{prefix}.weight_offset", torch.zeros_like(scales, dtype=torch.int8))])
    with pytest.raises(ValueError, match="requires a zero weight offset"):
        model.load_weights(checkpoint + [(f"{prefix}.weight_offset", scales.to(torch.int8))])
    with pytest.raises(ValueError, match="Missing Flash checkpoint weights.*weight_scale"):
        model.load_weights(checkpoint[:1])
    with pytest.raises(ValueError, match="precision mismatch"):
        model.load_weights([(f"{prefix}.weight", weights.bfloat16())])


@pytest.mark.parametrize("entry", ["checkpoint", "kernel", "transfer"])
def test_worker_reload_refreshes_kv_b_and_fp32_scale_in_place(entry):
    from vllm_ascend.worker.worker import NPUWorker

    decoder = nn.Module()
    attn = AscendHYV4FlashAttention.__new__(AscendHYV4FlashAttention)
    nn.Module.__init__(attn)
    attn.register_buffer("_kv_b_dense_weight", None, persistent=False)
    decoder.self_attn = attn
    proj = nn.Module()
    proj.params_dtype = torch.bfloat16
    proj._fused_preprocess_managed = True
    attn.kv_b_proj = proj
    scheme = AscendW8A8DynamicLinearMethod()
    proj.quant_method = AscendLinearMethod(scheme)
    params = scheme.get_weight(2, 3, proj.params_dtype) | scheme.get_perchannel_param(3, proj.params_dtype)
    for name, value in params.items():
        proj.register_parameter(name, nn.Parameter(value, requires_grad=False))
    model = _flash_model(decoder)
    record_metadata_for_reloading(model)
    prefix = "model.layers.0.self_attn.kv_b_proj"
    model.load_weights(
        [
            (prefix + ".weight", torch.ones(3, 2, dtype=torch.int8)),
            (prefix + ".weight_scale", torch.ones(3, dtype=torch.bfloat16)),
        ]
    )
    proj.quant_method.process_weights_after_loading(proj)
    model.process_weights_after_loading()
    dense_before = attn._kv_b_dense_weight
    scale_before = proj.weight_scale_fp32
    config = SimpleNamespace(
        model_config=SimpleNamespace(quantization="ascend", dtype=proj.params_dtype),
        load_config=model._load_config,
        lora_config=None,
    )
    weight = torch.arange(6, dtype=torch.int8).reshape(3, 2)
    scale = torch.tensor([1, 0.5, 2], dtype=torch.bfloat16)

    def reload_model(is_checkpoint_format):
        value = weight if is_checkpoint_format else weight.T.contiguous()
        reload_weights(
            config,
            model,
            iter([(prefix + ".weight", value), (prefix + ".weight_scale", scale)]),
            None,
            is_checkpoint_format,
        )

    worker = NPUWorker.__new__(NPUWorker)
    worker.model_runner = SimpleNamespace(
        get_model=lambda: model, reload_weights=reload_model, reset_lora_state=lambda: None
    )
    if entry == "transfer":
        worker._weight_update_active = True
        worker.weight_transfer_engine = SimpleNamespace(finish_weight_update=lambda: reload_model(False))
        worker.finish_weight_update()
    else:
        worker.reload_weights(is_checkpoint_format=(entry == "checkpoint"))

    assert attn._kv_b_dense_weight is dense_before
    assert proj.weight_scale_fp32 is scale_before
    torch.testing.assert_close(dense_before, weight.bfloat16() * scale[:, None])
    torch.testing.assert_close(scale_before, scale.float())

    if entry == "kernel":
        worker.model_runner.reload_weights = lambda: reload_weights(
            config, model, iter([(prefix + ".weight_offset", torch.ones_like(scale))]), None, False
        )
        with pytest.raises(ValueError, match="requires a zero weight offset"):
            worker.reload_weights()


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
