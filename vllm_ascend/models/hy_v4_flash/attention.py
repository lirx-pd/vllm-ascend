# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch
import torch_npu
from torch import nn
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear, RowParallelLinear
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.rotary_embedding.mrope import apply_interleaved_rope
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.models.extract_hidden_states import CacheOnlyAttentionBackend, CacheOnlyAttentionImpl
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.hy_v4_flash.configs import SLIDING_ATTENTION
from vllm.models.hy_v4_flash.reference import (
    INDEX_DIM,
    indexpool_fp8,
    mla_attention_probs,
    pooled_keys,
    sparse_candidates,
    window_candidates,
)
from vllm.triton_utils import HAS_TRITON
from vllm.v1.attention.backend import AttentionCGSupport, AttentionMetadataBuilder
from vllm.v1.kv_cache_interface import HiddenStateCacheSpec

from vllm_ascend.ops.triton.hy_v4_indexpool import fused_indexpool_fp8, gather_paged_rows

logger = init_logger(__name__)

QUERY_CHUNK = 16
SWA_PREFILL_QUERY_CHUNK = 512
INDEXPOOL_PREFILL_QUERY_CHUNK = 256
# Limit extra QK/PV work when scoring sparse candidates over a request's KV.
INDEXPOOL_PREFILL_MAX_KV_RATIO = 8
# At most 64 MiB for each FP32 IndexPool per-head score temporary.
INDEXPOOL_PREFILL_SCORE_ELEMENTS = 16 * 1024 * 1024
INDEXPOOL_FUSED_MAX_ROWS = 16
FIA_BATCHED_KV_LIMIT = 512
FIA_LONG_KV_LIMIT = 2048
FIA_LONG_KV_MAX_BATCH = 4
POSITION_AXES = 3
POSITION_LANES = POSITION_AXES * torch.int64.itemsize // torch.bfloat16.itemsize


def _quantize_indexpool(vectors):
    if (
        HAS_TRITON
        and vectors.device.type == "npu"
        and vectors.is_contiguous()
        and 0 < vectors.numel() <= INDEXPOOL_FUSED_MAX_ROWS * INDEX_DIM
    ):
        return fused_indexpool_fp8(vectors)
    return indexpool_fp8(vectors)


def _prefill_pool_mask(query, weights, pool, logical, ratio, budget, visible_len):
    current = logical[:, None] // ratio
    positions = torch.arange(visible_len, device=logical.device)
    token_groups = positions // ratio
    future = positions > logical[:, None]
    in_current = token_groups == current
    if budget == 0:
        return future | ~in_current

    per_head = torch.matmul(query.transpose(0, 1), pool.T)
    scores = (per_head.relu() * weights.T.unsqueeze(-1)).sum(0)
    groups = torch.arange(pool.shape[0], device=logical.device)
    scores.masked_fill_(groups >= current, -torch.inf)

    count = min(budget, pool.shape[0])
    scores, selected = scores.sort(dim=-1, descending=True, stable=True)
    # The shared KV matmul only needs membership, so keep the stable recall
    # order and scatter the chosen groups directly into the mask.
    selected = selected[:, :count].masked_fill(scores[:, :count] == -torch.inf, pool.shape[0])
    keep = torch.zeros((logical.numel(), pool.shape[0] + 1), dtype=torch.int32, device=logical.device)
    keep.scatter_(1, selected, 1)
    # Only the current group can contain a visible token beyond the complete pool.
    from_history = keep.index_select(1, token_groups.clamp_max(pool.shape[0] - 1)) != 0
    return future | ~(from_history | in_current)


@dataclass
class FlashAttentionMetadata:
    query_start_loc: list[int]
    seq_lens: list[int]
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    decode_seq_lens: torch.Tensor | None = None


class FlashMetadataBuilder(AttentionMetadataBuilder[FlashAttentionMetadata]):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        meta = common_attn_metadata
        return FlashAttentionMetadata(
            meta.query_start_loc_cpu[: meta.num_reqs + 1].tolist(),
            meta.seq_lens_cpu[: meta.num_reqs].tolist(),
            meta.block_table_tensor,
            meta.slot_mapping,
        )

    def build_for_cudagraph_capture(self, common_attn_metadata):
        metadata = self.build(0, common_attn_metadata)
        metadata.decode_seq_lens = common_attn_metadata.seq_lens
        return metadata


class FlashAttentionImpl(CacheOnlyAttentionImpl):
    @staticmethod
    def update_graph_params(*args, **kwargs):
        # Captured ops read the device buffers that MRV2 refreshes before replay.
        pass


class FlashAttentionBackend(CacheOnlyAttentionBackend):
    @staticmethod
    def get_name():
        return "HYV4_FLASH_ASCEND"

    @staticmethod
    def get_builder_cls():
        return FlashMetadataBuilder

    @staticmethod
    def get_impl_cls():
        return FlashAttentionImpl


class FlashIndexPool(nn.Module):
    def __init__(self, config, quant_config, rope_parameters, prefix):
        super().__init__()
        self.ratio = config.compressed_indexer_compress_ratio
        self.budget = config.compressed_indexer_group_topk - 1
        self.num_heads = config.compressed_indexer_num_attention_heads
        self.rope_dim = config.qk_rope_head_dim
        self.theta = rope_parameters.get("rope_theta", 10000.0)
        self.sections = rope_parameters.get("mrope_section")
        self.interleaved = rope_parameters.get("mrope_interleaved", True)
        self.eps = config.rms_norm_eps
        if rope_parameters.get("rope_type", "default") != "default":
            raise ValueError("HYV4 Flash IndexPool requires default RoPE")

        self.query_projection = ReplicatedLinear(
            config.q_lora_rank,
            self.num_heads * INDEX_DIM,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.query_projection",
        )
        self.head_weight_projection = ReplicatedLinear(
            config.hidden_size,
            self.num_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.head_weight_projection",
        )
        self.token_key_projection = ReplicatedLinear(
            config.hidden_size,
            INDEX_DIM,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.token_key_projection",
        )
        self.token_key_norm = nn.RMSNorm(INDEX_DIM, eps=self.eps)
        self.key_value_projection = nn.Linear(INDEX_DIM, INDEX_DIM, bias=False, dtype=torch.float32)
        self.key_gate_projection = nn.Linear(INDEX_DIM, INDEX_DIM, bias=False, dtype=torch.float32)
        self.key_norm = nn.RMSNorm(INDEX_DIM, eps=self.eps, dtype=torch.float32)
        self.intra_group_ape = nn.Parameter(torch.empty(self.ratio, INDEX_DIM, dtype=torch.float32))

    def rotate(self, vectors, positions):
        inverse = self.theta ** (
            -torch.arange(0, self.rope_dim, 2, device=vectors.device, dtype=torch.float32) / self.rope_dim
        )
        phase = positions.float().unsqueeze(-1) * inverse
        if positions.ndim == 2:
            if self.interleaved:
                phase = apply_interleaved_rope(phase, self.sections)
            else:
                parts = phase.split(self.sections, dim=-1)
                phase = torch.cat([part[axis] for axis, part in enumerate(parts)], dim=-1)
        if vectors.ndim == 3:
            phase = phase.unsqueeze(1)
        nope, rotary = vectors.split((INDEX_DIM - self.rope_dim, self.rope_dim), dim=-1)
        even, odd = rotary.float()[..., ::2], rotary.float()[..., 1::2]
        cos, sin = phase.cos(), phase.sin()
        return torch.cat(
            (nope, (even * cos - odd * sin).to(vectors.dtype), (odd * cos + even * sin).to(vectors.dtype)), dim=-1
        )

    def encode_queries(self, hidden, q_c, positions):
        query = self.query_projection(q_c)[0].reshape(-1, self.num_heads, INDEX_DIM)
        weight = self.head_weight_projection(hidden)[0].float() * (self.num_heads * INDEX_DIM) ** -0.5
        return _quantize_indexpool(self.rotate(query, positions)), weight

    def encode_keys(self, raw, positions):
        pooled = pooled_keys(
            raw.reshape(-1, self.ratio, INDEX_DIM),
            self.key_value_projection.weight,
            self.key_gate_projection.weight,
            self.intra_group_ape,
            self.key_norm.weight,
            self.eps,
        )
        return _quantize_indexpool(self.rotate(pooled, positions[..., :: self.ratio]))


class AscendHYV4FlashAttention(nn.Module, AttentionLayerBase):
    supports_dcp = False

    def __init__(self, vllm_config, config, layer_idx, prefix):
        super().__init__()
        self.layer_name = f"{prefix}.attn"
        is_window_layer = config.layer_types[layer_idx] == SLIDING_ATTENTION
        self.window = config.sliding_window
        self.max_model_len = vllm_config.model_config.max_model_len
        dim_prefix = "swa_" if is_window_layer else ""
        heads = getattr(config, f"{dim_prefix}num_attention_heads")
        self.local_heads = heads // get_tensor_model_parallel_world_size()
        self.nope_dim = getattr(config, f"{dim_prefix}qk_nope_head_dim")
        self.rope_dim = getattr(config, f"{dim_prefix}qk_rope_head_dim")
        self.v_dim = getattr(config, f"{dim_prefix}v_head_dim")
        self.latent_dim = config.kv_lora_rank
        self.scale = (self.nope_dim + self.rope_dim) ** -0.5
        self.kv_width = self.latent_dim + self.rope_dim
        # Equal-width cache pages let SWA and IndexPool share one scheduler group.
        self.cache_width = self.kv_width + INDEX_DIM + POSITION_LANES
        self.kv_cache = torch.empty(0)
        cache_layers = vllm_config.compilation_config.static_forward_context
        if self.layer_name in cache_layers:
            raise ValueError(f"Duplicate HYV4 Flash cache layer {self.layer_name}")
        cache_layers[self.layer_name] = self

        hidden = config.hidden_size
        quant = vllm_config.quant_config
        self.q_a_proj = ReplicatedLinear(
            hidden, config.q_lora_rank, bias=False, quant_config=quant, prefix=f"{prefix}.q_a_proj"
        )
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            hidden, self.kv_width, bias=False, quant_config=quant, prefix=f"{prefix}.kv_a_proj_with_mqa"
        )
        self.q_a_layernorm = RMSNorm(config.q_lora_rank, config.rms_norm_eps)
        self.kv_a_layernorm = RMSNorm(self.latent_dim, config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            config.q_lora_rank,
            heads * (self.nope_dim + self.rope_dim),
            bias=False,
            quant_config=quant,
            prefix=f"{prefix}.q_b_proj",
        )
        self.kv_b_proj = ColumnParallelLinear(
            self.latent_dim,
            heads * (self.nope_dim + self.v_dim),
            bias=False,
            quant_config=quant,
            prefix=f"{prefix}.kv_b_proj",
        )
        # This weight is consumed by absorbed MLA in ND layout.
        self.kv_b_proj.skip_weight_nz_conversion = True
        self.kv_b_proj._fused_preprocess_managed = True
        self.register_buffer("_kv_b_dense_weight", None, persistent=False)
        self.linear_gate = ColumnParallelLinear(
            hidden, heads * self.v_dim, bias=False, quant_config=quant, prefix=f"{prefix}.linear_gate"
        )
        self.o_proj = RowParallelLinear(
            heads * self.v_dim, hidden, bias=False, quant_config=quant, prefix=f"{prefix}.o_proj"
        )
        rope_parameters = config.rope_for_layer(layer_idx)
        self.rotary_emb = get_rope(
            self.rope_dim,
            max_position=vllm_config.model_config.max_model_len,
            rope_parameters=rope_parameters,
            is_neox_style=False,
        )
        self.indexer = None if is_window_layer else FlashIndexPool(config, quant, rope_parameters, f"{prefix}.indexer")
        self.learnable_sink_param = nn.Parameter(torch.zeros(self.local_heads, dtype=torch.float32))
        set_weight_attrs(self.learnable_sink_param, {"weight_loader": sharded_weight_loader(0)})
        self.post_proj_norm = (
            RMSNorm(hidden, config.rms_norm_eps)
            if config.sandwich_norm and (is_window_layer or not config.dsa_skip_attn_post_norm)
            else None
        )

    def get_attn_backend(self):
        return FlashAttentionBackend

    def get_kv_cache_spec(self, vllm_config):
        return HiddenStateCacheSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.cache_width,
            dtype=torch.bfloat16,
        )

    @torch.inference_mode()
    def prepare_kv_b_weight(self):
        proj = self.kv_b_proj
        if proj.weight.dtype != torch.int8:
            return
        # ModelSlim's post-load INT8 weights are [in, out], with one scale per
        # output channel. Restore the dense [out, in] MLA projection once.
        weight = (proj.weight.T.float() * proj.weight_scale.float().reshape(-1, 1)).to(proj.params_dtype)
        if self._kv_b_dense_weight is None:
            self._kv_b_dense_weight = weight.contiguous()
        else:
            self._kv_b_dense_weight.copy_(weight)

    def forward(self, positions, hidden_states):
        q_c = self.q_a_layernorm(self.q_a_proj(hidden_states)[0])
        q = self.q_b_proj(q_c)[0].reshape(-1, self.local_heads, self.nope_dim + self.rope_dim)
        kv_c, k_pe = self.kv_a_proj_with_mqa(hidden_states)[0].split((self.latent_dim, self.rope_dim), -1)
        kv_c = self.kv_a_layernorm(kv_c)
        q_nope, q_pe = q.split((self.nope_dim, self.rope_dim), dim=-1)
        q_pe, k_pe = self.rotary_emb(positions, q_pe.flatten(1).contiguous(), k_pe.contiguous())
        q_pe = q_pe.reshape(-1, self.local_heads, self.rope_dim)
        out = hidden_states.new_zeros((q.shape[0], self.local_heads * self.v_dim))

        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, dict) and (meta := metadata.get(self.layer_name)) is not None:
            num_tokens = meta.query_start_loc[-1]
            if num_tokens:
                self._write_cache_rows(meta, hidden_states, kv_c, k_pe, positions, num_tokens)
                weight = self._kv_b_dense_weight if self._kv_b_dense_weight is not None else self.kv_b_proj.weight
                weight = weight.reshape(self.local_heads, self.nope_dim + self.v_dim, self.latent_dim)
                wk, wv = weight.split((self.nope_dim, self.v_dim), dim=1)
                q_latent = torch.bmm(q_nope[:num_tokens].transpose(0, 1).contiguous(), wk.contiguous()).transpose(0, 1)
                query = torch.cat((q_latent, q_pe[:num_tokens]), dim=-1)
                wv = wv.transpose(1, 2).contiguous()
                if meta.decode_seq_lens is None:
                    self._attend(meta, hidden_states, q_c, positions, query, wv, out)
                else:
                    self._attend_decode_graph(meta, hidden_states, q_c, positions, query, wv, out)

        output = self.o_proj(out * self.linear_gate(hidden_states)[0].sigmoid())[0]
        return self.post_proj_norm(output) if self.post_proj_norm is not None else output

    def _write_cache_rows(self, meta, hidden, kv_c, k_pe, positions, num_tokens):
        rows = [kv_c[:num_tokens], k_pe[:num_tokens]]
        if self.indexer is not None:
            keys = self.indexer.token_key_norm(self.indexer.token_key_projection(hidden[:num_tokens])[0]).bfloat16()
            coordinates = positions[..., :num_tokens]
            if coordinates.ndim == 1:
                coordinates = coordinates[:, None].expand(-1, POSITION_AXES)
            else:
                coordinates = coordinates.T
            # Bitcast MRoPE coordinates to preserve exact positions in the BF16 page.
            rows.extend((keys, coordinates.long().contiguous().view(torch.bfloat16)))
        else:
            rows.append(kv_c.new_zeros((num_tokens, INDEX_DIM + POSITION_LANES)))
        slots = meta.slot_mapping[:num_tokens].long()
        valid = slots >= 0
        cache = self.kv_cache.view(-1, self.cache_width)
        packed = torch.cat(rows, dim=-1)
        if meta.decode_seq_lens is not None:
            # Static shape during capture. Padding writes back the original
            # value of slot zero in the scheduler's reserved page.
            cache.index_copy_(0, slots.clamp_min(0), torch.where(valid[:, None], packed, cache[:1]))
        else:
            cache.index_copy_(0, slots[valid], packed[valid])

    def _attend(self, meta, hidden, q_c, positions, query, wv, out):
        num_tokens = query.shape[0]
        indexer = self.indexer
        candidate_width = (indexer.budget + 1) * indexer.ratio if indexer is not None else self.window
        cache_groups = indexer is not None and self.max_model_len > (indexer.budget + 1) * indexer.ratio
        uses_indexpool = cache_groups and any(
            (length - 1) // indexer.ratio > indexer.budget for length in meta.seq_lens
        )
        if uses_indexpool:
            index_query, index_weight = indexer.encode_queries(
                hidden[:num_tokens], q_c[:num_tokens], positions[..., :num_tokens]
            )
        for request, seq_len in enumerate(meta.seq_lens):
            start, stop = meta.query_start_loc[request : request + 2]
            if start == stop:
                continue
            table = meta.block_table[request]
            batched = stop - start > QUERY_CHUNK and (
                indexer is None or seq_len <= INDEXPOOL_PREFILL_MAX_KV_RATIO * candidate_width
            )
            pool = None
            if cache_groups:
                ratio = indexer.ratio
                context_len = seq_len - (stop - start)
                first = context_len // ratio * ratio
                complete = seq_len // ratio * ratio
                if first < complete:
                    history = torch.arange(first, complete, device=query.device)
                    self._store_index_pool(table, history, positions.ndim)

            key_start = 0
            request_kv = None
            rows = None
            chunk = QUERY_CHUNK
            if batched:
                logger.info_once("HYV4 Flash uses shared KV for batched prefill attention.")
                if indexer is None:
                    key_start = max(0, seq_len - (stop - start) + 1 - self.window)
                    chunk = min(SWA_PREFILL_QUERY_CHUNK, self.window)
                else:
                    chunk = INDEXPOOL_PREFILL_QUERY_CHUNK
                keys = torch.arange(key_start, seq_len, device=query.device)
                rows = self._gather(table, keys)
                key = rows[:, : self.kv_width].float().contiguous()
                request_kv = (key, key[:, : self.latent_dim].contiguous())

            if cache_groups and (seq_len - 1) // ratio > indexer.budget:
                if rows is not None:
                    pool = rows[ratio - 1 : complete : ratio, self.kv_width : self.kv_width + INDEX_DIM].float()
                    scores_per_query = pool.shape[0] * indexer.num_heads
                    chunk = min(chunk, max(1, INDEXPOOL_PREFILL_SCORE_ELEMENTS // scores_per_query))
                else:
                    group_ends = torch.arange(ratio - 1, complete, ratio, device=query.device)
                    pool = self._read_pool(table, group_ends)

            for lo in range(start, stop, chunk):
                hi = min(lo + chunk, stop)
                visible_len = seq_len - (stop - hi) - key_start
                logical = torch.arange(visible_len - (hi - lo), visible_len, device=query.device)

                if request_kv is not None:
                    key, value = request_kv
                    first = 0
                    if pool is not None and (visible_len - 1) // indexer.ratio > indexer.budget:
                        masked = _prefill_pool_mask(
                            index_query[lo:hi],
                            index_weight[lo:hi],
                            pool,
                            logical,
                            indexer.ratio,
                            indexer.budget,
                            visible_len,
                        )
                    else:
                        if indexer is None:
                            first = max(0, visible_len - (hi - lo) + 1 - self.window)
                        visible = torch.arange(first, visible_len, device=query.device)
                        masked = visible > logical[:, None]
                        if indexer is None:
                            masked |= visible <= logical[:, None] - self.window
                    self._attend_shared_kv(
                        key[first:visible_len], value[first:visible_len], masked, query[lo:hi], wv, out[lo:hi]
                    )
                    continue

                if indexer is None:
                    candidates = window_candidates(logical, min(self.window, visible_len))
                elif pool is None:
                    visible = torch.arange(visible_len, device=query.device)
                    candidates = visible.expand(hi - lo, -1).masked_fill(visible > logical[:, None], -1)
                else:
                    candidates = sparse_candidates(
                        index_query[lo:hi], index_weight[lo:hi], pool, logical, indexer.ratio, indexer.budget
                    )
                self._attend_rows(table, candidates, query[lo:hi], wv, out[lo:hi])

    def _attend_shared_kv(self, key, value, masked, query, wv, out):
        tokens, heads, _ = query.shape
        scores = torch.mm(query.flatten(0, 1).float(), key.T).view(tokens, heads, -1)
        scores.mul_(self.scale).masked_fill_(masked[:, None, :], -torch.inf)
        sink = self.learnable_sink_param.view(1, heads, 1).expand(tokens, -1, -1)
        probs = torch.cat((scores, sink), dim=-1).softmax(dim=-1)[..., :-1]
        context = torch.mm(probs.flatten(0, 1), value).view(tokens, heads, self.latent_dim).to(query.dtype)
        self._project_values(context, wv, out)

    def _attend_decode_graph(self, meta, hidden, q_c, positions, query, wv, out):
        num_tokens = query.shape[0]
        # The 910C runner captures one target query per request.
        live = meta.slot_mapping[:num_tokens] >= 0
        logical = (meta.decode_seq_lens[:num_tokens].long() - 1).masked_fill(~live, -1)
        table = meta.block_table[:num_tokens]
        max_context = min(self.max_model_len, table.shape[-1] * self.kv_cache.shape[2])
        indexer = self.indexer
        fused_gather = HAS_TRITON and query.device.type == "npu"
        if indexer is None:
            candidates = window_candidates(logical, min(self.window, max_context))
        elif max_context > (indexer.budget + 1) * indexer.ratio:
            index_query, index_weight = indexer.encode_queries(
                hidden[:num_tokens], q_c[:num_tokens], positions[..., :num_tokens]
            )
            ratio = indexer.ratio
            offsets = torch.arange(ratio, device=query.device)
            visible = logical[:, None]
            closes_group = (logical >= 0) & ((logical + 1) % ratio == 0)
            group = visible + 1 - ratio + offsets
            self._store_index_pool(table, group.masked_fill(~closes_group[:, None], 0), positions.ndim, closes_group)
            group_ends = torch.arange(ratio - 1, max_context, ratio, device=query.device)
            pool_positions = group_ends.expand(num_tokens, -1).masked_fill(group_ends > visible, -1)
            if fused_gather:
                pool = gather_paged_rows(self.kv_cache, table, pool_positions, self.kv_width, INDEX_DIM, torch.float32)
            else:
                pool = self._read_pool(table, pool_positions.clamp_min(0))
            candidates = sparse_candidates(index_query, index_weight, pool, logical, ratio, indexer.budget)
            dense = torch.arange(candidates.shape[-1], device=query.device)
            dense = dense.expand(num_tokens, -1).masked_fill(dense > visible, -1)
            candidates = torch.where(visible // ratio > indexer.budget, candidates, dense)
        else:
            dense = torch.arange(max_context, device=query.device)
            candidates = dense.expand(num_tokens, -1).masked_fill(dense > logical[:, None], -1)
        self._attend_rows(table, candidates, query, wv, out[:num_tokens], fused_gather=fused_gather)

    def _store_index_pool(self, table, positions, position_ndim, valid=None):
        ratio = self.indexer.ratio
        cache = self.kv_cache.view(-1, self.cache_width)
        slots = self._cache_slots(table, positions)
        rows = cache.index_select(0, slots)
        position_start = self.kv_width + INDEX_DIM
        coords = rows[:, position_start:].contiguous().view(torch.int64)
        keys = self.indexer.encode_keys(
            rows[:, self.kv_width : position_start], coords[:, 0] if position_ndim == 1 else coords.T
        )
        # The terminal token owns the summary even when a prefix shares earlier pages.
        # Power-of-two scaled E4M3 values fit exactly in the BF16 key lanes.
        terminal = rows[ratio - 1 :: ratio]
        updated = torch.cat((terminal[:, : self.kv_width], keys.to(rows.dtype), terminal[:, position_start:]), dim=-1)
        slots = slots[ratio - 1 :: ratio]
        if valid is not None:
            slots = slots.masked_fill(~valid, 0)
            updated = torch.where(valid[:, None], updated, cache[:1])
        cache.index_copy_(0, slots, updated)

    def _read_pool(self, table, group_ends):
        rows = self._gather(table, group_ends)
        return rows[..., self.kv_width : self.kv_width + INDEX_DIM].float()

    def _attend_rows(self, table, candidates, query, wv, out, *, fused_gather=False):
        if fused_gather:
            kv = gather_paged_rows(self.kv_cache, table, candidates, 0, self.kv_width, self.kv_cache.dtype)
        else:
            kv = self._gather(table, candidates.clamp_min(0))[..., : self.kv_width]
            kv = kv.masked_fill(candidates[..., None] < 0, 0)
        batch, kv_tokens = kv.shape[:2]
        # shortcut: A3 FIA can miscompute larger batches with long KV, retest on a newer CANN.
        if (
            fused_gather
            and query.dtype == torch.bfloat16
            and (self.latent_dim, self.rope_dim) == (512, 64)
            and query.shape[1] in (4, 8)
            and (
                batch == 1
                or kv_tokens <= FIA_BATCHED_KV_LIMIT
                or batch <= FIA_LONG_KV_MAX_BATCH
                and kv_tokens <= FIA_LONG_KV_LIMIT
            )
        ):
            context = self._fused_decode_mla(query, kv, candidates)
        else:
            probs = mla_attention_probs(query, kv, candidates, self.learnable_sink_param, self.scale)
            context = torch.bmm(probs, kv[..., : self.latent_dim].float()).to(query.dtype)
        self._project_values(context, wv, out)

    def _fused_decode_mla(self, query, kv, candidates):
        query, query_rope = (part.unsqueeze(2).contiguous() for part in query.split((512, 64), dim=-1))
        key, key_rope = (part.unsqueeze(1).contiguous() for part in kv.split((512, 64), dim=-1))
        mask = (candidates < 0)[:, None, None, :]
        context, lse = torch_npu.npu_fused_infer_attention_score_v2(
            query,
            key,
            key,
            query_rope=query_rope,
            key_rope=key_rope,
            atten_mask=mask,
            num_query_heads=query.shape[1],
            num_key_value_heads=1,
            softmax_scale=self.scale,
            input_layout="BNSD",
            sparse_mode=0,
            return_softmax_lse=True,
        )
        # D=512 MLA returns LSE. The zero-valued sink only changes the denominator.
        sink = self.learnable_sink_param.view(1, -1, 1)
        weight = (lse.squeeze(2) - sink).sigmoid()
        context = (context.squeeze(2).float() * weight).to(query.dtype)
        return context.masked_fill(mask.all(dim=-1), 0)

    @staticmethod
    def _project_values(context, wv, out):
        value = torch.bmm(context.transpose(0, 1).contiguous(), wv).transpose(0, 1)
        out.copy_(value.reshape_as(out))

    def _cache_slots(self, block_table, positions):
        block_size = self.kv_cache.shape[2]
        pages = positions // block_size
        blocks = (
            block_table.gather(1, pages.long())
            if block_table.ndim == 2
            else block_table.index_select(0, pages.flatten())
        )
        return blocks.flatten().long() * block_size + positions.flatten() % block_size

    def _gather(self, block_table, positions):
        cache = self.kv_cache
        slots = self._cache_slots(block_table, positions)
        return cache.view(-1, self.cache_width).index_select(0, slots).view(*positions.shape, self.cache_width)
