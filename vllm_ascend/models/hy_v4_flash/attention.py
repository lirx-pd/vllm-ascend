# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch
from torch import nn
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear, RowParallelLinear
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.rotary_embedding.mrope import apply_interleaved_rope
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.models.extract_hidden_states import CacheOnlyAttentionBackend
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
from vllm.v1.attention.backend import AttentionMetadataBuilder
from vllm.v1.kv_cache_interface import HiddenStateCacheSpec

# Process bounded query tiles in eager mode. The cache remains scheduler-owned.
QUERY_CHUNK = 16
POSITION_AXES = 3
POSITION_LANES = POSITION_AXES * torch.int64.itemsize // torch.bfloat16.itemsize


@dataclass
class FlashAttentionMetadata:
    query_start_loc: list[int]
    seq_lens: list[int]
    block_table: torch.Tensor
    slot_mapping: torch.Tensor


class FlashMetadataBuilder(AttentionMetadataBuilder[FlashAttentionMetadata]):
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


class FlashAttentionBackend(CacheOnlyAttentionBackend):
    @staticmethod
    def get_name():
        return "HYV4_FLASH_ASCEND"

    @staticmethod
    def get_builder_cls():
        return FlashMetadataBuilder


class FlashIndexPool(nn.Module):
    def __init__(self, config, rope_parameters, prefix):
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
            config.q_lora_rank, self.num_heads * INDEX_DIM, bias=False, prefix=f"{prefix}.query_projection"
        )
        self.head_weight_projection = ReplicatedLinear(
            config.hidden_size, self.num_heads, bias=False, prefix=f"{prefix}.head_weight_projection"
        )
        self.token_key_projection = ReplicatedLinear(
            config.hidden_size, INDEX_DIM, bias=False, prefix=f"{prefix}.token_key_projection"
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
        return indexpool_fp8(self.rotate(query, positions)), weight

    def encode_keys(self, raw, positions):
        pooled = pooled_keys(
            raw.reshape(-1, self.ratio, INDEX_DIM),
            self.key_value_projection.weight,
            self.key_gate_projection.weight,
            self.intra_group_ape,
            self.key_norm.weight,
            self.eps,
        )
        return indexpool_fp8(self.rotate(pooled, positions[..., :: self.ratio]))


class AscendHYV4FlashAttention(nn.Module, AttentionLayerBase):
    supports_dcp = False

    def __init__(self, vllm_config, config, layer_idx, prefix):
        super().__init__()
        self.layer_name = f"{prefix}.attn"
        is_window_layer = config.layer_types[layer_idx] == SLIDING_ATTENTION
        self.window = config.sliding_window
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
        self.q_a_proj = ReplicatedLinear(hidden, config.q_lora_rank, bias=False, prefix=f"{prefix}.q_a_proj")
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            hidden, self.kv_width, bias=False, prefix=f"{prefix}.kv_a_proj_with_mqa"
        )
        self.q_a_layernorm = RMSNorm(config.q_lora_rank, config.rms_norm_eps)
        self.kv_a_layernorm = RMSNorm(self.latent_dim, config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            config.q_lora_rank, heads * (self.nope_dim + self.rope_dim), bias=False, prefix=f"{prefix}.q_b_proj"
        )
        self.kv_b_proj = ColumnParallelLinear(
            self.latent_dim, heads * (self.nope_dim + self.v_dim), bias=False, prefix=f"{prefix}.kv_b_proj"
        )
        self.kv_b_proj.skip_weight_nz_conversion = True
        self.linear_gate = ColumnParallelLinear(hidden, heads * self.v_dim, bias=False, prefix=f"{prefix}.linear_gate")
        self.o_proj = RowParallelLinear(heads * self.v_dim, hidden, bias=False, prefix=f"{prefix}.o_proj")
        rope_parameters = config.rope_for_layer(layer_idx)
        self.rotary_emb = get_rope(
            self.rope_dim,
            max_position=vllm_config.model_config.max_model_len,
            rope_parameters=rope_parameters,
            is_neox_style=False,
        )
        self.indexer = None if is_window_layer else FlashIndexPool(config, rope_parameters, f"{prefix}.indexer")
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
                weight = self.kv_b_proj.weight.reshape(self.local_heads, self.nope_dim + self.v_dim, self.latent_dim)
                wk, wv = weight.split((self.nope_dim, self.v_dim), dim=1)
                q_latent = torch.bmm(q_nope[:num_tokens].transpose(0, 1).contiguous(), wk.contiguous()).transpose(0, 1)
                query = torch.cat((q_latent, q_pe[:num_tokens]), dim=-1)
                self._attend(meta, hidden_states, q_c, positions, query, wv.transpose(1, 2).contiguous(), out)

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
        self.kv_cache.view(-1, self.cache_width).index_copy_(0, slots[valid], torch.cat(rows, dim=-1)[valid])

    def _attend(self, meta, hidden, q_c, positions, query, wv, out):
        num_tokens = query.shape[0]
        indexer = self.indexer
        uses_indexpool = indexer is not None and any(
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
            pool = None
            if uses_indexpool and (seq_len - 1) // indexer.ratio > indexer.budget:
                # shortcut: recompute complete groups in eager mode, cache them when tuning decode latency.
                complete = seq_len // indexer.ratio * indexer.ratio
                history = torch.arange(complete, device=query.device)
                rows = self._gather(table, history)[:, self.kv_width :]
                keys = rows[:, :INDEX_DIM]
                coords = rows[:, INDEX_DIM:].contiguous().view(torch.int64)
                pool = indexer.encode_keys(keys, coords[:, 0] if positions.ndim == 1 else coords.T)

            for lo in range(start, stop, QUERY_CHUNK):
                hi = min(lo + QUERY_CHUNK, stop)
                visible_len = seq_len - (stop - hi)
                logical = torch.arange(visible_len - (hi - lo), visible_len, device=query.device)
                if indexer is None:
                    candidates = window_candidates(logical, min(self.window, visible_len))
                elif pool is not None:
                    candidates = sparse_candidates(
                        index_query[lo:hi], index_weight[lo:hi], pool, logical, indexer.ratio, indexer.budget
                    )
                else:
                    visible = torch.arange(visible_len, device=query.device)
                    candidates = visible.expand(hi - lo, -1).masked_fill(visible > logical[:, None], -1)
                kv = self._gather(table, candidates.clamp_min(0))[..., : self.kv_width]
                probs = mla_attention_probs(query[lo:hi], kv, candidates, self.learnable_sink_param, self.scale)
                context = torch.bmm(probs, kv[..., : self.latent_dim].float()).to(query.dtype)
                value = torch.bmm(context.transpose(0, 1).contiguous(), wv).transpose(0, 1)
                out[lo:hi].copy_(value.reshape_as(out[lo:hi]))

    def _gather(self, block_table, positions):
        cache = self.kv_cache
        block_size = cache.shape[2]
        pages = (positions // block_size).flatten()
        blocks = block_table.index_select(0, pages).long()
        slots = blocks * block_size + positions.flatten() % block_size
        return cache.view(-1, self.cache_width).index_select(0, slots).view(*positions.shape, self.cache_width)
