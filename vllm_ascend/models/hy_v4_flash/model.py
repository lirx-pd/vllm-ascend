# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.config.compilation import CompilationMode
from vllm.models.hy_v4_flash.model import HYV4FlashDecoderLayer, HYV4FlashForCausalLM, HYV4FlashModel

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType

from .attention import AscendHYV4FlashAttention, FlashAttentionBackend
from .engram import FlashEngram, FlashNgramHashState


def validate_runtime(vllm_config: VllmConfig) -> None:
    model = vllm_config.model_config
    config = model.hf_text_config
    config.validate_flash_config()
    if get_ascend_device_type() is not AscendDeviceType.A3:
        raise ValueError("This HYV4 Flash implementation requires Atlas A3 (Ascend 910C)")
    compilation = vllm_config.compilation_config
    if (
        not vllm_config.use_v2_model_runner
        or compilation.mode != CompilationMode.NONE
        or compilation.cudagraph_mode != CUDAGraphMode.NONE
    ):
        raise ValueError("HYV4 Flash on 910C requires model runner V2 and --enforce-eager")
    if model.dtype != torch.bfloat16 or model.quantization is not None or vllm_config.quant_config is not None:
        raise ValueError("HYV4 Flash on 910C requires a BF16 checkpoint without weight quantization")
    if not FlashAttentionBackend.supports_kv_cache_dtype(vllm_config.cache_config.cache_dtype):
        raise ValueError("HYV4 Flash on 910C requires a BF16 KV cache")
    if vllm_config.speculative_config is not None or vllm_config.kv_transfer_config is not None:
        raise ValueError("HYV4 Flash on 910C requires speculative decoding and KV transfer to be disabled")
    if vllm_config.lora_config is not None:
        raise ValueError("HYV4 Flash on 910C does not support LoRA")
    parallel = vllm_config.parallel_config
    for field in ("pipeline_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size"):
        if getattr(parallel, field) != 1:
            raise ValueError(f"HYV4 Flash on 910C requires {field}=1")
    ascend_config = get_ascend_config()
    if (
        parallel.use_ubatching
        or parallel.enable_eplb
        or parallel.use_sequence_parallel_moe
        or ascend_config.enable_fused_mc2
        or ascend_config.enable_force_eplb
        or ascend_config.enable_shared_expert_dp
    ):
        raise ValueError(
            "HYV4 Flash on 910C requires microbatching, EPLB, FlashComm, fused MC2 and shared-expert DP off"
        )
    storage = vllm_config.engram_config
    if storage is not None and (
        storage.embedding_across_dp or storage.dp_shared_memory or getattr(storage, "disk_offload", False)
    ):
        raise ValueError("HYV4 Flash on 910C uses private TP-local Engram tables")
    if config.engram_layer_ids:
        if not config.enable_ihc:
            raise ValueError("HYV4 Flash Engram requires iHC")
        if parallel.tensor_parallel_size > (config.engram_max_ngram_size - 1) * config.engram_n_head_per_ngram:
            raise ValueError("HYV4 Flash Engram requires at least one hash head per tensor-parallel rank")
    if not (config.learnable_sink and config.gated_mla and config.gating_type == "elementwise"):
        raise ValueError("HYV4 Flash requires learned sinks and elementwise MLA gates")
    if (
        config.gate_lora_rank is not None
        or config.o_lora_rank is not None
        or config.gate_groups != 1
        or config.o_groups != 1
        or config.full_nope
        or not config.qk_norm
        or config.qk_rope_head_dim != config.swa_qk_rope_head_dim
    ):
        raise ValueError("HYV4 Flash on 910C requires ungrouped MLA projections and equal rotary widths")
    for field in ("num_attention_heads", "swa_num_attention_heads"):
        if getattr(config, field) % parallel.tensor_parallel_size:
            raise ValueError(f"{field} must be divisible by tensor_parallel_size")


class AscendHYV4FlashDecoderLayer(HYV4FlashDecoderLayer):
    attention_cls = AscendHYV4FlashAttention


class AscendHYV4FlashModel(HYV4FlashModel):
    decoder_layer_cls = AscendHYV4FlashDecoderLayer
    engram_cls = FlashEngram
    engram_hash_cls = FlashNgramHashState


class AscendHYV4FlashForCausalLM(HYV4FlashForCausalLM):
    model_cls = AscendHYV4FlashModel

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        validate_runtime(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    @staticmethod
    def get_model_state_cls():
        from .model_state import FlashModelState

        return FlashModelState
