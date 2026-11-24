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


def _validate_weight_offset(name, offset):
    if torch.count_nonzero(offset):
        raise ValueError(f"HYV4 Flash W8A8_DYNAMIC requires a zero weight offset for {name}")


def _with_symmetric_weight_offsets(weights):
    for name, value in weights:
        if name.endswith(".weight_offset"):
            _validate_weight_offset(name, value)
            continue
        yield name, value
        # Use the scale's shape and loader route for every TP/EP shard.
        if name.endswith(".weight_scale"):
            yield name.removesuffix("_scale") + "_offset", torch.zeros_like(value)
        elif ".experts." in name and name.endswith("_proj_scale"):
            yield name.removesuffix("_scale") + ".weight_offset", torch.zeros_like(value)


@torch.inference_mode()
def refresh_reloaded_weights(model):
    for name, offset in model.named_parameters():
        if name.endswith((".weight_offset", ".w13_weight_offset", ".w2_weight_offset")):
            _validate_weight_offset(name, offset)
    for module in model.modules():
        for name in ("weight_scale", "w13_weight_scale"):
            if (scale := getattr(module, f"{name}_fp32", None)) is not None:
                scale.copy_(getattr(module, name))
    model.process_weights_after_loading()


def validate_runtime(vllm_config: VllmConfig) -> None:
    model = vllm_config.model_config
    config = model.hf_text_config
    config.validate_flash_config()
    if get_ascend_device_type() is not AscendDeviceType.A3:
        raise ValueError("This HYV4 Flash implementation requires Atlas A3 (Ascend 910C)")
    if not vllm_config.use_v2_model_runner:
        raise ValueError("HYV4 Flash on 910C requires model runner V2")
    compilation = vllm_config.compilation_config
    if (
        compilation.cudagraph_mode not in (CUDAGraphMode.NONE, CUDAGraphMode.FULL_DECODE_ONLY)
        or compilation.cudagraph_mode == CUDAGraphMode.NONE
        and compilation.mode != CompilationMode.NONE
    ):
        raise ValueError("HYV4 Flash on 910C supports eager or FULL_DECODE_ONLY ACLGraph")
    if model.dtype != torch.bfloat16:
        raise ValueError("HYV4 Flash on 910C requires BF16 compute")
    quant_config = vllm_config.quant_config
    if (quant_config is not None or model.quantization is not None) and (
        model.quantization != "ascend" or quant_config is None
    ):
        raise ValueError("HYV4 Flash on 910C supports BF16 or ModelSlim W8A8_DYNAMIC checkpoints")
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
    supports_mrope = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        validate_runtime(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._uses_symmetric_w8a8 = vllm_config.quant_config is not None

    def prepare_checkpoint_weights(self, weights):
        return _with_symmetric_weight_offsets(weights) if self._uses_symmetric_w8a8 else weights

    def load_weights(self, weights):
        # The VL wrapper processes the entire stream before splitting off the tower.
        if self._weight_loading_depth == 0:
            weights = self.prepare_checkpoint_weights(weights)
        return super().load_weights(weights)

    def process_weights_after_loading(self):
        for layer in self.model.layers:
            layer.self_attn.prepare_kv_b_weight()

    def process_weights_after_reload(self):
        refresh_reloaded_weights(self)

    @staticmethod
    def get_mrope_input_positions(input_tokens, mm_features):
        return torch.arange(len(input_tokens), device="cpu").expand(3, -1), 0

    @staticmethod
    def get_model_state_cls():
        from .model_state import FlashModelState

        return FlashModelState
