# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm import envs
from vllm.models.hy_v4.nvidia.hc import HYV4HCLayer, HYV4HCPreLayer
from vllm.models.hy_v4.nvidia.triton_ihc import _ihc_pre_stage1, _ihc_pre_stage2
from vllm.triton_utils import HAS_TRITON, tl, triton


class AscendHYV4FlashHCPreLayer(HYV4HCPreLayer):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, channels, hidden = x.shape
        if not (
            HAS_TRITON
            and not envs.VLLM_BATCH_INVARIANT
            and x.device.type == "npu"
            and x.dtype == torch.bfloat16
            and x.is_contiguous()
            and self.hc_fn.weight.is_contiguous()
            and 0 < tokens <= 16
            and (channels, hidden) == (4, 3072)
        ):
            return super().forward(x)

        tile = 1024
        width = channels * hidden
        splits = width // tile
        partial_stride = 1 + 2 * channels
        partial = torch.empty((tokens, splits, partial_stride), device=x.device, dtype=torch.float32)
        output = x.new_empty((tokens, hidden))
        post = partial.new_empty((tokens, channels))
        _ihc_pre_stage1[(tokens, splits)](
            x,
            self.hc_fn.weight,
            partial,
            width,
            channels,
            channels,
            splits,
            tile,
            partial_stride,
            enable_fp_fusion=False,
        )
        _ihc_pre_stage2[(tokens, hidden // tile)](
            x,
            partial,
            self.hc_scale,
            self.hc_base,
            output,
            post,
            hidden,
            width,
            channels,
            channels,
            splits,
            partial_stride,
            tile,
            self.magnitude,
            self.layernorm_epsilon,
            self.hc_eps,
            enable_fp_fusion=False,
        )
        return output, post


@triton.jit
def _ihc_post_kernel(hidden, residual, gates, output, HIDDEN: tl.constexpr, CHANNELS: tl.constexpr, TILE: tl.constexpr):
    token = tl.program_id(0)
    channel = tl.program_id(1)
    col = tl.program_id(2) * TILE + tl.arange(0, TILE)
    lane = token * CHANNELS + channel
    valid = col < HIDDEN
    x = tl.load(hidden + token * HIDDEN + col, valid, other=0).to(tl.float32)
    r = tl.load(residual + lane * HIDDEN + col, valid, other=0).to(tl.float32)
    gate = tl.load(gates + lane)
    tl.store(output + lane * HIDDEN + col, r + gate * x, valid)


class AscendHYV4FlashHCLayer(HYV4HCLayer):
    pre_layer_cls = AscendHYV4FlashHCPreLayer

    def post(self, output_with_bias, residual, post_gates):
        if (
            HAS_TRITON
            and self.enable_ihc
            and output_with_bias.device.type == "npu"
            and output_with_bias.dtype == residual.dtype == torch.bfloat16
            and post_gates is not None
            and post_gates.dtype == torch.float32
            and output_with_bias.shape[0] > 0
            and output_with_bias.is_contiguous()
            and residual.is_contiguous()
            and post_gates.is_contiguous()
        ):
            tokens, channels, hidden_size = residual.shape
            output = torch.empty_like(residual)
            tile_size = 1024
            _ihc_post_kernel[(tokens, channels, triton.cdiv(hidden_size, tile_size))](
                output_with_bias,
                residual,
                post_gates,
                output,
                hidden_size,
                channels,
                tile_size,
                enable_fp_fusion=False,
            )
            return output
        return super().post(output_with_bias, residual, post_gates)
