# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.models.hy_v4.nvidia.hc import HYV4HCLayer
from vllm.triton_utils import HAS_TRITON, tl, triton


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
