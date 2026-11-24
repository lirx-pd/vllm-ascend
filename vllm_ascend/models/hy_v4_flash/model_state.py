# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import triton

from vllm_ascend.ops.triton.engram_lookback import _gather_lookback_kernel
from vllm_ascend.worker.v2.model_states.default import AscendModelState


class FlashModelState(AscendModelState):
    supports_prompt_embeds = False

    def prepare_engram_inputs(self, input_batch, req_states):
        depth = self.model.token_lookback_depth
        if not depth:
            return {}
        num_reqs = input_batch.num_reqs
        history = torch.empty((num_reqs, depth), dtype=torch.int32, device=self.device)
        if input_batch.is_dummy:
            history.fill_(-1)
        elif num_reqs:
            all_token_ids = req_states.all_token_ids.gpu
            _gather_lookback_kernel[(num_reqs,)](
                history,
                input_batch.idx_mapping,
                req_states.num_computed_tokens.gpu,
                all_token_ids,
                all_token_ids.stride(0),
                num_reqs,
                DEPTH=depth,
                BLOCK_DEPTH=triton.next_power_of_2(depth),
            )
        return {"lookback_token_ids": history}
