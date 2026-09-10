# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.worker.mm_encoder_model_runner import MMEncoderModelRunner

from vllm_ascend.worker.v2.model_runner import NPUModelRunner


class NPUEncoderModelRunner(MMEncoderModelRunner, NPUModelRunner):
    """Use the upstream encoder-only lifecycle with Ascend buffers and EC transfer.

    MMEncoderModelRunner delegates initialization through NPUModelRunner, and
    its encoder-only execution uses the Ascend input preparation methods.
    """
