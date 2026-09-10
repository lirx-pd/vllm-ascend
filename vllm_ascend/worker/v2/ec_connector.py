# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager

from vllm.config import VllmConfig
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu.ec_connector import ActiveECConnector
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache

from vllm_ascend.worker.ec_connector import ec_connector_output


class AscendECConnector(ActiveECConnector):
    def __init__(self, vllm_config: VllmConfig, encoder_cache: EncoderCache) -> None:
        super().__init__(vllm_config, encoder_cache.encoder_outputs)
        self.mm_features = encoder_cache.mm_features

    @contextmanager
    def maybe_get_output(self, scheduler_output: SchedulerOutput):
        if scheduler_output.ec_connector_metadata is None:
            yield None
            return
        with ec_connector_output(self.ec_connector, scheduler_output, self.encoder_cache) as output:
            # Track only scheduled cache misses, after consumer loads have finished.
            pending: dict[str, None] = {}
            if self.ec_connector.is_producer:
                for req_id, input_ids in scheduler_output.scheduled_encoder_inputs.items():
                    for input_id in input_ids:
                        feature = self.mm_features[req_id][input_id]
                        if feature.data is not None and feature.identifier not in self.encoder_cache:
                            pending[feature.identifier] = None
            yield output
            # Preserve batch-end publication without scanning resident cache entries.
            for mm_hash in pending:
                if mm_hash in self.encoder_cache:
                    self.ec_connector.save_caches(encoder_cache=self.encoder_cache, mm_hash=mm_hash)
