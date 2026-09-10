# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EC transfer lifecycle shared by the Ascend model runners."""

from collections.abc import Generator
from contextlib import contextmanager

import torch
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorBase
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ECConnectorOutput


@contextmanager
def ec_connector_output(
    connector: ECConnectorBase,
    scheduler_output: SchedulerOutput,
    encoder_cache: dict[str, torch.Tensor],
    **kwargs,
) -> Generator[ECConnectorOutput, None, None]:
    metadata = scheduler_output.ec_connector_metadata
    if metadata is None:
        raise RuntimeError("EC connector metadata is required")
    output = ECConnectorOutput()
    connector.bind_connector_metadata(metadata)
    try:
        if connector.is_producer:
            connector.start_save_caches(encoder_cache=encoder_cache, **kwargs)
        if connector.is_consumer:
            connector.start_load_caches(encoder_cache, **kwargs)

        yield output
    finally:
        try:
            output.finished_sending, output.finished_recving = connector.get_finished(scheduler_output.finished_req_ids)
            output.ec_connector_worker_meta = connector.build_connector_worker_meta()
        finally:
            connector.clear_connector_metadata()
