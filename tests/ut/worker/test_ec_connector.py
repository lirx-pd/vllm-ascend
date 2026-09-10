# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import torch
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT

from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake_ec_connector import ECMooncakeConnector
from vllm_ascend.worker.ec_connector import ec_connector_output
from vllm_ascend.worker.v2.ec_connector import AscendECConnector


class NoScanCache(dict):
    def keys(self):
        raise AssertionError("Do not scan the encoder cache")

    def __iter__(self):
        raise AssertionError("Do not scan the encoder cache")


@pytest.fixture
def transfer():
    connector = Mock(spec=ECMooncakeConnector)
    connector.is_producer = True
    connector.is_consumer = False
    connector.get_finished.return_value = ({"sent"}, {"received"})
    connector.build_connector_worker_meta.return_value = {"ready": True}
    scheduler_output = SimpleNamespace(
        ec_connector_metadata=object(), finished_req_ids={"finished"}, scheduled_encoder_inputs={}
    )
    cache = NoScanCache(cached=torch.zeros(1))
    adapter = object.__new__(AscendECConnector)
    adapter.ec_connector = connector
    adapter.encoder_cache = cache
    adapter.mm_features = {}
    return connector, scheduler_output, cache, adapter


def test_v1_does_not_save_encoded_items_twice(transfer):
    connector, scheduler_output, cache, _ = transfer
    model = object()
    with ec_connector_output(connector, scheduler_output, cache, model=model) as output:
        cache["new"] = torch.ones(1)
        connector.save_caches(encoder_cache=cache, mm_hash="new")

    connector.start_save_caches.assert_called_once_with(encoder_cache=cache, model=model)
    connector.save_caches.assert_called_once_with(encoder_cache=cache, mm_hash="new")
    connector.start_load_caches.assert_not_called()
    assert output.finished_sending == {"sent"}
    assert output.finished_recving == {"received"}
    assert output.ec_connector_worker_meta == {"ready": True}
    connector.clear_connector_metadata.assert_called_once_with()


@pytest.mark.parametrize("producer,consumer", [(True, False), (False, True), (True, True)])
def test_v2_lifecycle_does_not_scan_or_publish_cache(transfer, producer, consumer):
    connector, scheduler_output, cache, adapter = transfer
    connector.is_producer = producer
    connector.is_consumer = consumer

    def load_cache(encoder_cache):
        encoder_cache["loaded"] = torch.full((1,), 2.0)

    connector.start_load_caches.side_effect = load_cache
    with adapter.maybe_get_output(scheduler_output) as output:
        connector.bind_connector_metadata.assert_called_once_with(scheduler_output.ec_connector_metadata)
        assert connector.start_save_caches.call_count == int(producer)
        assert connector.start_load_caches.call_count == int(consumer)
        connector.save_caches.assert_not_called()
        cache["new"] = torch.ones(1)

    if producer:
        connector.start_save_caches.assert_called_once_with(encoder_cache=cache)
    connector.save_caches.assert_not_called()
    if consumer:
        connector.start_load_caches.assert_called_once_with(cache)
        assert "loaded" in cache
    connector.get_finished.assert_called_once_with(scheduler_output.finished_req_ids)
    assert output.finished_sending == {"sent"}
    assert output.finished_recving == {"received"}
    assert output.ec_connector_worker_meta == {"ready": True}
    assert connector.mock_calls[-1] == call.clear_connector_metadata()


def test_missing_metadata_is_noop_only_for_v2(transfer):
    connector, scheduler_output, cache, adapter = transfer
    scheduler_output.ec_connector_metadata = None
    with adapter.maybe_get_output(scheduler_output) as output:
        assert output is None
    assert adapter.no_forward(scheduler_output) is EMPTY_MODEL_RUNNER_OUTPUT
    with (
        pytest.raises(RuntimeError, match="EC connector metadata is required"),
        ec_connector_output(connector, scheduler_output, cache),
    ):
        pytest.fail("Missing metadata must fail before entering the context")
    assert connector.mock_calls == []


@pytest.mark.parametrize(
    "failure",
    ["start_save_caches", "start_load_caches", "forward", "save_caches", "get_finished", "build_connector_worker_meta"],
)
def test_failures_always_clear_bound_metadata(transfer, failure):
    connector, scheduler_output, cache, adapter = transfer
    connector.is_consumer = True
    if failure != "forward":
        getattr(connector, failure).side_effect = RuntimeError("transfer failed")

    with pytest.raises(RuntimeError, match="transfer failed"), adapter.maybe_get_output(scheduler_output):
        cache["new"] = torch.ones(1)
        if failure == "forward":
            raise RuntimeError("transfer failed")
        if failure == "save_caches":
            connector.save_caches(encoder_cache=cache, mm_hash="new")

    connector.get_finished.assert_called_once_with(scheduler_output.finished_req_ids)
    connector.clear_connector_metadata.assert_called_once_with()
    assert connector.mock_calls[-1] == call.clear_connector_metadata()
    if failure in ("start_save_caches", "start_load_caches", "forward"):
        connector.save_caches.assert_not_called()


def test_v2_no_forward_preserves_metadata_without_mutating_empty_output(transfer):
    connector, scheduler_output, _, adapter = transfer
    connector.get_finished.return_value = (None, None)
    original_ec_output = EMPTY_MODEL_RUNNER_OUTPUT.ec_connector_output

    output = adapter.no_forward(scheduler_output)

    assert output is not EMPTY_MODEL_RUNNER_OUTPUT
    assert output.req_ids == []
    assert output.ec_connector_output.ec_connector_worker_meta == {"ready": True}
    assert EMPTY_MODEL_RUNNER_OUTPUT.ec_connector_output is original_ec_output
    connector.start_save_caches.assert_called_once()
    connector.clear_connector_metadata.assert_called_once_with()


@pytest.mark.parametrize("consumer", [False, True])
def test_v2_publishes_scheduled_outputs_only_after_whole_batch(transfer, consumer):
    connector, scheduler, cache, adapter = transfer
    connector.is_consumer = consumer
    adapter.mm_features = {
        "request": [SimpleNamespace(identifier=name, data=object()) for name in ("cached", "loaded", "first", "second")]
    }
    scheduler.scheduled_encoder_inputs = {"request": [0, 1, 2, 2, 3]}
    if consumer:
        connector.start_load_caches.side_effect = lambda cache: cache.update(loaded=torch.zeros(1))
    with adapter.maybe_get_output(scheduler):
        cache["first"] = torch.ones(1)
        connector.save_caches.assert_not_called()
        cache["second"] = torch.ones(1)
        cache["unrelated"] = torch.ones(1)
        connector.save_caches.assert_not_called()
    assert [item.kwargs["mm_hash"] for item in connector.save_caches.call_args_list] == ["first", "second"]


def test_v2_failed_batch_does_not_publish_partial_outputs(transfer):
    connector, scheduler, cache, adapter = transfer
    adapter.mm_features = {"request": [SimpleNamespace(identifier="first", data=object())]}
    scheduler.scheduled_encoder_inputs = {"request": [0]}
    with pytest.raises(RuntimeError, match="encode failed"), adapter.maybe_get_output(scheduler):
        cache["first"] = torch.ones(1)
        raise RuntimeError("encode failed")
    connector.save_caches.assert_not_called()
    connector.clear_connector_metadata.assert_called_once()


def test_v2_passthrough_is_published_at_batch_end(transfer):
    connector, scheduler, cache, adapter = transfer
    adapter.mm_features = {
        "request": [SimpleNamespace(identifier="embedding", modality="prompt_embeds", data=object())]
    }
    scheduler.scheduled_encoder_inputs = {"request": [0]}
    with adapter.maybe_get_output(scheduler):
        cache["embedding"] = torch.ones(1)
        connector.save_caches.assert_not_called()
    connector.save_caches.assert_called_once_with(encoder_cache=cache, mm_hash="embedding")
