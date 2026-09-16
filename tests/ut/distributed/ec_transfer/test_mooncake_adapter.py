# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
import threading
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ECConnectorOutput

from vllm_ascend.distributed.ec_transfer import register_connector
from vllm_ascend.distributed.ec_transfer.mooncake import (
    AscendECMooncakeConnector,
    AscendMooncakeTransfer,
    _AscendECMooncakeWorker,
)
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend.worker.v2.mm_encoder_model_runner import NPUEncoderModelRunner
from vllm_ascend.worker.v2.model_runner import NPUModelRunner as NPUModelRunnerV2


def test_registers_concrete_ascend_connector() -> None:
    with patch.dict(ECConnectorFactory._registry):
        register_connector()
        connector = ECConnectorFactory._registry["ECMooncakeConnector"]()

    assert connector is AscendECMooncakeConnector
    assert not inspect.isabstract(connector)


@pytest.mark.parametrize(
    "runner_cls, is_producer",
    [(NPUEncoderModelRunner, True), (NPUModelRunnerV2, False)],
    ids=["encoder-producer", "pd-consumer"],
)
@pytest.mark.parametrize("has_metadata", [False, True])
def test_v2_idle_step_returns_ec_completions(runner_cls, has_metadata, is_producer):
    runner, connector = _make_v2_ec_runner(runner_cls, is_producer)
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=0,
        ec_connector_metadata=object() if has_metadata else None,
        finished_req_ids={"finished"},
    )
    output = runner.execute_model(scheduler_output)
    runner.model_state.execute_mm_encoder.assert_not_called()
    if has_metadata:
        if is_producer:
            assert output.ec_connector_output.finished_sending == {"image"}
            assert output.ec_connector_output.finished_recving is None
            connector.start_save_caches.assert_called_once()
            connector.start_load_caches.assert_not_called()
        else:
            assert output.ec_connector_output.finished_sending is None
            assert output.ec_connector_output.finished_recving == {"image"}
            connector.start_load_caches.assert_called_once_with(runner.ec_connector.encoder_cache)
            connector.start_save_caches.assert_not_called()
        connector.get_finished.assert_called_once_with({"finished"})
        connector.clear_connector_metadata.assert_called_once()
    else:
        assert output.ec_connector_output is None
        connector.bind_connector_metadata.assert_not_called()
    assert EMPTY_MODEL_RUNNER_OUTPUT.ec_connector_output is None


@pytest.mark.parametrize("fail_encoder", [False, True])
def test_v2_encoder_publishes_new_cache_and_clears_metadata(fail_encoder):
    runner, connector = _make_v2_ec_runner(NPUEncoderModelRunner)
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=4,
        num_scheduled_tokens={"request": 4},
        scheduled_encoder_inputs={"request": [0]},
        ec_connector_metadata=object(),
        finished_req_ids=set(),
    )
    cache = runner.ec_connector.encoder_cache
    cache["old"] = object()

    def encode(inputs):
        assert inputs == {"request": [0]}
        if fail_encoder:
            raise RuntimeError("encoder failed")
        cache["new"] = object()

    runner.model_state.execute_mm_encoder.side_effect = encode
    if fail_encoder:
        with pytest.raises(RuntimeError, match="encoder failed"):
            runner.execute_model(scheduler_output)
        connector.save_caches.assert_not_called()
    else:
        output = runner.execute_model(scheduler_output)
        assert output.req_ids == ["request"]
        assert output.sampled_token_ids == [[]]
        assert output.ec_connector_output.finished_sending == {"image"}
        connector.save_caches.assert_called_once_with(encoder_cache=cache, mm_hash="new")
    connector.start_save_caches.assert_called_once_with(encoder_cache=cache)
    connector.clear_connector_metadata.assert_called_once()
    assert runner.get_kv_cache_spec() == {}
    assert runner.capture_model() == 0


def _make_v2_ec_runner(runner_cls, is_producer=True):
    from vllm.v1.worker.gpu.ec_connector import ActiveECConnector
    from vllm.v1.worker.gpu.kv_connector import NO_OP_KV_CONNECTOR

    runner = runner_cls.__new__(runner_cls)
    runner._input_events = [Mock()]
    runner._input_event_idx = 0
    runner.lora_config = None
    runner.ascend_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(profiling_chunk_config=SimpleNamespace(need_timing=False))
    )
    runner.kvpp = Mock()
    runner.model_state = Mock()
    runner.kv_connector = NO_OP_KV_CONNECTOR
    for method in ("update_pp_decode_requests", "finish_requests", "free_states", "add_requests", "update_requests"):
        setattr(runner, method, Mock())
    runner.block_tables = Mock()
    runner.gather_batch_req_state = Mock(return_value=(SimpleNamespace(num_tokens=4), None))
    runner.prepare_inputs = Mock()
    connector = Mock(is_producer=is_producer, is_consumer=not is_producer)
    connector.get_finished.return_value = ({"image"}, None) if is_producer else (None, {"image"})
    connector.build_connector_worker_meta.return_value = None
    ec = ActiveECConnector.__new__(ActiveECConnector)
    ec.encoder_cache = {}
    ec.ec_connector = connector
    ec.save_new_caches = is_producer
    runner.ec_connector = ec
    return runner, connector


def test_rejects_non_ascend_transport() -> None:
    with pytest.raises(ValueError, match="mooncake_protocol='ascend'"):
        AscendMooncakeTransfer("127.0.0.1", "rdma")


def test_transfer_uses_shared_engine_for_registration_and_write() -> None:
    engine = Mock()
    engine.get_rpc_port.return_value = 1234
    engine.batch_register_memory.return_value = 0
    engine.batch_transfer_sync_write.return_value = 0
    tensor = Mock(nbytes=4096)
    tensor.data_ptr.return_value = 2097152
    with patch(
        "vllm_ascend.distributed.ec_transfer.mooncake.global_te.get_transfer_engine",
        return_value=engine,
    ) as get_engine:
        transfer = AscendMooncakeTransfer("127.0.0.1", "ascend")
        get_engine.assert_not_called()
        transfer.ensure_ready()
        assert transfer.local_session() == "127.0.0.1:1234"
        assert transfer.register_memory(tensor) == 0
        transfer.write("peer:1234", [2097152], [4194304], [4096])
        get_engine.assert_called_once_with("127.0.0.1", device_name=None)
    engine.batch_register_memory.assert_called_once_with([2097152], [4096])
    engine.batch_transfer_sync_write.assert_called_once_with("peer:1234", [2097152], [4194304], [4096])
    engine.batch_transfer_sync_write.return_value = -1
    with pytest.raises(RuntimeError, match="failed with status -1"):
        transfer.write("peer:1234", [2097152], [4194304], [4096])


@pytest.mark.parametrize("close_status", [0, -1])
def test_transfer_retains_failed_unregistration_until_close(close_status: int) -> None:
    engine = Mock()
    engine.unregister_memory.return_value = -1
    engine.batch_unregister_memory.return_value = close_status
    tensor = Mock(nbytes=4096)
    tensor.data_ptr.return_value = 2097152
    with patch(
        "vllm_ascend.distributed.ec_transfer.mooncake.global_te.get_transfer_engine",
        return_value=engine,
    ):
        transfer = AscendMooncakeTransfer("127.0.0.1", "ascend")
        transfer.register_memory(tensor)
        assert not transfer.unregister_memory(tensor)
        assert transfer._pending_unregister[2097152] is tensor
        transfer.close()
        transfer.close()
    engine.batch_unregister_memory.assert_called_once_with([2097152])
    assert transfer._pending_unregister == ({2097152: tensor} if close_status else {})
    engine.close.assert_not_called()


def test_transfer_rejects_direct_source_registration() -> None:
    transfer = AscendMooncakeTransfer("127.0.0.1", "ascend")
    with patch("vllm_ascend.distributed.ec_transfer.mooncake.global_te.get_transfer_engine") as get_engine:
        with pytest.raises(RuntimeError, match="aligned staging pool"):
            transfer.acquire_sources([Mock()])
        assert transfer.release_sources([])
        transfer.close()
        get_engine.assert_not_called()


@pytest.mark.parametrize("is_producer, is_consumer", [(True, False), (False, True), (True, True)])
def test_worker_reuses_upstream_lifecycle_with_lazy_ascend_data_plane(is_producer: bool, is_consumer: bool) -> None:
    config = SimpleNamespace(
        is_producer=is_producer,
        is_consumer=is_consumer,
        control_host="127.0.0.1",
        control_port=14579,
        control_timeout_ms=10,
        buffer_device="cuda",
        pool_size=4096,
        protocol="ascend",
    )
    vllm_config = SimpleNamespace(ec_transfer_config=SimpleNamespace(ec_connector_extra_config={}))
    with (
        patch(
            "vllm_ascend.distributed.ec_transfer.mooncake.MooncakeECConfig.from_vllm_config",
            return_value=config,
        ),
        patch(
            "vllm.distributed.ec_transfer.ec_connector.mooncake.transfer.MooncakeTransfer._ensure_engine",
            side_effect=AssertionError("upstream engine must stay lazy"),
        ),
        patch(
            "vllm_ascend.distributed.ec_transfer.mooncake.global_te.get_transfer_engine",
            side_effect=AssertionError("Ascend engine must stay lazy"),
        ),
    ):
        worker = _AscendECMooncakeWorker(vllm_config)

        try:
            assert worker._buffer_device == "npu"
            assert isinstance(worker._transfer, AscendMooncakeTransfer)
            assert worker._consumer_memory._transfer is worker._transfer
            assert worker._producer_memory._transfer is worker._transfer
            assert worker._reservations._memory is worker._consumer_memory
            assert worker._consumer_memory.tensor is None
            assert worker._producer_memory.tensor is None
            assert worker._control_server is None
            if is_producer:
                dispatched = threading.Event()

                def submit_batches(*args, **kwargs):
                    dispatched.set()
                    return False

                with patch.object(worker._producer_pushes, "submit_batches", side_effect=submit_batches):
                    worker._push_ready.set()
                    assert dispatched.wait(timeout=5), "upstream dispatcher did not process work"
            else:
                assert worker._dispatcher is None
        finally:
            worker.close()
        assert worker._shutdown
        if is_producer:
            assert not worker._dispatcher.is_alive()


def test_v1_no_forward_preserves_ec_output() -> None:
    runner = object.__new__(NPUModelRunner)
    runner.encoder_cache = {}
    runner.vllm_config = object()
    ec_output = ECConnectorOutput()

    @contextmanager
    def output_context(*args, **kwargs):
        yield ec_output
        ec_output.finished_sending = {"image"}

    runner.maybe_get_ec_connector_output = output_context
    with (
        patch("vllm_ascend.worker.model_runner_v1.has_ec_transfer", return_value=True),
        patch("vllm_ascend.worker.model_runner_v1.has_kv_transfer_group", return_value=False),
    ):
        output = runner._no_forward_output(SimpleNamespace(ec_connector_metadata=object()))

    assert output is not EMPTY_MODEL_RUNNER_OUTPUT
    assert output.ec_connector_output is ec_output
    assert output.ec_connector_output.finished_sending == {"image"}
    assert EMPTY_MODEL_RUNNER_OUTPUT.ec_connector_output is None


def test_v1_no_forward_without_metadata_is_noop() -> None:
    runner = object.__new__(NPUModelRunner)
    runner.encoder_cache = {}
    runner.vllm_config = object()
    with (
        patch("vllm_ascend.worker.model_runner_v1.has_ec_transfer", return_value=True),
        patch("vllm_ascend.worker.model_runner_v1.has_kv_transfer_group", return_value=False),
    ):
        output = runner._no_forward_output(SimpleNamespace(ec_connector_metadata=None))

    assert output is EMPTY_MODEL_RUNNER_OUTPUT
