# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for Ascend Mooncake EC configuration and slab ownership."""

from __future__ import annotations

import ast
import importlib.util
import logging
import os
import socket
import sys
import time
import types
import unittest
from collections import Counter
from concurrent.futures import Future
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

_ROOT = Path(__file__).resolve().parents[4]
_MOONCAKE = _ROOT / "vllm_ascend" / "distributed" / "ec_transfer" / "ec_connector" / "mooncake"


def _module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    sys.modules[name] = module
    return module


def _load(name: str, filename: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, _MOONCAKE / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _Role(Enum):
    SCHEDULER = 0
    WORKER = 1


class _FakeEngine:
    def __init__(self) -> None:
        self.registered: list[tuple[int, int]] = []
        self.unregistered: list[int] = []
        self.writes: list[tuple[str, list[int], list[int], list[int]]] = []
        self.fail_unregister_once = False

    def get_rpc_port(self) -> int:
        return 19001

    def register_memory(self, address: int, nbytes: int) -> int:
        self.registered.append((address, nbytes))
        return 0

    def unregister_memory(self, address: int) -> int:
        self.unregistered.append(address)
        if self.fail_unregister_once:
            self.fail_unregister_once = False
            return -1
        return 0

    def batch_transfer_sync_write(
        self,
        session: str,
        sources: list[int],
        destinations: list[int],
        lengths: list[int],
    ) -> int:
        self.writes.append((session, sources, destinations, lengths))
        return 0


class _FakeGlobalTransferEngine:
    def __init__(self, engine: _FakeEngine) -> None:
        self.engine = engine
        self.calls: list[tuple[str, None]] = []

    def get_transfer_engine(self, hostname: str, device_name: None) -> _FakeEngine:
        self.calls.append((hostname, device_name))
        return self.engine


class _FakeTensor:
    def __init__(self, address: int = 0x1000, nbytes: int = 4096) -> None:
        self.address = address
        self.nbytes = nbytes

    def data_ptr(self) -> int:
        return self.address

    def narrow(self, _dimension: int, start: int, length: int) -> _FakeTensor:
        return _FakeTensor(self.address + start, length)


class TestMooncakeNPUDataPlane(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module_names = (
            "ec_mooncake_test_config",
            "ec_mooncake_test_transfer",
            "ec_mooncake_test_memory",
            "vllm",
            "vllm.distributed",
            "vllm.distributed.ec_transfer",
            "vllm.distributed.ec_transfer.ec_connector",
            "vllm_ascend",
            "vllm_ascend.distributed",
            "vllm_ascend.distributed.kv_transfer",
            "vllm_ascend.distributed.kv_transfer.utils",
            "vllm.distributed.ec_transfer.ec_connector.base",
            "vllm.distributed.ec_transfer.ec_connector.cpu",
            "vllm.distributed.ec_transfer.ec_connector.cpu.common",
            "vllm.logger",
            "vllm.v1",
            "vllm.v1.core",
            "vllm.v1.core.sched",
            "vllm.v1.core.sched.output",
            "vllm.v1.outputs",
            "vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine",
            "vllm_ascend.distributed.ec_transfer",
            "vllm_ascend.distributed.ec_transfer.ec_connector",
            "vllm_ascend.distributed.ec_transfer.ec_connector.mooncake",
            "vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.transfer",
            "vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.metadata",
            "vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.state",
            "vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.config",
            "vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.control",
            "vllm_ascend.distributed.ec_transfer.ec_connector.mooncake._availability",
            "vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.scheduler",
            "vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.producer",
            "torch",
        )
        cls.original_modules = {name: sys.modules.get(name) for name in cls.module_names}
        for package in cls.module_names[3:11]:
            _module(package)
        for package in (
            "vllm_ascend.distributed.ec_transfer",
            "vllm_ascend.distributed.ec_transfer.ec_connector",
            "vllm_ascend.distributed.ec_transfer.ec_connector.mooncake",
        ):
            _module(package)

        base = _module("vllm.distributed.ec_transfer.ec_connector.base")
        base.ECConnectorRole = _Role
        base.ECConnectorMetadata = type("ECConnectorMetadata", (), {})
        common = _module("vllm.distributed.ec_transfer.ec_connector.cpu.common")
        common._get_encoder_cache_hidden_dim = lambda _config: 1
        scheduler_output = _module("vllm.v1.core.sched.output")
        scheduler_output.SchedulerOutput = object
        outputs = _module("vllm.v1.outputs")
        outputs.ECConnectorOutput = object
        logger = _module("vllm.logger")
        logger.init_logger = logging.getLogger

        cls.engine = _FakeEngine()
        cls.global_te = _FakeGlobalTransferEngine(cls.engine)
        shared_engine = _module("vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine")
        shared_engine.global_te = cls.global_te

        torch = _module("torch")
        torch.Tensor = _FakeTensor
        torch.uint8 = "uint8"
        torch.empty = lambda size, **_: _FakeTensor(0x401000, size)
        cls.config_module = _load("ec_mooncake_test_config", "config.py")
        cls.transfer_module = _load("ec_mooncake_test_transfer", "transfer.py")
        sys.modules["vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.transfer"] = cls.transfer_module
        cls.memory_module = _load("ec_mooncake_test_memory", "memory.py")

        prefix = "vllm_ascend.distributed.ec_transfer.ec_connector.mooncake"
        sys.modules[f"{prefix}.memory"] = cls.memory_module
        cls.metadata_module = _load(f"{prefix}.metadata", "metadata.py")
        cls.state_module = _load(f"{prefix}.state", "state.py")
        sys.modules[f"{prefix}.config"] = cls.config_module
        availability = _module(f"{prefix}._availability")
        availability.ensure_mooncake_available = lambda: None
        control = _module(f"{prefix}.control")
        control.ControlClient = object
        control.EventInbox = object
        control.make_cancel_request = lambda *args: args
        cls.scheduler_module = _load(f"{prefix}.scheduler", "scheduler.py")
        cls.producer_module = _load(f"{prefix}.producer", "producer.py")
        cls.control_module = _load(f"{prefix}.control", "control.py")

    @classmethod
    def tearDownClass(cls) -> None:
        for name, module in cls.original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def test_registered_pool_is_2_mib_aligned(self) -> None:
        alignment = 2 * 1024 * 1024
        pool = self.memory_module._allocate_registered_pool(
            3 * 1024 * 1024,
            "npu:0",
        )

        self.assertEqual(pool.data_ptr() % alignment, 0)
        self.assertEqual(pool.nbytes, 4 * 1024 * 1024)

    def test_fixed_cuda_default_maps_to_npu_and_ascend(self) -> None:
        ec_config = SimpleNamespace(
            is_ec_producer=True,
            is_ec_consumer=False,
            ec_buffer_size=4096,
            ec_buffer_device="cuda",
            ec_connector_extra_config={},
        )
        parallel_config = SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            data_parallel_index=0,
        )
        config = self.config_module.MooncakeECConfig.from_vllm_config(
            SimpleNamespace(
                ec_transfer_config=ec_config,
                parallel_config=parallel_config,
            ),
            _Role.WORKER,
        )
        self.assertEqual(config.protocol, "ascend")
        self.assertEqual(config.buffer_device, "npu")

    def test_parallel_topology_is_rejected(self) -> None:
        ec_config = SimpleNamespace(
            is_ec_producer=True,
            is_ec_consumer=False,
            ec_buffer_size=4096,
            ec_buffer_device="npu",
            ec_connector_extra_config={},
        )
        parallel_config = SimpleNamespace(
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            data_parallel_index=0,
        )
        with self.assertRaisesRegex(ValueError, "tensor_parallel_size=1"):
            self.config_module.MooncakeECConfig.from_vllm_config(
                SimpleNamespace(
                    ec_transfer_config=ec_config,
                    parallel_config=parallel_config,
                ),
                _Role.WORKER,
            )

    def _consumer_scheduler(self):
        scheduler = self.scheduler_module.ECMooncakeScheduler.__new__(self.scheduler_module.ECMooncakeScheduler)
        scheduler._is_producer = False
        scheduler._is_consumer = True
        scheduler._transfers = self.state_module.SchedulerTransferTable(1_000_000, 300.0)
        scheduler._consumer_scheduler_metrics = Counter()
        scheduler._scheduler_pending_work = False
        scheduler._drain_push_notifications = MagicMock()
        scheduler._note_awaiting_push = MagicMock()
        scheduler._maybe_log_consumer_scheduler_metrics = MagicMock()
        return scheduler

    def _push_spec(self, transfer_id):
        return self.metadata_module.ECMooncakeLoadSpec(
            mm_hash="image",
            num_token=2,
            nbytes=16,
            shape=(2, 2),
            dtype="bfloat16",
            pushed=True,
            transfer_id=transfer_id,
            reservation_id=f"reservation-{transfer_id}",
        )

    def test_failed_reservation_with_bound_source_does_not_poison_batch(self):
        manager = self.producer_module.ProducerPushManager()
        records = {}
        futures = {}
        for identity in ("failed", "healthy", "pending"):
            future = Future()
            spec = self.metadata_module.ECMooncakePushSpec(identity, 4, (1,), "float32", "consumer", identity, identity)
            records[identity], _ = manager.reserve(spec, lambda future=future: future)
            futures[identity] = future
            manager.bind_source(identity, object(), None)
        futures["failed"].set_exception(RuntimeError("EC consumer buffer pool is full"))
        futures["healthy"].set_result([{"reservation_id": "healthy"}])
        executor = MagicMock()
        executor.submit.return_value = Future()
        run_batch = MagicMock()
        manager.submit_batches(executor, run_batch, MagicMock(), max_batch_bytes=1024)

        executor.submit.assert_called_once_with(run_batch, [records["healthy"]])
        self.assertIs(records["failed"].state, self.producer_module.ProducerPushState.FAILED)
        self.assertIsNone(records["failed"].source)
        self.assertIsNone(records["pending"].batch_future)
        self.assertEqual(manager.poll(), [("failed", "EC consumer buffer pool is full")])

    def test_failed_event_notifies_current_or_future_request_without_timeout(self):
        for initial_state in ("before_request", "waiting", "available"):
            with self.subTest(initial_state=initial_state):
                scheduler = self._consumer_scheduler()
                scheduler._drain_pending = True
                scheduler._drained_at = 0.0
                scheduler._poll_pending_cancels = MagicMock()
                scheduler._expire_transfers = MagicMock()
                scheduler._reservation_zmq_addr = "consumer"
                scheduler._event_inbox = MagicMock()
                scheduler._event_inbox.drain.return_value = [
                    {"transfer_id": "failed", "mm_hash": "image", "failed": True, "error": "abandoned"}
                ]
                if initial_state == "available":
                    scheduler._transfers.observe_ready(self._push_spec("failed"), 60.0)
                if initial_state != "before_request":
                    scheduler._transfers.wait_for_event("failed", "request", "image", 60.0)
                self.scheduler_module.ECMooncakeScheduler._drain_push_notifications(scheduler)
                scheduler._transfers.wait_for_event("failed", "request", "image", 60.0)
                self.assertEqual(scheduler.take_unavailable_requests(), {"request"})
                self.assertIs(
                    scheduler._transfers.get("failed").state, self.state_module.SchedulerTransferState.UNAVAILABLE
                )

    def test_abandon_emits_failure_but_reservation_refresh_does_not(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        control = self.control_module
        server = control.ConsumerControlServer(
            "127.0.0.1", port, MagicMock(), MagicMock(), MagicMock(), MagicMock(return_value=True), lambda: 0, 0
        )
        client = control.ControlClient(1000)
        inbox = control.EventInbox(client)
        addr = f"tcp://127.0.0.1:{port}"
        try:
            server.start()
            inbox._connect(addr)
            for transfer_id, refresh in (("refresh", True), ("failed", False)):
                payload = control.make_cancel_request(transfer_id, "", abandon=True, refresh=refresh)
                payload["mm_hash"] = "image"
                self.assertTrue(client.request(addr, payload)["cancelled"])
            events = []
            deadline = time.monotonic() + 2
            while not events and time.monotonic() < deadline:
                events.extend(inbox.drain(addr))
                time.sleep(0.01)
            self.assertEqual(
                events,
                [{"transfer_id": "failed", "mm_hash": "image", "failed": True, "error": "producer abandoned transfer"}],
            )
        finally:
            inbox.close()
            client.close()
            server.close()

    def test_reserve_batch_isolates_failures_and_emits_all_ready_events(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]

        def reserve(item):
            if item["transfer_id"] == "failed":
                raise RuntimeError("buffer pool is full")
            return {
                "reservation_id": f"reservation-{item['transfer_id']}",
                "ready": item["transfer_id"] == "ready",
            }

        statuses = {
            "ready": {"mm_hash": "ready-hash", "ready": True},
        }
        follower_event = {
            "transfer_id": "follower",
            "mm_hash": "shared-hash",
            "ready": True,
        }
        drained = [[follower_event], []]
        control = self.control_module
        server = control.ConsumerControlServer(
            "127.0.0.1",
            port,
            reserve,
            statuses.get,
            MagicMock(),
            MagicMock(),
            lambda: 0,
            0,
            drain_events=lambda: drained.pop(0) if drained else [],
        )
        client = control.ControlClient(1000)
        inbox = control.EventInbox(client)
        addr = f"tcp://127.0.0.1:{port}"

        def item(transfer_id):
            return {
                "transfer_id": transfer_id,
                "mm_hash": f"{transfer_id}-hash",
                "nbytes": 4,
                "shape": [1],
                "dtype": "float32",
            }

        try:
            server.start()
            inbox._connect(addr)
            result = client.request(
                addr,
                {
                    "op": "reserve_batch",
                    "items": [item("failed"), item("ready"), item("new")],
                },
            )
            self.assertEqual(
                result["items"],
                [
                    {"ok": False, "error": "buffer pool is full"},
                    {
                        "ok": True,
                        "result": {"reservation_id": "reservation-ready", "ready": True},
                    },
                    {
                        "ok": True,
                        "result": {"reservation_id": "reservation-new", "ready": False},
                    },
                ],
            )
            with self.assertRaisesRegex(RuntimeError, "buffer pool is full"):
                client.request(addr, {"op": "reserve", **item("failed")})

            events = []
            deadline = time.monotonic() + 2
            while len(events) < 2 and time.monotonic() < deadline:
                events.extend(inbox.drain(addr))
                time.sleep(0.01)
            self.assertCountEqual(
                events,
                [
                    {"transfer_id": "ready", "mm_hash": "ready-hash", "ready": True},
                    follower_event,
                ],
            )
        finally:
            inbox.close()
            client.close()
            server.close()

    @staticmethod
    def _request(transfer_id):
        return SimpleNamespace(
            request_id=f"request-{transfer_id}",
            has_encoder_inputs=True,
            mm_features=[
                SimpleNamespace(
                    identifier="image",
                    mm_position=SimpleNamespace(
                        offset=0,
                        length=2,
                        get_num_embeds=lambda: 2,
                        get_embeds_indices_in_range=lambda start, end: (start, end),
                    ),
                )
            ],
            ec_transfer_params={"ec_items": [{"mm_hash": "image", "transfer_id": transfer_id}]},
            get_num_encoder_embeds=lambda _index: 2,
        )

    def test_ready_hash_is_reused_across_transfers(self) -> None:
        scheduler = self._consumer_scheduler()
        transfers = scheduler._transfers
        transfers.observe_ready(self._push_spec("old"), 100.0)
        transfers.begin_load("image", 2, "old", "old-request")
        transfers.take_loads_to_dispatch()
        transfers.complete_load("image")
        transfers.observe_ready(self._push_spec("fresh"), 100.0)

        self.assertTrue(scheduler.ensure_cache_available(self._request("fresh"), 0))
        self.assertEqual(transfers.take_loads_to_dispatch(), [])
        self.assertIs(transfers.get("fresh").state, self.state_module.SchedulerTransferState.AVAILABLE)

    def test_concurrent_requests_share_one_hash_load_and_ack(self) -> None:
        scheduler = self._consumer_scheduler()
        transfers = scheduler._transfers
        requests = [self._request(transfer_id) for transfer_id in ("first", "second")]
        for transfer_id in ("first", "second"):
            transfers.observe_ready(self._push_spec(transfer_id), 100.0)
        for request in requests:
            self.assertFalse(scheduler.ensure_cache_available(request, 0))
        self.assertEqual(len(transfers.take_loads_to_dispatch()), 1)
        transfers.complete_load("image")
        for request in requests:
            self.assertTrue(scheduler.ensure_cache_available(request, 0))
        self.assertEqual(transfers.take_loads_to_dispatch(), [])

    def test_running_and_resumed_inputs_wait_for_resident_reload(self) -> None:
        # Compile the real methods without importing NPU dependencies. The
        # upstream scheduler remains responsible for external-input selection.
        window = MagicMock(return_value=(0, 1))
        namespace = {"get_mm_features_in_window": window, "SchedulerInterface": object}
        sources = (
            (Path(os.environ.get("VLLM_PATH", _ROOT.parent / "vllm")) / "vllm/v1/core/sched/scheduler.py", "Scheduler"),
            (_ROOT / "vllm_ascend/patch/platform/patch_balance_schedule.py", "BalanceScheduler"),
        )
        for path, name in sources:
            tree = ast.parse(path.read_text())
            definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
            definition.decorator_list = []
            definition.body = [
                node
                for node in definition.body
                if isinstance(node, ast.FunctionDef) and node.name == "_try_schedule_encoder_inputs"
            ] or [ast.Pass()]
            module = ast.Module(body=[definition], type_ignores=[])
            ast.fix_missing_locations(module)
            exec(compile(module, str(path), "exec"), namespace)
        scheduler = namespace["BalanceScheduler"].__new__(namespace["BalanceScheduler"])
        scheduler._balance_enabled = False
        scheduler.is_encoder_decoder = False
        scheduler.scheduler_config = SimpleNamespace(disable_chunked_mm_input=False)
        scheduler.encoder_cache_manager = SimpleNamespace(
            check_and_update_cache=MagicMock(return_value=False),
            can_allocate=MagicMock(return_value=True),
        )
        for computed in (0, 1):
            with self.subTest(num_computed_tokens=computed):
                consumer = self._consumer_scheduler()
                scheduler.ec_connector = consumer
                transfers = consumer._transfers
                transfers.observe_ready(self._push_spec("old"), 100.0)
                transfers.begin_load("image", 2, "old", "old-request")
                transfers.take_loads_to_dispatch()
                transfers.complete_load("image")
                transfers.release_ready("image", 10.0)
                request = self._request("fresh")
                window.reset_mock()
                self.assertEqual(scheduler._try_schedule_encoder_inputs(request, computed, 1, 8, 1), ([], 0, 8, []))
                window.assert_not_called()
                loads = transfers.take_loads_to_dispatch()
                self.assertEqual(len(loads), 1)
                self.assertTrue(loads[0].spec.local)
                transfers.complete_load("image")
                self.assertEqual(scheduler._try_schedule_encoder_inputs(request, computed, 1, 8, 1), ([], 1, 8, [0]))
                window.assert_called_once_with(request.mm_features, start=computed, end=computed + 2)

        consumer.ensure_cache_available = MagicMock(side_effect=AssertionError("unexpected EC gate"))
        self.assertEqual(scheduler._try_schedule_encoder_inputs(request, 0, 0, 8), ([], 0, 8, []))
        request.has_encoder_inputs = False
        self.assertEqual(scheduler._try_schedule_encoder_inputs(request, 0, 1, 8), ([], 1, 8, []))
        request.has_encoder_inputs = True
        scheduler.ec_connector = None
        self.assertEqual(scheduler._try_schedule_encoder_inputs(request, 0, 1, 8), ([0], 1, 6, []))

    def test_eviction_during_load_requires_local_reload(self) -> None:
        state = self.state_module
        for same_step in (False, True):
            with self.subTest(same_step=same_step):
                transfers = state.SchedulerTransferTable(1_000_000, 300.0)
                spec = self.metadata_module.ECMooncakeLoadSpec(
                    mm_hash="image",
                    num_token=2,
                    nbytes=16,
                    shape=(2, 2),
                    dtype="bfloat16",
                    pushed=True,
                    transfer_id="fresh",
                )
                transfers.observe_ready(spec, 100.0)
                transfers.begin_load("image", 2, "fresh", "request")
                if not same_step:
                    transfers.take_loads_to_dispatch()
                # Workers free before loading within a batch. A later batch
                # can free the tensor before the earlier load ACK is consumed.
                transfers.release_ready("image", 10.0)
                if same_step:
                    transfers.take_loads_to_dispatch()
                transfers.complete_load("image")
                record = transfers.get("fresh")
                expected = state.SchedulerTransferState.READY if same_step else state.SchedulerTransferState.RESIDENT
                self.assertIs(record.state, expected)
                if not same_step:
                    self.assertTrue(record.spec.local)
                    transfers.begin_load("image", 2, "fresh", "request")
                    self.assertEqual(len(transfers.take_loads_to_dispatch()), 1)
                    transfers.complete_load("image")
                    self.assertIs(record.state, state.SchedulerTransferState.READY)

    def test_evicted_load_ack_does_not_invalidate_newer_load(self) -> None:
        state = self.state_module
        transfers = state.SchedulerTransferTable(1_000_000, 300.0)
        for transfer_id in ("old", "fresh"):
            spec = self.metadata_module.ECMooncakeLoadSpec(
                mm_hash="image",
                num_token=2,
                nbytes=16,
                shape=(2, 2),
                dtype="bfloat16",
                pushed=True,
                transfer_id=transfer_id,
            )
            transfers.observe_ready(spec, 100.0)
            transfers.begin_load("image", 2, transfer_id, transfer_id)
            if transfer_id == "old":
                transfers.take_loads_to_dispatch()
        transfers.release_ready("image", 10.0)
        transfers.take_loads_to_dispatch()
        transfers.complete_load("image")
        self.assertIs(transfers.get("old").state, state.SchedulerTransferState.RESIDENT)
        # The newer batch loads after its free, so its ACK must retain READY.
        transfers.complete_load("image")
        self.assertIs(transfers.get("fresh").state, state.SchedulerTransferState.READY)

    def test_shared_engine_and_explicit_unregister_retry(self) -> None:
        transfer = self.transfer_module.MooncakeTransfer("10.0.0.1", "ascend")
        tensor = _FakeTensor()

        self.assertEqual(transfer.local_session(), "10.0.0.1:19001")
        self.assertEqual(transfer.register_memory(tensor), 0)
        transfer.write("10.0.0.2:19001", [0x1000], [0x2000], [4096])
        self.engine.fail_unregister_once = True
        self.assertFalse(transfer.unregister_memory(tensor))
        transfer.close()

        self.assertEqual(self.global_te.calls[-1], ("10.0.0.1", None))
        self.assertEqual(self.engine.registered[-1], (0x1000, 4096))
        self.assertEqual(self.engine.unregistered[-2:], [0x1000, 0x1000])
        self.assertEqual(
            self.engine.writes[-1],
            ("10.0.0.2:19001", [0x1000], [0x2000], [4096]),
        )


if __name__ == "__main__":
    unittest.main()
