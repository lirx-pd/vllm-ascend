# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for asynchronous source dispatch and batched reservations."""

from __future__ import annotations

import ast
import logging
import sys
import threading
import types
import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

_ROOT = Path(__file__).resolve().parents[4]
_MOONCAKE = _ROOT / "vllm_ascend/distributed/ec_transfer/ec_connector/mooncake"


def _load_classes():
    name = "ec_dispatch_test_producer"
    module = types.ModuleType(name)
    sys.modules[name] = module
    module.torch = SimpleNamespace(Tensor=object)
    tree = ast.parse((_MOONCAKE / "producer.py").read_text())
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.Import) and any(alias.name == "torch" for alias in node.names))
        and not (isinstance(node, ast.ImportFrom) and node.module.startswith("vllm_ascend"))
    ]
    exec(compile(tree, str(_MOONCAKE / "producer.py"), "exec"), module.__dict__)
    worker_tree = ast.parse((_MOONCAKE / "worker.py").read_text())
    worker = next(
        node for node in worker_tree.body if isinstance(node, ast.ClassDef) and node.name == "ECMooncakeWorker"
    )
    methods = {
        "_dispatch_pushes",
        "_flush_pending_pushes",
        "_reserve_batch",
        "start_save_caches",
        "close",
        "_initialize_transfer_thread",
        "_cancel_orphaned_reservation",
        "_known_reservations",
    }
    worker.body = [node for node in worker.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    module.Future = Future
    module.logger = logging.getLogger(name)
    module._READY_EVENT_POLL_SECONDS = 0.001
    tree.body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), worker]
    ast.fix_missing_locations(tree)
    exec(compile(tree, str(_MOONCAKE / "worker.py"), "exec"), module.__dict__)
    return module


def _spec(index: str, consumer: str = "tcp://pd:1"):
    return SimpleNamespace(
        transfer_id=index,
        request_id=index,
        mm_hash=index,
        consumer_zmq=consumer,
        nbytes=16,
        shape=(2, 4),
        dtype="float16",
    )


class TestProducerDispatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_classes()

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop(cls.module.__name__)

    def _manager(self):
        wake = threading.Event()
        return self.module.ProducerPushManager(wake.set), wake

    def test_ready_event_filters_without_blocking_healthy_sources(self):
        manager, wake = self._manager()
        records = []
        for name in ("pending", "ready"):
            future = Future()
            future.set_result([{}])
            record, _ = manager.reserve(_spec(name), lambda future=future: future)
            event = MagicMock()
            event.query.return_value = name == "ready"
            manager.bind_source(name, object(), event)
            records.append(record)
        executor = MagicMock()
        self.assertTrue(manager.submit_batches(executor, MagicMock(), MagicMock()))
        self.assertEqual(executor.submit.call_args.args[1], [records[1]])
        records[0].source.ready_event.synchronize.assert_not_called()
        records[0].source.ready_event.query.return_value = True
        self.assertFalse(manager.submit_batches(executor, MagicMock(), MagicMock()))
        self.assertEqual(executor.submit.call_args.args[1], [records[0]])
        self.assertTrue(wake.is_set())

    def test_dispatch_progresses_without_model_step_and_after_device_ready(self):
        manager, wake = self._manager()
        record, _ = manager.reserve(_spec("source"), Future)
        ready = threading.Event()
        source_event = SimpleNamespace(query=ready.is_set)
        manager.bind_source("source", object(), source_event)
        worker = self.module.ECMooncakeWorker.__new__(self.module.ECMooncakeWorker)
        worker._device = "npu:0"
        worker._producer_pushes = manager
        worker._push_ready = wake
        worker._dispatch_stop = threading.Event()
        worker._note_push_batch_queued = MagicMock()
        dispatched = threading.Event()
        worker._push_batch = lambda records: dispatched.set()
        self.module.torch.npu = SimpleNamespace(set_device=lambda _: None)
        with ThreadPoolExecutor(max_workers=1) as executor:
            worker._io_executor = executor
            thread = threading.Thread(target=worker._dispatch_pushes)
            thread.start()
            try:
                self.assertFalse(dispatched.wait(0.02))
                record.reservation_future.set_result([{}])
                self.assertFalse(dispatched.wait(0.02))
                ready.set()
                self.assertTrue(dispatched.wait(1))
                worker._note_push_batch_queued.assert_called_once()
            finally:
                worker._dispatch_stop.set()
                wake.set()
                thread.join(1)
                self.assertFalse(thread.is_alive())

    def _batch_worker(self):
        worker = self.module.ECMooncakeWorker.__new__(self.module.ECMooncakeWorker)
        worker._control_client = MagicMock()
        worker._retry_cancel_reservations = MagicMock()
        worker._timing_enabled = False
        manager, _ = self._manager()
        worker._producer_pushes = manager
        return worker, [manager.reserve(_spec(str(index)), Future)[0] for index in range(3)]

    def test_batch_partial_failure_settles_each_future_and_only_cancels_failure(self):
        worker, records = self._batch_worker()
        worker._control_client.request.return_value = {
            "items": [
                {"ok": True, "result": {"reservation_id": "a"}},
                {"ok": False, "error": "pool full"},
                {"ok": True, "result": {"reservation_id": "c"}},
            ]
        }
        worker._reserve_batch(records)
        self.assertEqual(records[0].reservation_future.result()[0]["reservation_id"], "a")
        self.assertEqual(records[2].reservation_future.result()[0]["reservation_id"], "c")
        with self.assertRaisesRegex(RuntimeError, "pool full"):
            records[1].reservation_future.result()
        worker._retry_cancel_reservations.assert_called_once()
        self.assertIs(worker._retry_cancel_reservations.call_args.args[0], records[1].spec)
        self.assertEqual(worker._control_client.request.call_args.args[1]["op"], "reserve_batch")

    def test_lost_batch_reply_cancels_all_unacknowledged_destinations(self):
        worker, records = self._batch_worker()
        worker._control_client.request.side_effect = TimeoutError("reply lost")
        worker._reserve_batch(records)
        self.assertEqual(worker._retry_cancel_reservations.call_count, 3)
        for record in records:
            with self.assertRaisesRegex(RuntimeError, "reply lost"):
                record.reservation_future.result()

    def test_malformed_item_does_not_leave_other_futures_pending(self):
        worker, records = self._batch_worker()
        worker._control_client.request.return_value = {
            "items": [
                {},
                {"ok": True, "result": {}},
                {"ok": True, "result": {}},
            ]
        }
        worker._reserve_batch(records)
        self.assertTrue(all(record.reservation_future.done() for record in records))
        self.assertIsInstance(records[0].reservation_future.exception(), KeyError)
        self.assertIsNone(records[1].reservation_future.exception())

    def test_step_reserves_once_per_consumer_and_deduplicates_transfer_ids(self):
        worker, _ = self._batch_worker()
        worker._control_executor = MagicMock()
        worker._bind_push_source = MagicMock()
        specs = [_spec("a"), _spec("b"), _spec("c", "tcp://pd:2")]
        metadata = SimpleNamespace(pushes=specs)
        worker.start_save_caches(metadata, {"a": "cached tensor"})
        self.assertEqual(worker._control_executor.submit.call_count, 2)
        groups = [call.args[1] for call in worker._control_executor.submit.call_args_list]
        self.assertEqual([[record.spec.transfer_id for record in group] for group in groups], [["a", "b"], ["c"]])
        worker.start_save_caches(metadata, {})
        self.assertEqual(worker._control_executor.submit.call_count, 2)
        worker._bind_push_source.assert_called_once_with("cached tensor", "a")

    def test_shutdown_settles_reservations_and_cancels_sources_not_produced(self):
        worker, records = self._batch_worker()
        worker._shutdown = False
        worker._dispatcher = None
        worker._dispatch_stop = threading.Event()
        worker._push_ready = threading.Event()
        worker._fanout_pool = None
        worker._control_server = None
        worker._consumer_memory = MagicMock()
        worker._producer_memory = MagicMock()
        worker._transfer = MagicMock()
        worker._note_push_batch_queued = MagicMock()
        worker._push_batch = MagicMock()
        worker._producer_pushes.bind_source("0", object(), None)
        with ThreadPoolExecutor(max_workers=1) as control, ThreadPoolExecutor(max_workers=1) as io:
            worker._control_executor = control
            worker._io_executor = io
            for record in records:
                control.submit(record.reservation_future.set_result, [{"reservation_id": record.spec.transfer_id}])
            worker.close()
        worker._push_batch.assert_called_once_with([records[0]])
        self.assertEqual(worker._retry_cancel_reservations.call_count, 2)
        self.assertEqual(
            [call.args[0].transfer_id for call in worker._retry_cancel_reservations.call_args_list], ["1", "2"]
        )
        self.assertTrue(all(record.state is self.module.ProducerPushState.CANCELLED for record in records[1:]))
        worker._control_client.close.assert_called_once()

    def test_transfer_thread_uses_dedicated_stream(self):
        npu = MagicMock()
        self.module.torch.npu = npu
        worker = self.module.ECMooncakeWorker.__new__(self.module.ECMooncakeWorker)
        worker._device = "npu:0"
        worker._initialize_transfer_thread()
        npu.set_device.assert_called_once_with("npu:0")
        npu.Stream.assert_called_once_with(device="npu:0")
        npu.set_stream.assert_called_once_with(npu.Stream.return_value)


if __name__ == "__main__":
    unittest.main()
