# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression for resident reservation metrics without model execution."""

import ast
import json
import logging
import math
import threading
import time
import unittest
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


class TestResidentMetrics(unittest.TestCase):
    @staticmethod
    def _timing_worker_class(clock=None, npu=None):
        path = (
            Path(__file__).resolve().parents[4] / "vllm_ascend/distributed/ec_transfer/ec_connector/mooncake/worker.py"
        )
        tree = ast.parse(path.read_text())
        worker_class = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ECMooncakeWorker"
        )
        wanted = {"_log_timing", "_push_batch"}
        worker_class.body = [
            node for node in worker_class.body if isinstance(node, ast.FunctionDef) and node.name in wanted
        ]
        tree.body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), worker_class]
        ast.fix_missing_locations(tree)
        logger = logging.getLogger("structured_timing_cpu_test")
        namespace = dict(
            json=json,
            logger=logger,
            time=clock or time,
            torch=SimpleNamespace(npu=npu),
            partial=partial,
            Any=object,
            _PUSH_STAGES=("reserve", "device", "admission", "register", "transfer", "unregister", "complete"),
            _RESERVATION_REFRESH_SECONDS=150,
        )
        exec(compile(tree, str(path), "exec"), namespace)
        return namespace["ECMooncakeWorker"], logger

    def test_structured_timing_is_opt_in(self):
        worker_class, logger = self._timing_worker_class()
        worker = worker_class()
        worker._device = SimpleNamespace(index=3)
        worker._timing_enabled = False
        with self.assertNoLogs(logger, level="INFO"):
            worker._log_timing("load", 0.25, transfer_id="transfer-1")
        worker._timing_enabled = True
        with self.assertLogs(logger, level="INFO") as captured:
            worker._log_timing("load", 0.25, transfer_id="transfer-1", request_id="")
        payload = json.loads(captured.output[0].split("NPU_EPD_TIMING ", 1)[1])
        self.assertEqual(
            payload,
            {
                "component": "connector",
                "stage": "load",
                "duration_s": 0.25,
                "rank": 3,
                "transfer_id": "transfer-1",
            },
        )

    def test_push_batch_emits_one_event_per_stage(self):
        clock = SimpleNamespace(monotonic=Mock(side_effect=range(20)))
        worker_class, logger = self._timing_worker_class(clock)
        worker = worker_class()
        worker._device = SimpleNamespace(index=3)
        worker._push_perf_lock = threading.Lock()
        worker._staging_lock = threading.Lock()
        worker._queued_transfer_batches = 1
        worker._active_transfer_batches = 0
        worker._producer_pushes = Mock()
        worker._producer_pushes.resolve_reservations.return_value = [{"cached": True, "ready": True}]
        worker._validate_push_source = Mock()
        worker._notify_completions = Mock()
        worker._abandon_pushes = Mock()
        worker._record_push_perf = Mock()
        pushes = [
            SimpleNamespace(
                source_at=time.monotonic(),
                source=SimpleNamespace(tensor=object()),
                spec=SimpleNamespace(
                    request_id=f"request-{index}", transfer_id=f"transfer-{index}", mm_hash=f"hash-{index}", nbytes=4
                ),
            )
            for index in range(2)
        ]
        worker._timing_enabled = False
        with self.assertNoLogs(logger, level="INFO"):
            worker._push_batch(pushes)
        self.assertEqual(clock.monotonic.call_count, 8)
        worker._queued_transfer_batches = 1
        worker._timing_enabled = True
        with self.assertLogs(logger, level="INFO") as captured:
            worker._push_batch(pushes)
        payloads = [json.loads(line.split("NPU_EPD_TIMING ", 1)[1]) for line in captured.output]
        self.assertEqual(len(payloads), 10)
        self.assertEqual(len({payload["stage"] for payload in payloads}), 9)
        batches = [payload for payload in payloads if payload["scope"] == "batch"]
        self.assertTrue(all(payload["transfer_ids"] == ["transfer-0", "transfer-1"] for payload in batches))
        queues = [payload for payload in payloads if payload["scope"] == "item"]
        self.assertEqual([payload["request_id"] for payload in queues], ["request-0", "request-1"])
        self.assertEqual([payload["transfer_id"] for payload in queues], ["transfer-0", "transfer-1"])
        self.assertEqual([payload["mm_hash"] for payload in queues], ["hash-0", "hash-1"])
        for payload in queues:
            self.assertNotIn("request_ids", payload)
            self.assertNotIn("transfer_ids", payload)
            self.assertNotIn("mm_hashes", payload)

    def test_concurrent_batches_share_slab_and_release_after_write_failure(self):
        worker_class, logger = self._timing_worker_class(npu=Mock())
        worker = worker_class()
        worker._timing_enabled = False
        worker._push_perf_lock = threading.Lock()
        worker._staging_lock = threading.Lock()
        worker._queued_transfer_batches = 2
        worker._active_transfer_batches = 0
        worker._producer_pushes = Mock()
        worker._producer_pushes.resolve_reservations.return_value = [
            {"nbytes": 4, "dst_session": "consumer", "dst_ptr": 123}
        ]
        second_started = threading.Event()
        first_writing = threading.Event()
        allow_failure = threading.Event()
        second_staged = threading.Event()
        batches = [
            [SimpleNamespace(
                source_at=time.monotonic(),
                source=SimpleNamespace(tensor=Mock(nbytes=4, device="npu:0"), ready_event=None),
                spec=SimpleNamespace(transfer_id=str(index), mm_hash=str(index), nbytes=4),
            )]
            for index in range(2)
        ]
        worker._validate_push_source = lambda push: second_started.set() if push is batches[1][0] else None
        allocations = []
        order = []

        def stage(tensors):
            self.assertFalse(allocations)
            staged = SimpleNamespace(tensors=tensors)
            allocations.append(staged)
            order.append("stage")
            if tensors[0] is batches[1][0].source.tensor:
                second_staged.set()
            return staged

        def release(staged):
            self.assertIs(allocations.pop(), staged)
            order.append("release")

        def write(*_):
            if not first_writing.is_set():
                first_writing.set()
                if not allow_failure.wait(2):
                    raise TimeoutError("test did not release first writer")
                raise RuntimeError("write failed")

        worker._producer_memory = Mock(stage=stage, release=release)
        worker._transfer = Mock(write=write)
        worker._run_fanout = lambda calls, track: [call() for call in calls]
        worker._notify_completions = Mock()
        worker._abandon_pushes = Mock()
        worker._record_push_perf = Mock()
        with self.assertLogs(logger, level="ERROR"), ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(worker._push_batch, batches[0])
            try:
                self.assertTrue(first_writing.wait(1))
                second = executor.submit(worker._push_batch, batches[1])
                self.assertTrue(second_started.wait(1))
                self.assertFalse(second_staged.wait(0.05))
            finally:
                allow_failure.set()
            first.result(timeout=2)
            second.result(timeout=2)
        self.assertEqual(order, ["stage", "release", "stage", "release"])
        self.assertFalse(allocations)
        worker._producer_pushes.fail.assert_called_once()
        self.assertIs(worker._producer_pushes.fail.call_args.args[0], batches[0])
        worker._producer_pushes.complete.assert_called_once_with(batches[1])
        worker._abandon_pushes.assert_called_once_with(batches[0])

    def test_push_phase_durations_and_item_queues(self):
        clock = SimpleNamespace(now=10.0)
        clock.monotonic = lambda: clock.now

        def advance(seconds):
            clock.now += seconds

        stream = Mock()
        stream.synchronize.side_effect = lambda: advance(0.4)
        npu = Mock()
        npu.current_stream.return_value = stream
        worker_class, logger = self._timing_worker_class(clock, npu)
        worker = worker_class()
        worker._device = SimpleNamespace(index=0)
        worker._timing_enabled = True
        worker._push_perf_lock = threading.Lock()
        worker._staging_lock = threading.Lock()
        worker._queued_transfer_batches = 1
        worker._active_transfer_batches = 0
        worker._validate_push_source = Mock()
        worker._producer_pushes = Mock()

        def reserve(push):
            advance(0.1)
            return [{"nbytes": 4, "dst_session": "consumer", "dst_ptr": 123}]

        worker._producer_pushes.resolve_reservations.side_effect = reserve
        event = Mock()
        event.synchronize.side_effect = lambda: advance(0.2)
        tensor = Mock(nbytes=4, device="npu:0")
        tensor.data_ptr.return_value = 456
        pushes = [
            SimpleNamespace(
                source_at=8.0 + index,
                source=SimpleNamespace(tensor=tensor, ready_event=event),
                spec=SimpleNamespace(
                    request_id=f"request-{index}", transfer_id=f"transfer-{index}", mm_hash="shared", nbytes=4
                ),
            )
            for index in range(2)
        ]
        worker._producer_memory = Mock()

        def stage(tensors):
            advance(0.3)
            return SimpleNamespace(tensors=tensors)

        worker._producer_memory.stage.side_effect = stage
        worker._transfer = Mock()
        worker._transfer.write.side_effect = lambda *args: advance(0.5)
        worker._run_fanout = lambda calls, track: [call() for call in calls]
        worker._notify_completions = Mock(side_effect=lambda ready: advance(0.6))
        worker._record_push_perf = Mock()
        worker._abandon_pushes = Mock()
        with self.assertLogs(logger, level="INFO") as captured:
            worker._push_batch(pushes)
        payloads = [json.loads(line.split("NPU_EPD_TIMING ", 1)[1]) for line in captured.output]
        queues = [item for item in payloads if item["stage"] == "producer_queue_completed"]
        self.assertEqual([item["duration_s"] for item in queues], [2.0, 1.0])
        self.assertEqual([item["started_monotonic_s"] for item in queues], [8.0, 9.0])
        self.assertTrue(all(item["ended_monotonic_s"] == 10.0 and item["scope"] == "item" for item in queues))
        phases = {item["stage"]: item for item in payloads if item["scope"] == "batch"}
        expected = {
            "producer_reservation_wait_completed": 0.2,
            "producer_device_ready_completed": 0.4,
            "producer_staging_wait_completed": 0.0,
            "producer_staging_copy_completed": 0.7,
            "producer_staging_completed": 1.1,
            "producer_transfer_completed": 0.5,
            "producer_notification_completed": 0.6,
            "producer_push_completed": 2.4,
        }
        self.assertEqual(phases.keys(), expected.keys())
        for name, seconds in expected.items():
            self.assertAlmostEqual(phases[name]["duration_s"], seconds)
            self.assertEqual(phases[name]["batch_items"], 2)
            self.assertEqual(phases[name]["batch_bytes"], 8)
            self.assertEqual(phases[name]["write_bytes"], 8)
            self.assertEqual(phases[name]["unique_source_count"], 1)
        self.assertEqual(phases["producer_staging_completed"]["aggregation"], "aggregate")
        self.assertEqual(
            phases["producer_staging_completed"]["overlaps"],
            ["producer_device_ready_completed", "producer_staging_copy_completed"],
        )
        self.assertAlmostEqual(phases["producer_push_completed"]["ended_monotonic_s"], 12.4)
        self.assertNotIn("started_monotonic_s", phases["producer_device_ready_completed"])
        self.assertEqual(event.synchronize.call_count, 2)
        stream.synchronize.assert_called_once()
        npu.Stream.assert_not_called()
        worker._transfer.write.assert_called_once_with("consumer", [456, 456], [123, 123], [4, 4])
        worker._producer_pushes.complete.assert_called_once_with(pushes)
        worker._abandon_pushes.assert_not_called()

        def failed_write(*args):
            advance(0.35)
            raise RuntimeError("write failed")

        worker._producer_pushes.reset_mock()
        worker._producer_pushes.resolve_reservations.side_effect = reserve
        worker._transfer.write.side_effect = failed_write
        worker._abandon_pushes.reset_mock()
        worker._queued_transfer_batches = 1
        with self.assertLogs(logger, level="INFO") as captured:
            worker._push_batch(pushes)
        failed_payloads = [
            json.loads(line.split("NPU_EPD_TIMING ", 1)[1]) for line in captured.output if "NPU_EPD_TIMING " in line
        ]
        failed_transfer = next(item for item in failed_payloads if item["stage"] == "producer_transfer_completed")
        self.assertEqual(failed_transfer["status"], "error")
        self.assertAlmostEqual(failed_transfer["duration_s"], 0.35)
        worker._producer_pushes.settle_all.assert_called_once_with(pushes)
        worker._abandon_pushes.assert_called_once_with(pushes)
        worker._producer_pushes.fail.assert_called_once()

        def failed_reserve(push):
            advance(0.25)
            raise RuntimeError("reservation failed")

        worker._producer_pushes.reset_mock()
        worker._abandon_pushes.reset_mock()
        worker._producer_pushes.resolve_reservations.side_effect = failed_reserve
        worker._queued_transfer_batches = 1
        with self.assertLogs(logger, level="INFO") as captured:
            worker._push_batch(pushes)
        failed_payloads = [
            json.loads(line.split("NPU_EPD_TIMING ", 1)[1]) for line in captured.output if "NPU_EPD_TIMING " in line
        ]
        self.assertTrue(all(item["status"] == "error" for item in failed_payloads))
        failed_wait = next(item for item in failed_payloads if item["stage"] == "producer_reservation_wait_completed")
        self.assertAlmostEqual(failed_wait["duration_s"], 0.25)
        worker._producer_pushes.settle_all.assert_called_once_with(pushes)
        worker._abandon_pushes.assert_called_once_with(pushes)
        worker._producer_pushes.fail.assert_called_once()

    def test_cached_reservation_logs_without_a_later_model_step(self):
        path = (
            Path(__file__).resolve().parents[4] / "vllm_ascend/distributed/ec_transfer/ec_connector/mooncake/worker.py"
        )
        # Execute the production methods without importing the NPU runtime.
        tree = ast.parse(path.read_text())
        worker_class = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ECMooncakeWorker"
        )
        methods = {"_reserve_push_destination", "_record_consumer_metric", "_maybe_log_consumer_worker_metrics"}
        worker_class.body = [
            node for node in worker_class.body if isinstance(node, ast.FunctionDef) and node.name in methods
        ]
        tree.body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), worker_class]
        ast.fix_missing_locations(tree)
        logger = logging.getLogger("resident_metrics_cpu_test")
        states = SimpleNamespace(READY="ready", CANCEL_PENDING="cancel_pending", CANCELLED="cancelled")
        namespace = dict(
            math=math,
            time=time,
            logger=logger,
            torch=SimpleNamespace(dtype=SimpleNamespace, float32=SimpleNamespace(itemsize=4)),
            ConsumerReservationState=states,
        )
        exec(compile(tree, str(path), "exec"), namespace)
        worker = namespace["ECMooncakeWorker"]()
        worker._consumer_memory = Mock(lock=threading.RLock())
        worker._consumer_memory.take_metrics.return_value = {}
        worker._consumer_memory.stats.return_value = (1, 1, 0, 0)
        worker._consumer_metrics_log_interval = 3600
        worker._consumer_metrics_started_at = time.monotonic()
        worker._consumer_worker_metrics = Counter()
        worker._expire_push_reservations = Mock()
        worker._record_expiry_metrics = Mock()
        worker._transfer = Mock()
        worker._transfer.local_session.return_value = "consumer:1234"
        reservation = SimpleNamespace(
            state=states.READY,
            lease=object(),
            reservation_id="reservation-2",
            mm_hash="image",
            created_at=time.monotonic(),
            allocation=SimpleNamespace(tensor=Mock(nbytes=16)),
        )
        worker._reservations = Mock()
        worker._reservations.reserve.return_value = (reservation, False, False, (0, 0, 0))
        worker._reservations.active_records.return_value = [reservation]
        payload = dict(transfer_id="request-2", mm_hash="image", nbytes=16, shape=[4], dtype="float32")
        with self.assertLogs(logger, level="INFO") as captured:
            result = worker._reserve_push_destination(payload)
        self.assertTrue(result["cached"])
        self.assertFalse(result["write"])
        self.assertIn("'reservations_cached': 1", captured.output[0])
        self.assertEqual(worker._consumer_worker_metrics, {})
        worker._consumer_metrics_log_interval = 0
        with self.assertNoLogs(logger, level="INFO"):
            worker._reserve_push_destination(payload)
        self.assertEqual(worker._consumer_worker_metrics["reservations_cached"], 1)


if __name__ == "__main__":
    unittest.main()
