# SPDX-License-Identifier: Apache-2.0
"""CPU-only regressions for producer writes and staging slab ownership."""

import ast
import logging
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


class TestWorkerPush(unittest.TestCase):
    @staticmethod
    def _worker_class(npu):
        path = (
            Path(__file__).resolve().parents[4] / "vllm_ascend/distributed/ec_transfer/ec_connector/mooncake/worker.py"
        )
        tree = ast.parse(path.read_text())
        worker_class = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ECMooncakeWorker"
        )
        worker_class.body = [
            node for node in worker_class.body if isinstance(node, ast.FunctionDef) and node.name == "_push_batch"
        ]
        tree.body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), worker_class]
        ast.fix_missing_locations(tree)
        logger = logging.getLogger("worker_push_cpu_test")
        namespace = dict(
            logger=logger,
            time=time,
            torch=SimpleNamespace(npu=npu),
            partial=partial,
            _RESERVATION_REFRESH_SECONDS=150,
        )
        exec(compile(tree, str(path), "exec"), namespace)
        return namespace["ECMooncakeWorker"], logger

    def test_concurrent_batches_share_slab_and_release_after_write_failure(self):
        worker_class, logger = self._worker_class(npu=Mock())
        worker = worker_class()
        worker._staging_lock = threading.Lock()
        worker._producer_pushes = Mock()
        worker._producer_pushes.resolve_reservations.return_value = [
            {"nbytes": 4, "dst_session": "consumer", "dst_ptr": 123}
        ]
        second_started = threading.Event()
        first_writing = threading.Event()
        allow_failure = threading.Event()
        second_staged = threading.Event()
        batches = [
            [
                SimpleNamespace(
                    source=SimpleNamespace(tensor=Mock(nbytes=4, device="npu:0"), ready_event=None),
                    spec=SimpleNamespace(transfer_id=str(index), mm_hash=str(index), nbytes=4),
                )
            ]
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

    def test_push_batch_writes_shared_source_and_settles_failures(self):
        npu = Mock()
        worker_class, logger = self._worker_class(npu)
        worker = worker_class()
        worker._staging_lock = threading.Lock()
        worker._validate_push_source = Mock()
        worker._producer_pushes = Mock()
        worker._producer_pushes.resolve_reservations.return_value = [
            {"nbytes": 4, "dst_session": "consumer", "dst_ptr": 123}
        ]
        event = Mock()
        tensor = Mock(nbytes=4, device="npu:0")
        tensor.data_ptr.return_value = 456
        pushes = [
            SimpleNamespace(
                source=SimpleNamespace(tensor=tensor, ready_event=event),
                spec=SimpleNamespace(transfer_id=f"transfer-{index}", mm_hash="shared", nbytes=4),
            )
            for index in range(2)
        ]
        worker._producer_memory = Mock()
        worker._producer_memory.stage.side_effect = lambda tensors: SimpleNamespace(tensors=tensors)
        worker._transfer = Mock()
        worker._run_fanout = lambda calls, track: [call() for call in calls]
        worker._notify_completions = Mock()
        worker._abandon_pushes = Mock()
        worker._push_batch(pushes)
        self.assertEqual(event.synchronize.call_count, 2)
        npu.current_stream.return_value.synchronize.assert_called_once()
        npu.Stream.assert_not_called()
        worker._transfer.write.assert_called_once_with("consumer", [456, 456], [123, 123], [4, 4])
        worker._producer_pushes.complete.assert_called_once_with(pushes)
        worker._abandon_pushes.assert_not_called()

        worker._producer_pushes.reset_mock()
        worker._transfer.write.side_effect = RuntimeError("write failed")
        with self.assertLogs(logger, level="ERROR"):
            worker._push_batch(pushes)
        worker._producer_pushes.settle_all.assert_called_once_with(pushes)
        worker._abandon_pushes.assert_called_once_with(pushes)
        worker._producer_pushes.fail.assert_called_once()
        worker._producer_pushes.complete.assert_not_called()

        worker._producer_pushes.reset_mock()
        worker._abandon_pushes.reset_mock()
        worker._producer_pushes.resolve_reservations.side_effect = RuntimeError("reservation failed")
        with self.assertLogs(logger, level="ERROR"):
            worker._push_batch(pushes)
        worker._producer_pushes.settle_all.assert_called_once_with(pushes)
        worker._abandon_pushes.assert_called_once_with(pushes)
        worker._producer_pushes.fail.assert_called_once()
        worker._producer_pushes.complete.assert_not_called()


if __name__ == "__main__":
    unittest.main()
