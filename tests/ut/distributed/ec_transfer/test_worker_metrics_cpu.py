# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression for resident reservation metrics without model execution."""

import ast
import logging
import math
import threading
import time
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


class TestResidentMetrics(unittest.TestCase):
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
