# SPDX-License-Identifier: Apache-2.0
"""Exercise real reservation and allocator lifecycles without NPU hardware."""

from __future__ import annotations

import ast
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

_MOONCAKE = Path(__file__).resolve().parents[4] / "vllm_ascend/distributed/ec_transfer/ec_connector/mooncake"


class _Tensor:
    def __init__(self, nbytes, offset=0):
        self.nbytes = nbytes
        self.offset = offset
        self.device = "npu"
        self.shape = (nbytes,)
        self.dtype = "uint8"

    def narrow(self, dim, offset, nbytes):
        return _Tensor(nbytes, self.offset + offset)

    def view(self, value):
        if isinstance(value, tuple):
            self.shape = value
        else:
            self.dtype = value
        return self


class _Event:
    def __init__(self):
        self.done = False

    def record(self, stream):
        pass

    def query(self):
        return self.done


def _load(filename, namespace):
    path = _MOONCAKE / filename
    tree = ast.parse(path.read_text())
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.ImportFrom)
            and node.module.startswith(("vllm", "torch"))
            or isinstance(node, ast.Import)
            and any(alias.name == "torch" for alias in node.names)
        )
    ]
    module = types.ModuleType(f"reservation_sharing_{path.stem}")
    module.__dict__.update(namespace)
    with patch.dict(sys.modules, {module.__name__: module}):
        exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


class TestReservationSharing(unittest.TestCase):
    def setUp(self):
        self.events = []

        def event():
            result = _Event()
            self.events.append(result)
            return result

        torch = types.SimpleNamespace(npu=types.SimpleNamespace(Event=event, current_stream=lambda device: None))
        self.memory_module = _load("memory.py", {"torch": torch, "init_logger": lambda name: None})
        self.reservation_module = _load(
            "reservation.py",
            {
                "torch": torch,
                "ConsumerMemoryPool": self.memory_module.ConsumerMemoryPool,
                "MemoryAllocation": self.memory_module.MemoryAllocation,
                "ResidentLease": self.memory_module.ResidentLease,
            },
        )
        self.memory = self.memory_module.ConsumerMemoryPool(512, None)
        self.memory._pool = _Tensor(512)
        self.memory._allocator = self.memory_module.ContiguousAllocator(512)
        self.reservations = self.reservation_module.ConsumerReservationManager(self.memory, 300, 100)

    def reserve(self, transfer_id, mm_hash="image"):
        return self.reservations.reserve(transfer_id, mm_hash, 256, (256,), "uint8", "uint8")

    def complete(self, record):
        return self.reservations.complete(record.transfer_id, record.reservation_id)

    def cancel(self, record):
        return self.reservations.cancel(record.transfer_id, record.reservation_id)

    def test_ready_is_shared_before_take_and_one_cancel_preserves_other_owner(self):
        first, write, _, _ = self.reserve("first")
        self.assertTrue(write)
        self.complete(first)
        second, write, _, _ = self.reserve("second")
        self.assertFalse(write)
        self.assertIs(second.allocation, first.allocation)
        self.assertEqual(self.memory._allocator._free, [(256, 256)])
        self.cancel(first)
        taken = self.reservations.take("second", "image")
        self.assertEqual(taken.offset, 0)
        self.reservations.retire_stale({"image": taken.tensor})
        self.assertEqual(self.memory.stats()[:3], (1, 1, 0))

    def test_duplicate_writers_share_only_after_their_writes_complete(self):
        first, _, _, _ = self.reserve("first")
        second, write, _, _ = self.reserve("second")
        self.assertTrue(write)
        self.assertIsNot(first.allocation, second.allocation)
        self.complete(first)
        self.assertEqual(self.memory._allocator._free, [])
        self.complete(second)
        self.assertIs(second.allocation, first.allocation)
        self.assertEqual(self.memory._allocator._free, [(256, 256)])
        self.cancel(first)
        self.assertIsNone(self.memory.reclaim_and_allocate(512, (512,), "uint8"))
        self.cancel(second)
        self.assertIsNotNone(self.memory.reclaim_and_allocate(512, (512,), "uint8"))

    def test_cancelling_duplicate_writer_waits_for_completion(self):
        first, _, _, _ = self.reserve("first")
        second, _, _, _ = self.reserve("second")
        self.complete(first)
        outcome, _ = self.cancel(second)
        self.assertIs(outcome, self.reservation_module.CancellationOutcome.DEFERRED)
        self.assertEqual(self.memory._allocator._free, [])
        self.assertTrue(self.complete(second).discarded)
        self.assertEqual(self.memory._allocator._free, [(256, 256)])
        self.assertEqual(self.reservations.take("first", "image").offset, 0)

    def test_reserved_hash_does_not_keep_old_model_pin_after_cache_release(self):
        first, _, _, _ = self.reserve("first")
        self.complete(first)
        taken = self.reservations.take("first", "image")
        second, write, _, _ = self.reserve("second")
        self.assertFalse(write)
        self.reservations.retire_stale({})
        self.assertEqual(self.memory.stats()[:3], (1, 0, 0))
        self.cancel(second)
        self.assertEqual(self.memory.stats()[:3], (1, 0, 1))
        self.assertIsNone(self.memory.reclaim_and_allocate(512, (512,), "uint8"))
        self.assertEqual(self.memory._pending_frees[0][1], taken)
        self.events[0].done = True
        self.assertIsNotNone(self.memory.try_allocate(512, (512,), "uint8"))

    def test_expiry_releases_only_the_expired_ready_lease(self):
        first, _, _, _ = self.reserve("first")
        self.complete(first)
        second, _, _, _ = self.reserve("second")
        first.expires_at = 0
        expired, deferred, _ = self.reservations.expire()
        self.assertEqual((expired, deferred), (1, 0))
        self.assertIsNone(self.memory.reclaim_and_allocate(512, (512,), "uint8"))
        self.assertEqual(self.reservations.take(second.transfer_id, "image").offset, 0)


if __name__ == "__main__":
    unittest.main()
