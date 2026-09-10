# SPDX-License-Identifier: Apache-2.0
"""Check reservation-batch expiry work without NPU hardware."""

import ast
import logging
import socket
import sys
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def load_control():
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/distributed/ec_transfer/ec_connector/mooncake/control.py"
    tree = ast.parse(path.read_text())
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.ImportFrom)
            and node.module == "vllm.logger"
            or isinstance(node, ast.Import)
            and any(alias.name == "torch" for alias in node.names)
        )
    ]
    module = types.ModuleType("control_reap_cpu")
    module.init_logger = logging.getLogger
    with patch.dict(sys.modules, {module.__name__: module}):
        exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


class TestControlReap(unittest.TestCase):
    def test_each_reservation_request_reaps_once_before_items(self):
        control = load_control()
        calls = []

        def reap():
            calls.append("reap")
            return 0

        def reserve(item):
            transfer_id = item["transfer_id"]
            calls.append(transfer_id)
            if transfer_id == "failed":
                raise RuntimeError("full")
            return {"ready": True}

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        server = control.ConsumerControlServer(
            "127.0.0.1",
            port,
            reserve,
            lambda _: {"ready": True},
            lambda *_: None,
            lambda *_: False,
            reap,
        )
        client = control.ControlClient(1000)
        inbox = control.EventInbox(client)
        addr = f"tcp://127.0.0.1:{port}"
        try:
            server.start()
            inbox._connect(addr)
            result = client.request(
                addr,
                {
                    "op": "reserve_batch",
                    "items": [{"transfer_id": value} for value in ("first", "failed", "last")],
                },
            )
            self.assertEqual(
                result["items"],
                [
                    {"ok": True, "result": {"ready": True}},
                    {"ok": False, "error": "full"},
                    {"ok": True, "result": {"ready": True}},
                ],
            )
            self.assertEqual(calls, ["reap", "first", "failed", "last"])
            self.assertEqual(client.request(addr, {"op": "reserve", "transfer_id": "single"}), {"ready": True})
            self.assertEqual(calls, ["reap", "first", "failed", "last", "reap", "single"])
            events = []
            deadline = time.monotonic() + 2
            while len(events) < 3 and time.monotonic() < deadline:
                events.extend(inbox.drain(addr))
                time.sleep(0.01)
            self.assertCountEqual([event["transfer_id"] for event in events], ["first", "last", "single"])
        finally:
            inbox.close()
            client.close()
            server.close()


if __name__ == "__main__":
    unittest.main()
