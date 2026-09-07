# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression checks for file and console logging initialization."""

import importlib.util
import io
import logging
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


class TestLoggingInitialization(unittest.TestCase):
    def test_metrics_reach_console_and_file_in_either_init_order(self):
        # Isolate vLLM imports; exercise real stdlib loggers and handlers.
        vllm = ModuleType("vllm")
        vllm.envs = SimpleNamespace(
            VLLM_LOGGING_LEVEL="INFO",
            VLLM_LOGGING_STREAM="ext://sys.stderr",
            NO_COLOR=True,
            VLLM_LOGGING_COLOR="0",
        )
        formatters = ModuleType("vllm.logging_utils")
        formatters.NewLineFormatter = logging.Formatter
        formatters.ColoredFormatter = logging.Formatter
        path = Path(__file__).resolve().parents[2] / "vllm_ascend" / "logger.py"
        spec = importlib.util.spec_from_file_location("ascend_logger_cpu_test", path)
        with patch.dict("sys.modules", {"vllm": vllm, "vllm.logging_utils": formatters}):
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        # Formatting itself is covered by test_logger.py.
        module._FORMAT = "%(message)s"
        messages = (
            "EC Mooncake push perf: batches=1 items=1 bytes=16",
            "EC Mooncake consumer worker: lifecycle={'completions_accepted': 1},",
            "EC Mooncake consumer scheduler: decisions={'loads_completed': 1},",
        )
        for file_first in (True, False):
            with self.subTest(file_first=file_first), tempfile.TemporaryDirectory() as directory:
                manager = logging.Manager(logging.RootLogger(logging.WARNING))
                console = io.StringIO()
                module._file_logging_configured = False
                module._file_handler = None
                with patch.object(logging, "getLogger", manager.getLogger), patch("sys.stderr", console):
                    try:
                        if file_first:
                            module._setup_file_logging(directory)
                        module.configure_ascend_logging()
                        module._setup_file_logging(directory)
                        module.configure_ascend_logging()
                        parent = manager.getLogger("vllm_ascend")
                        worker = manager.getLogger("vllm_ascend.distributed.ec_transfer.mooncake.worker")
                        self.assertTrue(worker.isEnabledFor(logging.INFO))
                        self.assertFalse(parent.propagate)
                        self.assertEqual(len(parent.handlers), 2)
                        for message in messages:
                            worker.info(message)
                        file_text = Path(module._file_handler.baseFilename).read_text()
                        for message in messages:
                            self.assertEqual(console.getvalue().count(message), 1)
                            self.assertEqual(file_text.count(message), 1)
                    finally:
                        for handler in manager.getLogger("vllm_ascend").handlers[:]:
                            manager.getLogger("vllm_ascend").removeHandler(handler)
                            manager.getLogger("vllm").removeHandler(handler)
                            handler.close()


if __name__ == "__main__":
    unittest.main()
