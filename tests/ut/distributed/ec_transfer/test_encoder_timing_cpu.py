# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for opt-in multimodal encoder host timing."""

import ast
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from pydantic import TypeAdapter


class BaseRunner:
    def _execute_mm_encoder(self, scheduler_output):
        self.base_call(scheduler_output)
        if self.base_error is not None:
            raise self.base_error
        return self.base_result


class TestEncoderTiming(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/model_runner_v1.py"
        tree = ast.parse(cls.path.read_text())
        runner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
        execute = next(
            node for node in runner.body if isinstance(node, ast.FunctionDef) and node.name == "_execute_mm_encoder"
        )
        module = ast.parse("class Runner(BaseRunner):\n    pass")
        module.body[0].body = [execute]
        ast.fix_missing_locations(module)
        cls.clock = Mock()
        cls.logger = Mock()
        namespace = {
            "BaseRunner": BaseRunner,
            "json": json,
            "logger": cls.logger,
            "time": SimpleNamespace(monotonic=cls.clock),
            "torch": SimpleNamespace(Tensor=object),
        }
        exec(compile(module, str(cls.path), "exec"), namespace)
        cls.runner_type = namespace["Runner"]

    def make_runner(self, role, *, enabled=True, legacy_enabled=False):
        if role == "dp4":
            ec_config = None
        else:
            ec_config = SimpleNamespace(
                is_ec_producer=role == "encoder",
                is_ec_consumer=role == "pd",
                ec_connector_extra_config={"timing_enabled": legacy_enabled},
            )
        runner = self.runner_type()
        runner.vllm_config = SimpleNamespace(
            ec_transfer_config=ec_config,
        )
        runner.ascend_config = SimpleNamespace(epd_profile=enabled)
        runner.base_call = Mock()
        runner.base_error = None
        runner.base_result = [object()]
        return runner

    def setUp(self):
        self.clock.reset_mock(side_effect=True)
        self.clock.side_effect = [10.0, 10.25]
        self.logger.reset_mock()
        self.scheduler_output = SimpleNamespace(
            scheduled_encoder_inputs={
                "request-1": [0, 1],
                "request-2": [0],
            }
        )

    def test_epd_profile_config_is_typed_boolean(self):
        config_path = self.path.parents[1] / "ascend_config.py"
        tree = ast.parse(config_path.read_text())
        config = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendConfig")
        field = next(
            node
            for node in config.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "epd_profile"
        )
        self.assertEqual(ast.unparse(field.annotation), "bool")
        self.assertFalse(ast.literal_eval(field.value))
        self.assertTrue(TypeAdapter(bool).validate_python("true"))
        self.assertFalse(TypeAdapter(bool).validate_python("false"))

    def test_records_all_roles(self):
        for role in ("dp4", "encoder", "pd"):
            with self.subTest(role=role):
                self.clock.reset_mock(side_effect=True)
                self.clock.side_effect = [10.0, 10.25]
                self.logger.reset_mock()
                runner = self.make_runner(role)

                result = runner._execute_mm_encoder(self.scheduler_output)

                self.assertIs(result, runner.base_result)
                runner.base_call.assert_called_once_with(self.scheduler_output)
                prefix, payload = self.logger.info.call_args.args
                self.assertEqual(prefix, "NPU_EPD_TIMING %s")
                self.assertEqual(
                    json.loads(payload),
                    {
                        "component": "encoder",
                        "stage": "execute_mm_encoder_host",
                        "scope": "batch",
                        "status": "ok",
                        "started_monotonic_s": 10.0,
                        "ended_monotonic_s": 10.25,
                        "duration_s": 0.25,
                        "request_ids": ["request-1", "request-2"],
                        "batch_items": 3,
                        "role": role,
                    },
                )

    def test_empty_and_disabled_do_not_record(self):
        for enabled, inputs in ((False, {"request-1": [0]}), (True, {})):
            with self.subTest(enabled=enabled, inputs=inputs):
                self.clock.reset_mock(side_effect=True)
                self.logger.reset_mock()
                runner = self.make_runner("dp4", enabled=enabled)
                scheduler_output = SimpleNamespace(scheduled_encoder_inputs=inputs)

                result = runner._execute_mm_encoder(scheduler_output)

                self.assertIs(result, runner.base_result)
                runner.base_call.assert_called_once_with(scheduler_output)
                self.clock.assert_not_called()
                self.logger.info.assert_not_called()

    def test_removed_connector_toggle_does_not_enable_timing(self):
        runner = self.make_runner("encoder", enabled=False, legacy_enabled=True)

        runner._execute_mm_encoder(self.scheduler_output)

        self.clock.assert_not_called()
        self.logger.info.assert_not_called()

    def test_error_is_recorded_and_reraised(self):
        runner = self.make_runner("pd")
        runner.base_error = RuntimeError("encoder failed")

        with self.assertRaisesRegex(RuntimeError, "encoder failed"):
            runner._execute_mm_encoder(self.scheduler_output)

        event = json.loads(self.logger.info.call_args.args[1])
        self.assertEqual(event["status"], "error")
        self.assertEqual(event["error"], "RuntimeError: encoder failed")
        self.assertEqual(event["ended_monotonic_s"], 10.25)


if __name__ == "__main__":
    unittest.main()
