# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression tests for the proxy's Mooncake request identity."""

import copy
import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

_PROXY_PATH = Path(__file__).resolve().parents[4] / "examples/disaggregated_encoder/disagg_epd_proxy.py"
_spec = importlib.util.spec_from_file_location("npu_ec_proxy", _PROXY_PATH)
proxy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(proxy)


class ProxyIdentityTest(unittest.IsolatedAsyncioTestCase):
    async def test_encoder_and_consumer_keep_same_identity(self):
        for response_body in (b"{}", b'{"ec_transfer_params":{"ec_items":[{"image_grid_thw":[[1,32,32]]}]}}'):
            with self.subTest(response_body=response_body):
                image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
                request = {
                    "model": "qwen",
                    "messages": [{"role": "user", "content": [{"type": "text", "text": "Describe"}, image, image]}],
                    "ec_transfer_params": {"other": "preserved"},
                }
                original = copy.deepcopy(request)
                response = AsyncMock()
                response.status = 200
                response.read.return_value = response_body
                response.json.return_value = json.loads(response_body)
                session = AsyncMock()
                session.post.return_value = response
                with patch.object(proxy, "encode_session", session), patch.object(proxy, "NO_REWRITE", False):
                    identities = await proxy.fanout_encoder_primer(
                        request, ["http://encoder"], "request", "tcp://consumer:29600"
                    )
                    forwarded = proxy.rewrite_for_decode(request, identities)
                self.assertEqual(request, original)
                self.assertEqual(forwarded["ec_transfer_params"]["other"], "preserved")
                images = forwarded["messages"][0]["content"][1:]
                transfers = forwarded["ec_transfer_params"]["ec_items"]
                for index, call in enumerate(session.post.call_args_list):
                    encoded = call.kwargs["json"]
                    encoder_image = encoded["messages"][0]["content"][0]
                    self.assertEqual(images[index]["type"], "image_url")
                    self.assertEqual(images[index]["image_url"], image["image_url"])
                    self.assertEqual(images[index]["uuid"], encoder_image["uuid"])
                    self.assertEqual(transfers[index], encoded["ec_transfer_params"]["ec_items"][0])
                    self.assertEqual(transfers[index]["mm_hash"], images[index]["uuid"])
                self.assertEqual(images[0]["uuid"], images[1]["uuid"])
                self.assertNotEqual(transfers[0]["transfer_id"], transfers[1]["transfer_id"])
                self.assertEqual(response.read.await_count, 2)

    async def test_retry_preserves_cache_key_with_fresh_transfer(self):
        request = {
            "model": "qwen",
            "messages": [
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}
            ],
        }
        response = AsyncMock()
        response.status = 200
        response.read.return_value = b"{}"
        response.json.return_value = {}
        session = AsyncMock()
        session.post.return_value = response
        with patch.object(proxy, "encode_session", session), patch.object(proxy, "NO_REWRITE", False):
            first = await proxy.fanout_encoder_primer(request, ["http://encoder"], "first", "tcp://consumer:29600")
            second = await proxy.fanout_encoder_primer(request, ["http://encoder"], "second", "tcp://consumer:29600")
        self.assertEqual(first[0]["mm_hash"], second[0]["mm_hash"])
        self.assertNotEqual(first[0]["transfer_id"], second[0]["transfer_id"])

    def test_text_only_is_unchanged(self):
        request = {"messages": [{"role": "user", "content": "hello"}]}
        self.assertIs(proxy.rewrite_for_decode(request, {}), request)


class ProxyTimingTest(unittest.IsolatedAsyncioTestCase):
    async def test_retry_and_failure_records(self):
        for statuses in ((500, 200), (400,)):
            with self.subTest(statuses=statuses):
                session = MagicMock()
                contexts = []
                for status in statuses:
                    response = AsyncMock()
                    response.status = status
                    response.text.return_value = "failed"
                    response.json.return_value = {"answer": "ok"}
                    context = AsyncMock()
                    context.__aenter__.return_value = response
                    contexts.append(context)
                session.post.side_effect = contexts
                with (
                    patch.object(proxy, "PROFILE_ENABLED", True),
                    patch.object(proxy, "decode_session", session),
                    patch.object(proxy.logger, "info") as log,
                ):
                    call = proxy.forward_non_stream(
                        {"messages": []}, "parent", ["http://encoder"], None, "http://pd", None
                    )
                    if statuses == (400,):
                        with self.assertRaises(proxy.HTTPException):
                            await call
                    else:
                        self.assertEqual(await call, {"answer": "ok"})
                records = [json.loads(c.args[1]) for c in log.call_args_list if c.args[0] == "NPU_EPD_TIMING %s"]
                attempts = [r for r in records if r["stage"] == "attempt"]
                self.assertEqual([r["status"] for r in attempts], ["retry", "ok"] if len(statuses) == 2 else ["error"])
                self.assertEqual([r["attempt"] for r in attempts], list(range(len(statuses))))
                self.assertEqual(len([r for r in records if r["stage"] == "pd_http"]), len(statuses))
                self.assertEqual(records[-1]["stage"], "request")
                for record in records:
                    self.assertEqual(record["component"], "proxy")
                    self.assertEqual(record["request_id"], "parent")
                    self.assertGreaterEqual(record["duration_s"], 0)
                    self.assertAlmostEqual(
                        record["duration_s"], record["ended_monotonic_s"] - record["started_monotonic_s"]
                    )

    async def test_stream_first_byte_and_disabled_logging(self):
        async def chunks():
            yield b""
            yield b"data: role\n\n"
            yield b"data: [DONE]\n\n"

        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                response = MagicMock()
                response.status = 200
                response.content.iter_chunked.return_value = chunks()
                session = MagicMock()
                session.post.return_value.__aenter__ = AsyncMock(return_value=response)
                session.post.return_value.__aexit__ = AsyncMock(return_value=False)
                with (
                    patch.object(proxy, "PROFILE_ENABLED", enabled),
                    patch.object(proxy, "decode_session", session),
                    patch.object(proxy.logger, "info") as log,
                ):
                    result = [chunk async for chunk in proxy.forward_stream({}, "parent", [], None, "http://pd", None)]
                self.assertEqual(result, ["data: role\n\n", "data: [DONE]\n\n"])
                records = [json.loads(c.args[1]) for c in log.call_args_list if c.args[0] == "NPU_EPD_TIMING %s"]
                if enabled:
                    first = [r for r in records if r["stage"] == "pd_first_byte"]
                    self.assertEqual(len(first), 1)
                    self.assertEqual(first[0]["attempt"], 0)
                    self.assertEqual(records[-1]["status"], "ok")
                else:
                    self.assertEqual(records, [])

    async def test_encoder_child_correlation(self):
        session = AsyncMock()
        session.post.return_value.status = 200
        with (
            patch.object(proxy, "PROFILE_ENABLED", True),
            patch.object(proxy, "encode_session", session),
            patch.object(proxy.logger, "info") as log,
        ):
            await proxy.encoder_post("http://encoder", {}, {"x-request-id": "child"}, "parent", "transfer")
        record = json.loads(log.call_args.args[1])
        self.assertEqual(record["stage"], "encoder_http_headers")
        self.assertEqual(record["request_id"], "child")
        self.assertEqual(record["parent_request_id"], "parent")
        self.assertEqual(record["transfer_id"], "transfer")
        self.assertEqual(record["target_url"], "http://encoder")


if __name__ == "__main__":
    unittest.main()
