# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression tests for the proxy's Mooncake request identity."""

import copy
import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

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


if __name__ == "__main__":
    unittest.main()
