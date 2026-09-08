# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression tests for the proxy's Mooncake request identity."""

import asyncio
import copy
import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from aiohttp import web
from aiohttp.test_utils import TestServer

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
                response.release = MagicMock()
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
        response.release = MagicMock()
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
                    response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
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
                        self.assertEqual(await call, {"choices": [{"message": {"content": "ok"}}]})
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
            yield b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
            yield b"data: [DONE]\n\n"

        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                response = MagicMock()
                response.status = 200
                response.content.__aiter__.side_effect = chunks
                session = MagicMock()
                session.post.return_value.__aenter__ = AsyncMock(return_value=response)
                session.post.return_value.__aexit__ = AsyncMock(return_value=False)
                with (
                    patch.object(proxy, "PROFILE_ENABLED", enabled),
                    patch.object(proxy, "decode_session", session),
                    patch.object(proxy.logger, "info") as log,
                ):
                    result = [chunk async for chunk in proxy.forward_stream({}, "parent", [], None, "http://pd", None)]
                self.assertEqual(result, ['data: {"choices":[{"delta":{"content":"ok"}}]}\n\n', "data: [DONE]\n\n"])
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


class ProxyFailureTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def event(payload):
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()

    def context(self, chunks=(), status=200):
        async def lines():
            for chunk in chunks:
                if isinstance(chunk, Exception):
                    raise chunk
                yield chunk

        response = MagicMock(status=status)
        response.text = AsyncMock(return_value="upstream failure")
        response.content.__aiter__.side_effect = lines
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=response)
        context.__aexit__ = AsyncMock(return_value=False)
        return context

    async def run_stream(self, contexts, request=None):
        session = MagicMock()
        session.post.side_effect = contexts
        with (
            patch.object(proxy, "decode_session", session),
            patch.object(proxy, "prepare_for_decode", AsyncMock(return_value=({}, 0, 0))) as prepare,
            patch.object(proxy, "PROFILE_ENABLED", True),
            patch.object(proxy.logger, "info") as log,
        ):
            result = [
                chunk async for chunk in proxy.forward_stream(request or {}, "request", [], None, "http://pd", None)
            ]
        records = [json.loads(c.args[1]) for c in log.call_args_list if c.args[0] == "NPU_EPD_TIMING %s"]
        return result, session.post.call_count, prepare.await_count, records

    async def test_stream_retries_only_before_generated_output(self):
        role = self.event({"choices": [{"delta": {"role": "assistant"}}]})
        content = self.event({"choices": [{"delta": {"content": "answer"}}]})
        failures = (
            [role, self.event({"choices": [{"finish_reason": "error"}]})],
            [role, self.event({"error": {"message": "EC load failed"}})],
            [role, b"data: [DONE]\n\n"],
            [role],
            [role, proxy.aiohttp.ServerDisconnectedError()],
        )
        for failure in failures:
            with self.subTest(failure=failure):
                contexts = [self.context(failure), self.context([role, content, b"data: [DONE]\n\n"])]
                result, calls, prepared, records = await self.run_stream(contexts)
                self.assertEqual(result, [role.decode(), content.decode(), "data: [DONE]\n\n"])
                self.assertEqual((calls, prepared), (2, 2))
                self.assertEqual([r["status"] for r in records if r["stage"] == "attempt"], ["retry", "ok"])
                self.assertEqual(records[-1]["status"], "ok")
                for context in contexts:
                    context.__aexit__.assert_awaited_once()

    async def test_partial_stream_failure_never_replays_output(self):
        content = self.event({"choices": [{"delta": {"content": "partial"}}]})
        for failure in (
            [self.event({"choices": [{"finish_reason": "error"}]})],
            [proxy.aiohttp.ClientPayloadError("incomplete body")],
            [],
        ):
            with self.subTest(failure=failure):
                result, calls, prepared, records = await self.run_stream([self.context([content, *failure])])
                self.assertEqual((calls, prepared), (1, 1))
                self.assertEqual(result[0], content.decode())
                self.assertIn('"error"', result[1])
                self.assertEqual(result[2], "data: [DONE]\n\n")
                self.assertEqual(records[-1]["status"], "error")

    async def test_stream_exhaustion_and_non_retryable_http_error(self):
        for contexts in (
            [self.context(status=400)],
            [self.context(status=500), self.context(status=500)],
            [self.context(), self.context()],
        ):
            with self.subTest(contexts=contexts):
                result, calls, _, records = await self.run_stream(contexts)
                self.assertEqual(calls, len(contexts))
                self.assertEqual(len(result), 2)
                self.assertIn('"error"', result[0])
                self.assertEqual(result[1], "data: [DONE]\n\n")
                self.assertEqual(records[-1]["status"], "error")

    async def test_connection_failure_before_response_retries(self):
        first = self.context()
        first.__aenter__.side_effect = proxy.aiohttp.ServerDisconnectedError()
        content = self.event({"choices": [{"delta": {"content": "ok"}}]})
        result, calls, _, records = await self.run_stream([first, self.context([content, b"data: [DONE]\n\n"])])
        self.assertEqual(calls, 2)
        self.assertEqual(result, [content.decode(), "data: [DONE]\n\n"])
        self.assertEqual(records[-1]["status"], "ok")

    async def test_fragmented_utf8_and_tool_output(self):
        for delta in ({"content": "中文"}, {"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}):
            with self.subTest(delta=delta):
                event = self.event({"choices": [{"delta": delta}]})
                reader = proxy.aiohttp.StreamReader(MagicMock(), limit=1024, loop=asyncio.get_running_loop())
                for byte in event + b"data: [DONE]\n\n":
                    reader.feed_data(bytes([byte]))
                reader.feed_eof()
                context = self.context()
                context.__aenter__.return_value.content = reader
                result, calls, _, _ = await self.run_stream([context])
                self.assertEqual(calls, 1)
                self.assertEqual(result, [event.decode(), "data: [DONE]\n\n"])

    async def test_asgi_error_completes_response_after_headers(self):
        session = MagicMock()
        session.post.side_effect = [self.context(status=400)]
        response = proxy.StreamingResponse(proxy.forward_stream({}, "request", [], None, "http://pd", None))
        send = AsyncMock()
        with (
            patch.object(proxy, "decode_session", session),
            patch.object(proxy, "prepare_for_decode", AsyncMock(return_value=({}, 0, 0))),
        ):
            await response({"type": "http", "asgi": {"spec_version": "2.4"}}, AsyncMock(), send)
        messages = [call.args[0] for call in send.await_args_list]
        self.assertEqual(messages[0]["type"], "http.response.start")
        self.assertIn(b'"error"', messages[1]["body"])
        self.assertEqual(messages[-1], {"type": "http.response.body", "body": b"", "more_body": False})

    async def test_client_cancellation_releases_upstream(self):
        content = self.event({"choices": [{"delta": {"content": "partial"}}]})
        context = self.context([content])
        session = MagicMock()
        session.post.return_value = context
        with (
            patch.object(proxy, "decode_session", session),
            patch.object(proxy, "prepare_for_decode", AsyncMock(return_value=({}, 0, 0))),
        ):
            stream = proxy.forward_stream({}, "request", [], None, "http://pd", None)
            self.assertEqual(await anext(stream), content.decode())
            await stream.aclose()
        context.__aexit__.assert_awaited_once()
        self.assertEqual(session.post.call_count, 1)

    async def test_stream_accepts_empty_completions_only_after_all_choices_finish(self):
        for count, indices in ((1, [0]), (2, [0, 1]), (2, [0]), (2, [0, 0])):
            with self.subTest(count=count, indices=indices):
                chunks = [self.event({"choices": [{"index": 0, "delta": {"role": "assistant"}}]})]
                chunks.extend(
                    self.event({"choices": [{"index": index, "delta": {"content": ""}, "finish_reason": "stop"}]})
                    for index in indices
                )
                chunks.append(self.event({"choices": [], "usage": {"completion_tokens": len(indices)}}))
                chunks.append(b"data: [DONE]\n\n")
                complete = len(set(indices)) == count
                contexts = [self.context(chunks) for _ in range(1 if complete else 2)]
                result, calls, _, records = await self.run_stream(contexts, {"n": count})
                self.assertEqual(calls, len(contexts))
                if complete:
                    self.assertEqual(result, [chunk.decode() for chunk in chunks])
                    self.assertEqual(records[-1]["status"], "ok")
                else:
                    self.assertEqual(len(result), 2)
                    self.assertIn('"error"', result[0])
                    self.assertEqual(records[-1]["status"], "error")

    async def test_nonstream_accepts_legal_empty_stop(self):
        for content in ("", None):
            with self.subTest(content=content):
                result = {"choices": [{"index": 0, "message": {"content": content}, "finish_reason": "stop"}]}
                context = self.context()
                context.__aenter__.return_value.json = AsyncMock(return_value=result)
                session = MagicMock()
                session.post.return_value = context
                with (
                    patch.object(proxy, "decode_session", session),
                    patch.object(proxy, "prepare_for_decode", AsyncMock(return_value=({}, 0, 0))),
                ):
                    self.assertEqual(await proxy.forward_non_stream({}, "request", [], None, "http://pd", None), result)
                self.assertEqual(session.post.call_count, 1)

    async def test_nonstream_internal_error_and_missing_choices(self):
        success = {"choices": [{"message": {"content": "answer"}}]}
        for failed in ({"choices": [{"finish_reason": "error"}]}, {"error": {"message": "failed"}}, {"choices": []}):
            for recovered in (True, False):
                with self.subTest(failed=failed, recovered=recovered):
                    contexts = [self.context(), self.context()]
                    contexts[0].__aenter__.return_value.json = AsyncMock(return_value=failed)
                    contexts[1].__aenter__.return_value.json = AsyncMock(return_value=success if recovered else failed)
                    session = MagicMock()
                    session.post.side_effect = contexts
                    with (
                        patch.object(proxy, "decode_session", session),
                        patch.object(proxy, "prepare_for_decode", AsyncMock(return_value=({}, 0, 0))),
                    ):
                        call = proxy.forward_non_stream({}, "request", [], None, "http://pd", None)
                        if recovered:
                            self.assertEqual(await call, success)
                        else:
                            with self.assertRaises(proxy.HTTPException) as error:
                                await call
                            self.assertEqual(error.exception.status_code, 502)
                    self.assertEqual(session.post.call_count, 2)

    async def test_encoder_failure_releases_sibling_responses(self):
        failed = MagicMock(status=500)
        failed.text = AsyncMock(return_value="encoder failed")
        sibling = MagicMock(status=200)
        session = AsyncMock()
        session.post.side_effect = [failed, sibling]
        image = {"type": "image_url", "image_url": {"url": "test.png"}}
        with patch.object(proxy, "encode_session", session), self.assertRaises(proxy.HTTPException):
            await proxy.fanout_encoder_primer(
                {"messages": [{"content": [image, image]}]}, ["http://encoder"], "request"
            )
        failed.release.assert_called_once()
        sibling.release.assert_called_once()


class ProxyConnectionTest(unittest.IsolatedAsyncioTestCase):
    async def test_completed_requests_release_connections_without_reuse(self):
        transports = []
        event = b'data: {"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'

        async def handle(request):
            transports.append(request.transport)
            payload = await request.json()
            if not payload.get("stream"):
                return web.json_response({})
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await response.write(event)
            await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
            return response

        upstream = web.Application()
        upstream.router.add_post("/v1/chat/completions", handle)
        async with TestServer(upstream) as server:
            url = str(server.make_url("/")).rstrip("/")
            with (
                patch.object(proxy.app.state, "p_urls", [], create=True),
                patch.object(proxy, "encode_session", None),
                patch.object(proxy, "prefill_session", None),
                patch.object(proxy, "decode_session", None),
            ):
                await proxy.on_startup()
                try:
                    for index in range(2):
                        response = await proxy.encoder_post(url, {}, {"x-request-id": str(index)}, "parent", str(index))
                        self.assertEqual(await response.json(), {})
                        response.release()
                        chunks = [
                            chunk
                            async for chunk in proxy.forward_stream({"stream": True}, str(index), [], None, url, None)
                        ]
                        self.assertEqual(chunks, [event.decode(), "data: [DONE]\n\n"])
                    self.assertEqual(len(set(transports)), 4)
                    async with asyncio.timeout(1):
                        while not all(transport.is_closing() for transport in transports):
                            await asyncio.sleep(0.01)
                finally:
                    await proxy.on_shutdown()


if __name__ == "__main__":
    unittest.main()
