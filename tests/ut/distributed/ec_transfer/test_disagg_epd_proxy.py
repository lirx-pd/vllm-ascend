# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression tests for the proxy's Mooncake request identity."""

import asyncio
import base64
import copy
import importlib.util
import io
import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import torch
from aiohttp import web
from aiohttp.test_utils import TestServer

_PROXY_PATH = Path(__file__).resolve().parents[4] / "examples/disaggregated_encoder/disagg_epd_proxy.py"
_spec = importlib.util.spec_from_file_location("npu_ec_proxy", _PROXY_PATH)
proxy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(proxy)


class ProxyIdentityTest(unittest.IsolatedAsyncioTestCase):
    def test_fixed_item_positions_rotate_across_all_encoders(self):
        for encoder_count in (1, 2, 3, 4):
            encoders = [f"http://e{i}" for i in range(encoder_count)]
            for item_count in (1, 2, 3, 4, 6):
                with self.subTest(encoders=encoder_count, items=item_count):
                    cursor = 0
                    assignments = []
                    for _ in encoders:
                        urls, cursor = proxy.encoder_rr_assignment(encoders, cursor, item_count)
                        assignments.append(urls)
                    for item_index in range(item_count):
                        self.assertCountEqual([urls[item_index] for urls in assignments], encoders)
                    self.assertEqual(cursor, 0)
            self.assertEqual(proxy.encoder_rr_assignment(encoders, encoder_count - 1, 0), ([], encoder_count - 1))

    async def test_concurrent_four_image_requests_balance_large_image_over_http(self):
        images = [{"type": "image_url", "image_url": {"url": f"image-{i}.png"}, "uuid": f"image-{i}"} for i in range(4)]
        request = {"messages": [{"content": images}]}
        received = {}

        async def encode(http_request):
            body = await http_request.json()
            parent, item, _ = http_request.headers["x-request-id"].split(":")
            received[parent, int(item)] = http_request.host
            self.assertEqual(body["messages"][0]["content"][0], images[int(item)])
            self.assertEqual(body["ec_transfer_params"]["consumer_zmq"], "tcp://consumer")
            return web.json_response({})

        upstream0, upstream1 = web.Application(), web.Application()
        for upstream in (upstream0, upstream1):
            upstream.router.add_post("/v1/chat/completions", encode)
        async with (
            TestServer(upstream0) as server0,
            TestServer(upstream1) as server1,
            proxy.aiohttp.ClientSession() as session,
        ):
            encoders = [str(server.make_url("/")).rstrip("/") for server in (server0, server1)]
            hosts = [f"{server.host}:{server.port}" for server in (server0, server1)]
            with (
                patch.object(proxy, "encode_session", session),
                patch.object(proxy, "NO_REWRITE", False),
                patch.object(proxy, "encoder_rr_idx", 0),
            ):
                results = await asyncio.gather(
                    *(proxy.fanout_encoder_primer(request, encoders, str(i), "tcp://consumer") for i in range(32))
                )
        self.assertEqual(len(received), 128)
        for host in hosts:
            self.assertEqual(sum(target == host for (parent, item), target in received.items() if item == 0), 16)
            self.assertEqual(sum(target == host for target in received.values()), 64)
        for metadata in results:
            self.assertEqual([metadata[i]["mm_hash"] for i in range(4)], [image["uuid"] for image in images])

    async def test_explicit_image_uuid_reaches_encoder_and_consumer(self):
        images = [
            {"type": "image_url", "image_url": {"url": "same.png"}, "uuid": identity}
            for identity in ("image-first", "image-second")
        ]
        request = {"messages": [{"content": images}]}
        response = MagicMock(status=200)
        response.read = AsyncMock(return_value=b"{}")
        session = AsyncMock()
        session.post.return_value = response
        with patch.object(proxy, "encode_session", session), patch.object(proxy, "NO_REWRITE", False):
            metadata = await proxy.fanout_encoder_primer(request, ["http://encoder"], "request", "tcp://consumer")
            forwarded = proxy.rewrite_for_decode(request, metadata)
        for index, call in enumerate(session.post.call_args_list):
            identity = images[index]["uuid"]
            encoded = call.kwargs["json"]
            self.assertEqual(encoded["messages"][0]["content"][0]["uuid"], identity)
            self.assertEqual(encoded["ec_transfer_params"]["ec_items"][0]["mm_hash"], identity)
            self.assertEqual(forwarded["messages"][0]["content"][index]["uuid"], identity)
            self.assertEqual(forwarded["ec_transfer_params"]["ec_items"][index]["mm_hash"], identity)
        self.assertEqual(
            proxy.content_uuid({**images[0], "uuid": None}),
            proxy.content_uuid({"type": "image_url", "image_url": {"url": "same.png"}}),
        )

    async def test_per_image_round_robin_and_metadata_only_decode(self):
        image = {"type": "image_url", "image_url": {"url": "test.png"}}
        request = {"messages": [{"content": [image, image, image]}]}

        async def encode(url, **kwargs):
            transfer = kwargs["json"]["ec_transfer_params"]["ec_items"][0]
            response = MagicMock(status=200)
            response.read = AsyncMock(
                return_value=json.dumps(
                    {"ec_transfer_params": {"ec_items": [{**transfer, "image_grid_thw": [[1, 32, 32]]}]}}
                ).encode()
            )
            return response

        session = AsyncMock()
        session.post.side_effect = encode
        with (
            patch.object(proxy, "encode_session", session),
            patch.object(proxy, "NO_REWRITE", False),
            patch.object(proxy, "encoder_rr_idx", 0),
        ):
            for req_id in ("first", "second"):
                metadata = await proxy.fanout_encoder_primer(
                    request, ["http://e0", "http://e1"], req_id, "tcp://pd:29600"
                )
                forwarded = proxy.rewrite_for_decode(request, metadata)
                for item in forwarded["messages"][0]["content"]:
                    self.assertEqual(item["type"], "image_embeds")
                    self.assertNotIn("image_url", item)
                    grid = torch.load(
                        io.BytesIO(base64.b64decode(item["image_embeds"]["image_grid_thw"])), weights_only=True
                    )
                    self.assertEqual(grid.tolist(), [1, 32, 32])
                    self.assertEqual(item["uuid"], proxy.content_uuid(image))
        self.assertEqual(
            [call.args[0] for call in session.post.call_args_list],
            [f"http://e{index % 2}/v1/chat/completions" for index in range(6)],
        )
        for call in session.post.call_args_list:
            self.assertEqual(call.kwargs["json"]["ec_transfer_params"]["consumer_zmq"], "tcp://pd:29600")

    def test_partial_metadata_keeps_all_images_raw(self):
        image = {"type": "image_url", "image_url": {"url": "test.png"}}
        request = {"messages": [{"content": [image, image]}]}
        metadata = {
            0: {"mm_hash": "same", "transfer_id": "a", "image_grid_thw": [[1, 32, 32]]},
            1: {"mm_hash": "same", "transfer_id": "b"},
        }
        forwarded = proxy.rewrite_for_decode(request, metadata)
        self.assertEqual([item["type"] for item in forwarded["messages"][0]["content"]], ["image_url", "image_url"])

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
