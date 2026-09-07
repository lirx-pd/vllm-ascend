# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ZMQ control plane for Mooncake encoder-cache reservations and events.

Tensor bytes never travel through this module.  It coordinates destination
reservations, completion/cancellation, and readiness notifications while
:mod:`transfer` owns the Mooncake data plane.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, TypedDict, cast

import torch
import zmq
from typing_extensions import NotRequired
from vllm.logger import init_logger

logger = init_logger(__name__)

_MAX_PENDING_EVENTS = 4096
_RESERVATION_REAP_INTERVAL_SECONDS = 1


class ReservationItem(TypedDict):
    """Identify one Consumer reservation in a batch control request."""

    transfer_id: str
    reservation_id: str


class EventPortRequest(TypedDict):
    """Request the PUSH socket port used for readiness events."""

    op: Literal["event_port"]


class StatusRequest(TypedDict):
    """Request the current status of a transfer reservation."""

    op: Literal["status"]
    transfer_id: str


class ReserveRequest(TypedDict):
    """Request destination memory for an encoder-cache tensor."""

    op: Literal["reserve"]
    transfer_id: str
    mm_hash: str
    nbytes: int
    shape: list[int]
    dtype: str


class CompleteBatchRequest(TypedDict):
    """Mark several destination writes complete in one exchange."""

    op: Literal["complete_batch"]
    items: list[ReservationItem]


class ReservationActionRequest(ReservationItem):
    """Complete or cancel one previously created reservation."""

    op: Literal["complete", "cancel"]
    abandon: NotRequired[bool]
    refresh: NotRequired[bool]


ControlRequest = EventPortRequest | StatusRequest | ReserveRequest | ReservationActionRequest | CompleteBatchRequest


class ControlSuccess(TypedDict):
    """Successful wire response with an optional operation result."""

    ok: Literal[True]
    result: NotRequired[Any]


class ControlFailure(TypedDict):
    """Failed wire response containing a user-facing error message."""

    ok: Literal[False]
    error: str


ControlResponse = ControlSuccess | ControlFailure


@dataclass(frozen=True)
class ControlCompletion:
    """Summarize the effect of a Consumer completion request.

    Attributes:
        accepted: Whether the reservation identity was valid.
        became_ready: Whether this call newly made the tensor readable.
    """

    accepted: bool
    became_ready: bool = False


class ControlClient:
    """Send control requests through reusable, thread-local REQ sockets.

    Attributes:
        _context: ZMQ context that owns all client sockets.
        _timeout_ms: Send and receive timeout for each exchange.
        _local: Thread-local mapping from address to REQ socket.
        _closed: Whether the client context has been destroyed.
    """

    def __init__(self, timeout_ms: int) -> None:
        self._context = zmq.Context()
        self._timeout_ms = timeout_ms
        self._local = threading.local()
        self._closed = False

    def _sockets(self) -> dict[str, zmq.Socket]:
        sockets = getattr(self._local, "sockets", None)
        if sockets is None:
            sockets = {}
            self._local.sockets = sockets
        return sockets

    def _discard(self, addr: str) -> None:
        socket = self._sockets().pop(addr, None)
        if socket is not None:
            socket.close(linger=0)

    def _exchange(self, addr: str, payload: ControlRequest) -> ControlResponse:
        sockets = self._sockets()
        socket = sockets.get(addr)
        if socket is None:
            socket = self._context.socket(zmq.REQ)
            socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
            socket.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
            socket.setsockopt(zmq.LINGER, 0)
            socket.connect(addr)
            sockets[addr] = socket
        try:
            socket.send_json(payload)
            response = socket.recv_json()
        except Exception:
            self._discard(addr)
            raise
        assert isinstance(response, dict)
        return cast(ControlResponse, response)

    def request(self, addr: str, payload: ControlRequest) -> Any:
        response = self._exchange(addr, payload)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "EC control request failed"))
        return response.get("result")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._context.destroy(linger=0)


def make_cancel_request(
    transfer_id: str,
    reservation_id: str,
    *,
    abandon: bool = False,
    refresh: bool = False,
) -> ReservationActionRequest:
    request: ReservationActionRequest = {
        "op": "cancel",
        "transfer_id": transfer_id,
        "reservation_id": reservation_id,
    }
    if abandon:
        request["abandon"] = True
    if refresh:
        request["refresh"] = True
    return request


class EventInbox:
    """Receive Consumer readiness events without blocking the Scheduler."""

    def __init__(self, client: ControlClient) -> None:
        self._client = client
        self._context: zmq.Context | None = None
        self._socket: zmq.Socket | None = None
        self._closed = False

    def _connect(self, base_addr: str) -> None:
        if self._socket is not None:
            return
        try:
            event_port = self._client.request(base_addr, {"op": "event_port"})
            address, _ = base_addr.rsplit(":", 1)
            endpoint = f"{address}:{int(event_port)}"
        except Exception:
            logger.warning(
                "EC Mooncake could not subscribe to the event channel of consumer %s; retrying later.",
                base_addr,
                exc_info=True,
            )
            return
        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        socket.connect(endpoint)
        self._context = context
        self._socket = socket

    def drain(self, base_addr: str) -> list[dict[str, Any]]:
        self._connect(base_addr)
        if self._socket is None:
            return []
        events = []
        while True:
            try:
                events.append(self._socket.recv_json(flags=zmq.DONTWAIT))
            except zmq.Again:
                return events

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._socket is not None:
            self._socket.close(linger=0)
        if self._context is not None:
            self._context.term()


class ConsumerControlServer:
    """Expose Consumer reservations and readiness events over ZMQ.

    The REP channel handles reservation operations, while a PUSH channel
    publishes newly ready items to the Scheduler.

    Attributes:
        host: Interface on which the control server listens.
        port: Rank-local REP control port.
        event_port: Dynamically allocated PUSH event port after startup.
        _device: NPU device selected in the control thread.
        _reserve: Callback that allocates or reuses destination memory.
        _status: Callback that reports active reservation state.
        _complete: Callback that marks destination writes complete.
        _cancel: Callback that cancels or abandons reservations.
        _reap: Callback that expires stale reservations.
        _metrics_log_interval: Interval for aggregate control-plane logs.
        _stop: Signal requesting termination of the server loop.
        _started: Signal indicating that socket binding has completed.
        _thread: Background server thread.
        _startup_error: Socket binding error captured from the server thread.
    """

    def __init__(
        self,
        host: str,
        port: int,
        reserve: Callable[[dict[str, Any]], dict[str, Any]],
        status: Callable[[str], dict[str, Any] | None],
        complete: Callable[[str, str], ControlCompletion],
        cancel: Callable[[str, str, bool, bool], bool],
        reap: Callable[[], int],
        metrics_log_interval: float = 10,
        device: torch.device | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self._device = device
        self.event_port: int | None = None
        self._reserve = reserve
        self._status = status
        self._complete = complete
        self._cancel = cancel
        self._reap = reap
        self._metrics_log_interval = metrics_log_interval
        self._stop = threading.Event()
        self._started = threading.Event()
        self._thread: threading.Thread | None = None
        self._startup_error: Exception | None = None

    def start(self) -> None:
        def loop() -> None:
            if self._device is not None:
                torch.npu.set_device(self._device)
            context = zmq.Context()
            socket = context.socket(zmq.REP)
            event_socket = context.socket(zmq.PUSH)
            pending_events: deque[dict[str, Any]] = deque()
            metrics: Counter[str] = Counter()

            def queue_event(event: dict[str, Any]) -> None:
                if len(pending_events) >= _MAX_PENDING_EVENTS:
                    pending_events.popleft()
                    metrics["events_dropped"] += 1
                pending_events.append(event)
                metrics["events_queued"] += 1

            metrics_started_at = time.monotonic()
            last_reap_at = metrics_started_at
            socket.setsockopt(zmq.RCVTIMEO, 100)
            try:
                socket.bind(f"tcp://{self.host}:{self.port}")
                self.event_port = event_socket.bind_to_random_port(f"tcp://{self.host}")
            except Exception as e:
                self._startup_error = e
                self._started.set()
                socket.close(linger=0)
                event_socket.close(linger=0)
                context.term()
                return
            self._started.set()
            try:
                while not self._stop.is_set():
                    while pending_events:
                        try:
                            event_socket.send_json(pending_events[0], flags=zmq.DONTWAIT)
                        except zmq.Again:
                            break
                        pending_events.popleft()
                        metrics["events_sent"] += 1
                    now = time.monotonic()
                    if now - last_reap_at >= _RESERVATION_REAP_INTERVAL_SECONDS:
                        metrics["reservations_reaped"] += self._reap()
                        last_reap_at = now
                    if self._metrics_log_interval > 0 and now - metrics_started_at >= self._metrics_log_interval:
                        logger.info(
                            "EC Mooncake consumer control: requests=%s, "
                            "events_queued=%d, events_sent=%d, events_dropped=%d, "
                            "event_backlog=%d, reservations_reaped=%d",
                            {
                                key.removeprefix("request_"): value
                                for key, value in metrics.items()
                                if key.startswith("request_")
                            },
                            metrics["events_queued"],
                            metrics["events_sent"],
                            metrics["events_dropped"],
                            len(pending_events),
                            metrics["reservations_reaped"],
                        )
                        metrics.clear()
                        metrics_started_at = now
                    try:
                        request = socket.recv_json()
                    except zmq.Again:
                        continue
                    try:
                        op = request.get("op")
                        result: Any = None
                        metrics[f"request_{op}"] += 1
                        if op == "reserve":
                            result = self._reserve(request)
                            if result.get("ready"):
                                transfer_id = str(request["transfer_id"])
                                status = self._status(transfer_id)
                                if status is not None:
                                    queue_event({"transfer_id": transfer_id, **status})
                        elif op == "status":
                            result = self._status(str(request["transfer_id"]))
                        elif op == "event_port":
                            result = self.event_port
                        elif op in ("complete", "complete_batch"):
                            items = request["items"] if op == "complete_batch" else [request]
                            completions = []
                            for item in items:
                                transfer_id = str(item["transfer_id"])
                                completion = self._complete(transfer_id, str(item["reservation_id"]))
                                completions.append(
                                    {
                                        "completed": completion.accepted,
                                        "became_ready": completion.became_ready,
                                    }
                                )
                                if not completion.became_ready:
                                    continue
                                status = self._status(transfer_id)
                                if status is not None:
                                    queue_event({"transfer_id": transfer_id, **status})
                            result = {"items": completions} if op == "complete_batch" else completions[0]
                        elif op == "cancel":
                            result = {
                                "cancelled": self._cancel(
                                    str(request["transfer_id"]),
                                    str(request.get("reservation_id", "")),
                                    bool(request.get("abandon", False)),
                                    bool(request.get("refresh", False)),
                                )
                            }
                        else:
                            raise ValueError(f"unknown control op: {op!r}")
                        socket.send_json({"ok": True, "result": result})
                    except Exception as e:
                        socket.send_json({"ok": False, "error": str(e)})
            finally:
                socket.close(linger=0)
                event_socket.close(linger=0)
                context.term()

        self._thread = threading.Thread(target=loop, name="ec-mooncake-control", daemon=True)
        self._thread.start()
        if not self._started.wait(timeout=5):
            raise RuntimeError("EC Mooncake control channel failed to start")
        if self._startup_error is not None:
            raise RuntimeError("EC Mooncake control channel failed to bind") from (self._startup_error)
        logger.info(
            "EC Mooncake control channel listening on tcp://%s:%d (events tcp://%s:%d)",
            self.host,
            self.port,
            self.host,
            self.event_port,
        )

    def close(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        self._thread = None
