# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side orchestration of Mooncake control, memory, and data planes.

Consumer Workers expose rank-local reservations and publish received tensors.
Producer Workers reserve a destination, bind computed sources, run
batched Mooncake writes, and report asynchronous completion to the Scheduler.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from functools import partial
from typing import TYPE_CHECKING, Any, TypeVar, cast

import torch
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip

from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake._availability import (
    ensure_mooncake_available,
)
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.config import MooncakeECConfig
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.control import (
    ConsumerControlServer,
    ControlClient,
    ControlCompletion,
    make_cancel_request,
)
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.memory import (
    ConsumerMemoryPool,
    MemoryAllocation,
    ProducerMemoryPool,
)
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.metadata import (
    ECMooncakeConnectorMetadata,
    ECMooncakeLoadSpec,
    ECMooncakePushSpec,
    ECMooncakeWorkerMetadata,
)
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.producer import (
    ProducerPushManager,
    ProducerPushRecord,
)
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.reservation import (
    CancellationOutcome,
    ConsumerReservationManager,
    ConsumerReservationState,
)
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.transfer import (
    MooncakeTransfer,
)

logger = init_logger(__name__)

_T = TypeVar("_T")

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_LEASE_TTL_SECONDS = 300
_RESERVATION_REFRESH_SECONDS = _LEASE_TTL_SECONDS / 2
_MAX_CANCELLED_TRANSFER_IDS = 1 << 16
_CANCEL_ATTEMPTS = 2
_READY_EVENT_POLL_SECONDS = 0.001


class _FanoutError(RuntimeError):
    """Retain task outcomes after every started task settles."""

    def __init__(self, error: BaseException, results: list[Any | None]) -> None:
        self.results = results
        super().__init__(str(error))


class ECMooncakeWorker:
    """Orchestrate consumer reservations and producer push batches.

    ``mooncake_protocol`` selects the transfer protocol. Consumer workers use
    ``consumer_buffer_pool_size`` and ``reservation_zmq_port`` for their
    registered receive arena and rank-local control endpoint. Producers use
    ``producer_buffer_pool_size`` for staging. ``transfer_max_workers`` and
    ``control_max_workers`` bound the two executor pools.

    This first MRV1 delivery supports single-rank TP, PP, and DP only. The
    fixed configuration check keeps distributed topology out of the data-plane
    lifecycle until it has dedicated Ascend coverage.

    Attributes:
        is_producer: Whether this Worker originates encoder-cache pushes.
        is_consumer: Whether this Worker accepts encoder-cache pushes.
        _reservation_zmq_port: Consumer control port.
        _transfer: Owner of the Mooncake engine and memory registrations.
        _consumer_memory: Registered receive slab and resident cache.
        _reservations: Consumer destination reservation state manager.
        _control_server: Rank-local Consumer reservation server.
        _producer_memory: Registered Producer source staging slab.
        _staging_lock: Exclusive slab ownership through transfer completion.
        _control_client: Client for remote Consumer control operations.
        _io_executor: Executor that owns transfer and cancellation batches.
        _control_executor: Executor that creates remote reservations.
        _fanout_pool: Lazily created executor for concurrent destination work.
        _fanout_pool_lock: Lock protecting fan-out pool initialization.
        _producer_pushes: Producer lifecycle and source-ownership manager.
        _completed_loads: Successful Consumer loads awaiting reporting.
        _failed_loads: Failed Consumer loads awaiting reporting.
        _shutdown: Whether Worker resource shutdown has started.
    """

    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig) -> ECMooncakeWorker:
        ensure_mooncake_available()
        config = MooncakeECConfig.from_vllm_config(vllm_config, ECConnectorRole.WORKER)
        hostname = get_ip()
        control_client = ControlClient(config.control_timeout_ms)
        try:
            return cls(config, hostname, control_client)
        except Exception:
            control_client.close()
            raise

    def __init__(
        self,
        config: MooncakeECConfig,
        hostname: str,
        control_client: ControlClient,
    ) -> None:
        self.is_producer = config.is_producer
        self.is_consumer = config.is_consumer
        configured_device = torch.device(config.buffer_device)
        device_index = configured_device.index
        if device_index is None:
            device_index = torch.npu.current_device()
        self._device = torch.device("npu", device_index)
        self._reservation_zmq_port = config.reservation_port
        self._transfer = MooncakeTransfer(hostname, config.protocol)
        self._consumer_memory = ConsumerMemoryPool(
            config.consumer_pool_size,
            self._transfer,
        )
        self._reservations = ConsumerReservationManager(
            self._consumer_memory,
            _LEASE_TTL_SECONDS,
            _MAX_CANCELLED_TRANSFER_IDS,
        )
        self._control_server: ConsumerControlServer | None = None
        # Worker producer
        self._producer_memory = ProducerMemoryPool(
            config.producer_pool_size,
            self._transfer,
        )
        self._staging_lock = threading.Lock()
        self._control_client = control_client
        self._io_executor = ThreadPoolExecutor(
            max_workers=config.transfer_workers,
            thread_name_prefix="ec-mooncake-transfer",
            initializer=self._initialize_transfer_thread,
        )
        self._control_executor = ThreadPoolExecutor(
            max_workers=config.control_workers,
            thread_name_prefix="ec-mooncake-control",
            initializer=torch.npu.set_device,
            initargs=(self._device,),
        )
        self._fanout_pool: ThreadPoolExecutor | None = None
        self._fanout_pool_lock = threading.Lock()
        self._push_ready = threading.Event()
        self._dispatch_stop = threading.Event()
        self._dispatcher: threading.Thread | None = None
        self._producer_pushes = ProducerPushManager(self._push_ready.set)
        self._completed_loads: set[str] = set()
        self._failed_loads: set[str] = set()
        self._shutdown = False
        if self.is_producer:
            self._dispatcher = threading.Thread(target=self._dispatch_pushes, name="ec-mooncake-ready", daemon=True)
            self._dispatcher.start()

    def _initialize_transfer_thread(self) -> None:
        torch.npu.set_device(self._device)
        # Staging must not synchronize later encoder work on the default stream.
        torch.npu.set_stream(torch.npu.Stream(device=self._device))

    def _dispatch_pushes(self) -> None:
        torch.npu.set_device(self._device)
        pending_event = False
        while not self._dispatch_stop.is_set():
            self._push_ready.wait(timeout=_READY_EVENT_POLL_SECONDS if pending_event else None)
            self._push_ready.clear()
            if self._dispatch_stop.is_set():
                break
            pending_event = self._flush_pending_pushes()

    def start_services(self) -> None:
        if not self.is_consumer or self._reservation_zmq_port is None or self._control_server is not None:
            return
        self._consumer_memory.prepare(self._device, receiving_rank=True)
        consumer_pool = self._consumer_memory.tensor
        if consumer_pool is None:
            raise RuntimeError("Mooncake push mode requires a registered consumer buffer pool.")
        self._control_server = ConsumerControlServer(
            "0.0.0.0",
            self._reservation_zmq_port,
            self._reserve_push_destination,
            self._push_status,
            self._complete_push,
            self._cancel_push,
            self._expire_push_reservations,
            device=consumer_pool.device,
            drain_events=self._reservations.drain_events,
        )
        try:
            self._control_server.start()
        except Exception:
            self._control_server.close()
            self._control_server = None
            raise

    def _expire_push_reservations(self) -> int:
        expired, _, _ = self._reservations.expire()
        return expired

    def _reserve_push_destination(self, payload: dict[str, Any]) -> dict[str, Any]:
        transfer_id = str(payload["transfer_id"])
        mm_hash = str(payload["mm_hash"])
        nbytes = int(payload["nbytes"])
        shape = tuple(int(value) for value in payload["shape"])
        if nbytes <= 0:
            raise ValueError("EC reservation nbytes must be positive")
        if not shape or any(dimension <= 0 for dimension in shape):
            raise ValueError("EC reservation shape must contain positive dimensions")
        dtype_name = str(payload["dtype"])
        dtype = getattr(torch, dtype_name, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Unsupported torch dtype string: {dtype_name!r}")
        expected_nbytes = math.prod(shape) * dtype.itemsize
        if expected_nbytes != nbytes:
            raise ValueError("shape and dtype do not match nbytes")

        reservation, should_write, _, _ = self._reservations.reserve(
            transfer_id, mm_hash, nbytes, shape, dtype_name, dtype
        )
        if reservation is None:
            raise RuntimeError("EC consumer buffer pool is full")
        if reservation.state in {
            ConsumerReservationState.CANCEL_PENDING,
            ConsumerReservationState.CANCELLED,
        }:
            return {
                "reservation_id": "",
                "dst_session": "",
                "dst_ptr": 0,
                "nbytes": nbytes,
                "write": False,
                "ready": False,
                "cancelled": True,
            }
        assert reservation.allocation is not None

        return {
            "reservation_id": reservation.reservation_id,
            "dst_session": self._transfer.local_session(),
            "dst_ptr": reservation.allocation.tensor.data_ptr(),
            "nbytes": reservation.allocation.tensor.nbytes,
            "write": should_write,
            "ready": reservation.state is ConsumerReservationState.READY,
            "cached": reservation.lease is not None,
        }

    def _push_status(self, transfer_id: str) -> dict[str, Any] | None:
        reservation = self._reservations.status(transfer_id)
        if reservation is None:
            return None
        assert reservation.allocation is not None
        return {
            "mm_hash": reservation.mm_hash,
            "ready": reservation.state is ConsumerReservationState.READY,
            "reservation_id": reservation.reservation_id,
            "nbytes": reservation.allocation.tensor.nbytes,
            "shape": list(reservation.shape),
            "dtype": reservation.dtype,
        }

    def _complete_push(self, transfer_id: str, reservation_id: str) -> ControlCompletion:
        result = self._reservations.complete(transfer_id, reservation_id)
        return ControlCompletion(result.accepted, result.became_ready)

    def _cancel_push(
        self,
        transfer_id: str,
        reservation_id: str,
        abandon: bool = False,
        refresh: bool = False,
    ) -> bool:
        outcome, _ = self._reservations.cancel(transfer_id, reservation_id, abandon, refresh)
        return outcome is not CancellationOutcome.REJECTED

    def _take_pushed_tensor(self, spec: ECMooncakeLoadSpec) -> tuple[torch.Tensor, MemoryAllocation]:
        allocation = self._reservations.take(spec.transfer_id, spec.mm_hash)
        return allocation.tensor, allocation

    def _fanout_executor(self) -> ThreadPoolExecutor:
        """Use a separate pool so nested fan-out cannot deadlock."""
        with self._fanout_pool_lock:
            if self._fanout_pool is None:
                self._fanout_pool = ThreadPoolExecutor(
                    max_workers=32,
                    thread_name_prefix="ec-mooncake-fanout",
                    initializer=torch.npu.set_device,
                    initargs=(self._device,),
                )
            return self._fanout_pool

    def _reserve_one(self, addr: str, spec: ECMooncakePushSpec) -> dict[str, Any]:
        result = self._control_client.request(
            addr,
            {
                "op": "reserve",
                "transfer_id": spec.transfer_id,
                "mm_hash": spec.mm_hash,
                "nbytes": spec.nbytes,
                "shape": list(spec.shape),
                "dtype": spec.dtype,
            },
        )
        if not isinstance(result, dict):
            raise RuntimeError("Invalid EC reservation response")
        result["_received_at"] = time.monotonic()
        result["addr"] = addr
        return result

    def _run_fanout(
        self,
        tasks: list[Callable[[], _T]],
        on_submit: Callable[[int, Future[_T]], None] | None = None,
    ) -> list[_T]:
        if not tasks:
            return []
        futures: list[tuple[int, Future[_T]]] = []
        results: list[_T | None] = [None] * len(tasks)
        error: BaseException | None = None
        for index, task in enumerate(tasks[1:], 1):
            try:
                future = self._fanout_executor().submit(task)
            except Exception as exc:
                error = exc
                break
            futures.append((index, future))
            if on_submit is not None:
                on_submit(index, future)
        if error is None:
            try:
                results[0] = tasks[0]()
            except Exception as exc:
                error = exc
        for index, future in futures:
            try:
                results[index] = future.result()
            except Exception as exc:
                if error is None:
                    error = exc
        if error is not None:
            raise _FanoutError(error, results)
        return cast(list[_T], results)

    def _cancel_reservations(
        self,
        spec: ECMooncakePushSpec,
        reservations: list[dict[str, Any]],
        *,
        refresh: bool = False,
    ) -> None:
        for reservation in reservations:
            if reservation.get("cancelled", False):
                continue
            request = make_cancel_request(
                spec.transfer_id,
                str(reservation.get("reservation_id", "")),
                abandon=True,
                refresh=refresh,
            )
            request["mm_hash"] = spec.mm_hash
            result = self._control_client.request(str(reservation.get("addr", spec.consumer_zmq)), request)
            if not isinstance(result, dict) or not result.get("cancelled"):
                raise RuntimeError(f"Could not cancel EC reservation for mm_hash={spec.mm_hash}")

    def _retry_cancel_reservations(
        self,
        spec: ECMooncakePushSpec,
        reservations: list[dict[str, Any]],
    ) -> None:
        error: Exception | None = None
        for _ in range(_CANCEL_ATTEMPTS):
            try:
                self._cancel_reservations(spec, reservations)
            except Exception as exc:
                error = exc
                continue
            return
        assert error is not None
        raise error

    def _reserve_remote(self, spec: ECMooncakePushSpec) -> list[dict[str, Any]]:
        try:
            return [self._reserve_one(spec.consumer_zmq, spec)]
        except Exception:
            # No writer has started. Cancel even an unacknowledged reservation
            # and notify the consumer instead of leaving it to time out.
            try:
                self._retry_cancel_reservations(spec, [{"addr": spec.consumer_zmq, "reservation_id": ""}])
            except Exception:
                logger.exception("Failed to cancel rejected EC reservation for transfer_id=%s", spec.transfer_id)
            raise

    def _refresh_remote_reservations(
        self,
        spec: ECMooncakePushSpec,
        reservations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        stale = [
            shard
            for shard in reservations
            if not shard.get("ready", False) and not shard.get("cached", False) and not shard.get("cancelled", False)
        ]
        try:
            self._cancel_reservations(spec, stale, refresh=True)
        except Exception as exc:
            try:
                self._retry_cancel_reservations(spec, stale)
            except Exception as cleanup_error:
                raise exc from cleanup_error
            raise
        return self._reserve_remote(spec)

    @staticmethod
    def _validate_push_source(push: ProducerPushRecord) -> None:
        source = push.source
        assert source is not None
        tensor = source.tensor
        spec = push.spec
        if tuple(tensor.shape) != tuple(spec.shape):
            raise ValueError(f"EC source shape mismatch for mm_hash={spec.mm_hash}")
        if str(tensor.dtype).split(".")[-1] != spec.dtype:
            raise ValueError(f"EC source dtype mismatch for mm_hash={spec.mm_hash}")
        if not tensor.is_contiguous():
            raise ValueError(f"EC source must be contiguous for mm_hash={spec.mm_hash}")
        if tensor.nbytes != spec.nbytes:
            raise ValueError(f"EC source size mismatch for mm_hash={spec.mm_hash}")
        if tensor.device.type != "npu":
            raise ValueError(f"EC source must be on NPU for mm_hash={spec.mm_hash}")

    def start_save_caches(
        self,
        metadata: ECMooncakeConnectorMetadata,
        encoder_cache: dict[str, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> None:
        new: dict[str, list[ProducerPushRecord]] = {}
        for spec in metadata.pushes:
            record, created = self._producer_pushes.reserve(spec, Future)
            if created:
                new.setdefault(spec.consumer_zmq, []).append(record)
        for records in new.values():
            self._control_executor.submit(self._reserve_batch, records)
        if not isinstance(encoder_cache, dict):
            return
        for mm_hash in dict.fromkeys(spec.mm_hash for spec in metadata.pushes):
            tensor = encoder_cache.get(mm_hash)
            if tensor is not None:
                self._bind_push_source(tensor, mm_hash)

    def _reserve_batch(self, records: list[ProducerPushRecord]) -> None:
        if len(records) == 1:
            record = records[0]
            try:
                result = self._reserve_remote(record.spec)
            except Exception as exc:
                record.reservation_future.set_exception(exc)
            else:
                record.reservation_future.set_result(result)
            return
        addr = records[0].spec.consumer_zmq
        try:
            response = self._control_client.request(
                addr,
                {
                    "op": "reserve_batch",
                    "items": [
                        {
                            "transfer_id": record.spec.transfer_id,
                            "mm_hash": record.spec.mm_hash,
                            "nbytes": record.spec.nbytes,
                            "shape": list(record.spec.shape),
                            "dtype": record.spec.dtype,
                        }
                        for record in records
                    ],
                },
            )
            outcomes = response["items"]
            if len(outcomes) != len(records):
                raise RuntimeError("Malformed EC reservation batch response")
        except Exception as exc:
            outcomes = [{"ok": False, "error": str(exc)} for _ in records]
        for record, outcome in zip(records, outcomes):
            spec = record.spec
            error = None
            try:
                if not outcome["ok"]:
                    raise RuntimeError(outcome["error"])
                reservation = outcome["result"]
                reservation["addr"] = addr
                reservation["_received_at"] = time.monotonic()
            except Exception as exc:
                error = exc
            if error is None:
                record.reservation_future.set_result([reservation])
            else:
                try:
                    self._retry_cancel_reservations(spec, [{"addr": addr, "reservation_id": ""}])
                except Exception:
                    logger.exception("Failed to cancel rejected EC reservation for transfer_id=%s", spec.transfer_id)
                record.reservation_future.set_exception(error)

    def start_load_caches(
        self,
        metadata: ECMooncakeConnectorMetadata,
        encoder_cache: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> None:
        self._transfer.ensure_ready()
        if not torch.npu.is_available():
            raise RuntimeError("ECMooncakeConnector requires an available NPU")
        self._reservations.retire_stale(encoder_cache)

        for spec in metadata.loads:
            if spec.mm_hash in encoder_cache:
                if spec.pushed:
                    # The spec's id is one shard's; cancel by transfer.
                    self._cancel_push(spec.transfer_id, "")
                self._completed_loads.add(spec.mm_hash)
                continue
            if spec.local:
                tensor = self._consumer_memory.take_resident(spec.mm_hash, tuple(spec.shape), spec.dtype)
            elif spec.pushed:
                try:
                    tensor, _ = self._take_pushed_tensor(spec)
                except RuntimeError as e:
                    logger.warning("EC Mooncake pushed load failed: %s", e)
                    tensor = None
            else:
                logger.warning(
                    "EC Mooncake load for mm_hash=%s has no transfer to take",
                    spec.mm_hash,
                )
                tensor = None
            if tensor is None:
                self._failed_loads.add(spec.mm_hash)
            else:
                encoder_cache[spec.mm_hash] = tensor
                self._completed_loads.add(spec.mm_hash)

    def _push_batch(self, pushes: list[ProducerPushRecord]) -> None:
        started_at = time.monotonic()
        ready: list[tuple[ProducerPushRecord, dict[str, Any]]] = []
        written_pushes: dict[str, ProducerPushRecord] = {}
        failure: Exception | None = None
        try:
            for push in pushes:
                self._validate_push_source(push)
                reservations = self._producer_pushes.resolve_reservations(push)
                stale = [
                    index
                    for index, shard in enumerate(reservations)
                    if not shard.get("ready", False)
                    and not shard.get("cancelled", False)
                    and time.monotonic() - float(shard.get("_received_at", started_at)) >= _RESERVATION_REFRESH_SECONDS
                ]
                if stale:
                    reservations = self._refresh_remote_reservations(push.spec, reservations)
                    self._producer_pushes.replace_reservations(push, reservations)
                self._producer_pushes.begin_writing(push)
                writable = [
                    shard
                    for shard in reservations
                    if not shard.get("cached", False) and not shard.get("cancelled", False) and shard.get("write", True)
                ]
                source = push.source
                assert source is not None
                if writable and source.ready_event is not None:
                    source.ready_event.synchronize()
                for shard in writable:
                    if int(shard["nbytes"]) != source.tensor.nbytes:
                        raise RuntimeError(f"Reserved EC size does not match tensor for mm_hash={push.spec.mm_hash}")
                    ready.append((push, shard))
                    written_pushes.setdefault(push.spec.transfer_id, push)
            if ready:
                # Stage each source once, then write it to every destination.
                source_index = {push.spec.transfer_id: index for index, push in enumerate(written_pushes.values())}
                tensors = [push.source.tensor for push in written_pushes.values() if push.source]
                lengths = [tensor.nbytes for tensor in tensors]
                # ponytail: one transaction per slab; use byte leases only if transfers saturate it.
                with self._staging_lock:
                    staged = None
                    try:
                        staged = self._producer_memory.stage(tensors)
                        if staged is None:
                            raise RuntimeError("Mooncake EC producer batch exceeds NPU staging pool capacity")
                        sources = staged.tensors
                        # Mooncake reads outside the NPU stream.
                        if sources:
                            torch.npu.current_stream(sources[0].device).synchronize()
                        addresses = [tensor.data_ptr() for tensor in sources]
                        by_session: dict[str, list[tuple[int, int]]] = {}
                        session_records: dict[str, dict[str, ProducerPushRecord]] = {}
                        for push, shard in ready:
                            session = str(shard["dst_session"])
                            by_session.setdefault(session, []).append(
                                (source_index[push.spec.transfer_id], int(shard["dst_ptr"]))
                            )
                            session_records.setdefault(session, {})[push.spec.transfer_id] = push

                        def write(session: str, items: list[tuple[int, int]]) -> None:
                            self._transfer.write(
                                session,
                                [addresses[index] for index, _ in items],
                                [dst for _, dst in items],
                                [lengths[index] for index, _ in items],
                            )

                        sessions = list(by_session.items())

                        # Write destinations concurrently to avoid serial transfer latency.
                        def track_write(index: int, future: Future[None]) -> None:
                            session = sessions[index][0]
                            self._producer_pushes.track_io_futures(list(session_records[session].values()), [future])

                        writes: list[Callable[[], None]] = [partial(write, *session) for session in sessions]
                        self._run_fanout(writes, track_write)
                    finally:
                        if staged is not None:
                            self._producer_memory.release(staged)

            self._producer_pushes.begin_notifying(pushes)
            self._notify_completions(ready)
            self._producer_pushes.complete(pushes)
        except Exception as exc:
            # Report asynchronously; raising here would fail EngineCore.
            failure = exc
            logger.exception(
                "EC Mooncake push batch failed for mm_hashes=%s",
                [push.spec.mm_hash for push in pushes],
            )
            self._producer_pushes.settle_all(pushes)
            self._abandon_pushes(pushes)
        finally:
            if failure is not None:
                self._producer_pushes.fail(pushes, failure)

    def _notify_completions(self, notifications: list[tuple[ProducerPushRecord, dict[str, Any]]]) -> None:
        """Tell the consumer, in one message per destination, what landed."""
        if not notifications:
            return
        by_destination: dict[str, list[tuple[ProducerPushRecord, dict[str, Any]]]] = {}
        for push, reservation in notifications:
            by_destination.setdefault(str(reservation.get("addr", push.spec.consumer_zmq)), []).append(
                (push, reservation)
            )
        destinations = list(by_destination.items())

        def notify(
            consumer_zmq: str,
            items: list[tuple[ProducerPushRecord, dict[str, Any]]],
        ) -> None:
            result = self._control_client.request(
                consumer_zmq,
                {
                    "op": "complete_batch",
                    "items": [
                        {
                            "transfer_id": push.spec.transfer_id,
                            "reservation_id": reservation["reservation_id"],
                        }
                        for push, reservation in items
                    ],
                },
            )
            completions = result.get("items", []) if isinstance(result, dict) else []
            if len(completions) != len(items):
                raise RuntimeError("Malformed EC completion response")
            for (push, _), completion in zip(items, completions):
                if not completion.get("completed"):
                    raise RuntimeError(f"Unknown EC reservation for mm_hash={push.spec.mm_hash}")

        def track(index: int, future: Future[None]) -> None:
            records = {push.spec.transfer_id: push for push, _ in destinations[index][1]}
            self._producer_pushes.track_io_futures(list(records.values()), [future])

        self._run_fanout(
            [partial(notify, destination, items) for destination, items in destinations],
            track,
        )

    @staticmethod
    def _known_reservations(record: ProducerPushRecord) -> list[dict[str, Any]]:
        if record.reservations:
            return list(record.reservations)
        try:
            return list(record.reservation_future.result())
        except Exception:
            return []

    def _abandon_pushes(self, pushes: list[ProducerPushRecord]) -> None:
        """Release the consumer-side reservations of a batch that failed."""
        for push in pushes:
            shards = self._known_reservations(push)
            if not shards:
                shards = [{"addr": push.spec.consumer_zmq, "reservation_id": ""}]
            try:
                self._retry_cancel_reservations(push.spec, shards)
            except Exception:
                logger.exception(
                    "Failed to abandon EC reservations for transfer_id=%s",
                    push.spec.transfer_id,
                )

    def _flush_pending_pushes(self, *, wait: bool = False) -> bool:
        return self._producer_pushes.submit_batches(
            self._io_executor,
            self._push_batch,
            max_batch_bytes=self._producer_memory.capacity,
            wait=wait,
        )

    def _bind_push_source(self, tensor: torch.Tensor, mm_hash: str) -> None:
        if tensor.device.type != "npu":
            raise ValueError(f"EC source must be on NPU for mm_hash={mm_hash}")
        ready_event = torch.npu.Event()
        ready_event.record(torch.npu.current_stream(tensor.device))
        self._producer_pushes.bind_source(mm_hash, tensor, ready_event)

    def _cancel_orphaned_reservation(self, record: ProducerPushRecord) -> None:
        try:
            reservations = self._producer_pushes.resolve_reservations(record)
        except Exception:
            reservations = self._known_reservations(record)
        known = bool(reservations)
        reservations = [
            shard for shard in reservations if not shard.get("cached", False) and not shard.get("cancelled", False)
        ]
        if not known:
            reservations = [{"addr": record.spec.consumer_zmq, "reservation_id": ""}]
        error = None
        try:
            self._retry_cancel_reservations(record.spec, reservations)
        except Exception as exc:
            error = exc
        self._producer_pushes.finish_cancel(record)
        if error is not None:
            raise error

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        if not self.is_producer:
            return None, None

        for record in self._producer_pushes.cancel_requests(finished_req_ids):
            self._producer_pushes.submit_cancel(
                record,
                self._io_executor,
                self._cancel_orphaned_reservation,
            )
        return None, None

    def save_caches(self, encoder_cache: dict[str, torch.Tensor], mm_hash: str, **kwargs: Any) -> None:
        if not self.is_producer:
            return
        tensor = encoder_cache[mm_hash]
        self._bind_push_source(tensor, mm_hash)

    def build_connector_worker_meta(self) -> ECMooncakeWorkerMetadata | None:
        self._flush_pending_pushes()
        failures = self._producer_pushes.poll()
        for mm_hash, error in failures:
            logger.error(
                "EC Mooncake async save failed for mm_hash=%s: %s",
                mm_hash,
                error,
            )
        reclaimed = self._consumer_memory.drain_reclaimed()
        meta = ECMooncakeWorkerMetadata(
            loaded=self._completed_loads,
            failed_loads=self._failed_loads,
            reclaimed=reclaimed,
            pending_loads=False,
            pending_saves=self._producer_pushes.pending,
        )
        self._completed_loads = set()
        self._failed_loads = set()
        return meta

    def close(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self._dispatch_stop.set()
        self._push_ready.set()
        if self._dispatcher is not None:
            self._dispatcher.join()
        # Settle reservation futures before the final source drain.
        self._control_executor.shutdown(wait=True)
        for record in self._producer_pushes.cancel_requests(None):
            self._producer_pushes.submit_cancel(record, self._io_executor, self._cancel_orphaned_reservation)
        self._flush_pending_pushes(wait=True)
        self._io_executor.shutdown(wait=True)
        if self._fanout_pool is not None:
            self._fanout_pool.shutdown(wait=True, cancel_futures=True)
        # Every thread that could hold a control socket is stopped by now.
        self._control_client.close()
        if self._control_server is not None:
            self._control_server.close()
        self._consumer_memory.close()
        self._producer_memory.close()
        self._transfer.close()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()
