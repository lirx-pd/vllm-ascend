# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side orchestration of Mooncake control, memory, and data planes.

Consumer Workers expose rank-local reservations and publish received tensors.
Producer Workers reserve a destination, bind computed sources, run
batched Mooncake writes, and report asynchronous completion to the Scheduler.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field
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
_PUSH_STAGES = (
    "reserve",
    "device",
    "register",
    "transfer",
    "unregister",
    "complete",
)


@dataclass
class _PushPerfWindow:
    """Accumulate Producer batch metrics between periodic log messages.

    Attributes:
        started_at: Monotonic start time of the aggregation window.
        batches: Number of completed push batches.
        items: Number of push records included in those batches.
        bytes: Number of tensor bytes written over the data plane.
        skipped_items: Items satisfied by cache or cancellation without a write.
        failures: Number of batches that ended in failure.
        stage_totals_ms: Accumulated time for every push stage.
        stage_max_ms: Maximum observed time for every push stage.
    """

    started_at: float = field(default_factory=time.monotonic)
    batches: int = 0
    items: int = 0
    bytes: int = 0
    skipped_items: int = 0
    failures: int = 0
    stage_totals_ms: dict[str, float] = field(default_factory=dict)
    stage_max_ms: dict[str, float] = field(default_factory=dict)


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
    ``control_max_workers`` bound the two executor pools; the transfer and
    consumer metrics intervals control aggregate logging.

    This first MRV1 delivery supports single-rank TP, PP, and DP only. The
    fixed configuration check keeps distributed topology out of the data-plane
    lifecycle until it has dedicated Ascend coverage.

    Attributes:
        is_producer: Whether this Worker originates encoder-cache pushes.
        is_consumer: Whether this Worker accepts encoder-cache pushes.
        _reservation_zmq_port: Consumer control port.
        _transfer: Owner of the Mooncake engine and memory registrations.
        _consumer_worker_metrics: Consumer lifecycle metric counters.
        _consumer_memory: Registered receive slab and resident cache.
        _reservations: Consumer destination reservation state manager.
        _control_server: Rank-local Consumer reservation server.
        _consumer_metrics_log_interval: Consumer metrics log interval.
        _consumer_metrics_started_at: Start time of the Consumer metric window.
        _producer_memory: Registered Producer source staging slab.
        _transfer_metrics_log_interval: Producer performance log interval.
        _control_client: Client for remote Consumer control operations.
        _producer_metrics: Producer lifecycle metric counters.
        _io_executor: Executor that owns transfer and cancellation batches.
        _control_executor: Executor that creates remote reservations.
        _fanout_pool: Lazily created executor for concurrent destination work.
        _fanout_pool_lock: Lock protecting fan-out pool initialization.
        _producer_pushes: Producer lifecycle and source-ownership manager.
        _push_perf_lock: Lock protecting Producer performance counters.
        _push_perf: Current Producer performance aggregation window.
        _active_transfer_batches: Batches currently executing data-plane work.
        _queued_transfer_batches: Batches submitted but not yet executing.
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
        self._consumer_worker_metrics: Counter[str] = Counter()
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
        self._consumer_metrics_log_interval = config.consumer_metrics_log_interval
        self._consumer_metrics_started_at = time.monotonic()
        # Worker producer
        self._producer_memory = ProducerMemoryPool(
            config.producer_pool_size,
            self._transfer,
        )
        self._transfer_metrics_log_interval = config.transfer_metrics_log_interval
        self._timing_enabled = config.timing_enabled
        self._control_client = control_client
        self._producer_metrics: Counter[str] = Counter()
        self._io_executor = ThreadPoolExecutor(
            max_workers=config.transfer_workers,
            thread_name_prefix="ec-mooncake-transfer",
            initializer=torch.npu.set_device,
            initargs=(self._device,),
        )
        self._control_executor = ThreadPoolExecutor(
            max_workers=config.control_workers,
            thread_name_prefix="ec-mooncake-control",
            initializer=torch.npu.set_device,
            initargs=(self._device,),
        )
        self._fanout_pool: ThreadPoolExecutor | None = None
        self._fanout_pool_lock = threading.Lock()
        self._producer_pushes = ProducerPushManager()
        self._push_perf_lock = threading.Lock()
        self._push_perf = _PushPerfWindow()
        self._active_transfer_batches = 0
        self._queued_transfer_batches = 0
        self._completed_loads: set[str] = set()
        self._failed_loads: set[str] = set()
        self._shutdown = False

    def _log_timing(self, stage: str, duration_s: float, **fields: Any) -> None:
        if not self._timing_enabled:
            return
        logger.info(
            "NPU_EPD_TIMING %s",
            json.dumps(
                {
                    "component": "connector",
                    "stage": stage,
                    "duration_s": duration_s,
                    "rank": self._device.index,
                    **{key: value for key, value in fields.items() if value not in (None, "")},
                },
                separators=(",", ":"),
            ),
        )

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
            self._consumer_metrics_log_interval,
            device=consumer_pool.device,
        )
        try:
            self._control_server.start()
        except Exception:
            self._control_server.close()
            self._control_server = None
            raise

    def _record_consumer_metric(self, name: str, value: int = 1) -> None:
        if not value:
            return
        with self._consumer_memory.lock:
            self._consumer_worker_metrics[name] += value

    def _maybe_log_consumer_worker_metrics(self, *, force: bool = False) -> None:
        with self._consumer_memory.lock:
            now = time.monotonic()
            if self._consumer_metrics_log_interval <= 0 or (
                not force and now - self._consumer_metrics_started_at < self._consumer_metrics_log_interval
            ):
                return
            reservations = self._reservations.active_records()
            ready = [record.mm_hash for record in reservations if record.state is ConsumerReservationState.READY]
            pending = [record.mm_hash for record in reservations if record.state is not ConsumerReservationState.READY]
            metrics = dict(self._consumer_worker_metrics)
            self._consumer_worker_metrics.clear()
            metrics.update(self._consumer_memory.take_metrics())
            residents, live, retired, pending_frees = self._consumer_memory.stats()
            oldest_reservation_ms = max(
                ((now - reservation.created_at) * 1000 for reservation in reservations),
                default=0.0,
            )
            self._consumer_metrics_started_at = now
        logger.info(
            "EC Mooncake consumer worker: lifecycle=%s, reservations_ready=%d, "
            "reservations_pending=%d, residents=%d, live=%d, retired=%d, "
            "pending_frees=%d, "
            "oldest_reservation_ms=%.1f, ready_hashes=%s, pending_hashes=%s",
            metrics,
            len(ready),
            len(pending),
            residents,
            live,
            retired,
            pending_frees,
            oldest_reservation_ms,
            [value[:16] for value in ready[:5]],
            [value[:16] for value in pending[:5]],
        )

    def _expire_push_reservations(self) -> int:
        return self._record_expiry_metrics(self._reservations.expire())

    def _record_expiry_metrics(self, counts: tuple[int, int, int]) -> int:
        expired, deferred, tombstones_dropped = counts
        self._record_consumer_metric("reservations_expired", expired)
        self._record_consumer_metric("cancellations_deferred", deferred)
        self._record_consumer_metric("cancel_records_dropped", tombstones_dropped)
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

        self._expire_push_reservations()
        reservation, should_write, reused, expiry_counts = self._reservations.reserve(
            transfer_id, mm_hash, nbytes, shape, dtype_name, dtype
        )
        self._record_expiry_metrics(expiry_counts)
        if reservation is None:
            raise RuntimeError("EC consumer buffer pool is full")
        if reservation.state in {
            ConsumerReservationState.CANCEL_PENDING,
            ConsumerReservationState.CANCELLED,
        }:
            self._record_consumer_metric("reservations_cancelled_early")
            return {
                "reservation_id": "",
                "dst_session": "",
                "dst_ptr": 0,
                "nbytes": nbytes,
                "write": False,
                "ready": False,
                "cancelled": True,
            }
        if reused:
            key = (
                "reservations_reused_ready"
                if reservation.state is ConsumerReservationState.READY
                else "reservations_reused_pending"
            )
            self._record_consumer_metric(key)
        elif reservation.lease is not None:
            self._record_consumer_metric("reservations_cached")
            # Cached reservations skip completion and may need no later model step.
            self._maybe_log_consumer_worker_metrics(force=True)
        else:
            self._record_consumer_metric("reservations_created")
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
        if not result.accepted:
            self._record_consumer_metric("completions_rejected")
        elif result.repeated:
            self._record_consumer_metric("completions_repeated")
        else:
            self._record_consumer_metric("completions_accepted")
        if result.discarded:
            self._record_consumer_metric("reservations_discarded")
        self._maybe_log_consumer_worker_metrics(force=True)
        return ControlCompletion(result.accepted, result.became_ready)

    def _cancel_push(
        self,
        transfer_id: str,
        reservation_id: str,
        abandon: bool = False,
        refresh: bool = False,
    ) -> bool:
        outcome, tombstones_dropped = self._reservations.cancel(transfer_id, reservation_id, abandon, refresh)
        metrics = {
            CancellationOutcome.REJECTED: "cancellations_rejected",
            CancellationOutcome.PRE_RESERVED: "cancellations_pre_reserved",
            CancellationOutcome.DEFERRED: "cancellations_deferred",
            CancellationOutcome.CANCELLED: "reservations_cancelled",
        }
        self._record_consumer_metric(metrics[outcome])
        self._record_consumer_metric("cancel_records_dropped", tombstones_dropped)
        return outcome is not CancellationOutcome.REJECTED

    def _take_pushed_tensor(self, spec: ECMooncakeLoadSpec) -> tuple[torch.Tensor, MemoryAllocation]:
        try:
            allocation = self._reservations.take(spec.transfer_id, spec.mm_hash)
        except RuntimeError:
            self._record_consumer_metric("takes_rejected")
            raise
        self._record_consumer_metric("reservations_taken")
        self._maybe_log_consumer_worker_metrics(force=True)
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
        started_at = time.monotonic() if self._timing_enabled else None
        try:
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
        except Exception:
            if started_at is not None:
                self._log_timing(
                    "producer_reservation_completed",
                    time.monotonic() - started_at,
                    request_id=spec.request_id,
                    transfer_id=spec.transfer_id,
                    mm_hash=spec.mm_hash,
                    boundary="control_request_round_trip",
                    status="error",
                )
            raise
        if not isinstance(result, dict):
            if started_at is not None:
                self._log_timing(
                    "producer_reservation_completed",
                    time.monotonic() - started_at,
                    request_id=spec.request_id,
                    transfer_id=spec.transfer_id,
                    mm_hash=spec.mm_hash,
                    boundary="control_request_round_trip",
                    status="error",
                )
            raise RuntimeError("Invalid EC reservation response")
        result["_received_at"] = time.monotonic()
        result["addr"] = addr
        if started_at is not None:
            self._log_timing(
                "producer_reservation_completed",
                time.monotonic() - started_at,
                request_id=spec.request_id,
                transfer_id=spec.transfer_id,
                mm_hash=spec.mm_hash,
                boundary="control_request_round_trip",
                status="ok",
            )
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
            result = self._control_client.request(
                str(reservation.get("addr", spec.consumer_zmq)),
                make_cancel_request(
                    spec.transfer_id,
                    str(reservation.get("reservation_id", "")),
                    abandon=True,
                    refresh=refresh,
                ),
            )
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
        return [self._reserve_one(spec.consumer_zmq, spec)]

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
        started_at = time.monotonic() if self._timing_enabled else None
        for spec in metadata.pushes:
            self._producer_pushes.reserve(
                spec,
                partial(self._submit_reservation, spec),
            )
        if not isinstance(encoder_cache, dict):
            return
        for mm_hash in dict.fromkeys(spec.mm_hash for spec in metadata.pushes):
            tensor = encoder_cache.get(mm_hash)
            if tensor is not None:
                self._bind_push_source(tensor, mm_hash)
        if started_at is not None and metadata.pushes:
            self._log_timing(
                "producer_save_enqueued",
                time.monotonic() - started_at,
                request_ids=[spec.request_id for spec in metadata.pushes if spec.request_id],
                transfer_ids=[spec.transfer_id for spec in metadata.pushes],
                mm_hashes=[spec.mm_hash for spec in metadata.pushes],
                boundary="cpu_enqueue",
                status="ok",
            )

    def _submit_reservation(self, spec: ECMooncakePushSpec) -> Future[list[dict[str, Any]]]:
        return self._control_executor.submit(self._reserve_remote, spec)

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
            started_at = time.monotonic() if self._timing_enabled else None
            if spec.mm_hash in encoder_cache:
                if spec.pushed:
                    # The spec's id is one shard's; cancel by transfer.
                    self._cancel_push(spec.transfer_id, "")
                self._completed_loads.add(spec.mm_hash)
                if started_at is not None:
                    self._log_timing(
                        "consumer_load_completed",
                        time.monotonic() - started_at,
                        transfer_id=spec.transfer_id,
                        mm_hash=spec.mm_hash,
                        boundary="worker_cache_handoff",
                        status="cache_hit",
                    )
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
            if started_at is not None:
                self._log_timing(
                    "consumer_load_completed",
                    time.monotonic() - started_at,
                    transfer_id=spec.transfer_id,
                    mm_hash=spec.mm_hash,
                    boundary="worker_cache_handoff",
                    status="ok" if tensor is not None else "error",
                )

    def _push_batch(self, pushes: list[ProducerPushRecord]) -> None:
        started_at = time.monotonic()
        with self._push_perf_lock:
            self._queued_transfer_batches -= 1
            self._active_transfer_batches += 1

        queue_waits_ms = []
        for push in pushes:
            assert push.source_at is not None
            queue_waits_ms.append(max(0, started_at - push.source_at) * 1000)
        stage_ms = {"queue": sum(queue_waits_ms), **dict.fromkeys(_PUSH_STAGES, 0.0)}
        reservation_wait_ms = 0.0
        if self._timing_enabled:
            unique_source_count = len({id(push.source.tensor) for push in pushes if push.source is not None})
        ready: list[tuple[ProducerPushRecord, dict[str, Any]]] = []
        written_pushes: dict[str, ProducerPushRecord] = {}
        failed = False
        failure: Exception | None = None
        try:
            for push in pushes:
                self._validate_push_source(push)
                stage_started_at = time.monotonic()
                try:
                    reservations = self._producer_pushes.resolve_reservations(push)
                finally:
                    if self._timing_enabled:
                        reservation_wait_ms += (time.monotonic() - stage_started_at) * 1000
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
                stage_ms["reserve"] += (time.monotonic() - stage_started_at) * 1000
                self._producer_pushes.begin_writing(push)
                writable = [
                    shard
                    for shard in reservations
                    if not shard.get("cached", False) and not shard.get("cancelled", False) and shard.get("write", True)
                ]
                source = push.source
                assert source is not None
                if writable and source.ready_event is not None:
                    stage_started_at = time.monotonic()
                    try:
                        source.ready_event.synchronize()
                    finally:
                        stage_ms["device"] += (time.monotonic() - stage_started_at) * 1000
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
                stage_started_at = time.monotonic()
                try:
                    staged = self._producer_memory.stage(tensors)
                    if staged is None:
                        raise RuntimeError("Mooncake EC producer NPU staging pool is full")
                    sources = staged.tensors
                    # Mooncake reads outside the NPU stream.
                    if sources:
                        torch.npu.current_stream(sources[0].device).synchronize()
                    addresses = [tensor.data_ptr() for tensor in sources]
                finally:
                    stage_ms["register"] = (time.monotonic() - stage_started_at) * 1000
                try:
                    by_session: dict[str, list[tuple[int, int]]] = {}
                    session_records: dict[str, dict[str, ProducerPushRecord]] = {}
                    for push, shard in ready:
                        session = str(shard["dst_session"])
                        by_session.setdefault(session, []).append(
                            (source_index[push.spec.transfer_id], int(shard["dst_ptr"]))
                        )
                        session_records.setdefault(session, {})[push.spec.transfer_id] = push
                    stage_started_at = time.monotonic()

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
                    try:
                        self._run_fanout(writes, track_write)
                    finally:
                        stage_ms["transfer"] = (time.monotonic() - stage_started_at) * 1000
                finally:
                    stage_started_at = time.monotonic()
                    self._producer_memory.release(staged)
                    stage_ms["unregister"] = (time.monotonic() - stage_started_at) * 1000

            self._producer_pushes.begin_notifying(pushes)
            stage_started_at = time.monotonic()
            try:
                self._notify_completions(ready)
            finally:
                stage_ms["complete"] = (time.monotonic() - stage_started_at) * 1000
            self._producer_pushes.complete(pushes)
        except Exception as exc:
            # Report asynchronously; raising here would fail EngineCore.
            failed = True
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
            ended_at = time.monotonic()
            stage_ms["total"] = (ended_at - started_at) * 1000
            if self._timing_enabled:
                common = {
                    "status": "ok" if not failed else "error",
                    "batch_items": len(pushes),
                    "batch_bytes": sum(push.spec.nbytes for push in pushes),
                    "write_bytes": sum(int(shard["nbytes"]) for _, shard in ready),
                    "unique_source_count": unique_source_count,
                }
                # Queue durations overlap across items; they are not a batch phase.
                for push, queue_ms in zip(pushes, queue_waits_ms):
                    self._log_timing(
                        "producer_queue_completed",
                        queue_ms / 1000,
                        boundary="source_bound_to_batch_start",
                        scope="item",
                        started_monotonic_s=push.source_at,
                        ended_monotonic_s=started_at,
                        request_id=push.spec.request_id,
                        transfer_id=push.spec.transfer_id,
                        mm_hash=push.spec.mm_hash,
                        **common,
                    )
                for stage, duration_ms, boundary in (
                    ("producer_reservation_wait_completed", reservation_wait_ms, "batch_reservation_future_wait"),
                    ("producer_device_ready_completed", stage_ms["device"], "batch_source_event_wait"),
                    ("producer_staging_copy_completed", stage_ms["register"], "batch_staging_copy_and_stream_sync"),
                    (
                        "producer_staging_completed",
                        stage_ms["device"] + stage_ms["register"],
                        "batch_device_ready_and_staging",
                    ),
                    ("producer_transfer_completed", stage_ms["transfer"], "batch_mooncake_write"),
                    ("producer_notification_completed", stage_ms["complete"], "batch_completion_notification"),
                    ("producer_push_completed", stage_ms["total"], "batch_start_to_completion_notification"),
                ):
                    self._log_timing(
                        stage,
                        duration_ms / 1000,
                        boundary=boundary,
                        scope="batch",
                        request_ids=[push.spec.request_id for push in pushes if push.spec.request_id],
                        transfer_ids=[push.spec.transfer_id for push in pushes],
                        mm_hashes=[push.spec.mm_hash for push in pushes],
                        **(
                            {"started_monotonic_s": started_at, "ended_monotonic_s": ended_at}
                            if stage == "producer_push_completed"
                            else {}
                        ),
                        aggregation="aggregate"
                        if stage in ("producer_staging_completed", "producer_push_completed")
                        else "phase",
                        overlaps=(
                            ["producer_device_ready_completed", "producer_staging_copy_completed"]
                            if stage == "producer_staging_completed"
                            else []
                        ),
                        **common,
                    )
            self._record_push_perf(
                stage_ms,
                stage_max_ms={"queue": max(queue_waits_ms, default=0.0)},
                item_count=len(pushes),
                byte_count=sum(push.spec.nbytes for push in written_pushes.values()),
                skipped_items=len(pushes) - len(written_pushes),
                failed=failed,
            )

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

    def _record_push_perf(
        self,
        stage_ms: dict[str, float],
        *,
        stage_max_ms: dict[str, float],
        item_count: int,
        byte_count: int,
        skipped_items: int,
        failed: bool,
    ) -> None:
        now = time.monotonic()
        report: tuple[_PushPerfWindow, int, int] | None = None
        with self._push_perf_lock:
            self._active_transfer_batches -= 1
            perf = self._push_perf
            perf.batches += 1
            perf.items += item_count
            perf.bytes += byte_count
            perf.skipped_items += skipped_items
            perf.failures += int(failed)
            for stage, elapsed_ms in stage_ms.items():
                perf.stage_totals_ms[stage] = perf.stage_totals_ms.get(stage, 0.0) + elapsed_ms
                perf.stage_max_ms[stage] = max(
                    perf.stage_max_ms.get(stage, 0.0),
                    stage_max_ms.get(stage, elapsed_ms),
                )
            if self._transfer_metrics_log_interval > 0 and now - perf.started_at >= self._transfer_metrics_log_interval:
                report = (
                    perf,
                    self._active_transfer_batches,
                    self._queued_transfer_batches,
                )
                self._push_perf = _PushPerfWindow(started_at=now)
        if report is None:
            return
        perf, active_batches, queued_batches = report
        batches = max(perf.batches, 1)
        items = max(perf.items, 1)
        stage_parts = []
        for stage in ("queue", *_PUSH_STAGES, "total"):
            divisor = items if stage == "queue" else batches
            average = perf.stage_totals_ms.get(stage, 0.0) / divisor
            maximum = perf.stage_max_ms.get(stage, 0.0)
            stage_parts.append(f"{stage}_ms={average:.1f}/{maximum:.1f}")
        stage_summary = " ".join(stage_parts)
        producer_metrics = dict(self._producer_metrics)
        self._producer_metrics.clear()
        logger.info(
            "EC Mooncake push perf: batches=%d items=%d bytes=%d "
            "batch_items=%.1f skipped=%d failures=%d active=%d queued=%d "
            "producer=%s queue_item_avg/max and stage_batch_avg/max: %s",
            perf.batches,
            perf.items,
            perf.bytes,
            perf.items / batches,
            perf.skipped_items,
            perf.failures,
            active_batches,
            queued_batches,
            producer_metrics,
            stage_summary,
        )

    def _flush_pending_pushes(self) -> None:
        self._producer_pushes.submit_batches(
            self._io_executor,
            self._push_batch,
            self._note_push_batch_queued,
        )

    def _note_push_batch_queued(self) -> None:
        with self._push_perf_lock:
            self._queued_transfer_batches += 1

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
        self._producer_metrics["saves_failed"] += len(failures)
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
        if self.is_consumer:
            self._maybe_log_consumer_worker_metrics()
        return meta

    def close(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self._flush_pending_pushes()
        self._io_executor.shutdown(wait=True, cancel_futures=True)
        if self._fanout_pool is not None:
            self._fanout_pool.shutdown(wait=True, cancel_futures=True)
        self._control_executor.shutdown(wait=True, cancel_futures=True)
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
