# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mooncake data-plane engine and memory-registration ownership."""

from __future__ import annotations

import threading
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import (
    global_te,
)

logger = init_logger(__name__)


class MooncakeTransfer:
    """Share the process-local Ascend engine and own EC slab registrations.

    Attributes:
        _hostname: Address advertised in the Mooncake session identifier.
        _engine: Shared process-local Ascend Mooncake engine.
        _engine_lock: Lock serializing first engine initialization.
        _pending_unregister: Slabs retained after an unregister failure.
        _closed: Whether final data-plane cleanup has begun.
    """

    def __init__(self, hostname: str, protocol: str) -> None:
        if protocol != "ascend":
            raise ValueError("MooncakeTransfer requires the Ascend protocol.")
        self._hostname = hostname
        self._engine: Any | None = None
        self._engine_lock = threading.Lock()
        self._pending_unregister: dict[int, torch.Tensor] = {}
        self._closed = False

    def _ensure_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        with self._engine_lock:
            if self._engine is not None:
                return self._engine
            engine = global_te.get_transfer_engine(self._hostname, device_name=None)
            self._engine = engine
            logger.info(
                "ECMooncakeConnector TransferEngine ready at %s:%d",
                self._hostname,
                engine.get_rpc_port(),
            )
        return self._engine

    def ensure_ready(self) -> None:
        self._ensure_engine()

    def local_session(self) -> str:
        engine = self._ensure_engine()
        return f"{self._hostname}:{engine.get_rpc_port()}"

    def register_memory(self, tensor: torch.Tensor) -> int:
        return self._ensure_engine().register_memory(tensor.data_ptr(), tensor.nbytes)

    def unregister_memory(self, tensor: torch.Tensor) -> bool:
        engine = self._ensure_engine()
        address = tensor.data_ptr()
        ret = engine.unregister_memory(address)
        if ret != 0:
            logger.error(
                "Mooncake EC memory unregistration failed for address %d: %d",
                address,
                ret,
            )
            self._pending_unregister[address] = tensor
            return False
        self._pending_unregister.pop(address, None)
        return True

    def write(
        self,
        session: str,
        sources: list[int],
        destinations: list[int],
        lengths: list[int],
    ) -> None:
        """Write one synchronous batch, returning only at terminal status."""
        ret = self._ensure_engine().batch_transfer_sync_write(session, sources, destinations, lengths)
        if ret != 0:
            raise RuntimeError(f"Mooncake EC push to {session} failed with status {ret}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        engine = self._engine
        if engine is None:
            return
        for address in list(self._pending_unregister):
            ret = engine.unregister_memory(address)
            if ret != 0:
                logger.error(
                    "Mooncake EC memory unregistration retry failed for address %d: %d",
                    address,
                    ret,
                )
                continue
            self._pending_unregister.pop(address, None)
