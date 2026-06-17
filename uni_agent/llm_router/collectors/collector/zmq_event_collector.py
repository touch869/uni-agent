"""ZMQEventCollector — ZMQ-based event collector with replay + sub dual sockets.

ZMQ is a generic transport protocol, not tied to any specific backend.
Subclasses implement ``_consume_payload(payload, node_id)`` to decode
and process ZMQ messages.
"""

from __future__ import annotations

import asyncio
from abc import abstractmethod

from dataclasses import dataclass

import zmq
import zmq.asyncio

from uni_agent.llm_router.config.router import CollectorConfig
from uni_agent.llm_router.collectors.collector.event_collector import EventCollector


@dataclass
class _ReplicaSocketSet:
    """Per-replica ZMQ socket bundle (internal — not exported)."""

    node_id: str
    context: zmq.asyncio.Context
    sub_socket: zmq.asyncio.Socket
    replay_socket: zmq.asyncio.Socket


class ZMQEventCollector(EventCollector):
    """ZMQ event collector — connects replay + sub dual socket per replica,
    subscribes to live events concurrently.

    ZMQ is a generic transport protocol, not tied to any specific backend.
    Subclasses implement ``_consume_payload(payload, node_id)`` to decode
    and process ZMQ messages.

    Each replica endpoint gets its own ZMQ context, socket pair, and
    background coroutine — all replicas subscribe concurrently.

    Args:
        config: ``CollectorConfig`` — long_connection retry params
                (base_retry_delay / max_retry_attempts / retry_backoff_factor).
    """

    def __init__(self, config) -> None:
        super().__init__()
        self._topic = "kv-events"
        long_conn = config.long_connection
        self._base_retry_delay = long_conn["base_retry_delay"]
        self._max_retry_delay = long_conn["max_retry_delay"]
        self._max_retry_attempts = long_conn["max_retry_attempts"]
        self._retry_backoff_factor = long_conn["retry_backoff_factor"]
        # TODO(kv_event_address): the per-replica ZMQ endpoints
        # {replica_id: [sub_addr, replay_addr]} are allocated dynamically when
        # the vLLM servers start and passed down at runtime — they must NOT
        # live in the static CollectorConfig. Hardcoded placeholder for bring-up;
        # real injection is a collectors-module design item.
        kv_event_address: dict[str, list[str]] = {
            "127.0.0.1:8000": ["127.0.0.1:5555", "127.0.0.1:5556"],
        }
        self._sub_endpoints: dict[str, str] = {}
        self._replay_endpoints: dict[str, str] = {}
        for replica_id, addresses in kv_event_address.items():
            self._sub_endpoints[replica_id] = f"tcp://{addresses[0]}"
            self._replay_endpoints[replica_id] = f"tcp://{addresses[1]}"
        self._stopped = False
        # Per-replica socket bundles
        self._replica_sockets: dict[str, _ReplicaSocketSet] = {}
        # Per-replica retry counters
        self._retry_counts: dict[str, int] = {}
        # Per-replica background tasks
        self._sub_tasks: dict[str, asyncio.Task] = {}

    def stop(self) -> None:
        """Stop ZMQ subscription and close all sockets."""
        self._stopped = True
        super().stop()  # cancels self._task
        for task in self._sub_tasks.values():
            task.cancel()
        self._sub_tasks.clear()
        self._close_all_zmq_sockets()

    # ── ZMQ connection management ───────────────────────────────────────

    async def _connect_zmq_for(
        self, node_id: str, sub_addr: str, replay_addr: str,
    ) -> bool:
        """Create ZMQ context and replay + sub dual socket for a single replica."""
        try:
            ctx = zmq.asyncio.Context()

            sub_socket = ctx.socket(zmq.SUB)
            sub_socket.connect(sub_addr)
            sub_socket.setsockopt_string(zmq.SUBSCRIBE, self._topic)

            replay_socket = ctx.socket(zmq.REQ)
            replay_socket.connect(replay_addr)

            self._replica_sockets[node_id] = _ReplicaSocketSet(
                node_id=node_id,
                context=ctx,
                sub_socket=sub_socket,
                replay_socket=replay_socket,
            )
            self._retry_counts[node_id] = 0
            return True

        except zmq.ZMQError:
            self._close_zmq_sockets_for(node_id)
            return False

    def _close_zmq_sockets_for(self, node_id: str) -> None:
        """Safely close ZMQ sockets and context for a single replica."""
        sockets = self._replica_sockets.pop(node_id, None)
        if sockets is None:
            return
        sockets.sub_socket.close()
        sockets.replay_socket.close()
        sockets.context.term()

    def _close_all_zmq_sockets(self) -> None:
        """Close all per-replica ZMQ sockets and contexts."""
        for node_id in list(self._replica_sockets.keys()):
            self._close_zmq_sockets_for(node_id)

    async def _reconnect_with_backoff_for(
        self, node_id: str, sub_addr: str, replay_addr: str,
    ) -> bool:
        """Exponential backoff reconnect for a single replica."""
        retry_count = self._retry_counts.get(node_id, 0)
        while retry_count < self._max_retry_attempts:
            delay = min(
                self._base_retry_delay * (self._retry_backoff_factor ** retry_count),
                self._max_retry_delay,
            )
            await asyncio.sleep(delay)
            retry_count += 1
            self._retry_counts[node_id] = retry_count

            if await self._connect_zmq_for(node_id, sub_addr, replay_addr):
                return True

        return False

    # ── Background subscription loop ────────────────────────────────────

    async def _subscribe_loop(self) -> None:
        """Spawn one subscription coroutine per replica endpoint."""
        try:
            sub_tasks = []
            for node_id in self._sub_endpoints:
                sub_addr = self._sub_endpoints[node_id]
                replay_addr = self._replay_endpoints[node_id]
                t = asyncio.create_task(
                    self._subscribe_for_replica(node_id, sub_addr, replay_addr)
                )
                self._sub_tasks[node_id] = t
                sub_tasks.append(t)
            await asyncio.gather(*sub_tasks, return_exceptions=True)
        except asyncio.CancelledError:
            for t in self._sub_tasks.values():
                t.cancel()
        finally:
            self._close_all_zmq_sockets()

    async def _subscribe_for_replica(
        self, node_id: str, sub_addr: str, replay_addr: str,
    ) -> None:
        """Per-replica subscription: connect → replay → subscribe loop."""
        try:
            if not await self._connect_zmq_for(node_id, sub_addr, replay_addr):
                if not await self._reconnect_with_backoff_for(node_id, sub_addr, replay_addr):
                    return

            await self._replay_historical_data_for(node_id)

            sockets = self._replica_sockets.get(node_id)
            if sockets is None:
                return

            while not self._stopped:
                try:
                    parts = await sockets.sub_socket.recv_multipart()
                    payload = parts[-1]
                    self._consume_payload(payload, node_id)
                except zmq.ZMQError:
                    self._close_zmq_sockets_for(node_id)
                    if not await self._reconnect_with_backoff_for(node_id, sub_addr, replay_addr):
                        break
                    await self._replay_historical_data_for(node_id)

        except asyncio.CancelledError:
            pass
        finally:
            self._close_zmq_sockets_for(node_id)

    # ── Replay ──────────────────────────────────────────────────────────

    async def _replay_historical_data_for(self, node_id: str) -> None:
        """Request replay of historical KVCache data for a single replica.
        Degrade to subscription-only on failure."""
        sockets = self._replica_sockets.get(node_id)
        if sockets is None or sockets.replay_socket is None:
            return

        try:
            await sockets.replay_socket.send(b"replay")

            try:
                replay_data = await asyncio.wait_for(
                    sockets.replay_socket.recv(), timeout=5.0,
                )
            except asyncio.TimeoutError:
                return  # timeout → degrade to subscription-only

            self._process_replay_data(replay_data, node_id)

        except zmq.ZMQError:
            pass

    def _process_replay_data(self, data: bytes, node_id: str) -> None:
        """Parse replay response data (newline-delimited events)."""
        if not data:
            return
        for line in data.splitlines():
            if line.strip():
                self._consume_payload(line, node_id)

    # ── Message consumption (abstract — subclass implements) ────────────

    @abstractmethod
    def _consume_payload(self, payload: bytes, node_id: str) -> None:
        """Decode a single ZMQ payload and apply its events.

        Subclasses implement backend-specific decoding.

        Args:
            payload: Raw ZMQ message bytes.
            node_id: The replica that sent this payload.
        """
        ...
