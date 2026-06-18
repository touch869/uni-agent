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
from uni_agent.llm_router.logging import get_router_logger


logger = get_router_logger("zmq-event-collector")


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
        kv_event_addresses: Per-replica ZMQ endpoints injected by the
                balancer at runtime. Key = replica_id, value = list of
                [sub_addr, replay_addr]. Each pair gets its own ZMQ context
                and socket pair.
    """

    def __init__(self, config, kv_event_addresses: dict[str, list[str]] | None = None) -> None:
        super().__init__()
        self._topic = "kv-events"
        long_conn = config.long_connection
        self._base_retry_delay = long_conn["base_retry_delay"]
        self._max_retry_delay = long_conn["max_retry_delay"]
        self._max_retry_attempts = long_conn["max_retry_attempts"]
        self._retry_backoff_factor = long_conn["retry_backoff_factor"]
        # Per-replica ZMQ endpoints injected by balancer at runtime.
        # Key = replica_id, value = [sub_addr, replay_addr].
        addresses = kv_event_addresses or {}
        self._sub_endpoints: dict[str, str] = {}
        self._replay_endpoints: dict[str, str] = {}
        for replica_id, addrs in addresses.items():
            self._sub_endpoints[replica_id] = f"tcp://{addrs[0]}"
            self._replay_endpoints[replica_id] = f"tcp://{addrs[1]}"
        self._stopped = False
        # Per-replica socket bundles
        self._replica_sockets: dict[str, _ReplicaSocketSet] = {}
        # Per-replica retry counters
        self._retry_counts: dict[str, int] = {}
        # Per-replica background tasks
        self._sub_tasks: dict[str, asyncio.Task] = {}

        logger.info(
            f"ZMQEventCollector initialized: sub_endpoints={self._sub_endpoints}, "
            f"replay_endpoints={self._replay_endpoints}, "
            f"retry: base_delay={self._base_retry_delay}s, max_delay={self._max_retry_delay}s, "
            f"max_attempts={self._max_retry_attempts}, backoff_factor={self._retry_backoff_factor}",
        )

    def stop(self) -> None:
        """Stop ZMQ subscription and close all sockets.

        Cancels all subscription tasks *inside* the background loop so
        that ``CancelledError`` is properly caught and coroutines finish
        cleanly — this avoids ``Task was destroyed but it is pending!``
        warnings.
        """
        logger.info("stopping ZMQ event collector")
        self._stopped = True
        # inside the loop thread, then await their cleanup.
        if self._task is not None and self._loop is not None:
            async def _cancel_and_wait():
                self._task.cancel()
                for t in self._sub_tasks.values():
                    t.cancel()
                # Wait for the main task to finish (it will cancel
                # sub-tasks in its CancelledError handler too, but we
                # already did it above for safety).
                try:
                    await self._task
                except (asyncio.CancelledError, Exception):
                    pass
                # Wait for all sub-tasks to finish their CancelledError
                # cleanup (close sockets per-replica in their finally).
                for t in list(self._sub_tasks.values()):
                    try:
                        await t
                    except (asyncio.CancelledError, Exception):
                        pass
            fut = asyncio.run_coroutine_threadsafe(_cancel_and_wait(), self._loop)
            try:
                fut.result(timeout=3)
            except Exception:
                pass
            self._task = None
            logger.info("ZMQ subscribe task cancelled and cleaned up")
        self._sub_tasks.clear()
        # Now stop the loop and join the thread (EventCollector.stop
        # does this, but we already cancelled the task ourselves above,
        # so we skip the super's cancel and only do loop/thread cleanup).
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        if self._loop is not None:
            self._loop.close()
            self._loop = None
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
            logger.debug(f"connected ZMQ sockets for {node_id}: sub={sub_addr}, replay={replay_addr}")
            return True

        except zmq.ZMQError as e:
            logger.warning(f"ZMQ connect failed for {node_id}: {e}")
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
        logger.debug(f"closed ZMQ sockets for {node_id}")

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
            logger.warning(f"ZMQ reconnect attempt {retry_count + 1}/{self._max_retry_attempts} for {node_id}, delay={delay:.1f}s")
            await asyncio.sleep(delay)
            retry_count += 1
            self._retry_counts[node_id] = retry_count

            if await self._connect_zmq_for(node_id, sub_addr, replay_addr):
                logger.info(f"ZMQ reconnect succeeded for {node_id} after {retry_count} attempts")
                return True

        logger.error(f"ZMQ reconnect exhausted all {self._max_retry_attempts} attempts for {node_id}")
        return False

    # ── Background subscription loop ────────────────────────────────────

    async def _subscribe_loop(self) -> None:
        """Spawn one subscription coroutine per replica endpoint."""
        logger.info(f"starting ZMQ subscribe loop for {len(self._sub_endpoints)} replicas: {list(self._sub_endpoints.keys())}")
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
            logger.info("ZMQ subscribe loop cancelled")
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
                logger.warning(f"initial ZMQ connect failed for {node_id}, starting reconnect")
                if not await self._reconnect_with_backoff_for(node_id, sub_addr, replay_addr):
                    logger.error(f"failed to connect ZMQ for {node_id} after all retries, giving up")
                    return

            logger.debug(f"ZMQ connected for {node_id}, requesting replay")
            await self._replay_historical_data_for(node_id)
            logger.debug(f"replay completed for {node_id}, entering subscribe loop")

            sockets = self._replica_sockets.get(node_id)
            if sockets is None:
                return

            while not self._stopped:
                try:
                    parts = await sockets.sub_socket.recv_multipart()
                    payload = parts[-1]
                    self._consume_payload(payload, node_id)
                except zmq.ZMQError as e:
                    logger.warning(f"ZMQ recv error for {node_id}: {e}, reconnecting")
                    self._close_zmq_sockets_for(node_id)
                    if not await self._reconnect_with_backoff_for(node_id, sub_addr, replay_addr):
                        break
                    await self._replay_historical_data_for(node_id)

        except asyncio.CancelledError:
            logger.info(f"subscription task cancelled for {node_id}")
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
                logger.warning(f"replay timeout for {node_id}, degrading to subscription-only")
                return  # timeout → degrade to subscription-only

            self._process_replay_data(replay_data, node_id)

        except zmq.ZMQError as e:
            logger.warning(f"replay ZMQ error for {node_id}: {e}")

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
