"""HTTPTransport — Prometheus HTTP polling transport.

Polls ``http://{address}/metrics`` for each replica at a fixed interval
and delivers response text to the handler callback.
"""

from __future__ import annotations

import asyncio
import logging

from typing import Callable

import httpx

from uni_agent.llm_router.collectors.transport.base import Transport

logger = logging.getLogger(__name__)


class HTTPTransport(Transport):
    """HTTP polling transport — fetches Prometheus metrics from replicas.

    Each replica endpoint is polled at ``interval`` via ``httpx.AsyncClient``.
    Response text is delivered to the handler callback for decoding.

    Args:
        endpoints: ``{replica_id: ip:port}`` — each address polls
            ``http://{address}/metrics``.
        interval: Polling interval in seconds.
        http_timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        endpoints: dict[str, str],
        interval: float = 5.0,
        http_timeout: float = 10.0,
    ) -> None:
        self._endpoints: dict[str, str] = {
            nid: f"http://{addr}/metrics" for nid, addr in endpoints.items()
        }
        self._interval = interval
        self._http_timeout = http_timeout
        self._client: httpx.AsyncClient | None = None

    async def subscribe(self, handler: Callable[[bytes | str, str], None]) -> None:
        """Start the HTTP polling loop — delivers response text to handler."""
        self._client = httpx.AsyncClient(timeout=self._http_timeout)
        try:
            while True:
                coros = {nid: self._client.get(url) for nid, url in self._endpoints.items()}
                responses = await asyncio.gather(*coros.values(), return_exceptions=True)
                for nid, resp in zip(coros.keys(), responses):
                    if isinstance(resp, Exception):
                        continue  # failed node — handler falls back to defaults
                    try:
                        handler(resp.text, nid)
                    except Exception as exc:
                        logger.debug("Handler error for node %s: %s", nid, exc)
                await asyncio.sleep(self._interval)
        except asyncio.CancelledError:
            pass
        finally:
            if self._client is not None:
                await self._client.aclose()
                self._client = None

    def stop(self) -> None:
        """Stop HTTP polling — lifecycle managed by Collector, nothing to do here."""
