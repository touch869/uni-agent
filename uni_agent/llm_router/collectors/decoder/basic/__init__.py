"""Statistic decoders fed by the Balancer callback transport.

- ``StickyDecoder``   — on_acquire/on_servers_removed → ``StickyUpdate``.
- ``InflightDecoder`` — on_acquire/on_release → inflight ``MetricsUpdate`` delta.

The pack contract ``StatisticEvent`` is defined on the transport side
(``collectors.transport.callback``) next to its producer.
"""

from uni_agent.llm_router.collectors.decoder.basic.inflight import InflightDecoder
from uni_agent.llm_router.collectors.decoder.basic.sticky import StickyDecoder

__all__ = ["InflightDecoder", "StickyDecoder"]
