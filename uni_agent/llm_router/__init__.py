"""KV-cache-aware LLM Router configuration and routing primitives."""

from uni_agent.llm_router.config import (
    CacheStoreConfig,
    CollectorConfig,
    ConfigError,
    KVCAwareConfig,
    KVCAwareStrategyConfig,
    StrategyConfig,
)
from uni_agent.llm_router.strategy import (
    KVCacheAwareStrategy,
    RoutingStrategy,
    StrategyError,
    StrategyRegistry,
    route,
)
from uni_agent.llm_router.types import MetricKey, ReplicaInfo, ReplicaMetrics, RouteContext, RouteDataProvider

__all__ = [
    "CacheStoreConfig",
    "CollectorConfig",
    "ConfigError",
    "KVCAwareConfig",
    "KVCAwareStrategyConfig",
    "StrategyConfig",
    "KVCacheAwareStrategy",
    "RoutingStrategy",
    "StrategyError",
    "StrategyRegistry",
    "route",
    "MetricKey",
    "ReplicaInfo",
    "ReplicaMetrics",
    "RouteContext",
    "RouteDataProvider",
]
