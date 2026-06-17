"""KV-cache-aware LLM Router configuration and routing primitives."""

from uni_agent.llm_router.config import (
    CacheStoreConfig,
    CollectorConfig,
    ConfigError,
    KVCAwareConfig,
    KVCAwareStrategyConfig,
    StrategyConfig,
)
from uni_agent.llm_router.strategies import (
    KVCacheAwareStrategy,
    RoutingStrategy,
    StrategyError,
    StrategyRegistry,
    route,
)
from uni_agent.llm_router.metric_spec import MetricKey
from uni_agent.llm_router.collectors import CollectorProvider
from uni_agent.llm_router.strategies import ReplicaInfo

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
    "CollectorProvider",
]
