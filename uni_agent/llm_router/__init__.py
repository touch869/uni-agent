"""KV-cache-aware LLM Router configuration and routing primitives."""

from uni_agent.llm_router.config import (
    CacheStoreConfig,
    ConfigError,
    KVCAwareConfig,
    MetricsBackendConfig,
    MetricsConfig,
    StrategyConfig,
)
from uni_agent.llm_router.metrics.mooncake_prometheus import MooncakePrometheusConfig
from uni_agent.llm_router.metrics.vllm_prometheus import VllmPrometheusConfig
from uni_agent.llm_router.metrics.vllm_zmq import VllmZmqConfig
from uni_agent.llm_router.strategies.kvc_aware import KVCAwareStrategyConfig

# Strategy layer — scoring interface, registry, composition, builtin strategy.
from uni_agent.llm_router.strategy import (
    KVCacheAwareStrategy,
    RoutingStrategy,
    StrategyError,
    StrategyRegistry,
    route,
)
from uni_agent.llm_router.types import MetricsProvider, ReplicaInfo, RouteContext

__all__ = [
    # Config
    "CacheStoreConfig",
    "ConfigError",
    "KVCAwareConfig",
    "KVCAwareStrategyConfig",
    "MetricsBackendConfig",
    "MetricsConfig",
    "MooncakePrometheusConfig",
    "StrategyConfig",
    "VllmPrometheusConfig",
    "VllmZmqConfig",
    # Strategy
    "KVCacheAwareStrategy",
    "RoutingStrategy",
    "StrategyError",
    "StrategyRegistry",
    "route",
    # Shared types
    "MetricsProvider",
    "ReplicaInfo",
    "RouteContext",
]
