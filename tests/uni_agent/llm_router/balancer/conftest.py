"""conftest for balancer tests.

Applies the _FakeCollectorManager patch ONLY when balancer ut tests are being run.
When st-cpu/e2e tests run (different pytest invocation, different -m filter),
no balancer ut tests are selected, so the patch is a no-op.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True, scope="session")
def _conditional_patch(request):
    """Patch CollectorManager + _init_manager + _resolve_max_num_seqs — only if balancer ut tests run."""
    has_balancer_ut = any(
        "balancer" in str(item.fspath) and item.get_closest_marker("ut") for item in request.session.items
    )
    if not has_balancer_ut:
        yield
        return

    import uni_agent.llm_router.collectors as _collectors_mod
    from tests.uni_agent.llm_router.balancer._helpers import (
        _fake_init_manager,
        _fake_resolve_max_num_seqs,
        _FakeCollectorManager,
    )
    from uni_agent.llm_router.balancer import KVCAwareBalancer

    _orig_provider = _collectors_mod.CollectorManager
    _orig_init = KVCAwareBalancer._init_manager
    _orig_resolve = KVCAwareBalancer._resolve_max_num_seqs

    _collectors_mod.CollectorManager = _FakeCollectorManager
    KVCAwareBalancer._init_manager = _fake_init_manager
    KVCAwareBalancer._resolve_max_num_seqs = staticmethod(_fake_resolve_max_num_seqs)

    yield

    _collectors_mod.CollectorManager = _orig_provider
    KVCAwareBalancer._init_manager = _orig_init
    KVCAwareBalancer._resolve_max_num_seqs = _orig_resolve


@pytest.fixture(autouse=True)
def _reset_store_singletons():
    """Reset the singleton-backed stores between balancer tests (function-scoped)."""
    from uni_agent.llm_router.store.kv_cache_store import KVCacheStore
    from uni_agent.llm_router.store.metrics_store import MetricsStore
    from uni_agent.llm_router.store.sticky_session_store import StickySessionStore

    for cls in (MetricsStore, KVCacheStore, StickySessionStore):
        cls._instance = None
    yield
    for cls in (MetricsStore, KVCacheStore, StickySessionStore):
        cls._instance = None
