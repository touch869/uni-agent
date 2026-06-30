"""conftest for balancer tests — re-exports the session fixture from _helpers.

The autouse session-scoped fixture ``_patch_provider`` lives in ``_helpers.py``
and handles patching/unpatching ``RouteDataProvider`` + ``_init_provider``.
conftest.py is the standard place for autouse fixtures that affect an entire
directory — importing it here makes pytest discover it.
"""

from tests.uni_agent.llm_router.balancer._helpers import _patch_provider  # noqa: F401
