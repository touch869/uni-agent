"""Business event names shared by publishers and subscribers."""

GENERATION_FINISHED = "GenerationFinished"
GENERATION_PREPARED = "GenerationPrepared"
SESSION_CLOSED = "SessionClosed"
SESSION_OPENED = "SessionOpened"
GATEWAY_SESSION_DIRECT_SUBSCRIPTION = "router-gateway-session-shadow"
GATEWAY_SESSION_STATE_SCOPE = "gateway_sessions"

__all__ = [
    "GENERATION_FINISHED",
    "GENERATION_PREPARED",
    "GATEWAY_SESSION_DIRECT_SUBSCRIPTION",
    "GATEWAY_SESSION_STATE_SCOPE",
    "SESSION_CLOSED",
    "SESSION_OPENED",
]
