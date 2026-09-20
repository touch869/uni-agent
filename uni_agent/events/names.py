"""Business event names shared by publishers and subscribers."""

GENERATION_FINISHED = "GenerationFinished"
GENERATION_PREPARED = "GenerationPrepared"
SESSION_CLOSED = "SessionClosed"
SESSION_OPENED = "SessionOpened"
GATEWAY_SESSION_DIRECT_SUBSCRIPTION = "router-gateway-session-shadow"
GATEWAY_SESSION_STATE_SCOPE = "gateway_sessions"
GATEWAY_GLOBAL_FORWARDER_SUBSCRIPTION = "gateway-global-forwarder"
GATEWAY_TELEMETRY_SUBSCRIPTION = "gateway-telemetry-shadow"

__all__ = [
    "GENERATION_FINISHED",
    "GENERATION_PREPARED",
    "GATEWAY_GLOBAL_FORWARDER_SUBSCRIPTION",
    "GATEWAY_SESSION_DIRECT_SUBSCRIPTION",
    "GATEWAY_SESSION_STATE_SCOPE",
    "GATEWAY_TELEMETRY_SUBSCRIPTION",
    "SESSION_CLOSED",
    "SESSION_OPENED",
]
