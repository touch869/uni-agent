from __future__ import annotations

import pytest

from tests.uni_agent.support import FakeTokenizer
from verl.workers.rollout.replica import TokenOutput


class _SpecAwareBackend:
    def __init__(self):
        self.calls = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        return TokenOutput(
            token_ids=[79, 75],
            log_probs=[-0.1, -0.2],
            stop_reason="completed",
            extra_fields={
                "global_steps": 3,
                "spec_num_draft_tokens": 4,
                "spec_num_accepted_tokens": 2,
                "spec_num_verify_steps": 1,
                "spec_cache_hits": 1,
                "spec_cache_misses": 0,
                "spec_fallbacks": 0,
                "spec_saved_tokens": 2,
                "spec_version_newer_verify": 1,
                "spec_verify_ms": 2.25,
                "spec_continuation_ms": 3.5,
            },
        )


@pytest.mark.asyncio
async def test_gateway_enables_specrl_only_for_train_and_materializes_stats():
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.gateway import _GatewayActor

    backend = _SpecAwareBackend()
    actor = _GatewayActor(
        GatewayActorConfig(
            tokenizer=FakeTokenizer(),
            base_sampling_params={"temperature": 1.0, "top_p": 1.0, "top_k": -1, "max_tokens": 512},
        ),
        backend,
    )
    await actor.start()
    try:
        await actor.create_session("train", metadata={"partition_id": "train"})
        await actor._handle_chat_completions("train", {"messages": [{"role": "user", "content": "hi"}]})
        train_trajectory = (await actor.finalize_session("train"))[0]

        await actor.create_session("val", metadata={"partition_id": "val"})
        await actor._handle_chat_completions("val", {"messages": [{"role": "user", "content": "hi"}]})
        await actor.finalize_session("val")

        assert backend.calls[0]["specrl_enabled"] is True
        assert backend.calls[0]["sampling_params"] == {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_tokens": 512,
        }
        assert "specrl_enabled" not in backend.calls[1]
        assert train_trajectory.extra_fields["spec_num_draft_tokens"] == 4
        assert train_trajectory.extra_fields["spec_num_accepted_tokens"] == 2
        assert train_trajectory.extra_fields["spec_version_newer_verify"] == 1
        assert train_trajectory.extra_fields["spec_verify_ms"] == 2.25
        assert isinstance(train_trajectory.extra_fields["spec_verify_ms"], float)
        assert train_trajectory.extra_fields["min_global_steps"] == 3
        assert train_trajectory.extra_fields["max_global_steps"] == 3
    finally:
        await actor.shutdown()
