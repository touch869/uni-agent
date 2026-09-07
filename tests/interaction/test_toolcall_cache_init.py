import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from uni_agent.agent_loop import UniAgentLoop


class StopAfterInit(Exception):
    pass


def test_cache_selection():
    old_cache = UniAgentLoop._observation_cache
    old_semaphore = UniAgentLoop._semaphore
    UniAgentLoop._observation_cache = None
    captured = []

    def capture(**kwargs):
        captured.append(kwargs['observation_cache'])
        raise StopAfterInit

    def run(controlled):
        loop = object.__new__(UniAgentLoop)
        loop.config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(agent=SimpleNamespace(num_workers=1))))
        loop._init_config = lambda *a, **kw: dict(
            skip_reward_evaluation=controlled, model={}, tools=[], env={},
            log_dir='/tmp/cache-init-test', interaction={})
        for name in ('_init_chat_model', '_init_tools_manager', '_init_skills_manager', '_init_env'):
            setattr(loop, name, lambda *a, **kw: None)
        try:
            asyncio.run(loop.run({}, raw_prompt=[]))
        except StopAfterInit:
            pass
        else:
            raise AssertionError('Interaction initialization was not reached')
        assert captured[-1] is loop.observation_cache
        return loop.observation_cache

    try:
        with patch('uni_agent.agent_loop.AgentInteraction', side_effect=capture):
            first = run(False)
            assert first is not None
            assert run(False) is first
            isolated = run(True)
            assert isolated is not first
            assert run(True) is not isolated
            assert run(False) is first
    finally:
        UniAgentLoop._observation_cache = old_cache
        UniAgentLoop._semaphore = old_semaphore
