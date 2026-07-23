from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from uni_agent.specrl import (
    SpecRLDraftCache,
    SpecRLLLMServerClient,
    SpecRLSettings,
    accepted_prefix_length,
)
from verl.workers.rollout.llm_server import LLMServerClient
from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.vllm_rollout import vllm_async_server


class _SequencedClient:
    def __init__(self, outputs, *, versions=None, verify_outputs=None):
        self.outputs = list(outputs)
        self.versions = list(versions or [])
        self.verify_outputs = list(verify_outputs or [])
        self.calls = []
        self.version_calls = []
        self.verify_calls = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output

    async def get_policy_version(self, **kwargs):
        self.version_calls.append(kwargs)
        version = self.versions.pop(0)
        if isinstance(version, Exception):
            raise version
        return version

    async def verify_and_generate(self, **kwargs):
        self.verify_calls.append(kwargs)
        output = self.verify_outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


def _output(tokens, logprobs, version, **extra):
    return TokenOutput(
        token_ids=tokens,
        log_probs=logprobs,
        stop_reason="completed",
        extra_fields={"global_steps": version, "min_global_steps": version, "max_global_steps": version, **extra},
    )


def _score_output(prompt_length, draft_logprobs, version):
    rows = [[0.0] for _ in range(prompt_length)] + [[value] for value in draft_logprobs]
    return _output([999], None, version, prompt_logprobs=rows)


def _params(**updates):
    params = {"max_tokens": 3, "temperature": 1.0, "top_p": 1.0, "top_k": -1}
    params.update(updates)
    return params


@pytest.mark.parametrize(
    ("old_logprobs", "new_logprobs", "expected"),
    [
        ([-1.0, -2.0], [-1.0, -2.0], 2),
        ([-1.0, -1.0], [-100.0, -1.0], 0),
    ],
)
def test_accepted_prefix_length(old_logprobs, new_logprobs, expected):
    assert accepted_prefix_length(old_logprobs, new_logprobs, bias=0.5, seed=7) == expected


def test_draft_cache_respects_version_and_token_lru():
    cache = SpecRLDraftCache(max_entries=2, max_tokens=3)
    assert cache.put("a", [1, 2], [-1.0, -1.0], "completed", 2)
    assert not cache.put("a", [3], [-1.0], "completed", 1)
    assert cache.put("b", [4, 5], [-1.0, -1.0], "completed", 2)
    assert cache.get("a") is None
    assert cache.stats() == {"entries": 1, "tokens": 2}


@pytest.mark.asyncio
async def test_client_cache_miss_then_full_reuse_on_newer_policy():
    prompt = [1, 2]
    draft = [10, 11, 12]
    verified = _output(
        draft,
        [-1.0, -1.0, -1.0],
        2,
        spec_num_draft_tokens=3,
        spec_num_accepted_tokens=3,
        spec_num_verify_steps=1,
        spec_saved_tokens=3,
        spec_verify_prompt_tokens=2,
        spec_verify_draft_tokens=3,
        spec_verify_ms=2.5,
    )
    base = _SequencedClient([_output(draft, [-1.0, -1.0, -1.0], 1)], versions=[2], verify_outputs=[verified])
    client = SpecRLLLMServerClient(base, SpecRLDraftCache(max_entries=10, max_tokens=100), SpecRLSettings(enabled=True))

    first = await client.generate("r1", prompt_ids=prompt, sampling_params=_params(), specrl_enabled=True)
    second = await client.generate("r2", prompt_ids=prompt, sampling_params=_params(), specrl_enabled=True)

    assert first.extra_fields["spec_cache_misses"] == 1
    assert second.token_ids == draft
    assert second.extra_fields["spec_num_accepted_tokens"] == 3
    assert second.extra_fields["spec_version_newer_verify"] == 1
    assert second.extra_fields["spec_verify_ms"] == 2.5
    assert len(base.calls) == 1
    assert len(base.verify_calls) == 1


@pytest.mark.asyncio
async def test_client_rejects_draft_uses_one_server_operation():
    prompt = [1, 2]
    draft = [10, 11, 12]
    replacement = [20, 21, 22]
    verified = _output(
        replacement,
        [-2.0, -2.0, -2.0],
        2,
        spec_num_draft_tokens=3,
        spec_num_accepted_tokens=0,
        spec_num_verify_steps=1,
        spec_saved_tokens=0,
        spec_continuation_tokens=3,
    )
    base = _SequencedClient([_output(draft, [-1.0, -1.0, -1.0], 1)], versions=[2], verify_outputs=[verified])
    client = SpecRLLLMServerClient(base, SpecRLDraftCache(max_entries=10, max_tokens=100), SpecRLSettings(enabled=True))

    await client.generate("r1", prompt_ids=prompt, sampling_params=_params(), specrl_enabled=True)
    output = await client.generate("r2", prompt_ids=prompt, sampling_params=_params(), specrl_enabled=True)

    assert output.token_ids == replacement
    assert output.extra_fields["spec_num_accepted_tokens"] == 0
    assert len(base.calls) == 1
    assert len(base.verify_calls) == 1
    assert base.verify_calls[0]["prompt_ids"] == prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("current_version", "counter", "reason"),
    [
        (0, "spec_version_older_bypass", "older_policy"),
        (1, "spec_version_equal_bypass", "equal_policy"),
    ],
)
async def test_policy_version_bypass_never_calls_verify(current_version, counter, reason):
    draft = [10, 11, 12]
    base = _SequencedClient(
        [
            _output(draft, [-1.0, -1.0, -1.0], 1),
            _output([20, 21, 22], [-2.0, -2.0, -2.0], current_version),
        ],
        versions=[current_version],
    )
    client = SpecRLLLMServerClient(base, SpecRLDraftCache(max_entries=10, max_tokens=100), SpecRLSettings(enabled=True))

    await client.generate("r1", prompt_ids=[1, 2], sampling_params=_params(), specrl_enabled=True)
    output = await client.generate("r2", prompt_ids=[1, 2], sampling_params=_params(), specrl_enabled=True)

    assert output.extra_fields[counter] == 1
    assert output.extra_fields["spec_fallback_reason"] == reason
    assert output.extra_fields["spec_num_verify_steps"] == 0
    assert len(base.verify_calls) == 0
    assert len(base.calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reason", "counter"),
    [
        ("verification_error", "spec_fallback_verification_error"),
        ("missing_logprobs", "spec_fallback_missing_logprobs"),
        ("continuation_error", "spec_fallback_continuation_error"),
    ],
)
async def test_verify_error_falls_back_with_reason(reason, counter):
    draft = [10, 11, 12]
    failed = TokenOutput(
        token_ids=[],
        log_probs=None,
        stop_reason="error",
        extra_fields={"global_steps": 2, "_specrl_error_reason": reason, "spec_verify_ms": 1.25},
    )
    base = _SequencedClient(
        [
            _output(draft, [-1.0, -1.0, -1.0], 1),
            _output([20, 21, 22], [-2.0, -2.0, -2.0], 2),
        ],
        versions=[2],
        verify_outputs=[failed],
    )
    client = SpecRLLLMServerClient(base, SpecRLDraftCache(max_entries=10, max_tokens=100), SpecRLSettings(enabled=True))

    await client.generate("r1", prompt_ids=[1, 2], sampling_params=_params(), specrl_enabled=True)
    output = await client.generate("r2", prompt_ids=[1, 2], sampling_params=_params(), specrl_enabled=True)

    assert output.extra_fields[counter] == 1
    assert output.extra_fields["spec_fallback_reason"] == reason
    assert output.extra_fields["spec_verify_ms"] == 1.25
    assert len(base.verify_calls) == 1
    assert len(base.calls) == 2


@pytest.mark.asyncio
async def test_client_falls_back_for_unsupported_sampling():
    base = _SequencedClient([_output([10], [-1.0], 1)])
    client = SpecRLLLMServerClient(base, None, SpecRLSettings(enabled=True))

    output = await client.generate(
        "r1",
        prompt_ids=[1],
        sampling_params=_params(temperature=0.8),
        specrl_enabled=True,
    )

    assert output.extra_fields["spec_fallbacks"] == 1
    assert output.extra_fields["spec_fallback_reason"] == "unsupported"
    assert len(base.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("accepted", "max_tokens", "expected_calls", "expected_tokens"),
    [
        (0, 3, 2, [20, 21, 22]),
        (1, 3, 2, [10, 20, 21]),
        (3, 3, 1, [10, 11, 12]),
        (3, 2, 1, [10, 11]),
    ],
)
async def test_server_verify_and_generate_acceptance_and_cached_continuation(
    monkeypatch, accepted, max_tokens, expected_calls, expected_tokens
):
    calls = []

    async def generate(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return _output(
                [99],
                None,
                2,
                target_prompt_logprobs=[-1.0, -2.0, -3.0],
            )
        continuation_prompt = kwargs["prompt_ids"]
        count = kwargs["sampling_params"]["max_tokens"]
        return _output(
            [20, 21, 22][:count],
            [-4.0, -5.0, -6.0][:count],
            2,
            vllm_num_cached_tokens=len(continuation_prompt),
        )

    def accept(old_logprobs, new_logprobs, **kwargs):
        assert old_logprobs == [-0.5, -0.6, -0.7]
        assert new_logprobs == [-1.0, -2.0, -3.0]
        return accepted

    monkeypatch.setattr(vllm_async_server, "_accepted_prefix_length", accept)
    server = SimpleNamespace(global_steps=2, generate=generate)
    output = await vllm_async_server.vLLMHttpServer.verify_and_generate(
        server,
        request_id="request",
        prompt_ids=[1, 2],
        draft_ids=[10, 11, 12],
        old_logprobs=[-0.5, -0.6, -0.7],
        sampling_params=_params(max_tokens=max_tokens),
        bias=0.5,
        accept_seed=7,
        expected_policy_version=2,
        cached_stop_reason="completed",
    )

    assert len(calls) == expected_calls
    assert output.token_ids == expected_tokens
    assert output.extra_fields["spec_num_accepted_tokens"] == min(accepted, max_tokens)
    assert output.extra_fields["spec_num_verify_steps"] == 1
    if expected_calls == 2:
        assert output.extra_fields["spec_continuation_prefill_tokens"] == 0
        assert output.extra_fields["spec_continuation_cached_tokens"] == 2 + accepted
    else:
        assert output.extra_fields["spec_continuation_tokens"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["missing_logprobs", "continuation_error"])
async def test_server_verify_and_generate_reports_structured_failure(monkeypatch, failure):
    calls = []

    async def generate(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            extra = {} if failure == "missing_logprobs" else {"target_prompt_logprobs": [-1.0, -2.0, -3.0]}
            return _output([99], None, 2, **extra)
        raise RuntimeError("continuation failed")

    monkeypatch.setattr(vllm_async_server, "_accepted_prefix_length", lambda *args, **kwargs: 1)
    server = SimpleNamespace(global_steps=2, generate=generate)
    output = await vllm_async_server.vLLMHttpServer.verify_and_generate(
        server,
        request_id="request",
        prompt_ids=[1, 2],
        draft_ids=[10, 11, 12],
        old_logprobs=[-0.5, -0.6, -0.7],
        sampling_params=_params(),
        bias=0.5,
        accept_seed=7,
        expected_policy_version=2,
    )

    assert output.extra_fields["_specrl_error_reason"] == failure
    assert len(calls) == (1 if failure == "missing_logprobs" else 2)


@pytest.mark.asyncio
async def test_policy_version_commit_is_atomic_and_consistent_across_replicas():
    release_clear = asyncio.Event()
    clear_started = asyncio.Event()

    async def blocked_clear():
        clear_started.set()
        await release_clear.wait()

    replicas = [
        SimpleNamespace(
            global_steps=4,
            _policy_version_lock=asyncio.Lock(),
            clear_kv_cache=blocked_clear,
        )
        for _ in range(2)
    ]
    commits = [
        asyncio.create_task(vllm_async_server.vLLMHttpServer.commit_weight_update(replica, 5)) for replica in replicas
    ]
    await clear_started.wait()
    reads = [asyncio.create_task(vllm_async_server.vLLMHttpServer.get_policy_version(replica)) for replica in replicas]
    await asyncio.sleep(0)
    assert not any(read.done() for read in reads)

    release_clear.set()
    await asyncio.gather(*commits)
    assert await asyncio.gather(*reads) == [5, 5]


class _RemoteCall:
    def __init__(self, function):
        self.function = function

    def remote(self, **kwargs):
        async def invoke():
            return self.function(**kwargs)

        return invoke()


class _StickyServer:
    def __init__(self):
        self.get_policy_version = _RemoteCall(lambda: 7)
        self.verify_and_generate = _RemoteCall(
            lambda **kwargs: _output([10], [-1.0], 7, routed_request=kwargs["request_id"])
        )


class _StickyLoadBalancer:
    def __init__(self, server):
        self.keys = []
        self.releases = []
        self.acquire_server = _RemoteCall(self._acquire)
        self.release_server = SimpleNamespace(remote=lambda **kwargs: self.releases.append(kwargs["server_id"]))
        self.server = server

    def _acquire(self, request_id):
        self.keys.append(request_id)
        return "replica-0", self.server


@pytest.mark.asyncio
async def test_policy_version_and_verify_use_same_sticky_replica_key():
    server = _StickyServer()
    load_balancer = _StickyLoadBalancer(server)
    client = LLMServerClient(config=None, load_balancer_handle=load_balancer)

    assert await client.get_policy_version("session-1") == 7
    output = await client.verify_and_generate("session-1", prompt_ids=[1])

    assert output.extra_fields["global_steps"] == 7
    assert load_balancer.keys == ["session-1", "session-1"]
    assert load_balancer.releases == ["replica-0", "replica-0"]
    assert output.extra_fields["routed_request"] != "session-1"


def test_extracts_only_requested_target_prompt_logprobs():
    output = SimpleNamespace(
        prompt_logprobs=[
            None,
            {2: SimpleNamespace(logprob=-0.1)},
            {"10": SimpleNamespace(logprob=-1.0)},
            {11: SimpleNamespace(logprob=-2.0)},
            {12: SimpleNamespace(logprob=-3.0)},
        ]
    )

    assert vllm_async_server._extract_target_prompt_logprobs(output, [1, 2, 10, 11, 12], start=2, count=3) == [
        -1.0,
        -2.0,
        -3.0,
    ]
    assert vllm_async_server._extract_target_prompt_logprobs(output, [1, 2, 10, 11, 12], start=0, count=3) is None
