from __future__ import annotations

import pytest

from uni_agent.specrl import (
    SpecRLDraftCache,
    SpecRLLLMServerClient,
    SpecRLSettings,
    accepted_prefix_length,
)
from verl.workers.rollout.replica import TokenOutput


class _SequencedClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        output = self.outputs.pop(0)
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
    base = _SequencedClient(
        [
            _output(draft, [-1.0, -1.0, -1.0], 1),
            _score_output(len(prompt), [-1.0, -1.0, -1.0], 2),
        ]
    )
    client = SpecRLLLMServerClient(
        base,
        SpecRLDraftCache(max_entries=10, max_tokens=100),
        SpecRLSettings(enabled=True),
    )

    first = await client.generate(
        "r1", prompt_ids=prompt, sampling_params=_params(), specrl_enabled=True
    )
    second = await client.generate(
        "r2", prompt_ids=prompt, sampling_params=_params(), specrl_enabled=True
    )

    assert first.token_ids == draft
    assert first.extra_fields["spec_cache_misses"] == 1
    assert second.token_ids == draft
    assert second.log_probs == [-1.0, -1.0, -1.0]
    assert second.extra_fields["spec_num_accepted_tokens"] == 3
    assert second.extra_fields["spec_saved_tokens"] == 3
    assert len(base.calls) == 2


@pytest.mark.asyncio
async def test_client_rejects_draft_and_generates_from_original_context():
    prompt = [1, 2]
    draft = [10, 11, 12]
    replacement = [20, 21, 22]
    base = _SequencedClient(
        [
            _output(draft, [-1.0, -1.0, -1.0], 1),
            _score_output(len(prompt), [-100.0, -100.0, -100.0], 2),
            _output(replacement, [-2.0, -2.0, -2.0], 2),
        ]
    )
    client = SpecRLLLMServerClient(
        base,
        SpecRLDraftCache(max_entries=10, max_tokens=100),
        SpecRLSettings(enabled=True),
    )

    await client.generate("r1", prompt_ids=prompt, sampling_params=_params(), specrl_enabled=True)
    output = await client.generate("r2", prompt_ids=prompt, sampling_params=_params(), specrl_enabled=True)

    assert output.token_ids == replacement
    assert output.extra_fields["spec_num_accepted_tokens"] == 0
    assert base.calls[-1]["prompt_ids"] == prompt
    assert base.calls[-1]["sampling_params"]["max_tokens"] == 3


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
    assert output.extra_fields["spec_fallback_reason"] == "unsupported_sampling"
    assert len(base.calls) == 1
