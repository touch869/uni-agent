import asyncio

from uni_agent.interaction.interaction import AgentInteraction
from uni_agent.interaction.model import AgentChatModel, CandidateQueryOutput
from uni_agent.interaction.tool_schemas import OpenAIFunctionToolCall
from uni_agent.toolcall_spec import (
    ObservationCache,
    canonical_arguments,
    canonical_tool_call,
    exact_tool_call_signature,
    finalize_toolcall_metrics,
    initialize_toolcall_metrics,
    is_eligible_tool_call,
)


def test_canonical_arguments_ignores_order_but_preserves_types():
    assert canonical_arguments({"b": 2, "a": 1}) == '{"a":1,"b":2}'
    assert canonical_arguments({"value": 1}) != canonical_arguments({"value": "1"})


def test_tool_call_key_ignores_id():
    schema = "schema"
    first = canonical_tool_call(
        {"id": "call-a", "function": {"name": "execute_bash", "arguments": {"command": "pwd"}}},
        schema,
    )
    second = canonical_tool_call(
        {"id": "call-b", "function": {"name": "execute_bash", "arguments": {"command": "pwd"}}},
        schema,
    )
    assert first == second


def test_shell_tool_aliases_share_an_equivalence_key():
    bash = canonical_tool_call({"function": {"name": "bash", "arguments": {"command": "pwd"}}}, "schema")
    execute_bash = canonical_tool_call(
        {"function": {"name": "execute_bash", "arguments": {"command": "pwd"}}}, "schema"
    )
    assert bash == execute_bash


def test_equivalent_cat_spellings_share_a_read_file_key():
    plain = canonical_tool_call({"function": {"name": "execute_bash", "arguments": {"command": "cat a.py"}}}, "schema")
    quoted = canonical_tool_call({"function": {"name": "bash", "arguments": {"command": "cat -- 'a.py'"}}}, "schema")
    assert plain == quoted
    assert plain.tool_name == "read_file"
    assert plain.canonical_arguments == '{"path":"a.py"}'


def test_shell_equivalence_rejects_options_multiple_files_and_command_chains():
    calls = [
        "cat -n a.py",
        "cat a.py b.py",
        "cat a.py | head",
        "cat a.py; echo done",
        "cat -secret",
        "cat $FILE",
        "cat *.py",
        "cat $(pwd)/a.py",
        "cat ~/a.py",
    ]
    keys = [
        canonical_tool_call({"function": {"name": "execute_bash", "arguments": {"command": command}}}, "schema")
        for command in calls
    ]
    assert all(key.tool_name == "execute_bash" for key in keys)
    assert len(set(keys)) == len(calls)


def test_file_view_tools_share_an_equivalence_key_for_read_only_calls():
    editor_view = canonical_tool_call(
        {
            "function": {
                "name": "str_replace_editor",
                "arguments": {
                    "command": "view",
                    "path": "/testbed/a.py",
                    "view_range": None,
                    "old_str": None,
                },
            }
        },
        "schema",
    )
    standalone_view = canonical_tool_call(
        {"function": {"name": "view", "arguments": {"path": "/testbed/a.py"}}}, "schema"
    )
    assert editor_view == standalone_view


def test_file_view_equivalence_preserves_semantic_parameters_and_rejects_edits():
    full_view = canonical_tool_call(
        {
            "function": {
                "name": "str_replace_editor",
                "arguments": {"command": "view", "path": "/testbed/a.py"},
            }
        },
        "schema",
    )
    ranged_view = canonical_tool_call(
        {
            "function": {
                "name": "view",
                "arguments": {"path": "/testbed/a.py", "view_range": [1, 10]},
            }
        },
        "schema",
    )
    edit = canonical_tool_call(
        {
            "function": {
                "name": "str_replace_editor",
                "arguments": {
                    "command": "str_replace",
                    "path": "/testbed/a.py",
                    "old_str": "old",
                    "new_str": "new",
                },
            }
        },
        "schema",
    )
    assert full_view != ranged_view
    assert full_view != edit


def test_observation_cache_only_stores_success_and_enforces_lru_limits():
    cache = ObservationCache(max_entries=2, max_tokens=4, token_counter=lambda value: len(value))
    key_a = canonical_tool_call({"function": {"name": "view", "arguments": {"path": "a"}}}, "s")
    key_b = canonical_tool_call({"function": {"name": "view", "arguments": {"path": "b"}}}, "s")
    key_c = canonical_tool_call({"function": {"name": "view", "arguments": {"path": "c"}}}, "s")
    assert not cache.put(key_a, "bad", status="timeout", source_trajectory_id="t")
    assert cache.put(key_a, "aa", status="ok", source_trajectory_id="t")
    assert cache.put(key_b, "bb", status="ok", source_trajectory_id="t")
    assert cache.get(key_a) is not None
    assert cache.put(key_c, "cc", status="ok", source_trajectory_id="t")
    assert cache.get(key_b) is None
    assert cache.get(key_a) is not None
    assert cache.token_count == 4


def test_eligibility_and_derived_metrics():
    cache = ObservationCache()
    key = canonical_tool_call({"function": {"name": "execute_bash", "arguments": {"command": "pwd"}}}, "s")
    cache.put(key, "ok", status="ok", source_trajectory_id="t")
    assert is_eligible_tool_call(
        enabled=True,
        tool_calls=[{"function": {"name": "execute_bash", "arguments": {"command": "pwd"}}}],
        cache=cache,
        cache_key=key,
    )
    editor_key = canonical_tool_call(
        {"function": {"name": "str_replace_editor", "arguments": {"command": "view", "path": "x"}}}, "s"
    )
    cache.put(editor_key, "view", status="ok", source_trajectory_id="t")
    assert is_eligible_tool_call(
        enabled=True,
        tool_calls=[{"function": {"name": "str_replace_editor", "arguments": {"command": "view", "path": "x"}}}],
        cache=cache,
        cache_key=editor_key,
        allowed_tools=frozenset({"str_replace_editor"}),
    )
    edit_call = {
        "function": {
            "name": "str_replace_editor",
            "arguments": {
                "command": "str_replace",
                "path": "x",
                "old_str": "old",
                "new_str": "new",
            },
        }
    }
    edit_key = canonical_tool_call(edit_call, "s")
    cache.put(edit_key, "edited", status="ok", source_trajectory_id="t")
    assert not is_eligible_tool_call(
        enabled=True,
        tool_calls=[edit_call],
        cache=cache,
        cache_key=edit_key,
    )
    metrics = {}
    initialize_toolcall_metrics(metrics)
    metrics["toolcall_lookup_attempts"] = 2
    metrics["canonical_call_matches"] = 1
    metrics["equivalent_tool_matches"] = 1
    metrics["candidate_started"] = 1
    metrics["candidate_committed"] = 1
    metrics["observation_matches"] = 1
    finalize_toolcall_metrics(metrics, cache)
    assert metrics["canonical_call_match_rate"] == 0.5
    assert metrics["equivalent_tool_match_rate"] == 1.0
    assert metrics["observation_match_rate"] == 1.0
    assert metrics["committed_candidate_rate"] == 1.0


class _FakeEnv:
    def __init__(self, observation="OK"):
        self.observation = observation
        self.calls = 0

    async def run_action(self, command, action_timeout=60):
        self.calls += 1
        return self.observation


class _FakeTools:
    tools_schemas = []

    def __init__(self, tool_name="execute_bash"):
        self.tool_name = tool_name

    async def parse_structured_action(self, content, tool_calls_data, step_idx=0):
        return content, [OpenAIFunctionToolCall(**item) for item in tool_calls_data]

    async def parse_action(self, model_output, step_idx=0):
        return model_output, []

    def get_tool_bash_command(self, tool_call):
        return tool_call.function.arguments.get("command", "done")


class _FakeInteractionModel:
    supports_toolcall_speculation = True
    sampling_params = {"temperature": 0}
    tool_schema_signature = "schema"
    current_policy_version = 1
    sampling_signature = "sampling"

    def __init__(
        self,
        *,
        tool_name="execute_bash",
        candidate_error=False,
        candidate_hangs=False,
        first_encode_hangs=False,
        tool_arguments=None,
    ):
        self.tool_name = tool_name
        self.tool_arguments = tool_arguments or {"command": "pwd"}
        self.candidate_error = candidate_error
        self.candidate_hangs = candidate_hangs
        self.first_encode_hangs = first_encode_hangs
        self.encode_calls = 0
        self.normal_queries = 0

    async def prepare_rollout_cache(self, messages):
        return {
            "request_id": "request",
            "prompt_ids": [1],
            "response_mask": [],
            "response_logprobs": [],
            "metrics": {},
            "extra_fields": {},
        }

    async def query(self, messages, rollout_cache, **kwargs):
        self.normal_queries += 1
        if self.normal_queries == 1:
            arguments = self.tool_arguments
            return (
                "thought",
                [
                    {
                        "id": "current-id",
                        "type": "function",
                        "function": {"name": self.tool_name, "arguments": arguments},
                    }
                ],
                rollout_cache,
                {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "prompt_hash": "prompt-normal",
                    "response_hash": "response-normal",
                    "hash_basis": "token_ids_sha256",
                },
            )
        return (
            "done",
            [],
            rollout_cache,
            {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "prompt_hash": "prompt-fallback",
                "response_hash": "response-fallback",
                "hash_basis": "token_ids_sha256",
            },
        )

    def clone_rollout_cache(self, rollout_cache, candidate_id=None):
        return {
            **rollout_cache,
            "request_id": candidate_id,
            "prompt_ids": list(rollout_cache["prompt_ids"]),
            "response_mask": list(rollout_cache["response_mask"]),
            "response_logprobs": list(rollout_cache["response_logprobs"]),
            "metrics": dict(rollout_cache["metrics"]),
            "extra_fields": dict(rollout_cache["extra_fields"]),
        }

    async def encode_tool_messages(self, messages):
        self.encode_calls += 1
        if self.first_encode_hangs and self.encode_calls == 1:
            await asyncio.Event().wait()
        return [99] if messages[0]["content"] == "OK" else [98]

    async def query_candidate(self, rollout_cache, predicted_tool_message_ids, sampling_params=None):
        if self.candidate_error:
            raise RuntimeError("candidate failed")
        if self.candidate_hangs:
            await asyncio.Event().wait()
        rollout_cache["prompt_ids"] += predicted_tool_message_ids
        return CandidateQueryOutput(
            response="done",
            tool_calls=[],
            rollout_cache=rollout_cache,
            generation_info={
                "prompt_tokens": 2,
                "completion_tokens": 3,
                "prompt_hash": "prompt-candidate",
                "response_hash": "response-candidate",
                "hash_basis": "token_ids_sha256",
            },
            context_ids=list(rollout_cache["prompt_ids"]),
        )

    def append_encoded_messages_to_rollout_cache(self, ids, rollout_cache):
        rollout_cache["prompt_ids"] += ids
        rollout_cache["response_mask"] += [0] * len(ids)
        return rollout_cache


def _run_interaction(
    *,
    actual_observation="OK",
    cached_observation="OK",
    tool_name="execute_bash",
    candidate_error=False,
    candidate_hangs=False,
    first_encode_hangs=False,
    cached_tool_name=None,
    tool_arguments=None,
    cached_tool_arguments=None,
):
    async def execute():
        model = _FakeInteractionModel(
            tool_name=tool_name,
            candidate_error=candidate_error,
            candidate_hangs=candidate_hangs,
            first_encode_hangs=first_encode_hangs,
            tool_arguments=tool_arguments,
        )
        env = _FakeEnv(actual_observation)
        cache = ObservationCache()
        cached_call = {
            "function": {
                "name": cached_tool_name or tool_name,
                "arguments": cached_tool_arguments or tool_arguments or {"command": "pwd"},
            }
        }
        key = canonical_tool_call(cached_call, model.tool_schema_signature)
        cache.put(
            key,
            cached_observation,
            status="ok",
            source_trajectory_id="seed",
            source_tool_name=cached_tool_name or tool_name,
            source_call_signature=exact_tool_call_signature(cached_call),
        )
        interaction = AgentInteraction(
            "run",
            env,
            model,
            _FakeTools(tool_name),
            [{"role": "user", "content": "go"}],
            max_turns=3,
            chat_mode=True,
            toolcall_speculation_enabled=True,
            observation_cache=cache,
            toolcall_encoding_timeout=0.01,
            toolcall_candidate_wait_timeout=0.01,
        )
        return env, model, await interaction.run()

    return asyncio.run(execute())


def test_equivalent_tool_cache_hit_commits_after_strict_observation_validation():
    env, model, result = _run_interaction(cached_tool_name="bash")
    metrics = result["rollout_cache"]["metrics"]
    assert env.calls == 1
    assert model.normal_queries == 1
    assert metrics["canonical_call_matches"] == 1
    assert metrics["equivalent_tool_matches"] == 1
    assert metrics["candidate_committed"] == 1


def test_equivalent_cat_spellings_hit_in_the_full_interaction():
    env, model, result = _run_interaction(
        tool_arguments={"command": "cat a.py"},
        cached_tool_arguments={"command": "cat -- 'a.py'"},
    )
    metrics = result["rollout_cache"]["metrics"]
    assert env.calls == 1
    assert model.normal_queries == 1
    assert metrics["canonical_call_matches"] == 1
    assert metrics["equivalent_tool_matches"] == 1
    assert metrics["candidate_committed"] == 1


def test_file_view_equivalence_hits_across_concrete_tools():
    env, model, result = _run_interaction(
        tool_name="str_replace_editor",
        tool_arguments={"command": "view", "path": "/testbed/a.py"},
        cached_tool_name="view",
        cached_tool_arguments={"path": "/testbed/a.py"},
    )
    metrics = result["rollout_cache"]["metrics"]
    assert env.calls == 1
    assert model.normal_queries == 1
    assert metrics["equivalent_tool_matches"] == 1
    assert metrics["candidate_committed"] == 1


def test_equivalent_tool_observation_mismatch_aborts_and_falls_back():
    env, model, result = _run_interaction(
        actual_observation="CHANGED",
        cached_tool_name="bash",
    )
    metrics = result["rollout_cache"]["metrics"]
    assert env.calls == 1
    assert model.normal_queries == 2
    assert metrics["equivalent_tool_matches"] == 1
    assert metrics["candidate_committed"] == 0
    assert metrics["candidate_abort_observation_mismatch"] == 1


def test_candidate_match_commits_and_real_tool_runs_once():
    env, model, result = _run_interaction()
    metrics = result["rollout_cache"]["metrics"]
    assert env.calls == 1
    assert model.normal_queries == 1
    assert metrics["candidate_started"] == 1
    assert metrics["candidate_committed"] == 1
    assert metrics["observation_matches"] == 1
    assert metrics["candidate_committed_after_wait"] == 1
    assert metrics["candidate_committed_without_wait"] == 0
    assert metrics["candidate_wait_committed_ms"] >= 0
    assert metrics["candidate_abort_wait_timeout"] == 0
    assert metrics["candidate_saved_overlap_ms"] >= 0
    assert result["trajectory"][0].generation_source == "normal"
    assert result["trajectory"][0].prompt_hash == "prompt-normal"
    assert result["trajectory"][0].response_hash == "response-normal"
    assert result["trajectory"][1].generation_source == "candidate"
    assert result["trajectory"][1].prompt_hash == "prompt-candidate"
    assert result["trajectory"][1].response_hash == "response-candidate"


def test_candidate_mismatch_waits_discards_and_falls_back():
    env, model, result = _run_interaction(actual_observation="CHANGED")
    metrics = result["rollout_cache"]["metrics"]
    assert env.calls == 1
    assert model.normal_queries == 2
    assert metrics["candidate_aborted"] == 1
    assert metrics["candidate_cancel_requested"] == 1
    assert metrics["candidate_cancelled"] == 1
    assert metrics["candidate_abort_observation_mismatch"] == 1
    assert metrics["candidate_abort_wait_timeout"] == 0
    assert metrics["candidate_wait_timeout_ms"] == 0
    assert metrics["wasted_candidate_tokens"] == 0
    assert metrics["candidate_cancel_overhead_ms"] >= 0
    assert metrics["context_validation_failures"] == 1


def test_candidate_error_falls_back_without_failing_rollout():
    env, model, result = _run_interaction(candidate_error=True)
    metrics = result["rollout_cache"]["metrics"]
    assert env.calls == 1
    assert model.normal_queries == 2
    assert metrics["candidate_aborted"] == 1
    assert result["trajectory"][-1].exit_reason == "turn_done"


def test_candidate_encoding_timeout_does_not_block_real_tool_or_rollout():
    env, model, result = _run_interaction(first_encode_hangs=True)
    metrics = result["rollout_cache"]["metrics"]
    assert env.calls == 1
    assert model.normal_queries == 2
    assert metrics["candidate_started"] == 0
    assert result["trajectory"][-1].exit_reason == "turn_done"


def test_candidate_wait_timeout_falls_back_to_normal_generation():
    env, model, result = _run_interaction(candidate_hangs=True)
    metrics = result["rollout_cache"]["metrics"]
    assert env.calls == 1
    assert model.normal_queries == 2
    assert metrics["candidate_started"] == 1
    assert metrics["candidate_aborted"] == 1
    assert metrics["candidate_abort_wait_timeout"] == 1
    assert metrics["candidate_wait_timeout_ms"] > 0
    assert metrics["candidate_wait_committed_ms"] == 0
    assert result["trajectory"][-1].exit_reason == "turn_done"


def test_terminal_tool_bypasses_candidate_generation():
    env, model, result = _run_interaction(tool_name="submit")
    metrics = result["rollout_cache"]["metrics"]
    assert env.calls == 1
    assert model.normal_queries == 1
    assert metrics["candidate_started"] == 0
    assert result["trajectory"][-1].exit_reason == "finished"


def test_rollout_cache_clone_has_no_shared_mutable_state():
    model = object.__new__(AgentChatModel)
    original = {
        "request_id": "request",
        "prompt_ids": [1],
        "response_mask": [1],
        "response_logprobs": [0.1],
        "metrics": {"nested": {"value": 1}},
        "extra_fields": {"nested": [1]},
        "routed_experts": object(),
    }
    clone = model.clone_rollout_cache(original, "candidate")
    clone["prompt_ids"].append(2)
    clone["metrics"]["nested"]["value"] = 2
    clone["extra_fields"]["nested"].append(2)
    assert original["prompt_ids"] == [1]
    assert original["metrics"]["nested"]["value"] == 1
    assert original["extra_fields"]["nested"] == [1]
    assert clone["routed_experts"] is original["routed_experts"]
