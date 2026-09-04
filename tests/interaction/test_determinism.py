import argparse
import asyncio
from pathlib import Path
from types import SimpleNamespace

from examples.agent_interaction.parallel_infer import apply_controlled_tool_prompt, init_config
from uni_agent.agent_loop import UniAgentLoop
from uni_agent.interaction.env import normalize_volatile_observation
from uni_agent.interaction.model import AgentChatModel, token_ids_hash
from uni_agent.interaction.tool_parser import HermesToolParser, XMLToolParser
from uni_agent.interaction.tool_schemas import OpenAIFunctionToolSchema
from uni_agent.interaction.tools_manager import ToolsManager

TOOLS = [
    OpenAIFunctionToolSchema(
        type="function",
        function={
            "name": "execute_bash",
            "description": "Run a command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    )
]

def test_controlled_prompt_removes_reward_without_mutating_source():
    source = [{
        "prompt": [{"role": "system", "content": "system"}, {"role": "user", "content": "task"}],
        "extra_info": {
            "tools_kwargs": {
                "reward": {"name": "swe_bench", "metadata": {"instance_id": "sample-1"}},
                "env": {"deployment": {"image": "image:tag"}},
            }
        },
    }]

    controlled = apply_controlled_tool_prompt(source, delay_s=0.5, repeats=4)

    assert source[0]["extra_info"]["tools_kwargs"]["reward"]["name"] == "swe_bench"
    assert controlled[0]["controlled_instance_id"] == "sample-1"
    assert controlled[0]["extra_info"]["tools_kwargs"]["reward"] is None
    assert controlled[0]["extra_info"]["tools_kwargs"]["skip_reward_evaluation"] is True
    assert "exactly 4 times" in controlled[0]["prompt"][-1]["content"]

def test_volatile_python_repr_addresses_are_normalized_narrowly():
    observation = (
        "fn=<function _cstack at 0x7ed79ad1a430> "
        "obj=<Thing object at 0xABCDEF12> value=0xdeadbeef"
    )
    assert normalize_volatile_observation(observation) == (
        "fn=<function _cstack at 0xADDR> obj=<Thing object at 0xADDR> value=0xdeadbeef"
    )


def manager(parser) -> ToolsManager:
    instance = object.__new__(ToolsManager)
    instance.tools_schemas = [tool.model_dump() for tool in TOOLS]
    instance._tool_parser = parser
    return instance


def parse_text(instance, output, step_idx):
    return asyncio.run(instance.parse_action(output, step_idx=step_idx))[1]


def test_tool_call_ids_are_stable_step_aware_and_ordinal_aware():
    instance = manager(HermesToolParser())
    single = '<tool_call>{"name":"execute_bash","arguments":{"command":"pwd"}}</tool_call>'
    first = parse_text(instance, single, step_idx=7)
    repeated = parse_text(instance, single, step_idx=7)
    next_step = parse_text(instance, single, step_idx=8)
    multiple = parse_text(instance, single + single, step_idx=7)

    assert first[0].id == repeated[0].id
    assert first[0].id != next_step[0].id
    assert multiple[0].id != multiple[1].id


def test_text_xml_and_structured_calls_share_the_same_id_policy():
    hermes = manager(HermesToolParser())
    xml = manager(XMLToolParser())
    hermes_output = '<tool_call>{"name":"execute_bash","arguments":{"command":"pwd"}}</tool_call>'
    xml_output = (
        "<tool_call><function=execute_bash>"
        "<parameter=command>pwd</parameter></function></tool_call>"
    )

    hermes_call = parse_text(hermes, hermes_output, step_idx=3)[0]
    xml_call = parse_text(xml, xml_output, step_idx=3)[0]

    async def parse_structured():
        return await hermes.parse_structured_action(
            "",
            [
                {
                    "id": "provider-random-id",
                    "type": "function",
                    "function": {"name": "execute_bash", "arguments": {"command": "pwd"}},
                }
            ],
            step_idx=3,
        )

    structured_call = asyncio.run(parse_structured())[1][0]
    assert hermes_call.id == xml_call.id == structured_call.id
    assert structured_call.id.startswith("call_")


def test_rejected_structured_calls_also_replace_provider_ids_deterministically():
    instance = manager(HermesToolParser())
    malformed = [
        {
            "id": "provider-random-id-a",
            "type": "function",
            "function": {"name": "execute_bash", "arguments": "{bad json"},
        }
    ]
    other_provider_id = [{**malformed[0], "id": "provider-random-id-b"}]

    first = instance.normalize_structured_tool_call_ids(malformed, step_idx=4)
    repeated = instance.normalize_structured_tool_call_ids(other_provider_id, step_idx=4)
    assert first[0]["id"] == repeated[0]["id"]
    assert first[0]["id"].startswith("call_")
    assert first[0]["id"] != "provider-random-id-a"


class FakeClient:
    async def generate(self, **kwargs):
        return SimpleNamespace(
            token_ids=[31, 32],
            num_preempted=None,
            log_probs=None,
            routed_experts=None,
            extra_fields={},
        )


class FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return [7, 8, len(messages)]

    def decode(self, token_ids):
        return "decoded:" + ",".join(str(token_id) for token_id in token_ids)


def test_normal_and_candidate_generation_expose_stable_token_hashes():
    async def execute():
        model = object.__new__(AgentChatModel)
        model.client = FakeClient()
        model.tokenizer = FakeTokenizer()
        model.max_model_len = 100
        model.sampling_params = {"temperature": 0}
        model.loop = asyncio.get_running_loop()
        cache = {
            "request_id": "request",
            "prompt_ids": [11, 12],
            "response_mask": [],
            "response_logprobs": [],
            "metrics": {},
            "extra_fields": {},
            "routed_experts": None,
        }
        _, _, _, normal_info = await model.query([], cache)

        candidate_cache = {
            "request_id": "candidate",
            "prompt_ids": [11, 12],
            "response_mask": [],
            "response_logprobs": [],
            "metrics": {},
            "extra_fields": {},
            "routed_experts": None,
        }
        candidate = await model.query_candidate(candidate_cache, [21])
        return normal_info, candidate

    normal_info, candidate = asyncio.run(execute())
    assert normal_info["prompt_hash"] == token_ids_hash([11, 12])
    assert normal_info["response_hash"] == token_ids_hash([31, 32])
    assert normal_info["hash_basis"] == "token_ids_sha256"
    assert candidate.context_ids == [11, 12, 21]
    assert candidate.generation_info["prompt_hash"] == token_ids_hash([11, 12, 21])
    assert candidate.generation_info["response_hash"] == token_ids_hash([31, 32])
    assert token_ids_hash([1, 2]) != token_ids_hash([1, 3])


def test_initial_request_id_is_stable_and_sample_aware():
    async def prepare(request_key):
        model = object.__new__(AgentChatModel)
        model.tokenizer = FakeTokenizer()
        model.tools_schemas = []
        model.request_key = request_key
        model.loop = asyncio.get_running_loop()
        return await model.prepare_rollout_cache([{"role": "user", "content": "same"}])

    first = asyncio.run(prepare("0"))
    repeated = asyncio.run(prepare("0"))
    other_sample = asyncio.run(prepare("1"))
    assert first["request_id"] == repeated["request_id"]
    assert first["request_id"] != other_sample["request_id"]


def test_init_config_enables_full_determinism_and_seed():
    args = argparse.Namespace(
        agent_config_path="examples/agent_interaction/agent_config.yaml",
        num_workers=1,
        max_turns=8,
        temperature=0.0,
        top_p=1.0,
        full_determinism=True,
        seed=123,
        nnodes=1,
        n_gpus_per_node=4,
        model_path="/tmp/model",
        engine="vllm",
        prompt_length=4096,
        response_length=4096,
        n=1,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.85,
    )
    config = init_config(args)
    assert config.actor_rollout_ref.rollout.full_determinism is True
    assert config.actor_rollout_ref.rollout.seed == 123
def test_controlled_skip_flag_overrides_dataset_reward_config():
    args = argparse.Namespace(
        agent_config_path="examples/agent_interaction/agent_config.yaml",
        num_workers=1,
        max_turns=8,
        temperature=0.0,
        top_p=1.0,
        full_determinism=True,
        seed=123,
        nnodes=1,
        n_gpus_per_node=4,
        model_path="/tmp/model",
        engine="vllm",
        prompt_length=4096,
        response_length=4096,
        n=1,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.85,
    )
    loop = object.__new__(UniAgentLoop)
    loop.config = init_config(args)
    loop.config.actor_rollout_ref.rollout.agent.agent_loop_config_path = str(
        Path(__file__).resolve().parents[2]
        / "examples/agent_interaction/agent_config_openyuanrong.yaml"
    )
    loop.server_manager = object()
    loop.tokenizer = object()

    effective = loop._init_config(
        {},
        tools_kwargs={
            "reward": {"name": "swe_bench"},
            "skip_reward_evaluation": True,
        },
    )

    assert effective["reward"] is None
    assert effective["skip_reward_evaluation"] is True
