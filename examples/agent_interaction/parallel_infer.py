# ruff: noqa: E402

import argparse
import copy
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Prefer this checkout over any editable uni-agent installation and propagate
# the same import path to Ray workers.
REPO_ROOT = str(Path(__file__).resolve().parents[2])
python_path = os.environ.get("PYTHONPATH", "").split(os.pathsep)
if REPO_ROOT not in python_path:
    sys.path.insert(0, REPO_ROOT)
    os.environ["PYTHONPATH"] = os.pathsep.join([REPO_ROOT, *filter(None, python_path)])

import numpy as np
import ray
from datasets import load_dataset
from omegaconf import DictConfig

import verl
from verl import DataProto
from verl.experimental.agent_loop import AgentLoopManager
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.workers.rollout.llm_server import LLMServerManager

# Setup basic logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=os.getenv("VERL_LOGGING_LEVEL", "INFO")
)
logger = logging.getLogger(__name__)

CONTROLLED_TOOL_SENTINEL = "TOOLCALL_BENCH_OK"


def controlled_tool_command(delay_s: float) -> str:
    """Return the exact repeatable command used by the latency benchmark."""
    return f"sleep {delay_s:g} && printf '{CONTROLLED_TOOL_SENTINEL}\\n'"


def controlled_tool_prompt(delay_s: float, repeats: int) -> str:
    """Build a strict prompt that repeatedly exercises one deterministic tool call."""
    command = controlled_tool_command(delay_s)
    return f"""This is a controlled tool-latency benchmark, not a software task.
Call the execute_bash tool exactly {repeats} times, one call per assistant turn.
Every execute_bash call must use exactly this command, byte for byte:
{command}
Do not inspect or modify files. Do not use any other command or tool while these calls are running.
After observing {CONTROLLED_TOOL_SENTINEL} from the {repeats}th call, call submit exactly once.
Each assistant response must contain exactly one tool call and no explanatory text.
Do not stop before all {repeats} execute_bash calls have succeeded."""


def apply_controlled_tool_prompt(samples: list[dict], delay_s: float, repeats: int) -> list[dict]:
    """Replace task prompts and explicitly disable task reward evaluation."""
    prompt_text = controlled_tool_prompt(delay_s, repeats)
    overridden = copy.deepcopy(samples)
    for sample in overridden:
        system_messages = [
            message
            for message in sample.get("prompt", [])
            if isinstance(message, dict) and message.get("role") == "system"
        ]
        sample["prompt"] = [*system_messages, {"role": "user", "content": prompt_text}]
        tools_kwargs = sample.setdefault("extra_info", {}).setdefault("tools_kwargs", {})
        reward_metadata = (tools_kwargs.get("reward") or {}).get("metadata", {})
        sample["controlled_instance_id"] = reward_metadata.get("instance_id")
        tools_kwargs["reward"] = None
        tools_kwargs["skip_reward_evaluation"] = True
    return overridden


def _output_values(output: DataProto, key: str, count: int, default=None) -> list:
    """Return a JSON-friendly per-sample non-tensor output field."""
    values = output.non_tensor_batch.get(key)
    if values is None:
        return [default for _ in range(count)]
    result = []
    for value in values:
        if hasattr(value, "item"):
            try:
                value = value.item()
            except ValueError:
                pass
        result.append(value)
    return result


def init_config(args: argparse.Namespace) -> DictConfig:
    """Initialize the configuration from hydra and override with command-line arguments."""
    from hydra import compose, initialize_config_dir

    # config_dir = os.path.abspath("verl/trainer/config")
    config_dir = str(Path(verl.__file__).resolve().parent / "trainer" / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="ppo_trainer")

    # Override rollout configs
    config.actor_rollout_ref.rollout.agent.agent_loop_config_path = os.path.expanduser(args.agent_config_path)
    config.actor_rollout_ref.rollout.agent.num_workers = args.num_workers
    config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns = args.max_turns
    config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls = 1

    # Validation / sampling kwargs
    config.actor_rollout_ref.rollout.temperature = args.temperature
    config.actor_rollout_ref.rollout.top_p = args.top_p
    config.actor_rollout_ref.rollout.val_kwargs.temperature = args.temperature
    config.actor_rollout_ref.rollout.val_kwargs.top_p = args.top_p
    config.actor_rollout_ref.rollout.calculate_log_probs = True
    config.actor_rollout_ref.rollout.full_determinism = args.full_determinism
    config.actor_rollout_ref.rollout.seed = args.seed

    # Hardware configs
    config.actor_rollout_ref.rollout.nnodes = args.nnodes
    config.actor_rollout_ref.rollout.n_gpus_per_node = args.n_gpus_per_node
    config.trainer.nnodes = args.nnodes
    config.trainer.n_gpus_per_node = args.n_gpus_per_node

    # Model and engine configs
    config.actor_rollout_ref.model.path = os.path.expanduser(args.model_path)
    config.actor_rollout_ref.rollout.name = args.engine
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.prompt_length = args.prompt_length
    config.actor_rollout_ref.rollout.response_length = args.response_length
    config.actor_rollout_ref.rollout.max_model_len = getattr(
        args, "max_model_len", args.prompt_length + args.response_length
    )
    config.actor_rollout_ref.rollout.max_num_seqs = getattr(args, "max_num_seqs", 4)
    config.actor_rollout_ref.rollout.n = args.n
    config.actor_rollout_ref.rollout.tensor_model_parallel_size = args.tensor_parallel_size
    config.actor_rollout_ref.rollout.gpu_memory_utilization = args.gpu_memory_utilization

    # Data configs
    config.data.return_raw_chat = True
    config.data.max_prompt_length = args.prompt_length
    config.data.max_response_length = args.response_length

    return config



def apply_specrl_plugin():
    import vllm

    if vllm.__version__.startswith("0.17."):
        from uni_agent.histospec_vllm017_plugin import install

        install()
    else:
        from recipe.specRL.histoSpec.vllm_plugin.patch import specRL_plugin

        specRL_plugin()



def update_specrl_cache(output, responses_per_prompt: int, server_addresses: list[str]):
    response_length = output.batch["responses"].shape[-1]
    prompt_mask = output.batch["attention_mask"][:, :-response_length]
    response_mask = output.batch["attention_mask"][:, -response_length:]
    payload = {
        "prompts": output.batch["prompts"].cpu().tolist(),
        "responses": output.batch["responses"].cpu().tolist(),
        "prompt_lengths": prompt_mask.sum(-1).float().cpu().tolist(),
        "response_lengths": response_mask.sum(-1).float().cpu().tolist(),
        "responses_per_prompt": responses_per_prompt,
        "server_addresses": server_addresses,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        payload_path = f.name

    code_lines = [
        "import json",
        "import sys",
        "from specrl.cache_updater import SuffixCacheUpdater",
        "with open(sys.argv[1]) as f:",
        "    payload = json.load(f)",
        "updater = SuffixCacheUpdater(payload[\"server_addresses\"])",
        "updater.update_response_cache(",
        "    prompts=payload[\"prompts\"],",
        "    responses=payload[\"responses\"],",
        "    prompt_lengths=payload[\"prompt_lengths\"],",
        "    response_lengths=payload[\"response_lengths\"],",
        "    responses_per_prompt=payload[\"responses_per_prompt\"],",
        ")",
        "print(\"specrl cache updated\")",
    ]
    try:
        subprocess.run([sys.executable, "-c", "\n".join(code_lines), payload_path], check=True)
    finally:
        try:
            os.unlink(payload_path)
        except OSError:
            pass
def run_inference(args: argparse.Namespace):
    """Run the inference pipeline using the provided arguments."""
    run_started_at = time.time()
    # Ray/vLLM child processes are started below and inherit this value. VERL's
    # deterministic request routing uses Python's hash(), so its hash seed must
    # be stable across the two benchmark processes as well.
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.environ["UNI_AGENT_MAX_TURNS"] = str(args.max_turns)
    if args.toolcall_speculation is not None:
        os.environ["UNI_AGENT_TOOLCALL_SPECULATION"] = str(args.toolcall_speculation).lower()

    if args.enable_specrl:
        apply_specrl_plugin()

    # 1. Init Ray
    ray.init()

    # 2. Init rollout manager
    logger.info("Initializing configuration and AgentLoopManager...")
    config = init_config(args)
    llm_server_manager = LLMServerManager.create(config=config)
    agent_loop_manager = AgentLoopManager.create(
        config=config,
        llm_client=llm_server_manager.get_client(),
    )

    # 3. Load dataset
    data_path = os.path.expanduser(args.data_path)
    logger.info(f"Loading dataset from: {data_path}")
    samples = load_dataset("parquet", data_files=data_path, split="train").to_list()

    if args.instance_id:
        samples = [
            sample
            for sample in samples
            if sample.get("extra_info", {})
            .get("tools_kwargs", {})
            .get("reward", {})
            .get("metadata", {})
            .get("instance_id")
            == args.instance_id
        ]
        if not samples:
            raise ValueError(f"Instance ID not found in dataset: {args.instance_id}")
        logger.info("Selected instance_id=%s", args.instance_id)

    # Limit number of samples (-1 = no limit)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
        logger.info("Using first %d samples (--max-samples=%d)", len(samples), args.max_samples)

    if args.controlled_tool_benchmark:
        samples = apply_controlled_tool_prompt(
            samples,
            delay_s=args.controlled_tool_delay,
            repeats=args.controlled_tool_repeats,
        )
        logger.info(
            "Controlled tool benchmark enabled: repeats=%d command=%r",
            args.controlled_tool_repeats,
            controlled_tool_command(args.controlled_tool_delay),
        )

    # 4. Prepare batch data
    logger.info("Preparing data batch...")
    batch = DataProto(
        non_tensor_batch={
            "raw_prompt": np.array([sample["prompt"] for sample in samples], dtype=object),
            "agent_name": np.array([sample["agent_name"] for sample in samples], dtype=object),
            "tools_kwargs": np.array([sample["extra_info"]["tools_kwargs"] for sample in samples], dtype=object),
        },
        meta_info={"validate": True},
    ).repeat(config.actor_rollout_ref.rollout.n)

    # 5. Generate sequences. Driver startup is kept separate from the
    # interaction batch so initialization cannot be reported as Toolcall+ work.
    logger.info("Starting sequence generation...")
    size_divisor = config.actor_rollout_ref.rollout.agent.num_workers
    batch_padded, pad_size = pad_dataproto_to_divisor(batch, size_divisor)
    startup_finished_at = time.time()
    generation_started_at = time.time()
    output_padded = agent_loop_manager.generate_sequences(batch_padded)
    generation_finished_at = time.time()
    output = unpad_dataproto(output_padded, pad_size=pad_size)

    if args.specrl_update_cache:
        update_specrl_cache(
            output,
            responses_per_prompt=config.actor_rollout_ref.rollout.n,
            server_addresses=args.specrl_server_address.split(","),
        )

    # 6. Process results
    reward_scores_available = "rm_scores" in output.batch.keys()
    rm_scores = (
        output.batch["rm_scores"].sum(dim=-1).tolist()
        if reward_scores_available
        else [None] * len(samples)
    )
    numeric_scores = [score for score in rm_scores if score is not None]
    mean_score = float(np.mean(numeric_scores)) if numeric_scores else None
    run_finished_at = time.time()

    if mean_score is None:
        logger.info("Generation completed. Reward evaluation was skipped.")
        print("\n=> Reward evaluation skipped\n")
    else:
        logger.info(f"Generation completed. Mean RM Score: {mean_score:.4f}")
        print(f"\n=> Mean RM Score: {mean_score:.4f}\n")

    # 7. Optionally persist a machine-readable result file (used by eval_checkpoints.py).
    if args.result_path:
        result_path = os.path.expanduser(args.result_path)
        os.makedirs(os.path.dirname(result_path) or ".", exist_ok=True)
        sample_count = len(samples)
        run_ids = _output_values(output, "run_id", sample_count)
        phase_timings = _output_values(output, "phase_timings_s", sample_count, {})
        reward_executed_values = _output_values(output, "reward_executed", sample_count, False)
        reward_skipped_values = _output_values(
            output, "reward_evaluation_skipped", sample_count, False
        )
        sample_details = []
        for sample_index, (sample, score) in enumerate(zip(samples, rm_scores, strict=True)):
            tools_kwargs = sample.get("extra_info", {}).get("tools_kwargs", {})
            reward_metadata = (tools_kwargs.get("reward") or {}).get("metadata", {})
            deployment = tools_kwargs.get("env", {}).get("deployment", {})
            sample_details.append(
                {
                    "sample_index": sample_index,
                    "instance_id": sample.get("controlled_instance_id")
                    or reward_metadata.get("instance_id"),
                    "image": deployment.get("image"),
                    "score": score,
                    "run_id": run_ids[sample_index],
                    "phase_timings_s": phase_timings[sample_index] or {},
                    "reward_executed": bool(reward_executed_values[sample_index]),
                    "reward_evaluation_skipped": bool(reward_skipped_values[sample_index]),
                }
            )

        controlled_reward_config_removed = args.controlled_tool_benchmark and all(
            sample.get("extra_info", {}).get("tools_kwargs", {}).get("reward") is None
            and sample.get("extra_info", {})
            .get("tools_kwargs", {})
            .get("skip_reward_evaluation")
            is True
            for sample in samples
        )
        controlled_reward_skip_verified = (
            controlled_reward_config_removed
            and not reward_scores_available
            and all(not sample["reward_executed"] for sample in sample_details)
            and all(sample["reward_evaluation_skipped"] for sample in sample_details)
            and all(float(sample["phase_timings_s"].get("reward", -1)) == 0.0 for sample in sample_details)
        )
        result = {
            "model_path": os.path.expanduser(args.model_path),
            "data_path": data_path,
            "agent_config_path": os.path.expanduser(args.agent_config_path),
            "engine": args.engine,
            "n": config.actor_rollout_ref.rollout.n,
            "num_samples": len(rm_scores),
            "max_turns": args.max_turns,
            "prompt_length": args.prompt_length,
            "response_length": args.response_length,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "full_determinism": args.full_determinism,
            "seed": args.seed,
            "num_workers": args.num_workers,
            "n_gpus_per_node": args.n_gpus_per_node,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enable_specrl": args.enable_specrl,
            "specrl_update_cache": args.specrl_update_cache,
            "specrl_server_address": args.specrl_server_address,
            "controlled_tool_benchmark": args.controlled_tool_benchmark,
            "controlled_tool_delay_s": args.controlled_tool_delay,
            "controlled_tool_repeats": args.controlled_tool_repeats,
            "controlled_tool_command": (
                controlled_tool_command(args.controlled_tool_delay)
                if args.controlled_tool_benchmark
                else None
            ),
            "controlled_tool_sentinel": (
                CONTROLLED_TOOL_SENTINEL if args.controlled_tool_benchmark else None
            ),
            "reward_scores_available": reward_scores_available,
            "reward_executed_count": sum(sample["reward_executed"] for sample in sample_details),
            "reward_skipped_count": sum(
                sample["reward_evaluation_skipped"] for sample in sample_details
            ),
            "controlled_reward_config_removed": controlled_reward_config_removed,
            "controlled_reward_skip_verified": controlled_reward_skip_verified,
            "started_at_unix": run_started_at,
            "finished_at_unix": run_finished_at,
            "wall_time_s": run_finished_at - run_started_at,
            "generation_time_s": generation_finished_at - generation_started_at,
            "phase_timings_s": {
                "driver_startup": startup_finished_at - run_started_at,
                "interaction_batch": generation_finished_at - generation_started_at,
                "driver_postprocess": run_finished_at - generation_finished_at,
            },
            "mean_rm_score": mean_score,
            "rm_scores": rm_scores,
            "samples": sample_details,
        }
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Wrote result file to: {result_path}")

    return mean_score


def main():
    parser = argparse.ArgumentParser(description="Uni-Agent Inference Runner")

    # Input / Output configs
    parser.add_argument(
        "--data-path",
        type=str,
        default="~/data/swe_agent/swe_bench_verified.parquet",
        help="Path to the input dataset (Parquet format).",
    )
    parser.add_argument(
        "--model-path",
        "--model",
        type=str,
        default="~/models/Qwen3-Coder-30B-A3B-Instruct",
        help="Path to the local model checkpoint.",
    )
    parser.add_argument(
        "--agent-config-path",
        type=str,
        default="examples/agent_interaction/agent_config.yaml",
        help="Path to the agent loop configuration YAML.",
    )
    parser.add_argument(
        "--instance-id",
        type=str,
        default=None,
        help="Run only the matching benchmark instance.",
    )
    parser.add_argument(
        "--result-path",
        type=str,
        default=None,
        help="Optional path to write a JSON result file (mean reward and per-rollout scores).",
    )
    parser.add_argument(
        "--controlled-tool-benchmark",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Replace the selected sample prompt with a deterministic repeated slow-tool benchmark.",
    )
    parser.add_argument(
        "--controlled-tool-delay",
        type=float,
        default=5.0,
        help="Seconds slept by each controlled execute_bash call (default: 5).",
    )
    parser.add_argument(
        "--controlled-tool-repeats",
        type=int,
        default=10,
        help="Number of identical controlled execute_bash calls requested (default: 10).",
    )

    # Inference parameters
    parser.add_argument("--max-turns", type=int, default=100, help="Maximum number of interaction turns per episode.")
    parser.add_argument("--prompt-length", type=int, default=4096, help="Maximum prompt length (tokens).")
    parser.add_argument("--response-length", type=int, default=65536, help="Maximum response length (tokens).")
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=131072,
        help="Maximum vLLM engine context length.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=4,
        help="Maximum number of sequences concurrently handled by vLLM.",
    )
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Sampling top-p (nucleus sampling).")
    parser.add_argument(
        "--full-determinism",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable deterministic VERL routing, kernels, and per-request sampling (default: enabled).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Rollout seed used by VERL and vLLM.")
    parser.add_argument("--toolcall-speculation", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--n", type=int, default=1, help="Number of rollouts per prompt (N).")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="vLLM GPU memory utilization fraction.",
    )
    parser.add_argument("--enable-specrl", action="store_true", help="Enable Uni-Agent histoSpec vLLM plugin.")
    parser.add_argument(
        "--specrl-update-cache",
        action="store_true",
        help="Update the histoSpec suffix cache after generation using a helper subprocess.",
    )
    parser.add_argument(
        "--specrl-server-address",
        type=str,
        default="localhost:6378",
        help="Comma-separated histoSpec cache server address list for cache updates.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Max number of samples to run (default -1). Use -1 for no limit (full dataset).",
    )

    # Execution / Engine configs
    parser.add_argument(
        "--engine",
        type=str,
        default="vllm",
        choices=["vllm", "sglang"],
        help="Inference engine backend (e.g., vllm or sglang).",
    )
    parser.add_argument("--num-workers", type=int, default=8, help="Number of agent rollout workers.")
    parser.add_argument("--nnodes", type=int, default=1, help="Number of nodes to run the job.")
    parser.add_argument("--n-gpus-per-node", type=int, default=8, help="Number of GPUs per node.")
    parser.add_argument(
        "--tensor-parallel-size", "--tp", type=int, default=4, help="Tensor parallel size for the model."
    )

    args = parser.parse_args()
    if args.controlled_tool_delay <= 0:
        parser.error("--controlled-tool-delay must be greater than zero")
    if args.controlled_tool_repeats <= 0:
        parser.error("--controlled-tool-repeats must be a positive integer")
    run_inference(args)


if __name__ == "__main__":
    main()
