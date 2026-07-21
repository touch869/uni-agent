import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

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

    # Limit number of samples (-1 = no limit)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
        logger.info("Using first %d samples (--max-samples=%d)", len(samples), args.max_samples)

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

    # 5. Generate sequences
    logger.info("Starting sequence generation...")
    size_divisor = config.actor_rollout_ref.rollout.agent.num_workers
    batch_padded, pad_size = pad_dataproto_to_divisor(batch, size_divisor)
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
    rm_scores = output.batch["rm_scores"].sum(dim=-1).tolist()
    mean_score = float(np.mean(rm_scores)) if len(rm_scores) > 0 else 0.0
    run_finished_at = time.time()

    logger.info(f"Generation completed. Mean RM Score: {mean_score:.4f}")
    print(f"\n=> Mean RM Score: {mean_score:.4f}\n")

    # 7. Optionally persist a machine-readable result file (used by eval_checkpoints.py).
    if args.result_path:
        result_path = os.path.expanduser(args.result_path)
        os.makedirs(os.path.dirname(result_path) or ".", exist_ok=True)
        sample_details = []
        for sample, score in zip(samples, rm_scores):
            tools_kwargs = sample.get("extra_info", {}).get("tools_kwargs", {})
            reward_metadata = tools_kwargs.get("reward", {}).get("metadata", {})
            deployment = tools_kwargs.get("env", {}).get("deployment", {})
            sample_details.append(
                {
                    "instance_id": reward_metadata.get("instance_id"),
                    "image": deployment.get("image"),
                    "score": score,
                }
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
            "num_workers": args.num_workers,
            "n_gpus_per_node": args.n_gpus_per_node,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enable_specrl": args.enable_specrl,
            "specrl_update_cache": args.specrl_update_cache,
            "specrl_server_address": args.specrl_server_address,
            "started_at_unix": run_started_at,
            "finished_at_unix": run_finished_at,
            "wall_time_s": run_finished_at - run_started_at,
            "generation_time_s": generation_finished_at - generation_started_at,
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
        "--result-path",
        type=str,
        default=None,
        help="Optional path to write a JSON result file (mean reward and per-rollout scores).",
    )

    # Inference parameters
    parser.add_argument("--max-turns", type=int, default=100, help="Maximum number of interaction turns per episode.")
    parser.add_argument("--prompt-length", type=int, default=4096, help="Maximum prompt length (tokens).")
    parser.add_argument("--response-length", type=int, default=65536, help="Maximum response length (tokens).")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Sampling top-p (nucleus sampling).")
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
    run_inference(args)


if __name__ == "__main__":
    main()
