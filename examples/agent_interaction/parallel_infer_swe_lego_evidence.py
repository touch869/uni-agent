#!/usr/bin/env python3
"""SWE-Lego inference runner with top-k and durable per-rollout evidence."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import ray
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf

import verl
from examples.agent_interaction.parallel_infer import apply_specrl_plugin, update_specrl_cache
from verl import DataProto
from verl.experimental.agent_loop import AgentLoopManager
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.workers.rollout.llm_server import LLMServerManager

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=os.getenv("VERL_LOGGING_LEVEL", "INFO"),
)
logger = logging.getLogger(__name__)


def _public_swebench_image(image: str) -> str:
    private = "swr.cn-east-3.myhuaweicloud.com/openyuanrong/swe-bench-verified/"
    public = "enterprise-public-cn-beijing.cr.volces.com/swe-bench-verified/"
    return public + image[len(private) :] if image.startswith(private) else image


def init_config(args: argparse.Namespace) -> DictConfig:
    from hydra import compose, initialize_config_dir

    config_dir = str(Path(verl.__file__).resolve().parent / "trainer" / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="ppo_trainer")
    rollout = config.actor_rollout_ref.rollout
    rollout.agent.agent_loop_config_path = os.path.expanduser(args.agent_config_path)
    rollout.agent.num_workers = args.num_workers
    rollout.multi_turn.max_assistant_turns = args.max_turns
    rollout.multi_turn.max_parallel_calls = 1
    rollout.temperature = args.temperature
    rollout.top_p = args.top_p
    rollout.top_k = args.top_k
    rollout.val_kwargs.temperature = args.temperature
    rollout.val_kwargs.top_p = args.top_p
    rollout.val_kwargs.top_k = args.top_k
    rollout.calculate_log_probs = True
    rollout.nnodes = args.nnodes
    rollout.n_gpus_per_node = args.n_gpus_per_node
    config.trainer.nnodes = args.nnodes
    config.trainer.n_gpus_per_node = args.n_gpus_per_node
    config.actor_rollout_ref.model.path = os.path.expanduser(args.model_path)
    rollout.name = args.engine
    rollout.mode = "async"
    rollout.prompt_length = args.prompt_length
    rollout.response_length = args.response_length
    # The checkpoint advertises 163,840 tokens, but this evaluation explicitly
    # uses prompt+response as its context budget. Profiling the full native
    # window on two 24 GiB 3090s aborted before KV-cache initialization.
    rollout.max_model_len = args.prompt_length + args.response_length
    rollout.n = args.n
    rollout.tensor_model_parallel_size = args.tensor_parallel_size
    rollout.gpu_memory_utilization = args.gpu_memory_utilization
    rollout.seed = args.seed
    rollout.enforce_eager = args.enforce_eager
    if args.enable_specrl:
        OmegaConf.update(
            rollout,
            "engine_kwargs.vllm.speculative_config",
            {
                "method": "suffix",
                "num_speculative_tokens": args.specrl_num_speculative_tokens,
                "suffix_decoding_min_token_prob": 0.1,
            },
            merge=True,
            force_add=True,
        )
        cache_update_finished = time.time()
    config.data.return_raw_chat = True
    config.data.max_prompt_length = args.prompt_length
    config.data.max_response_length = args.response_length
    return config


def _prepare_rollouts(samples: list[dict], args: argparse.Namespace) -> dict[str, np.ndarray]:
    raw_prompts = []
    agent_names = []
    tools_kwargs = []
    for sample in samples:
        for rollout_index in range(args.n):
            kwargs = copy.deepcopy(sample["extra_info"]["tools_kwargs"])
            if not kwargs.setdefault("env", {}).get("post_setup_cmd"):
                kwargs["env"].pop("post_setup_cmd", None)
            deployment = kwargs.setdefault("env", {}).setdefault("deployment", {})
            if deployment.get("image"):
                deployment["image"] = _public_swebench_image(deployment["image"])
            kwargs.setdefault("reward", {})["name"] = "swe_bench_evidence"
            kwargs["interaction"] = {
                **kwargs.get("interaction", {}),
                "max_turns": args.max_turns,
            }
            kwargs["log_dir"] = str(Path(args.evidence_dir) / args.run_name)
            kwargs["evidence"] = {
                "run_name": args.run_name,
                "rollout_index": rollout_index,
            }
            raw_prompts.append(sample["prompt"])
            agent_names.append(sample["agent_name"])
            tools_kwargs.append(kwargs)
    return {
        "raw_prompt": np.array(raw_prompts, dtype=object),
        "agent_name": np.array(agent_names, dtype=object),
        "tools_kwargs": np.array(tools_kwargs, dtype=object),
    }


def _collect_evidence(args: argparse.Namespace) -> tuple[list[dict], list[str]]:
    run_dir = Path(args.evidence_dir) / args.run_name
    records = []
    errors = []
    for path in sorted(run_dir.glob("*/evidence.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            record["evidence_path"] = str(path)
            records.append(record)
            if record.get("fatal_error"):
                errors.append(f"{path}: fatal_error={record['fatal_error']}")
            elif not record.get("instance_id"):
                errors.append(f"{path}: missing instance_id")
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
    jsonl_path = Path(args.evidence_jsonl)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records, errors


def run(args: argparse.Namespace) -> None:
    wall_started = time.time()
    os.environ["VLLM_PLUGINS"] = "histospec_v017" if args.enable_specrl else "none"
    if args.enable_specrl:
        apply_specrl_plugin()
    ray.init()
    config = init_config(args)
    server_manager = LLMServerManager.create(config=config)
    loop_manager = AgentLoopManager.create(config=config, llm_client=server_manager.get_client())
    samples = load_dataset("parquet", data_files=os.path.expanduser(args.data_path), split="train").to_list()
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    batch = DataProto(
        non_tensor_batch=_prepare_rollouts(samples, args),
        meta_info={"validate": True},
    )
    padded, pad_size = pad_dataproto_to_divisor(batch, config.actor_rollout_ref.rollout.agent.num_workers)
    generation_started = time.time()
    output_padded = loop_manager.generate_sequences(padded)
    generation_finished = time.time()
    output = unpad_dataproto(output_padded, pad_size=pad_size)
    response_length = output.batch["responses"].shape[-1]
    attention_mask = output.batch["attention_mask"]
    prompt_tokens = int(attention_mask[:, :-response_length].sum().item())
    response_tokens = int(attention_mask[:, -response_length:].sum().item())
    cache_update_started = None
    cache_update_finished = None
    if args.specrl_update_cache:
        cache_update_started = time.time()
        update_specrl_cache(
            output,
            responses_per_prompt=args.n,
            server_addresses=args.specrl_server_address.split(","),
        )
        cache_update_finished = time.time()
    scores = output.batch["rm_scores"].sum(dim=-1).tolist()
    records, evidence_errors = _collect_evidence(args)
    expected_rollouts = len(samples) * args.n
    result = {
        "run_name": args.run_name,
        "model_path": os.path.expanduser(args.model_path),
        "data_path": os.path.expanduser(args.data_path),
        "agent_config_path": os.path.expanduser(args.agent_config_path),
        "engine": args.engine,
        "n": args.n,
        "num_samples": len(scores),
        "unique_instances": len(
            {
                sample.get("instance_id")
                for sample in records
                if sample.get("instance_id") is not None
            }
        ),
        "max_turns": args.max_turns,
        "prompt_length": args.prompt_length,
        "response_length": args.response_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "num_workers": args.num_workers,
        "n_gpus_per_node": args.n_gpus_per_node,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
        "enforce_eager": args.enforce_eager,
        "enable_specrl": args.enable_specrl,
        "specrl_update_cache": args.specrl_update_cache,
        "specrl_num_speculative_tokens": args.specrl_num_speculative_tokens,
        "wall_time_s": time.time() - wall_started,
        "evaluation_wall_time_s": generation_finished - generation_started,
        "cache_update_time_s": (
            cache_update_finished - cache_update_started
            if cache_update_started is not None and cache_update_finished is not None
            else 0.0
        ),
        "generation_time_s": generation_finished - generation_started,
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "total_tokens": prompt_tokens + response_tokens,
        "mean_rm_score": float(np.mean(scores)) if scores else 0.0,
        "resolved": sum(score > 0 for score in scores),
        "rm_scores": scores,
        "expected_evidence_records": expected_rollouts,
        "evidence_records": len(records),
        "evidence_complete": len(records) == expected_rollouts and not evidence_errors,
        "evidence_jsonl": os.path.expanduser(args.evidence_jsonl),
        "evidence_errors": evidence_errors,
        "instances": [
            {
                "instance_id": record.get("instance_id"),
                "rollout_index": record.get("rollout_index"),
                "rm_score": record.get("rm_score"),
                "resolved": record.get("resolved"),
                "patch_empty": record.get("patch_empty"),
                "actual_turns": record.get("actual_turns"),
                "termination_reason": record.get("termination_reason"),
                "evidence_path": record.get("evidence_path"),
            }
            for record in records
        ],
    }
    result_path = Path(os.path.expanduser(args.result_path))
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"rm_scores", "instances"}}, indent=2))
    if not result["evidence_complete"]:
        raise RuntimeError(
            f"Evidence incomplete: expected {expected_rollouts}, found {len(records)}, errors={evidence_errors}"
        )
        cache_update_finished = time.time()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--agent-config-path", required=True)
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--evidence-jsonl", required=True)
    parser.add_argument("--engine", default="vllm", choices=["vllm"])
    parser.add_argument("--max-samples", type=int, required=True)
    parser.add_argument("--max-turns", type=int, required=True)
    parser.add_argument("--prompt-length", type=int, required=True)
    parser.add_argument("--response-length", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--n-gpus-per-node", type=int, default=2)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-specrl", action="store_true")
    parser.add_argument("--specrl-num-speculative-tokens", type=int, default=5)
    parser.add_argument("--specrl-update-cache", action="store_true")
    parser.add_argument("--specrl-server-address", default="localhost:6378")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
