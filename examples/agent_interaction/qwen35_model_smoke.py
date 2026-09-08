#!/usr/bin/env python3
"""Load the local Qwen3.5 checkpoint and generate one token."""

from vllm import LLM, SamplingParams


def main() -> None:
    llm = LLM(
        model="/workspace/models/Qwen3.5-9B",
        tensor_parallel_size=2,
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
    )
    output = llm.generate(["Say OK"], SamplingParams(max_tokens=1, temperature=0))
    completion = output[0].outputs[0]
    print("MODEL_SMOKE_OK", completion.token_ids, completion.text)


if __name__ == "__main__":
    main()
