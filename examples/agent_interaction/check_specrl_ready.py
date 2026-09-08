#!/usr/bin/env python3
import importlib.util
import json
import sys


def main():
    result = {
        "ready": True,
        "checks": {},
        "errors": [],
    }

    try:
        import vllm
        result["checks"]["vllm_version"] = vllm.__version__
        if not vllm.__version__.startswith("0.10.0"):
            result["ready"] = False
            result["errors"].append(
                f"histoSpec plugin in this repo requires vllm==0.10.0, found {vllm.__version__}"
            )
    except Exception as exc:
        result["ready"] = False
        result["checks"]["vllm_version"] = None
        result["errors"].append(f"failed to import vllm: {exc}")

    for module in ["specrl", "specrl.suffix_cache", "specrl.cache_updater"]:
        try:
            spec = importlib.util.find_spec(module)
        except Exception as exc:
            spec = None
            result["errors"].append(f"failed to inspect {module}: {exc}")
        result["checks"][module] = bool(spec)
        if spec is None:
            result["ready"] = False
            result["errors"].append(f"missing Python module: {module}")

    plugin = importlib.util.find_spec("recipe.specRL.histoSpec.vllm_plugin.patch")
    result["checks"]["uniagent_histoSpec_plugin"] = bool(plugin)
    if plugin is None:
        result["ready"] = False
        result["errors"].append("missing Uni-Agent histoSpec vLLM plugin")

    print(json.dumps(result, indent=2))
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
