"""Parse manually collected LLM responses into experiment_results format.

Usage:
    python parse_live_responses.py chatgpt
    python parse_live_responses.py claude
    python parse_live_responses.py llama3
    python parse_live_responses.py --merge   (merge all 3 into cross-model comparison)
"""
import json
import sys
from pathlib import Path

RESULTS_DIR = Path("results")
MODELS = ["chatgpt", "claude", "llama3"]


def parse_responses(model_key):
    """Read a JSON array or JSONL file of responses and structure them."""
    json_path = RESULTS_DIR / f"live_responses_{model_key}.json"
    jsonl_path = RESULTS_DIR / f"live_responses_{model_key}.jsonl"

    responses = []
    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                responses = data
            else:
                responses = [data]
        print(f"  Loaded {len(responses)} responses from {json_path.name}")
    elif jsonl_path.exists():
        with open(jsonl_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    responses.append(obj)
                except json.JSONDecodeError as e:
                    print(f"  WARNING: line {line_num} is not valid JSON: {e}")
        print(f"  Loaded {len(responses)} responses from {jsonl_path.name}")
    else:
        print(f"ERROR: Neither {json_path} nor {jsonl_path} found.")
        return None

    print(f"  Parsed {len(responses)} responses from {model_key}")

    with open(RESULTS_DIR / "eval_batch_compact.json") as f:
        main_tasks = json.load(f)
    with open(RESULTS_DIR / "eval_batch_adversarial.json") as f:
        adv_tasks = json.load(f)

    task_map = {}
    for t in main_tasks:
        task_map[t["task_id"]] = {"task": t, "is_adversarial": False}
    for t in adv_tasks:
        task_map[t["task_id"]] = {"task": t, "is_adversarial": True}

    main_results = []
    adversarial_results = []

    for resp in responses:
        tid = resp.get("task_id")
        if not tid or tid not in task_map:
            print(f"  WARNING: unknown task_id {tid}")
            continue

        info = task_map[tid]
        task = info["task"]

        result = {
            "task_id": tid,
            "archetype": task.get("archetype", "Critical Analyst"),
            "effective_archetype": task.get("archetype", "Critical Analyst"),
            "explanation_type": task.get("explanation_type", task.get("flaw_type", "unknown")),
            "strategy": "A",
            "persona_granularity": "thick",
            "calibration_level": "calibrated",
            "panel_size": 3,
            "debate_rounds": 0,
            "agents": resp.get("agents", []),
            "model": model_key,
        }

        if info["is_adversarial"]:
            adversarial_results.append(result)
        else:
            main_results.append(result)

    output = {
        "config": {
            "strategy": "A",
            "granularity": "thick",
            "calibration": "calibrated",
            "panel_size": 3,
            "debate_rounds": 0,
            "model": model_key,
            "n_main_tasks": len(main_results),
            "n_adversarial_tasks": len(adversarial_results),
        },
        "main_results": main_results,
        "adversarial_results": adversarial_results,
    }

    outpath = RESULTS_DIR / f"experiment_results_live_{model_key}.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved {outpath}")
    return output


def merge_all():
    """Merge all 3 model results into a cross-model comparison file."""
    all_results = {}
    for model in MODELS:
        path = RESULTS_DIR / f"experiment_results_live_{model}.json"
        if path.exists():
            with open(path) as f:
                all_results[model] = json.load(f)
            print(f"  Loaded {model}: {len(all_results[model]['main_results'])} main tasks")
        else:
            print(f"  SKIP {model}: {path} not found")

    if len(all_results) < 2:
        print("  Need at least 2 models to merge.")
        return

    outpath = RESULTS_DIR / "cross_model_comparison.json"
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Merged {len(all_results)} models into {outpath}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_live_responses.py <chatgpt|claude|llama3|--merge>")
        sys.exit(1)

    arg = sys.argv[1]
    if arg == "--merge":
        merge_all()
    elif arg in MODELS:
        parse_responses(arg)
    else:
        print(f"Unknown model: {arg}. Use one of {MODELS} or --merge")
