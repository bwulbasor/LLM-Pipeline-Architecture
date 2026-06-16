"""Step 6: Run local ablations for persona granularity, debate rounds, calibration, and more."""
import json
import sys

import numpy as np

from config import RESULTS_DIR
from step4_generate_evaluations import simulate_focus_group

ABLATION_GRID = {
    "persona_granularity": {
        "conditions": [
            {"name": "thin", "granularity": "thin"},
            {"name": "standard", "granularity": "standard"},
            {"name": "thick", "granularity": "thick"},
        ],
        "fixed": {"strategy": "A", "calibration": "calibrated", "panel_size": 5, "debate_rounds": 1},
    },
    "debate_rounds": {
        "conditions": [
            {"name": "no_debate", "debate_rounds": 0},
            {"name": "single_debate", "debate_rounds": 1},
            {"name": "multi_debate", "debate_rounds": 2},
        ],
        "fixed": {"strategy": "A", "granularity": "thick", "calibration": "calibrated", "panel_size": 5},
    },
    "calibration": {
        "conditions": [
            {"name": "neutral", "calibration": "none"},
            {"name": "instruction_only", "calibration": "instruction_only"},
            {"name": "calibrated", "calibration": "calibrated"},
        ],
        "fixed": {"strategy": "A", "granularity": "thick", "panel_size": 5, "debate_rounds": 1},
    },
    "panel_size": {
        "conditions": [
            {"name": "minimal_3", "panel_size": 3},
            {"name": "standard_5", "panel_size": 5},
            {"name": "extended_7", "panel_size": 7},
        ],
        "fixed": {"strategy": "A", "granularity": "thick", "calibration": "calibrated", "debate_rounds": 1},
    },
    "persona_strategy": {
        "conditions": [
            {"name": "trait_only", "strategy": "A"},
            {"name": "demographic", "strategy": "B"},
        ],
        "fixed": {"granularity": "thick", "calibration": "calibrated", "panel_size": 5, "debate_rounds": 1},
    },
    "temperature": {
        "conditions": [
            {"name": "temp_0_0", "temperature": 0.0},
            {"name": "temp_0_3", "temperature": 0.3},
            {"name": "temp_0_7", "temperature": 0.7},
            {"name": "temp_1_0", "temperature": 1.0},
        ],
        "fixed": {"strategy": "A", "granularity": "thick", "calibration": "calibrated", "panel_size": 5, "debate_rounds": 1},
    },
    "history_length": {
        "conditions": [
            {"name": "history_short", "history_length": "short"},
            {"name": "history_long", "history_length": "long"},
        ],
        "fixed": {"strategy": "A", "granularity": "thick", "calibration": "calibrated", "panel_size": 5, "debate_rounds": 1},
    },
    "random_seed": {
        "conditions": [
            {"name": "seed_13", "random_seed": 13},
            {"name": "seed_42", "random_seed": 42},
            {"name": "seed_99", "random_seed": 99},
        ],
        "fixed": {"strategy": "A", "granularity": "thick", "calibration": "calibrated", "panel_size": 5, "debate_rounds": 1},
    },
}


def run_condition(tasks, config):
    results = []
    for task in tasks:
        result = simulate_focus_group(task, config=config, is_adversarial=False)
        results.append(result)
    return results


def summarize_condition(results):
    summary = {"utility": [], "trust": [], "persuasiveness": [], "cognitive_load": []}
    stds = {"utility": [], "trust": [], "persuasiveness": [], "cognitive_load": []}
    for result in results:
        for dimension in summary:
            scores = [agent[dimension]["score"] for agent in result["agents"]]
            summary[dimension].append(float(np.mean(scores)))
            stds[dimension].append(float(np.std(scores)))
    return {
        "n_tasks": len(results),
        "mean_scores": {dimension: round(float(np.mean(values)), 3) for dimension, values in summary.items()},
        "mean_panel_std": {dimension: round(float(np.mean(values)), 3) for dimension, values in stds.items()},
    }


def main(sample_size=None):
    with open(RESULTS_DIR / "eval_batch_compact.json") as handle:
        tasks = json.load(handle)

    if sample_size is None:
        sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    tasks = tasks[:sample_size]
    print(f"Running local ablations on {len(tasks)} sampled tasks")

    all_results = {}
    summary = {}

    for ablation_name, specification in ABLATION_GRID.items():
        print(f"\n=== {ablation_name} ===")
        all_results[ablation_name] = {}
        summary[ablation_name] = {}
        for condition in specification["conditions"]:
            condition_name = condition["name"]
            config = dict(specification["fixed"])
            config.update({key: value for key, value in condition.items() if key != "name"})
            print(f"  {condition_name}: {config}")
            results = run_condition(tasks, config)
            all_results[ablation_name][condition_name] = results
            summary[ablation_name][condition_name] = summarize_condition(results)

    with open(RESULTS_DIR / "ablation_results.json", "w") as handle:
        json.dump(all_results, handle, indent=2)
    with open(RESULTS_DIR / "ablation_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\nSaved ablation results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
