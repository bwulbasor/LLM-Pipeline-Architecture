"""Flexible pipeline runner for local and tiny-live persona evaluation workflows."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import (
    DEFAULT_ABLATION_SAMPLE_SIZE,
    DEFAULT_LIVE_DEBATE_ROUNDS,
    DEFAULT_LIVE_PANEL_SIZE,
    DEFAULT_LIVE_REASONING,
    DEFAULT_LIVE_SAMPLE_SIZE,
    FIGURES_DIR,
    RESULTS_DIR,
)


def stage_is_fresh(outputs, dependencies):
    output_paths = [Path(path) for path in outputs]
    dependency_paths = [Path(path) for path in dependencies]
    if not output_paths or any(not path.exists() for path in output_paths):
        return False
    if not dependency_paths or any(not path.exists() for path in dependency_paths):
        return True
    newest_dependency = max(path.stat().st_mtime for path in dependency_paths)
    oldest_output = min(path.stat().st_mtime for path in output_paths)
    return oldest_output >= newest_dependency


def execute_stage(index, total, stage, force=False):
    stage_name = stage["name"]
    outputs = stage.get("outputs", [])
    dependencies = stage.get("dependencies", [])

    print(f"\n[STEP {index}/{total}] {stage_name}...")
    if not force and outputs and stage_is_fresh(outputs, dependencies):
        print("  Outputs are fresh, skipping.")
        return {"name": stage_name, "status": "skipped", "elapsed_seconds": 0.0}

    start = time.time()
    stage["callable"]()
    elapsed = time.time() - start
    print(f"  Completed in {elapsed:.1f}s")
    return {"name": stage_name, "status": "completed", "elapsed_seconds": round(elapsed, 2)}


def build_local_stages(ablation_sample_size):
    from step1_preprocess import main as preprocess_main
    from step2_generate_explanations import main as explanations_main
    from step3_personas import main as personas_main
    from step4_batch_eval import main as batch_main
    from step4_generate_evaluations import main as evaluation_main
    from step5_analysis import main as analysis_main
    from step6_ablations import main as ablations_main

    return [
        {
            "name": "Preprocessing MovieLens data and clustering archetypes",
            "callable": preprocess_main,
            "outputs": [
                Path("data/user_features.csv"),
                Path("data/filtered_ratings.csv"),
                Path("data/movies.csv"),
                RESULTS_DIR / "seed_users.json",
                RESULTS_DIR / "archetype_profiles.json",
            ],
            "dependencies": [Path("data/ml-10M100K/ratings.dat"), Path("data/ml-10M100K/movies.dat")],
        },
        {
            "name": "Generating grounded explanation tasks",
            "callable": explanations_main,
            "outputs": [
                RESULTS_DIR / "evaluation_tasks.json",
                RESULTS_DIR / "seed_user_histories.json",
                RESULTS_DIR / "adversarial_explanations.json",
            ],
            "dependencies": [
                Path("data/user_features.csv"),
                Path("data/filtered_ratings.csv"),
                Path("data/movies.csv"),
                RESULTS_DIR / "seed_users.json",
            ],
        },
        {
            "name": "Exporting persona prompt variants",
            "callable": personas_main,
            "outputs": [RESULTS_DIR / "persona_templates.json"],
            "dependencies": [Path("src/step3_personas.py")],
        },
        {
            "name": "Building stratified evaluation batches",
            "callable": batch_main,
            "outputs": [
                RESULTS_DIR / "eval_batch_compact.json",
                RESULTS_DIR / "eval_batch_full.json",
                RESULTS_DIR / "eval_batch_adversarial.json",
            ],
            "dependencies": [
                RESULTS_DIR / "evaluation_tasks.json",
                RESULTS_DIR / "adversarial_explanations.json",
            ],
        },
        {
            "name": "Generating local panel evaluations and baselines",
            "callable": evaluation_main,
            "outputs": [
                RESULTS_DIR / "experiment_results.json",
                RESULTS_DIR / "baseline_results.json",
            ],
            "dependencies": [
                RESULTS_DIR / "eval_batch_compact.json",
                RESULTS_DIR / "eval_batch_adversarial.json",
            ],
        },
        {
            "name": "Computing metrics and generating figures",
            "callable": analysis_main,
            "outputs": [
                RESULTS_DIR / "metrics_report.json",
                FIGURES_DIR / "scores_by_dimension.png",
                FIGURES_DIR / "scores_by_explanation_type.png",
                FIGURES_DIR / "score_distribution.png",
                FIGURES_DIR / "inter_agent_agreement.png",
                FIGURES_DIR / "cognitive_load.png",
                FIGURES_DIR / "heatmap_scores.png",
                FIGURES_DIR / "adversarial_comparison.png",
            ],
            "dependencies": [
                RESULTS_DIR / "experiment_results.json",
                RESULTS_DIR / "baseline_results.json",
                RESULTS_DIR / "eval_batch_compact.json",
                Path("data/lides_benchmark.json"),
            ],
        },
        {
            "name": "Running scaled-down ablations",
            "callable": lambda: ablations_main(sample_size=ablation_sample_size),
            "outputs": [
                RESULTS_DIR / "ablation_results.json",
                RESULTS_DIR / "ablation_summary.json",
            ],
            "dependencies": [RESULTS_DIR / "eval_batch_compact.json", Path("src/step6_ablations.py")],
        },
    ]


def run_tiny_live_sample(sample_size, panel_size, debate_rounds, reasoning_enabled):
    from step4_multi_agent_eval import save_live_experiment

    os.environ["OPENAI_COMPAT_REASONING"] = "true" if reasoning_enabled else "false"
    config = {
        "strategy": "A",
        "persona_granularity": "thick",
        "calibration_level": "calibrated",
        "panel_size": panel_size,
        "debate_rounds": debate_rounds,
    }
    return save_live_experiment(
        sample_size=sample_size,
        config=config,
        output_name="experiment_results_live_small.json",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run the persona-evaluation pipeline.")
    parser.add_argument("--mode", choices=["local", "local_with_live_small"], default="local")
    parser.add_argument("--force", action="store_true", help="Rerun stages even if outputs look fresh.")
    parser.add_argument("--start-at", type=int, default=1, help="1-based start stage for the local pipeline.")
    parser.add_argument("--end-at", type=int, default=7, help="1-based end stage for the local pipeline.")
    parser.add_argument("--ablation-sample-size", type=int, default=DEFAULT_ABLATION_SAMPLE_SIZE)
    parser.add_argument("--live-sample-size", type=int, default=DEFAULT_LIVE_SAMPLE_SIZE)
    parser.add_argument("--live-panel-size", type=int, default=DEFAULT_LIVE_PANEL_SIZE)
    parser.add_argument("--live-debate-rounds", type=int, default=DEFAULT_LIVE_DEBATE_ROUNDS)
    parser.add_argument(
        "--live-reasoning",
        choices=["true", "false"],
        default="true" if DEFAULT_LIVE_REASONING else "false",
        help="Enable or disable reasoning/thinking mode for the tiny live sample.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    stages = build_local_stages(args.ablation_sample_size)
    total = len(stages)
    start_at = max(1, args.start_at)
    end_at = min(total, args.end_at)
    if start_at > end_at:
        raise ValueError("--start-at cannot be greater than --end-at")

    start = time.time()
    stage_records = []

    print("=" * 72)
    print("PERSONA EVALUATION PIPELINE")
    print("=" * 72)
    print(f"Mode: {args.mode}")
    print(f"Local stage range: {start_at}..{end_at}")

    for idx in range(start_at, end_at + 1):
        stage = stages[idx - 1]
        record = execute_stage(idx, total, stage, force=args.force)
        stage_records.append(record)

    live_record = None
    if args.mode == "local_with_live_small":
        print("\n[LIVE] Running tiny live sample...")
        live_start = time.time()
        live_output = run_tiny_live_sample(
            sample_size=args.live_sample_size,
            panel_size=args.live_panel_size,
            debate_rounds=args.live_debate_rounds,
            reasoning_enabled=args.live_reasoning == "true",
        )
        live_elapsed = time.time() - live_start
        live_record = {
            "name": "tiny_live_sample",
            "status": "completed",
            "elapsed_seconds": round(live_elapsed, 2),
            "output": str(live_output),
        }
        print(f"  Live sample saved to {live_output} in {live_elapsed:.1f}s")

    elapsed = time.time() - start
    summary = {
        "mode": args.mode,
        "local_stage_range": {"start_at": start_at, "end_at": end_at},
        "force": args.force,
        "stages": stage_records,
        "live": live_record,
        "total_elapsed_seconds": round(elapsed, 2),
    }
    summary_path = RESULTS_DIR / "pipeline_run_summary.json"
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\n{'=' * 72}")
    print(f"Pipeline complete in {elapsed:.1f}s")
    print(f"{'=' * 72}")
    print("Outputs:")
    print("  data/user_features.csv")
    print("  results/seed_users.json")
    print("  results/seed_user_histories.json")
    print("  results/evaluation_tasks.json")
    print("  results/eval_batch_compact.json")
    print("  results/experiment_results.json")
    print("  results/baseline_results.json")
    print("  results/metrics_report.json")
    print("  results/ablation_summary.json")
    if live_record:
        print(f"  {live_record['output']}")
    print(f"  {summary_path}")
    print("  figures/")


if __name__ == "__main__":
    main()
