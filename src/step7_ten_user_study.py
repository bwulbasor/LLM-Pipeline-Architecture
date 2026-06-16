"""Prepare and analyze a lightweight 10-user study package.

This module adds a small, human-checkable study on top of the main pipeline.
It does not replace the main methodology; it packages a simpler subset that a
team can execute quickly:

1. select 10 seed users (2 per archetype),
2. generate short personas,
3. generate 5 explanation cases per user,
4. score them with the existing local persona-evaluation logic,
5. export a human-rating sheet,
6. compare human ratings against the LLM-side ratings once filled.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import RESULTS_DIR
from step3_personas import ARCHETYPE_TRAITS, build_persona_prompt_strategy_a
from step4_generate_evaluations import simulate_focus_group

STUDY_DIR = RESULTS_DIR / "ten_user_study"
DIMENSIONS = ["utility", "trust", "persuasiveness"]


def load_inputs():
    with open(RESULTS_DIR / "seed_users.json") as handle:
        seed_users = json.load(handle)
    with open(RESULTS_DIR / "evaluation_tasks.json") as handle:
        tasks = json.load(handle)
    return seed_users, tasks


def tasks_by_user(tasks: List[dict]) -> Dict[int, List[dict]]:
    grouped: Dict[int, List[dict]] = {}
    for task in tasks:
        grouped.setdefault(int(task["user_id"]), []).append(task)
    return grouped


def select_ten_users(seed_users: dict, available_tasks: Dict[int, List[dict]], per_archetype: int = 2) -> List[dict]:
    selection = []
    for archetype, user_ids in seed_users.items():
        count = 0
        for user_id in user_ids:
            uid = int(user_id)
            if uid not in available_tasks:
                continue
            selection.append({"user_id": uid, "archetype": archetype})
            count += 1
            if count >= per_archetype:
                break
    return selection


def generate_short_persona(archetype: str, task: dict) -> str:
    genres = ", ".join(task.get("user_top_genres", [])[:3])
    top_movies = task.get("user_top_movies", [])[:2]
    movie_text = " and ".join(top_movies) if top_movies else "a few well-rated movies"
    summary = task.get("user_history_summary_short", "")

    if archetype == "Blockbuster Follower":
        framing = "This user prefers familiar, crowd-pleasing films and responds well to simple, confidence-building explanations."
    elif archetype == "Niche Explorer":
        framing = "This user looks for novelty and dislikes explanations that rely too heavily on popularity or generic claims."
    elif archetype == "Genre Specialist":
        framing = "This user has strong tastes and expects explanations to show real understanding of their preferred genres."
    elif archetype == "Casual Positive Rater":
        framing = "This user is easy to please, but prefers quick explanations that do not demand much effort."
    else:
        framing = "This user is more skeptical and analytical, and tends to reward specific evidence over vague persuasion."

    return (
        f"User archetype: {archetype}. "
        f"They show positive signals for {genres}. "
        f"Some of their strongest anchors are {movie_text}. "
        f"{framing} "
        f"Behavior summary: {summary}"
    )


def get_base_tasks_for_user(user_tasks: List[dict]) -> Dict[str, dict]:
    mapping = {}
    for task in user_tasks:
        mapping[task["explanation_type"]] = task
    return mapping


def build_extra_templates(base_feature: dict, base_neighbor: dict) -> List[dict]:
    anchor_titles = base_feature.get("user_top_movies", [])[:2]
    anchor_text = " and ".join([f"'{title}'" for title in anchor_titles]) if anchor_titles else "movies you rated highly"
    top_genres = ", ".join(base_feature.get("user_top_genres", [])[:3])
    movie = base_feature["recommended_movie"]
    support = base_neighbor.get("neighbor_support", 0)
    avg = base_neighbor.get("neighbor_avg_rating", 0.0)

    return [
        {
            "study_template": "preference_summary",
            "scoring_profile": "feature",
            "explanation_text": (
                f"We recommend '{movie}' because your history shows a strong preference for {top_genres}. "
                f"This title fits that pattern and is likely to match the kind of movies you usually reward."
            ),
        },
        {
            "study_template": "hybrid_social",
            "scoring_profile": "neighbor",
            "explanation_text": (
                f"'{movie}' fits the taste pattern suggested by {anchor_text}, and it is also supported by users with similar behavior. "
                f"In our neighbor pool, {support} similar users rated it at least 4/5, with an average of {avg:.1f}/5."
            ),
        },
    ]


def build_case_records(selected_users: List[dict], grouped_tasks: Dict[int, List[dict]]) -> List[dict]:
    cases = []
    for row in selected_users:
        user_id = row["user_id"]
        archetype = row["archetype"]
        base = get_base_tasks_for_user(grouped_tasks[user_id])

        if not {"feature", "neighbor", "counterfactual"} <= set(base):
            continue

        feature_task = base["feature"]
        neighbor_task = base["neighbor"]
        counterfactual_task = base["counterfactual"]

        persona_short = generate_short_persona(archetype, feature_task)
        persona_prompt = build_persona_prompt_strategy_a(
            archetype_name=archetype,
            quirk_idx=0,
            granularity="thick",
            user_history_summary=feature_task.get("user_history_summary_short"),
        )

        standard_cases = [
            {
                "study_template": "feature",
                "scoring_profile": "feature",
                "source_task": feature_task,
                "explanation_text": feature_task["explanation_text"],
            },
            {
                "study_template": "neighbor",
                "scoring_profile": "neighbor",
                "source_task": neighbor_task,
                "explanation_text": neighbor_task["explanation_text"],
            },
            {
                "study_template": "counterfactual",
                "scoring_profile": "counterfactual",
                "source_task": counterfactual_task,
                "explanation_text": counterfactual_task["explanation_text"],
            },
        ]

        extra_cases = build_extra_templates(feature_task, neighbor_task)
        for extra in extra_cases:
            extra["source_task"] = feature_task
        combined_cases = standard_cases + extra_cases

        for idx, case in enumerate(combined_cases, start=1):
            source = case["source_task"]
            case_id = f"U{user_id}_{case['study_template']}"
            cases.append(
                {
                    "case_id": case_id,
                    "user_id": user_id,
                    "archetype": archetype,
                    "recommended_movie": source["recommended_movie"],
                    "recommended_movie_genres": source["recommended_movie_genres"],
                    "study_template": case["study_template"],
                    "scoring_profile": case["scoring_profile"],
                    "persona_short": persona_short,
                    "persona_prompt": persona_prompt,
                    "explanation_text": case["explanation_text"],
                    "user_top_movies": source.get("user_top_movies", []),
                    "user_top_genres": source.get("user_top_genres", []),
                    "user_history_summary_short": source.get("user_history_summary_short", ""),
                    "user_history_summary_long": source.get("user_history_summary_long", ""),
                    "order_within_user": idx,
                }
            )
    return cases


def llm_rate_cases(cases: List[dict]) -> pd.DataFrame:
    rows = []
    for case in cases:
        task = {
            "task_id": case["case_id"],
            "archetype": case["archetype"],
            "explanation_type": case["scoring_profile"],
            "explanation_text": case["explanation_text"],
            "recommended_movie": case["recommended_movie"],
            "user_history_summary_short": case["user_history_summary_short"],
            "user_history_summary_long": case["user_history_summary_long"],
        }
        result = simulate_focus_group(
            task,
            config={
                "strategy": "A",
                "granularity": "thick",
                "calibration": "calibrated",
                "panel_size": 1,
                "debate_rounds": 0,
                "history_length": "long",
                "temperature": 0.7,
                "random_seed": 42,
            },
        )
        agent = result["agents"][0]
        row = {
            "case_id": case["case_id"],
            "user_id": case["user_id"],
            "archetype": case["archetype"],
            "study_template": case["study_template"],
            "recommended_movie": case["recommended_movie"],
        }
        for dimension in DIMENSIONS:
            row[f"llm_{dimension}"] = agent[dimension]["score"]
            row[f"llm_{dimension}_justification"] = agent[dimension]["justification"]
        row["llm_cognitive_load"] = agent["cognitive_load"]["score"]
        rows.append(row)
    return pd.DataFrame(rows)


def export_study_packet(cases: List[dict], llm_df: pd.DataFrame):
    STUDY_DIR.mkdir(parents=True, exist_ok=True)

    case_df = pd.DataFrame(cases)
    case_df.to_json(STUDY_DIR / "study_cases.json", orient="records", indent=2)
    case_df.to_csv(STUDY_DIR / "study_cases.csv", index=False)

    llm_df.to_csv(STUDY_DIR / "llm_ratings_local.csv", index=False)

    human_template = case_df[
        [
            "case_id",
            "user_id",
            "archetype",
            "order_within_user",
            "recommended_movie",
            "recommended_movie_genres",
            "study_template",
            "persona_short",
            "explanation_text",
        ]
    ].copy()
    human_template["human_utility"] = ""
    human_template["human_trust"] = ""
    human_template["human_persuasiveness"] = ""
    human_template["notes"] = ""
    human_template.to_csv(STUDY_DIR / "human_rating_template.csv", index=False)

    packet_lines = [
        "# Ten-User Study Packet",
        "",
        "This folder contains a small study package derived from the main pipeline.",
        "",
        "## What to do",
        "",
        "1. Open `human_rating_template.csv`.",
        "2. For each row, read the short persona and the explanation.",
        "3. Rate utility, trust, and persuasiveness from 1 to 5.",
        "4. Save the completed file as `human_ratings_completed.csv` in the same folder.",
        "5. Run `python src/step7_ten_user_study.py analyze` to compare the human ratings against the LLM-side ratings.",
        "",
        "## Study design",
        "",
        "- 10 users total",
        "- 2 users sampled from each archetype",
        "- 5 explanation cases per user",
        "- 50 total cases",
        "- LLM-side ratings generated with the existing local calibrated evaluator",
        "",
        "## Files",
        "",
        "- `study_cases.json` / `study_cases.csv`: full case metadata",
        "- `llm_ratings_local.csv`: local evaluator ratings",
        "- `human_rating_template.csv`: sheet for your teammates to fill",
        "",
    ]
    (STUDY_DIR / "README.md").write_text("\n".join(packet_lines), encoding="utf-8")


def prepare_study():
    seed_users, tasks = load_inputs()
    grouped = tasks_by_user(tasks)
    selected_users = select_ten_users(seed_users, grouped, per_archetype=2)
    cases = build_case_records(selected_users, grouped)
    llm_df = llm_rate_cases(cases)
    export_study_packet(cases, llm_df)
    print(f"Prepared 10-user study package in {STUDY_DIR}")


def validate_human_ratings(human_df: pd.DataFrame):
    missing = human_df[["human_utility", "human_trust", "human_persuasiveness"]].isna().any(axis=1)
    blanks = (human_df[["human_utility", "human_trust", "human_persuasiveness"]].astype(str).applymap(lambda x: x.strip() == "")).any(axis=1)
    invalid = human_df[missing | blanks]
    if not invalid.empty:
        raise ValueError(
            "Human ratings are incomplete. Fill `human_utility`, `human_trust`, and `human_persuasiveness` for every row."
        )


def analyze_study():
    llm_path = STUDY_DIR / "llm_ratings_local.csv"
    human_path = STUDY_DIR / "human_ratings_completed.csv"
    if not llm_path.exists():
        raise FileNotFoundError("Missing llm_ratings_local.csv. Run prepare first.")
    if not human_path.exists():
        raise FileNotFoundError("Missing human_ratings_completed.csv. Ask a teammate to fill the template first.")

    llm_df = pd.read_csv(llm_path)
    human_df = pd.read_csv(human_path)
    validate_human_ratings(human_df)

    merged = human_df.merge(llm_df, on=["case_id", "user_id", "archetype", "study_template", "recommended_movie"], how="inner")
    if merged.empty:
        raise ValueError("No matching rows found between human ratings and LLM ratings.")

    summary = {
        "n_cases": int(len(merged)),
        "per_dimension": {},
    }

    for dimension in DIMENSIONS:
        human_col = f"human_{dimension}"
        llm_col = f"llm_{dimension}"
        merged[human_col] = merged[human_col].astype(float)
        merged[llm_col] = merged[llm_col].astype(float)
        mae = (merged[human_col] - merged[llm_col]).abs().mean()
        corr = merged[[human_col, llm_col]].corr(method="pearson").iloc[0, 1]
        summary["per_dimension"][dimension] = {
            "human_mean": round(float(merged[human_col].mean()), 3),
            "llm_mean": round(float(merged[llm_col].mean()), 3),
            "mae": round(float(mae), 3),
            "pearson_r": round(float(corr), 3) if pd.notna(corr) else None,
        }

    merged["mean_abs_diff"] = (
        (merged["human_utility"].astype(float) - merged["llm_utility"].astype(float)).abs()
        + (merged["human_trust"].astype(float) - merged["llm_trust"].astype(float)).abs()
        + (merged["human_persuasiveness"].astype(float) - merged["llm_persuasiveness"].astype(float)).abs()
    ) / 3.0

    merged.to_csv(STUDY_DIR / "human_vs_llm_case_table.csv", index=False)
    with open(STUDY_DIR / "human_vs_llm_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    plot_rows = []
    for dimension in DIMENSIONS:
        plot_rows.append({"dimension": dimension.title(), "source": "Human", "mean": summary["per_dimension"][dimension]["human_mean"]})
        plot_rows.append({"dimension": dimension.title(), "source": "LLM", "mean": summary["per_dimension"][dimension]["llm_mean"]})
    plot_df = pd.DataFrame(plot_rows)

    plt.figure(figsize=(8, 4.5))
    for idx, source in enumerate(["Human", "LLM"]):
        subset = plot_df[plot_df["source"] == source]
        x_positions = [i + (-0.18 if source == "Human" else 0.18) for i in range(len(subset))]
        plt.bar(x_positions, subset["mean"], width=0.35, label=source)
    plt.xticks(range(len(DIMENSIONS)), [d.title() for d in DIMENSIONS])
    plt.ylim(0, 5)
    plt.ylabel("Mean Rating")
    plt.title("10-User Study: Human vs LLM Mean Ratings")
    plt.legend()
    plt.tight_layout()
    plt.savefig(STUDY_DIR / "human_vs_llm_means.png", dpi=200)
    plt.close()

    print(f"Saved comparison outputs to {STUDY_DIR}")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare or analyze the lightweight 10-user study.")
    parser.add_argument("mode", choices=["prepare", "analyze"], nargs="?", default="prepare")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "prepare":
        prepare_study()
    else:
        analyze_study()


if __name__ == "__main__":
    main()
