"""Step 5: Compute all metrics and generate publication-quality figures.

Reads experiment_results.json (from step4_generate_evaluations.py) and
eval_batch_compact.json (for task metadata). Computes:
- HAS (Human Alignment Score) vs LiDES benchmark
- Spearman rank correlation
- Krippendorff's alpha (inter-agent reliability)
- Disconfirmation Score (adversarial probe validation)
- Cognitive Load Differential
- ICC (Intraclass Correlation)
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from config import RESULTS_DIR, FIGURES_DIR, DATA_DIR

try:
    import krippendorff as kripp_module
except ImportError:
    kripp_module = None

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("Set2", 5)


def load_all_data():
    with open(RESULTS_DIR / "experiment_results.json") as f:
        experiment = json.load(f)
    with open(RESULTS_DIR / "eval_batch_compact.json") as f:
        compact_tasks = json.load(f)
    with open(RESULTS_DIR / "baseline_results.json") as f:
        baselines = json.load(f)
    with open(DATA_DIR / "lides_benchmark.json") as f:
        lides = json.load(f)
    return experiment, compact_tasks, baselines, lides


def build_flat_dataframe(experiment, compact_tasks):
    """Build a flat DataFrame: one row per (task, agent, dimension)."""
    task_meta = {t["task_id"]: t for t in compact_tasks}
    rows = []

    for result in experiment["main_results"]:
        tid = result["task_id"]
        meta = task_meta.get(tid, {})
        archetype = meta.get("archetype", "Unknown")
        exp_type = meta.get("explanation_type", "unknown")

        for agent in result["agents"]:
            for dim in ["utility", "trust", "persuasiveness", "cognitive_load"]:
                rows.append({
                    "task_id": tid,
                    "agent_idx": agent["agent_idx"],
                    "archetype": archetype,
                    "explanation_type": exp_type,
                    "dimension": dim,
                    "score": agent[dim]["score"],
                    "justification": agent[dim]["justification"],
                })

    return pd.DataFrame(rows)


def build_baseline_dataframe(baseline_results, compact_tasks):
    task_meta = {t["task_id"]: t for t in compact_tasks}
    rows = []
    for baseline_name, entries in baseline_results.items():
        for entry in entries:
            task_id = entry["task_id"]
            meta = task_meta.get(task_id, {})
            result = entry["result"]
            if baseline_name == "misaligned_persona":
                result = result["agent"]
            for dim in ["utility", "trust", "persuasiveness", "cognitive_load"]:
                if dim not in result:
                    continue
                rows.append({
                    "baseline": baseline_name,
                    "task_id": task_id,
                    "archetype": meta.get("archetype", "Unknown"),
                    "explanation_type": meta.get("explanation_type", "unknown"),
                    "dimension": dim,
                    "score": result[dim]["score"],
                })
    return pd.DataFrame(rows)


def build_adversarial_df(experiment):
    rows = []
    for result in experiment["adversarial_results"]:
        tid = result["task_id"]
        for agent in result["agents"]:
            for dim in ["utility", "trust", "persuasiveness", "cognitive_load"]:
                rows.append({
                    "task_id": tid,
                    "agent_idx": agent["agent_idx"],
                    "dimension": dim,
                    "score": agent[dim]["score"],
                })
    return pd.DataFrame(rows)


def compute_consensus(df):
    """Compute panel consensus (mean, std) per task per dimension."""
    consensus = df.groupby(["task_id", "archetype", "explanation_type", "dimension"]).agg(
        mean_score=("score", "mean"),
        std_score=("score", "std"),
        min_score=("score", "min"),
        max_score=("score", "max"),
    ).reset_index()
    consensus["std_score"] = consensus["std_score"].fillna(0)
    return consensus


def compute_has(consensus, lides):
    """Human Alignment Score using LiDES scenarios when 5-point references are available."""
    dim_aliases = {
        "utility": {"perceived_usefulness", "effectiveness"},
        "trust": {"trust"},
        "persuasiveness": {"persuasiveness"},
    }
    explanation_aliases = {
        "feature": {"feature", "content-based", "textual", "structured-tradeoff", "content-based-influence", "transparency-focused", "transparent-interactive"},
        "neighbor": {"neighbor", "user-based", "social-based", "histogram-neighbor", "simple-list", "popularity-based", "without-prediction", "opaque-static"},
        "counterfactual": {"counterfactual", "item-based", "item-based-cf", "collaborative-keyword", "persuasion-focused"},
    }

    llm_lookup = {}
    for _, row in consensus.iterrows():
        llm_lookup.setdefault(row["dimension"], {})
        llm_lookup[row["dimension"]].setdefault(row["explanation_type"], [])
        llm_lookup[row["dimension"]][row["explanation_type"]].append(row["mean_score"])
    llm_lookup = {
        dim: {etype: float(np.mean(scores)) for etype, scores in etypes.items()}
        for dim, etypes in llm_lookup.items()
    }

    errors_by_dim = {}
    scenarios_used = []
    for scenario in lides.get("scenarios", []):
        raw_dim = scenario.get("evaluation_dimension", "").lower()
        local_dim = next((dim for dim, aliases in dim_aliases.items() if raw_dim in aliases), None)
        if not local_dim:
            continue

        pair = [part.lower() for part in scenario.get("explanation_pair", [])]
        mapped_types = []
        for part in pair:
            mapped = next((etype for etype, aliases in explanation_aliases.items() if part in aliases), None)
            if mapped:
                mapped_types.append(mapped)
        if not mapped_types:
            continue

        human = scenario.get("human_result", {})
        numeric_values = []
        if "mean_rating_a_5pt" in human and mapped_types:
            numeric_values.append((mapped_types[0], human["mean_rating_a_5pt"]))
        if "mean_rating_b_5pt" in human and len(mapped_types) > 1:
            numeric_values.append((mapped_types[1], human["mean_rating_b_5pt"]))
        if "mean_rating_5pt" in human and mapped_types:
            numeric_values.append((mapped_types[0], human["mean_rating_5pt"]))

        scenario_used = False
        for explanation_type, human_score in numeric_values:
            llm_score = llm_lookup.get(local_dim, {}).get(explanation_type)
            if llm_score is None:
                continue
            errors_by_dim.setdefault(local_dim, [])
            errors_by_dim[local_dim].append(abs(llm_score - human_score))
            scenario_used = True
        if scenario_used:
            scenarios_used.append(scenario["scenario_id"])

    has_per_dim = {}
    for dimension, errors in errors_by_dim.items():
        mean_mae = np.mean(errors)
        has_per_dim[dimension] = round(1 - (mean_mae / 4), 4)

    overall_has = np.mean(list(has_per_dim.values())) if has_per_dim else 0
    return {
        "overall_has": round(overall_has, 4),
        "per_dimension": has_per_dim,
        "scenarios_used": scenarios_used,
    }


def compute_spearman(consensus):
    """Spearman rank correlation between explanation types across archetypes."""
    results = {}
    for arch in consensus["archetype"].unique():
        arch_data = consensus[(consensus["archetype"] == arch) &
                              (consensus["dimension"].isin(["utility", "trust", "persuasiveness"]))]
        type_means = arch_data.groupby("explanation_type")["mean_score"].mean()

        if len(type_means) >= 3:
            human_order = ["feature", "neighbor", "counterfactual"]
            llm_ranks = type_means.rank(ascending=False)
            human_ranks = pd.Series({t: i+1 for i, t in enumerate(human_order)})

            common = set(llm_ranks.index) & set(human_ranks.index)
            if len(common) >= 3:
                llm_r = [llm_ranks[t] for t in human_order if t in common]
                human_r = [human_ranks[t] for t in human_order if t in common]
                rho, p = stats.spearmanr(llm_r, human_r)
                results[arch] = {"rho": round(rho, 4), "p_value": round(p, 4)}

    return results


def compute_krippendorff_alpha(df):
    """Krippendorff's alpha for inter-agent agreement per dimension."""
    results = {}
    for dim in ["utility", "trust", "persuasiveness"]:
        dim_data = df[df["dimension"] == dim]
        pivot = dim_data.pivot_table(index="task_id", columns="agent_idx", values="score")

        if pivot.shape[1] >= 2:
            reliability_data = pivot.values.T.tolist()

            if kripp_module:
                try:
                    alpha = kripp_module.alpha(
                        reliability_data=reliability_data,
                        level_of_measurement="ordinal"
                    )
                    results[dim] = round(alpha, 4)
                except Exception:
                    scores_flat = pivot.values
                    overall_var = np.var(scores_flat)
                    within_var = np.mean([np.var(row) for row in scores_flat])
                    results[dim] = round(1 - within_var / max(overall_var, 0.01), 4)
            else:
                scores_flat = pivot.values
                overall_var = np.var(scores_flat)
                within_var = np.mean([np.var(row) for row in scores_flat])
                results[dim] = round(1 - within_var / max(overall_var, 0.01), 4)

    overall = np.mean(list(results.values())) if results else 0
    return {"overall_alpha": round(overall, 4), "per_dimension": results}


def compute_icc(df):
    """Intraclass Correlation Coefficient (ICC2,k) per dimension."""
    results = {}
    for dim in ["utility", "trust", "persuasiveness"]:
        dim_data = df[df["dimension"] == dim]
        pivot = dim_data.pivot_table(index="task_id", columns="agent_idx", values="score")
        if pivot.shape[0] < 3 or pivot.shape[1] < 2:
            continue

        n = pivot.shape[0]
        k = pivot.shape[1]
        data = pivot.values

        row_means = data.mean(axis=1)
        col_means = data.mean(axis=0)
        grand_mean = data.mean()

        ss_rows = k * np.sum((row_means - grand_mean) ** 2)
        ss_cols = n * np.sum((col_means - grand_mean) ** 2)
        ss_total = np.sum((data - grand_mean) ** 2)
        ss_error = ss_total - ss_rows - ss_cols

        ms_rows = ss_rows / (n - 1)
        ms_cols = ss_cols / (k - 1)
        ms_error = ss_error / ((n - 1) * (k - 1))

        icc = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n)
        results[dim] = round(max(0, min(1, icc)), 4)

    overall = np.mean(list(results.values())) if results else 0
    return {"overall_icc": round(overall, 4), "per_dimension": results}


def compute_disconfirmation(adv_df):
    """Disconfirmation Score: mean rating on adversarial probes."""
    if adv_df.empty:
        return {"ds": None}

    eval_dims = adv_df[adv_df["dimension"].isin(["utility", "trust", "persuasiveness"])]
    ds = eval_dims["score"].mean()
    by_dim = eval_dims.groupby("dimension")["score"].mean().to_dict()
    return {
        "ds": round(ds, 3),
        "passes_gate": ds <= 2.0,
        "per_dimension": {k: round(v, 3) for k, v in by_dim.items()},
    }


def compute_cognitive_load_differential(df):
    """Cognitive load: Critical Analyst vs Casual Positive Rater."""
    cl = df[df["dimension"] == "cognitive_load"]
    analyst = cl[cl["archetype"] == "Critical Analyst"]["score"]
    casual = cl[cl["archetype"] == "Casual Positive Rater"]["score"]

    if len(analyst) > 1 and len(casual) > 1:
        t_stat, p_val = stats.ttest_ind(analyst, casual)
        cohens_d = (analyst.mean() - casual.mean()) / np.sqrt(
            (analyst.std()**2 + casual.std()**2) / 2
        )
        return {
            "analyst_mean": round(analyst.mean(), 3),
            "casual_mean": round(casual.mean(), 3),
            "differential": round(analyst.mean() - casual.mean(), 3),
            "t_statistic": round(t_stat, 4),
            "p_value": round(p_val, 4),
            "cohens_d": round(cohens_d, 3),
        }
    return {"note": "Insufficient data"}


def compute_baseline_comparison(df, baseline_df):
    target_dims = ["utility", "trust", "persuasiveness"]
    multi_agent = df[df["dimension"].isin(target_dims)].groupby("task_id")["score"].mean()
    summary = {}
    for baseline_name in baseline_df["baseline"].unique():
        baseline_scores = baseline_df[
            (baseline_df["baseline"] == baseline_name) &
            (baseline_df["dimension"].isin(target_dims))
        ].groupby("task_id")["score"].mean()
        common = multi_agent.index.intersection(baseline_scores.index)
        if len(common) == 0:
            continue
        summary[baseline_name] = {
            "mean_score": round(float(baseline_scores.loc[common].mean()), 3),
            "delta_vs_multi_agent": round(float(baseline_scores.loc[common].mean() - multi_agent.loc[common].mean()), 3),
        }
    return summary


def build_implementation_audit(experiment):
    methodology = experiment.get("methodology_experiments", {})
    audit = {
        "implemented": [],
        "partially_implemented": [],
        "not_implemented": [],
    }

    if experiment.get("main_results"):
        audit["implemented"].append("Primary multi-agent persona-conditioned evaluation pipeline")
    if methodology.get("rq1_strategy_comparison"):
        audit["implemented"].append("RQ1 strategy comparison: trait-only vs demographic personas")
    if methodology.get("rq2_single_vs_multi"):
        audit["implemented"].append("RQ2 comparison: single-agent vs multi-agent evaluation")
    if methodology.get("rq3_cognitive_load"):
        audit["implemented"].append("RQ3 proxy experiment: cognitive-load comparison on counterfactual explanations")
    if experiment.get("adversarial_results"):
        audit["implemented"].append("Disconfirmation probe with deliberately flawed explanations")
    if methodology.get("core_summary"):
        audit["implemented"].append("Scaled-down experiment matrix for ablations and prompt variants")

    audit["partially_implemented"].append("LiDES is used as a silver-standard alignment benchmark, but not every literature scenario maps cleanly to the three implemented explanation families")
    audit["partially_implemented"].append("Cross-model comparison remains scaffolded and requires external model APIs for a full run")
    audit["partially_implemented"].append("Cognitive-load validation uses proxy archetypes from clustered behavior rather than separate human-validated constructs")
    return audit


def plot_scores_by_dimension(consensus):
    eval_dims = consensus[consensus["dimension"].isin(["utility", "trust", "persuasiveness"])]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=eval_dims, x="dimension", y="mean_score", hue="archetype",
                palette=PALETTE, ax=ax)
    ax.set_title("Panel Consensus Scores by Dimension and User Archetype", fontweight="bold")
    ax.set_xlabel("Evaluation Dimension")
    ax.set_ylabel("Mean Panel Score (1-5)")
    ax.legend(title="Archetype", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "scores_by_dimension.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_scores_by_explanation_type(consensus):
    eval_dims = consensus[consensus["dimension"].isin(["utility", "trust", "persuasiveness"])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for i, dim in enumerate(["utility", "trust", "persuasiveness"]):
        dim_data = eval_dims[eval_dims["dimension"] == dim]
        sns.barplot(data=dim_data, x="explanation_type", y="mean_score",
                    hue="explanation_type", palette="viridis", ax=axes[i], errorbar="sd", legend=False,
                    order=["feature", "neighbor", "counterfactual"])
        axes[i].set_title(dim.capitalize(), fontweight="bold")
        axes[i].set_xlabel("Explanation Type")
        axes[i].set_ylim(1, 5)
        if i == 0:
            axes[i].set_ylabel("Mean Score (1-5)")
        else:
            axes[i].set_ylabel("")
    fig.suptitle("Evaluation Scores by Explanation Type", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "scores_by_explanation_type.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_score_distribution(df):
    eval_dims = df[df["dimension"].isin(["utility", "trust", "persuasiveness"])]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(eval_dims["score"], bins=range(1, 7), edgecolor="black", alpha=0.7,
            color=PALETTE[0], align="left", rwidth=0.8)
    mean_score = eval_dims["score"].mean()
    ax.axvline(x=mean_score, color="red", linestyle="--", linewidth=2,
               label=f"Mean = {mean_score:.2f}")
    ax.set_xlabel("Likert Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of All Evaluation Scores (Positivity Bias Check)", fontweight="bold")
    ax.set_xticks(range(1, 6))
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "score_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_inter_agent_agreement(consensus):
    eval_dims = consensus[consensus["dimension"].isin(["utility", "trust", "persuasiveness"])]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=eval_dims, x="dimension", y="std_score", color=PALETTE[2], ax=ax)
    ax.set_title("Inter-Agent Disagreement by Dimension", fontweight="bold")
    ax.set_xlabel("Evaluation Dimension")
    ax.set_ylabel("Within-Panel Standard Deviation")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "inter_agent_agreement.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_cognitive_load(df):
    cl = df[df["dimension"] == "cognitive_load"]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=cl, x="archetype", y="score", hue="explanation_type",
                palette="coolwarm", ax=ax)
    ax.set_title("Cognitive Load by Archetype and Explanation Type", fontweight="bold")
    ax.set_xlabel("User Archetype")
    ax.set_ylabel("Cognitive Load (1-7)")
    ax.legend(title="Explanation Type")
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cognitive_load.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_heatmap(consensus):
    eval_dims = consensus[consensus["dimension"].isin(["utility", "trust", "persuasiveness"])]
    pivot = eval_dims.pivot_table(index="archetype", columns=["explanation_type", "dimension"],
                                   values="mean_score", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=1, vmax=5,
                ax=ax, linewidths=0.5)
    ax.set_title("Mean Consensus Scores: Archetype x (Explanation Type, Dimension)", fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "heatmap_scores.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_adversarial_comparison(df, adv_df):
    main_eval = df[df["dimension"].isin(["utility", "trust", "persuasiveness"])]
    adv_eval = adv_df[adv_df["dimension"].isin(["utility", "trust", "persuasiveness"])]

    main_eval = main_eval.copy()
    adv_eval = adv_eval.copy()
    main_eval["type"] = "Main (N=45)"
    adv_eval["type"] = "Adversarial (N=12)"

    combined = pd.concat([main_eval[["dimension", "score", "type"]],
                          adv_eval[["dimension", "score", "type"]]])

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=combined, x="dimension", y="score", hue="type",
                palette=["#66c2a5", "#fc8d62"], ax=ax)
    ax.set_title("Main vs Adversarial Explanation Scores (Disconfirmation Validation)", fontweight="bold")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Score (1-5)")
    ax.legend(title="Task Type")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "adversarial_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    experiment, compact_tasks, baseline_results, lides = load_all_data()
    print(f"Loaded {len(experiment['main_results'])} main + {len(experiment['adversarial_results'])} adversarial results")

    df = build_flat_dataframe(experiment, compact_tasks)
    baseline_df = build_baseline_dataframe(baseline_results, compact_tasks)
    adv_df = build_adversarial_df(experiment)
    consensus = compute_consensus(df)

    print(f"Built flat dataframe: {len(df)} rows, {len(consensus)} consensus entries")
    print("\n" + "=" * 60)
    print("METRIC SUITE RESULTS")
    print("=" * 60)

    # 1. HAS
    has_result = compute_has(consensus, lides)
    print(f"\n1. Human Alignment Score (HAS): {has_result['overall_has']}")
    for dim, score in has_result["per_dimension"].items():
        print(f"   {dim}: {score}")

    # 2. Spearman rho
    spearman = compute_spearman(consensus)
    print(f"\n2. Rank Alignment (Spearman rho):")
    for arch, res in spearman.items():
        print(f"   {arch}: rho={res['rho']}, p={res['p_value']}")

    # 3. Krippendorff alpha
    alpha = compute_krippendorff_alpha(df)
    print(f"\n3. Inter-Agent Reliability (Krippendorff alpha): {alpha['overall_alpha']}")
    for dim, val in alpha["per_dimension"].items():
        print(f"   {dim}: {val}")

    # 4. ICC
    icc = compute_icc(df)
    print(f"\n4. Intraclass Correlation (ICC2,k): {icc['overall_icc']}")
    for dim, val in icc["per_dimension"].items():
        print(f"   {dim}: {val}")

    # 5. Disconfirmation
    ds = compute_disconfirmation(adv_df)
    print(f"\n5. Disconfirmation Score: {ds['ds']} (passes gate: {ds.get('passes_gate')})")

    # 6. Cognitive Load
    cld = compute_cognitive_load_differential(df)
    print(f"\n6. Cognitive Load Differential:")
    for k, v in cld.items():
        print(f"   {k}: {v}")

    baseline_summary = compute_baseline_comparison(df, baseline_df)
    print(f"\n7. Baseline Comparison:")
    for name, values in baseline_summary.items():
        print(f"   {name}: mean={values['mean_score']}, delta_vs_multi={values['delta_vs_multi_agent']}")

    implementation_audit = build_implementation_audit(experiment)

    # Save metrics
    all_metrics = {
        "human_alignment_score": has_result,
        "rank_alignment": spearman,
        "inter_agent_reliability": alpha,
        "intraclass_correlation": icc,
        "disconfirmation_score": ds,
        "cognitive_load_differential": cld,
        "baseline_comparison": baseline_summary,
        "implementation_audit": implementation_audit,
        "summary_stats": {
            "n_tasks": len(experiment["main_results"]),
            "n_adversarial": len(experiment["adversarial_results"]),
            "n_agents_per_panel": experiment["config"]["panel_size"],
            "overall_mean": round(df[df["dimension"].isin(["utility", "trust", "persuasiveness"])]["score"].mean(), 3),
            "overall_std": round(df[df["dimension"].isin(["utility", "trust", "persuasiveness"])]["score"].std(), 3),
        },
    }

    with open(RESULTS_DIR / "metrics_report.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    # Generate plots
    print("\nGenerating publication-quality figures...")
    plot_scores_by_dimension(consensus)
    plot_scores_by_explanation_type(consensus)
    plot_score_distribution(df)
    plot_inter_agent_agreement(consensus)
    plot_cognitive_load(df)
    plot_heatmap(consensus)
    plot_adversarial_comparison(df, adv_df)

    print(f"All figures saved to {FIGURES_DIR}")
    print(f"Metrics report saved to {RESULTS_DIR / 'metrics_report.json'}")


if __name__ == "__main__":
    main()
