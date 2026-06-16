"""Step 6: Cross-model validation analysis.

Computes agreement metrics across 3 LLMs (ChatGPT 5.4, Claude Sonnet 4.6, MiniMax M2.7)
on the trimmed 19-task subset, and generates a comparison figure.
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

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)

MODEL_LABELS = {
    "chatgpt": "ChatGPT 5.4",
    "claude": "Claude Sonnet 4.6",
    "llama3": "MiniMax M2.7",
}

EVAL_DIMS = ["utility", "trust", "persuasiveness"]


def load_cross_model():
    with open(RESULTS_DIR / "cross_model_comparison.json") as f:
        return json.load(f)


def build_cross_model_df(data):
    rows = []
    for model_key, model_data in data.items():
        for result in model_data["main_results"]:
            tid = result["task_id"]
            archetype = result["archetype"]
            exp_type = result["explanation_type"]
            for agent in result["agents"]:
                for dim in EVAL_DIMS + ["cognitive_load"]:
                    rows.append({
                        "model": model_key,
                        "model_label": MODEL_LABELS.get(model_key, model_key),
                        "task_id": tid,
                        "archetype": archetype,
                        "explanation_type": exp_type,
                        "agent_idx": agent["agent_idx"],
                        "dimension": dim,
                        "score": agent[dim]["score"],
                    })
    return pd.DataFrame(rows)


def build_adversarial_df(data):
    rows = []
    for model_key, model_data in data.items():
        for result in model_data["adversarial_results"]:
            tid = result["task_id"]
            for agent in result["agents"]:
                for dim in EVAL_DIMS + ["cognitive_load"]:
                    rows.append({
                        "model": model_key,
                        "model_label": MODEL_LABELS.get(model_key, model_key),
                        "task_id": tid,
                        "agent_idx": agent["agent_idx"],
                        "dimension": dim,
                        "score": agent[dim]["score"],
                    })
    return pd.DataFrame(rows)


def compute_per_model_stats(df):
    stats_out = {}
    for model in df["model"].unique():
        mdf = df[(df["model"] == model) & (df["dimension"].isin(EVAL_DIMS))]
        stats_out[MODEL_LABELS.get(model, model)] = {
            "mean": round(mdf["score"].mean(), 3),
            "std": round(mdf["score"].std(), 3),
            "median": round(mdf["score"].median(), 3),
            "n_ratings": len(mdf),
        }
    return stats_out


def compute_cross_model_icc(df):
    """ICC(2,k) treating models as raters, task-dimension means as subjects."""
    results = {}
    for dim in EVAL_DIMS:
        dim_data = df[df["dimension"] == dim]
        task_model_means = dim_data.groupby(["task_id", "model"])["score"].mean().reset_index()
        pivot = task_model_means.pivot(index="task_id", columns="model", values="score").dropna()

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

        denom = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
        icc = (ms_rows - ms_error) / denom if denom != 0 else 0
        results[dim] = round(max(-1, min(1, icc)), 4)

    overall = np.mean(list(results.values())) if results else 0
    return {"overall_icc": round(overall, 4), "per_dimension": results}


def compute_pairwise_correlations(df):
    """Spearman correlations between each pair of models on task-level mean scores."""
    models = sorted(df["model"].unique())
    eval_df = df[df["dimension"].isin(EVAL_DIMS)]
    task_means = eval_df.groupby(["task_id", "model"])["score"].mean().reset_index()
    pivot = task_means.pivot(index="task_id", columns="model", values="score").dropna()

    pairs = {}
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m1, m2 = models[i], models[j]
            if m1 in pivot.columns and m2 in pivot.columns:
                rho, p = stats.spearmanr(pivot[m1], pivot[m2])
                label = f"{MODEL_LABELS.get(m1, m1)} vs {MODEL_LABELS.get(m2, m2)}"
                pairs[label] = {"rho": round(rho, 4), "p_value": round(p, 4)}
    return pairs


def compute_adversarial_agreement(adv_df):
    """Check all 3 models correctly flag adversarial probes (mean <= 2.0)."""
    results = {}
    for model in adv_df["model"].unique():
        mdf = adv_df[(adv_df["model"] == model) & (adv_df["dimension"].isin(EVAL_DIMS))]
        mean_score = mdf["score"].mean()
        results[MODEL_LABELS.get(model, model)] = {
            "adversarial_mean": round(mean_score, 3),
            "passes_gate": bool(mean_score <= 2.0),
        }
    return results


def compute_score_spread(df):
    """Per-model score spread across explanation types."""
    results = {}
    for model in df["model"].unique():
        mdf = df[(df["model"] == model) & (df["dimension"].isin(EVAL_DIMS))]
        type_means = mdf.groupby("explanation_type")["score"].mean()
        results[MODEL_LABELS.get(model, model)] = {
            etype: round(score, 3) for etype, score in type_means.items()
        }
    return results


def plot_cross_model_comparison(df):
    eval_df = df[df["dimension"].isin(EVAL_DIMS)]
    task_model = eval_df.groupby(["task_id", "model_label", "dimension"])["score"].mean().reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for i, dim in enumerate(EVAL_DIMS):
        dim_data = task_model[task_model["dimension"] == dim]
        sns.boxplot(data=dim_data, x="model_label", y="score",
                    palette="Set2", ax=axes[i])
        axes[i].set_title(dim.capitalize(), fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].set_ylim(0.5, 5.5)
        if i == 0:
            axes[i].set_ylabel("Mean Panel Score (1-5)")
        else:
            axes[i].set_ylabel("")
        axes[i].tick_params(axis='x', rotation=15)

    fig.suptitle("Cross-Model Validation: Score Distributions by Dimension", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cross_model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {FIGURES_DIR / 'cross_model_comparison.png'}")


def plot_cross_model_heatmap(df):
    eval_df = df[df["dimension"].isin(EVAL_DIMS)]
    pivot = eval_df.groupby(["model_label", "explanation_type"])["score"].mean().reset_index()
    heatmap_data = pivot.pivot(index="model_label", columns="explanation_type", values="score")
    heatmap_data = heatmap_data[["feature", "neighbor", "counterfactual"]]

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=1, vmax=5, ax=ax, linewidths=0.5)
    ax.set_title("Cross-Model Mean Scores by Explanation Type", fontweight="bold")
    ax.set_ylabel("")
    ax.set_xlabel("Explanation Type")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cross_model_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {FIGURES_DIR / 'cross_model_heatmap.png'}")


def main():
    data = load_cross_model()
    print(f"Loaded {len(data)} models: {list(data.keys())}")

    df = build_cross_model_df(data)
    adv_df = build_adversarial_df(data)
    print(f"Main ratings: {len(df)} rows, Adversarial: {len(adv_df)} rows")

    print("\n" + "=" * 60)
    print("CROSS-MODEL VALIDATION RESULTS")
    print("=" * 60)

    # 1. Per-model summary
    per_model = compute_per_model_stats(df)
    print("\n1. Per-Model Summary Statistics:")
    for model, s in per_model.items():
        print(f"   {model}: mean={s['mean']}, std={s['std']}, n={s['n_ratings']}")

    # 2. Cross-model ICC
    icc = compute_cross_model_icc(df)
    print(f"\n2. Cross-Model ICC(2,k): {icc['overall_icc']}")
    for dim, val in icc["per_dimension"].items():
        print(f"   {dim}: {val}")

    # 3. Pairwise correlations
    pairwise = compute_pairwise_correlations(df)
    print(f"\n3. Pairwise Spearman Correlations:")
    for pair, res in pairwise.items():
        print(f"   {pair}: rho={res['rho']}, p={res['p_value']}")

    # 4. Adversarial agreement
    adv_agreement = compute_adversarial_agreement(adv_df)
    print(f"\n4. Adversarial Disconfirmation (per model):")
    for model, res in adv_agreement.items():
        print(f"   {model}: mean={res['adversarial_mean']}, passes={res['passes_gate']}")

    # 5. Score spread by explanation type
    spread = compute_score_spread(df)
    print(f"\n5. Mean Score by Explanation Type (per model):")
    for model, types in spread.items():
        print(f"   {model}: {types}")

    # Save report
    report = {
        "models": list(MODEL_LABELS.values()),
        "n_main_tasks": 15,
        "n_adversarial_tasks": 4,
        "n_agents_per_panel": 3,
        "per_model_stats": per_model,
        "cross_model_icc": icc,
        "pairwise_correlations": pairwise,
        "adversarial_agreement": adv_agreement,
        "score_by_explanation_type": spread,
    }

    outpath = RESULTS_DIR / "cross_model_metrics.json"
    with open(outpath, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nMetrics saved to {outpath}")

    # Generate figures
    print("\nGenerating cross-model figures...")
    plot_cross_model_comparison(df)
    plot_cross_model_heatmap(df)

    print("\nDone.")


if __name__ == "__main__":
    main()
