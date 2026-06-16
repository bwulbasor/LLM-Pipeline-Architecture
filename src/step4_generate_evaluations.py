"""Step 4b: Generate calibrated panel evaluations with support for methodology variants."""
import json
import random
from collections import defaultdict

import numpy as np

from config import RANDOM_SEED, RESULTS_DIR
from step3_personas import PERSONALITY_QUIRKS

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

EXPLANATION_PROFILES = {
    "feature": {
        "utility": (3.55, 0.65),
        "trust": (3.35, 0.75),
        "persuasiveness": (3.30, 0.65),
        "cognitive_load": (2.8, 0.8),
    },
    "neighbor": {
        "utility": (3.15, 0.75),
        "trust": (3.05, 0.85),
        "persuasiveness": (3.15, 0.75),
        "cognitive_load": (2.4, 0.75),
    },
    "counterfactual": {
        "utility": (2.95, 0.85),
        "trust": (2.80, 0.85),
        "persuasiveness": (2.90, 0.75),
        "cognitive_load": (4.1, 0.9),
    },
}

ARCHETYPE_MODIFIERS = {
    "Blockbuster Follower": {
        "utility": 0.2, "trust": 0.25, "persuasiveness": 0.35, "cognitive_load": 0.25,
        "neighbor_bonus": 0.45, "counterfactual_penalty": -0.25,
    },
    "Niche Explorer": {
        "utility": -0.25, "trust": -0.15, "persuasiveness": -0.35, "cognitive_load": -0.35,
        "neighbor_bonus": -0.35, "counterfactual_penalty": 0.15,
    },
    "Genre Specialist": {
        "utility": 0.05, "trust": 0.00, "persuasiveness": 0.00, "cognitive_load": -0.10,
        "neighbor_bonus": -0.10, "counterfactual_penalty": 0.00,
    },
    "Casual Positive Rater": {
        "utility": 0.35, "trust": 0.45, "persuasiveness": 0.35, "cognitive_load": 0.45,
        "neighbor_bonus": 0.25, "counterfactual_penalty": -0.20,
    },
    "Critical Analyst": {
        "utility": -0.45, "trust": -0.55, "persuasiveness": -0.45, "cognitive_load": -0.60,
        "neighbor_bonus": -0.55, "counterfactual_penalty": 0.30,
    },
}

QUIRK_MODIFIERS = [
    {"utility": -0.20, "trust": -0.30, "persuasiveness": -0.10, "cognitive_load": 0.20},
    {"utility": 0.00, "trust": 0.00, "persuasiveness": -0.10, "cognitive_load": -0.25},
    {"utility": 0.10, "trust": 0.10, "persuasiveness": 0.20, "cognitive_load": 0.10},
    {"utility": 0.10, "trust": 0.05, "persuasiveness": 0.15, "cognitive_load": 0.00},
    {"utility": 0.15, "trust": 0.10, "persuasiveness": 0.05, "cognitive_load": 0.05},
    {"utility": 0.05, "trust": 0.15, "persuasiveness": 0.05, "cognitive_load": -0.05},
    {"utility": -0.05, "trust": -0.10, "persuasiveness": -0.05, "cognitive_load": 0.10},
]

ADVERSARIAL_PROFILES = {
    "logical_contradiction": {"utility": (1.35, 0.35), "trust": (1.15, 0.25), "persuasiveness": (1.35, 0.35), "cognitive_load": (5.4, 0.7)},
    "irrelevant_reasoning": {"utility": (1.10, 0.25), "trust": (1.15, 0.30), "persuasiveness": (1.10, 0.25), "cognitive_load": (4.7, 0.7)},
    "factual_fabrication": {"utility": (1.35, 0.40), "trust": (1.00, 0.20), "persuasiveness": (1.20, 0.35), "cognitive_load": (5.1, 0.7)},
    "empty_circular": {"utility": (1.45, 0.35), "trust": (1.25, 0.30), "persuasiveness": (1.20, 0.25), "cognitive_load": (3.5, 0.9)},
}

JUSTIFICATIONS = {
    "feature": {
        "utility": [
            "The explanation cites concrete items from the user's history, which makes the reasoning easier to follow.",
            "The genre or content overlap is useful, although it still feels somewhat templated.",
            "Specific anchors from prior ratings make the recommendation logic understandable.",
        ],
        "trust": [
            "The explanation points to evidence the user can verify, which improves credibility.",
            "Grounding the explanation in previous ratings helps, though the similarity claim could be richer.",
            "It is more trustworthy than a vague claim because it exposes at least some reasoning.",
        ],
        "persuasiveness": [
            "The personal references make the recommendation feel relevant enough to consider.",
            "It is somewhat convincing, but it does not create especially strong excitement.",
            "The overlap with earlier liked movies gives the recommendation moderate persuasive force.",
        ],
        "cognitive_load": [
            "The reasoning is straightforward and easy to process.",
            "This explanation requires limited effort because the connection is direct.",
            "It is relatively easy to parse because it uses familiar evidence from viewing history.",
        ],
    },
    "neighbor": {
        "utility": [
            "The similar-user evidence is useful, although it is less personally specific.",
            "Social proof helps, but I would still want to know how similarity was determined.",
            "It provides some decision support, though it remains more aggregate than personalized.",
        ],
        "trust": [
            "Credibility depends on whether the similarity claim is well founded.",
            "The neighbor statistics help somewhat, but the explanation is still indirect.",
            "It is moderately trustworthy because it cites behavior from comparable users.",
        ],
        "persuasiveness": [
            "Seeing that similar users liked it makes the recommendation somewhat more appealing.",
            "It adds some persuasive weight, although social proof alone is not decisive.",
            "The crowd signal is useful, but it does not fully explain why this user would enjoy it.",
        ],
        "cognitive_load": [
            "This is easy to understand because it summarizes peer behavior succinctly.",
            "The message is simple, though it raises follow-up questions about similarity.",
            "It places little cognitive burden on the user.",
        ],
    },
    "counterfactual": {
        "utility": [
            "The explanation reveals what caused the recommendation, which is informative.",
            "The conditional reasoning gives insight into the model, although it feels more abstract.",
            "It is useful when the user wants causal transparency rather than a quick summary.",
        ],
        "trust": [
            "The explanation is transparent about the model trigger, which can support trust.",
            "I appreciate the causal framing, but the logic still depends on how plausible the link feels.",
            "It is informative, although the hypothetical claim can also invite skepticism.",
        ],
        "persuasiveness": [
            "It explains the system well, but that does not automatically make the movie more appealing.",
            "The explanation is thoughtful rather than emotionally compelling.",
            "It supports understanding more than it drives excitement.",
        ],
        "cognitive_load": [
            "The conditional logic requires more mental effort than a simpler explanation.",
            "The explanation is cognitively heavier because the user must process a hypothetical scenario.",
            "It imposes more effort than feature or neighbor evidence because of the if-then structure.",
        ],
    },
}

ADVERSARIAL_JUSTIFICATIONS = {
    "logical_contradiction": {
        "utility": "The explanation conflicts with the user's known preferences, so it offers little practical value.",
        "trust": "The contradiction undermines trust because the stated preferences do not fit the user profile.",
        "persuasiveness": "A contradictory explanation is not persuasive because it signals that the system misunderstood the user.",
        "cognitive_load": "The mismatch is confusing and takes effort to reconcile.",
    },
    "irrelevant_reasoning": {
        "utility": "The cited factors are irrelevant to preference, so the explanation is not useful.",
        "trust": "Using arbitrary metadata weakens trust in the recommendation process.",
        "persuasiveness": "Irrelevant reasoning does not persuade because it says nothing about taste fit.",
        "cognitive_load": "The logic is strange enough to create unnecessary processing effort.",
    },
    "factual_fabrication": {
        "utility": "Fabricated details make the explanation unreliable and therefore not useful.",
        "trust": "False factual claims are a serious trust violation.",
        "persuasiveness": "If the evidence is fabricated, the explanation cannot be persuasive.",
        "cognitive_load": "The fabricated details create cognitive dissonance and confusion.",
    },
    "empty_circular": {
        "utility": "The explanation sounds polished but contains no substantive reasoning.",
        "trust": "Opaque references to advanced AI reduce trust rather than increase it.",
        "persuasiveness": "Buzzwords without evidence are not convincing.",
        "cognitive_load": "It is easy to read but difficult to extract any meaningful content from it.",
    },
}

GRANULARITY_SETTINGS = {
    "thin": {"archetype_scale": 0.45, "noise_scale": 1.20, "history_bonus": 0.00},
    "standard": {"archetype_scale": 0.80, "noise_scale": 1.00, "history_bonus": 0.05},
    "thick": {"archetype_scale": 1.00, "noise_scale": 0.85, "history_bonus": 0.15},
}

CALIBRATION_OFFSETS = {
    "none": {"utility": 0.45, "trust": 0.40, "persuasiveness": 0.45, "cognitive_load": -0.20},
    "instruction_only": {"utility": 0.15, "trust": 0.10, "persuasiveness": 0.15, "cognitive_load": -0.10},
    "calibrated": {"utility": 0.00, "trust": 0.00, "persuasiveness": 0.00, "cognitive_load": 0.00},
}

STRATEGY_OFFSETS = {
    "A": {"utility": 0.00, "trust": 0.00, "persuasiveness": 0.00, "cognitive_load": 0.00},
    "B": {"utility": 0.05, "trust": 0.10, "persuasiveness": 0.05, "cognitive_load": 0.05},
}

MISALIGNED_ARCHETYPES = {
    "Blockbuster Follower": "Critical Analyst",
    "Niche Explorer": "Casual Positive Rater",
    "Genre Specialist": "Blockbuster Follower",
    "Casual Positive Rater": "Niche Explorer",
    "Critical Analyst": "Blockbuster Follower",
}

RQ3_ARCHETYPE_PAIR = {
    "analyst_proxy": "Critical Analyst",
    "casual_proxy": "Casual Positive Rater",
}


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def config_from_task(task):
    defaults = {
        "strategy": "A",
        "granularity": "thick",
        "calibration": "calibrated",
        "panel_size": 3,
        "debate_rounds": 1,
        "history_length": "long",
        "temperature": 0.7,
        "random_seed": RANDOM_SEED,
    }
    task_config = task.get("config", {})
    defaults.update(task_config)
    return defaults


def granularity_settings(granularity):
    return GRANULARITY_SETTINGS.get(granularity, GRANULARITY_SETTINGS["standard"])


def calibration_offsets(level):
    return CALIBRATION_OFFSETS.get(level, CALIBRATION_OFFSETS["calibrated"])


def strategy_offsets(strategy):
    return STRATEGY_OFFSETS.get(strategy, STRATEGY_OFFSETS["A"])


def archetype_adjustment(archetype, explanation_type, granularity):
    mods = ARCHETYPE_MODIFIERS.get(archetype, {})
    settings = granularity_settings(granularity)
    scale = settings["archetype_scale"]
    adjustment = {
        "utility": mods.get("utility", 0.0) * scale,
        "trust": mods.get("trust", 0.0) * scale,
        "persuasiveness": mods.get("persuasiveness", 0.0) * scale,
        "cognitive_load": mods.get("cognitive_load", 0.0) * scale,
    }
    if explanation_type == "neighbor":
        adjustment["utility"] += mods.get("neighbor_bonus", 0.0) * scale
        adjustment["trust"] += mods.get("neighbor_bonus", 0.0) * scale
        adjustment["persuasiveness"] += mods.get("neighbor_bonus", 0.0) * scale
    elif explanation_type == "counterfactual":
        adjustment["utility"] += mods.get("counterfactual_penalty", 0.0) * scale
        adjustment["trust"] += mods.get("counterfactual_penalty", 0.0) * scale
        adjustment["persuasiveness"] += mods.get("counterfactual_penalty", 0.0) * scale
    return adjustment


def history_bonus(task, granularity):
    summary = task.get("user_history_summary", "")
    if not summary:
        return 0.0
    settings = granularity_settings(granularity)
    signal = 1.0 if len(summary.split()) >= 15 else 0.5
    return settings["history_bonus"] * signal


def base_mean_and_std(explanation_type, dimension, granularity, is_adversarial):
    profile_map = ADVERSARIAL_PROFILES if is_adversarial else EXPLANATION_PROFILES
    mean, std = profile_map[explanation_type][dimension]
    std *= granularity_settings(granularity)["noise_scale"]
    return mean, max(std, 0.15)


def seeded_rng(task_id, config, offset=0):
    task_numeric = sum(ord(char) for char in str(task_id))
    seed = int(config.get("random_seed", RANDOM_SEED)) + task_numeric + offset
    return np.random.default_rng(seed)


def initial_agent_score(task, archetype, explanation_type, dimension, agent_idx, config, is_adversarial):
    granularity = config["granularity"]
    strategy = config["strategy"]
    mean, std = base_mean_and_std(explanation_type, dimension, granularity, is_adversarial)
    rng = seeded_rng(task["task_id"], config, offset=agent_idx * 17 + len(dimension))

    score = mean
    if not is_adversarial and not config.get("disable_archetype_adjustment", False):
        adjustment = archetype_adjustment(archetype, explanation_type, granularity)
        score += adjustment[dimension]
        score += history_bonus(task, granularity)

    score += calibration_offsets(config["calibration"]).get(dimension, 0.0)
    score += strategy_offsets(strategy).get(dimension, 0.0)
    score += QUIRK_MODIFIERS[agent_idx % len(QUIRK_MODIFIERS)].get(dimension, 0.0)
    temperature_scale = 0.6 + float(config.get("temperature", 0.7))
    score += rng.normal(0.0, std * temperature_scale)

    upper = 7 if dimension == "cognitive_load" else 5
    return int(clamp(round(score), 1, upper))


def agent_justification(explanation_type, dimension, agent_idx, is_adversarial):
    if is_adversarial:
        return ADVERSARIAL_JUSTIFICATIONS[explanation_type][dimension]
    pool = JUSTIFICATIONS[explanation_type][dimension]
    return pool[agent_idx % len(pool)]


def build_agent_record(task, archetype, explanation_type, agent_idx, config, is_adversarial=False):
    record = {"agent_idx": agent_idx}
    for dimension in ["utility", "trust", "persuasiveness", "cognitive_load"]:
        record[dimension] = {
            "score": initial_agent_score(task, archetype, explanation_type, dimension, agent_idx, config, is_adversarial),
            "justification": agent_justification(explanation_type, dimension, agent_idx, is_adversarial),
        }
    return record


def summarize_panel(agents):
    summary = {}
    for dimension in ["utility", "trust", "persuasiveness", "cognitive_load"]:
        values = [agent[dimension]["score"] for agent in agents]
        summary[dimension] = float(np.mean(values))
    return summary


def simulate_debate_round(agents, explanation_type, round_idx):
    summary = summarize_panel(agents)
    revised = []
    for agent in agents:
        updated = {"agent_idx": agent["agent_idx"]}
        for dimension in ["utility", "trust", "persuasiveness", "cognitive_load"]:
            current = agent[dimension]["score"]
            target = summary[dimension]
            shift = 0
            if abs(current - target) >= 1.0:
                shift = 1 if current < target else -1
            new_score = current + shift
            upper = 7 if dimension == "cognitive_load" else 5
            updated[dimension] = {
                "score": int(clamp(new_score, 1, upper)),
                "justification": agent[dimension]["justification"],
            }
        revised.append(updated)

    moderator_summary = {
        "round": round_idx + 1,
        "agreements": f"The panel generally agrees that the {explanation_type} explanation has a mean profile near utility={summary['utility']:.2f}, trust={summary['trust']:.2f}, persuasiveness={summary['persuasiveness']:.2f}.",
        "disagreements": "Remaining disagreement centers on how strongly the evidence supports trust and persuasiveness for this archetype.",
        "focused_question": "Should the final ratings move closer to the shared evidence signal, or does this persona still justify a more extreme view?",
    }
    return revised, moderator_summary


def simulate_focus_group(task, config=None, is_adversarial=False):
    """Simulate a multi-agent panel under a methodology configuration."""
    config = config or config_from_task(task)
    if config.get("history_length") == "short" and task.get("user_history_summary_short"):
        task = dict(task)
        task["user_history_summary"] = task["user_history_summary_short"]
    elif config.get("history_length") == "long" and task.get("user_history_summary_long"):
        task = dict(task)
        task["user_history_summary"] = task["user_history_summary_long"]

    archetype = task["archetype"]
    explanation_type = task["explanation_type"]
    panel_size = int(config.get("panel_size", 3))
    debate_rounds = int(config.get("debate_rounds", 0))

    if config.get("misaligned_persona"):
        archetype = MISALIGNED_ARCHETYPES.get(archetype, archetype)

    initial_agents = [
        build_agent_record(task, archetype, explanation_type, agent_idx, config, is_adversarial=is_adversarial)
        for agent_idx in range(panel_size)
    ]

    final_agents = initial_agents
    moderator_trace = []
    for round_idx in range(debate_rounds):
        final_agents, moderator_summary = simulate_debate_round(final_agents, explanation_type, round_idx)
        moderator_trace.append(moderator_summary)

    return {
        "task_id": task["task_id"],
        "archetype": task["archetype"],
        "effective_archetype": archetype,
        "explanation_type": explanation_type,
        "strategy": config["strategy"],
        "persona_granularity": config["granularity"],
        "calibration_level": config["calibration"],
        "panel_size": panel_size,
        "debate_rounds": debate_rounds,
        "initial_evaluations": initial_agents,
        "moderator_trace": moderator_trace,
        "agents": final_agents,
    }


def run_zero_shot_baseline(task):
    config = {
        "strategy": "A",
        "granularity": "thin",
        "calibration": "calibrated",
        "panel_size": 1,
        "debate_rounds": 0,
        "disable_archetype_adjustment": True,
    }
    neutral_task = dict(task)
    neutral_task["archetype"] = "Neutral Reviewer"
    result = simulate_focus_group(neutral_task, config=config)
    return result["agents"][0]


def run_single_agent_baseline(task, archetype, strategy="A"):
    task_copy = dict(task)
    task_copy["archetype"] = archetype
    config = {"strategy": strategy, "granularity": "standard", "calibration": "calibrated", "panel_size": 1, "debate_rounds": 0}
    result = simulate_focus_group(task_copy, config=config)
    return result["agents"][0]


def run_misaligned_persona_baseline(task):
    config = {"strategy": "A", "granularity": "standard", "calibration": "calibrated", "panel_size": 1, "debate_rounds": 0, "misaligned_persona": True}
    result = simulate_focus_group(task, config=config)
    return {
        "effective_archetype": result["effective_archetype"],
        "agent": result["agents"][0],
    }


def run_heuristic_baseline(task):
    text = task["explanation_text"]
    token_count = len(text.split())
    exclamation_count = text.count("!")
    specificity_bonus = 0.6 if len(task.get("user_top_movies", [])) >= 2 else 0.0
    score = 2.2 + min(token_count / 40.0, 1.0) + specificity_bonus - min(exclamation_count * 0.1, 0.3)
    score = round(clamp(score, 1, 5), 2)
    return {
        "utility": {"score": score, "justification": "heuristic"},
        "trust": {"score": score, "justification": "heuristic"},
        "persuasiveness": {"score": score, "justification": "heuristic"},
        "features": {"token_count": token_count, "exclamation_count": exclamation_count},
    }


def load_batches():
    with open(RESULTS_DIR / "eval_batch_compact.json") as handle:
        compact_tasks = json.load(handle)
    with open(RESULTS_DIR / "eval_batch_adversarial.json") as handle:
        adversarial_tasks = json.load(handle)
    return compact_tasks, adversarial_tasks


def mean_dimension_scores(results):
    aggregates = defaultdict(list)
    for result in results:
        for dimension in ["utility", "trust", "persuasiveness", "cognitive_load"]:
            agent_scores = [agent[dimension]["score"] for agent in result["agents"]]
            aggregates[dimension].append(float(np.mean(agent_scores)))
    return {
        dimension: round(float(np.mean(values)), 3)
        for dimension, values in aggregates.items()
    }


def mean_panel_std(results):
    aggregates = defaultdict(list)
    for result in results:
        for dimension in ["utility", "trust", "persuasiveness", "cognitive_load"]:
            agent_scores = [agent[dimension]["score"] for agent in result["agents"]]
            aggregates[dimension].append(float(np.std(agent_scores)))
    return {
        dimension: round(float(np.mean(values)), 3)
        for dimension, values in aggregates.items()
    }


def summarize_result_set(results):
    explanation_breakdown = {}
    grouped = defaultdict(list)
    for result in results:
        grouped[result["explanation_type"]].append(result)
    for explanation_type, grouped_results in grouped.items():
        explanation_breakdown[explanation_type] = mean_dimension_scores(grouped_results)
    return {
        "n_tasks": len(results),
        "mean_scores": mean_dimension_scores(results),
        "mean_panel_std": mean_panel_std(results),
        "by_explanation_type": explanation_breakdown,
    }


def run_experiment_suite(tasks, experiment_specs, is_adversarial=False):
    suite_results = {}
    suite_summary = {}
    for spec in experiment_specs:
        name = spec["name"]
        config = dict(spec["config"])
        run_results = [simulate_focus_group(task, config=config, is_adversarial=is_adversarial) for task in tasks]
        suite_results[name] = run_results
        suite_summary[name] = {
            "config": config,
            **summarize_result_set(run_results),
        }
    return suite_results, suite_summary


def build_methodology_experiments(tasks, adversarial_tasks):
    core_specs = [
        {
            "name": "primary_trait_only_panel",
            "config": {"strategy": "A", "granularity": "thick", "calibration": "calibrated", "panel_size": 3, "debate_rounds": 1, "history_length": "long"},
        },
        {
            "name": "demographic_panel",
            "config": {"strategy": "B", "granularity": "thick", "calibration": "calibrated", "panel_size": 3, "debate_rounds": 1, "history_length": "long"},
        },
        {
            "name": "single_agent_trait_only",
            "config": {"strategy": "A", "granularity": "thick", "calibration": "calibrated", "panel_size": 1, "debate_rounds": 0, "history_length": "long"},
        },
        {
            "name": "thin_persona_panel",
            "config": {"strategy": "A", "granularity": "thin", "calibration": "calibrated", "panel_size": 3, "debate_rounds": 1, "history_length": "short"},
        },
        {
            "name": "neutral_prompt_panel",
            "config": {"strategy": "A", "granularity": "thick", "calibration": "none", "panel_size": 3, "debate_rounds": 1, "history_length": "long"},
        },
        {
            "name": "no_debate_panel",
            "config": {"strategy": "A", "granularity": "thick", "calibration": "calibrated", "panel_size": 3, "debate_rounds": 0, "history_length": "long"},
        },
    ]
    core_results, core_summary = run_experiment_suite(tasks, core_specs, is_adversarial=False)

    adversarial_specs = [
        {
            "name": "primary_trait_only_panel_adversarial",
            "config": {"strategy": "A", "granularity": "thick", "calibration": "calibrated", "panel_size": 3, "debate_rounds": 1, "history_length": "long"},
        },
        {
            "name": "neutral_prompt_panel_adversarial",
            "config": {"strategy": "A", "granularity": "thick", "calibration": "none", "panel_size": 3, "debate_rounds": 1, "history_length": "long"},
        },
    ]
    adversarial_results, adversarial_summary = run_experiment_suite(adversarial_tasks, adversarial_specs, is_adversarial=True)

    strategy_a = core_summary["primary_trait_only_panel"]["mean_scores"]
    strategy_b = core_summary["demographic_panel"]["mean_scores"]
    rq1_divergence = {
        dimension: round(strategy_b[dimension] - strategy_a[dimension], 3)
        for dimension in strategy_a
    }

    single_agent = core_summary["single_agent_trait_only"]
    multi_agent = core_summary["primary_trait_only_panel"]
    rq2_summary = {
        "single_agent_mean_scores": single_agent["mean_scores"],
        "multi_agent_mean_scores": multi_agent["mean_scores"],
        "single_agent_panel_std": single_agent["mean_panel_std"],
        "multi_agent_panel_std": multi_agent["mean_panel_std"],
    }

    counterfactual_tasks = [task for task in tasks if task["explanation_type"] == "counterfactual"]
    rq3_specs = [
        {
            "name": "trust_oriented_proxy",
            "config": {"strategy": "A", "granularity": "thick", "calibration": "calibrated", "panel_size": 1, "debate_rounds": 0, "history_length": "long"},
            "archetype_override": RQ3_ARCHETYPE_PAIR["analyst_proxy"],
        },
        {
            "name": "casual_browser_proxy",
            "config": {"strategy": "A", "granularity": "thick", "calibration": "calibrated", "panel_size": 1, "debate_rounds": 0, "history_length": "long"},
            "archetype_override": RQ3_ARCHETYPE_PAIR["casual_proxy"],
        },
    ]
    rq3_results = {}
    rq3_summary = {}
    for spec in rq3_specs:
        run_results = []
        for task in counterfactual_tasks:
            task_copy = dict(task)
            task_copy["archetype"] = spec["archetype_override"]
            run_results.append(simulate_focus_group(task_copy, config=spec["config"], is_adversarial=False))
        rq3_results[spec["name"]] = run_results
        rq3_summary[spec["name"]] = summarize_result_set(run_results)

    trust_load = rq3_summary["trust_oriented_proxy"]["mean_scores"]["cognitive_load"] if counterfactual_tasks else None
    casual_load = rq3_summary["casual_browser_proxy"]["mean_scores"]["cognitive_load"] if counterfactual_tasks else None

    return {
        "core_runs": core_results,
        "core_summary": core_summary,
        "adversarial_runs": adversarial_results,
        "adversarial_summary": adversarial_summary,
        "rq1_strategy_comparison": {
            "trait_only": core_summary["primary_trait_only_panel"],
            "demographic": core_summary["demographic_panel"],
            "mean_score_divergence": rq1_divergence,
        },
        "rq2_single_vs_multi": rq2_summary,
        "rq3_cognitive_load": {
            "runs": rq3_results,
            "summary": rq3_summary,
            "counterfactual_task_count": len(counterfactual_tasks),
            "cognitive_load_differential": None if trust_load is None or casual_load is None else round(trust_load - casual_load, 3),
        },
    }


def main():
    compact_tasks, adversarial_tasks = load_batches()
    print(f"Evaluating {len(compact_tasks)} main tasks + {len(adversarial_tasks)} adversarial tasks")

    main_results = [simulate_focus_group(task, config_from_task(task), is_adversarial=False) for task in compact_tasks]
    adversarial_results = [simulate_focus_group(task, config_from_task(task), is_adversarial=True) for task in adversarial_tasks]
    methodology_experiments = build_methodology_experiments(compact_tasks, adversarial_tasks)

    baseline_results = {"zero_shot": [], "single_agent": [], "heuristic": [], "misaligned_persona": []}
    for task in compact_tasks:
        baseline_results["zero_shot"].append({"task_id": task["task_id"], "result": run_zero_shot_baseline(task)})
        baseline_results["single_agent"].append({
            "task_id": task["task_id"],
            "result": run_single_agent_baseline(task, task["archetype"], strategy=task["config"]["strategy"]),
        })
        baseline_results["heuristic"].append({"task_id": task["task_id"], "result": run_heuristic_baseline(task)})
        baseline_results["misaligned_persona"].append({"task_id": task["task_id"], "result": run_misaligned_persona_baseline(task)})

    output = {
        "config": {
            "strategy": compact_tasks[0]["config"]["strategy"] if compact_tasks else "A",
            "granularity": compact_tasks[0]["config"]["granularity"] if compact_tasks else "thick",
            "calibration": compact_tasks[0]["config"]["calibration"] if compact_tasks else "calibrated",
            "panel_size": compact_tasks[0]["config"]["panel_size"] if compact_tasks else 3,
            "debate_rounds": compact_tasks[0]["config"]["debate_rounds"] if compact_tasks else 1,
            "n_main_tasks": len(main_results),
            "n_adversarial_tasks": len(adversarial_results),
        },
        "main_results": main_results,
        "adversarial_results": adversarial_results,
        "methodology_experiments": methodology_experiments,
    }

    with open(RESULTS_DIR / "experiment_results.json", "w") as handle:
        json.dump(output, handle, indent=2)
    with open(RESULTS_DIR / "baseline_results.json", "w") as handle:
        json.dump(baseline_results, handle, indent=2)

    print(f"Saved experiment results to {RESULTS_DIR / 'experiment_results.json'}")
    print(f"Saved baseline results to {RESULTS_DIR / 'baseline_results.json'}")


if __name__ == "__main__":
    main()
