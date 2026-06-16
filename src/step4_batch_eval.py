"""Step 4a: Build stratified evaluation batches for live or simulated panel runs."""
import json
import random
from collections import Counter, defaultdict

from config import (
    PRIMARY_ADVERSARIAL_COUNT,
    PRIMARY_CALIBRATION_LEVEL,
    PRIMARY_DEBATE_ROUNDS,
    PRIMARY_PANEL_SIZE,
    PRIMARY_PERSONA_GRANULARITY,
    PRIMARY_PERSONA_STRATEGY,
    RANDOM_SEED,
    RESULTS_DIR,
    TASKS_PER_ARCHETYPE_TYPE,
)
from step3_personas import (
    CALIBRATION_INSTRUCTION_ONLY,
    CALIBRATION_NONE,
    CALIBRATION_WITH_EXAMPLES,
    build_persona_prompt_strategy_a,
    build_persona_prompt_strategy_b,
)

random.seed(RANDOM_SEED)


def sample_tasks(tasks, tasks_per_cell=TASKS_PER_ARCHETYPE_TYPE):
    """Stratified sample with exact balance by archetype and explanation type."""
    grouped = defaultdict(list)
    for task in tasks:
        grouped[(task["archetype"], task["explanation_type"])].append(task)

    sampled = []
    for key, candidates in grouped.items():
        random.shuffle(candidates)
        sampled.extend(candidates[:tasks_per_cell])
    return sampled


def calibration_text(level):
    mapping = {
        "none": CALIBRATION_NONE,
        "instruction_only": CALIBRATION_INSTRUCTION_ONLY,
        "calibrated": CALIBRATION_WITH_EXAMPLES,
    }
    return mapping.get(level, CALIBRATION_WITH_EXAMPLES)


def persona_builder(strategy):
    return build_persona_prompt_strategy_a if strategy == "A" else build_persona_prompt_strategy_b


def compact_persona_block(archetype, strategy, granularity, agent_idx, user_history_summary):
    builder = persona_builder(strategy)
    return builder(
        archetype,
        quirk_idx=agent_idx,
        granularity=granularity,
        user_history_summary=user_history_summary,
    )


def history_summary_for_task(task, history_length="long"):
    if history_length == "short":
        return task.get("user_history_summary_short") or task.get("user_history_summary")
    return task.get("user_history_summary_long") or task.get("user_history_summary")


def generate_full_batch(tasks, strategy, granularity, calibration, panel_size, debate_rounds, history_length="long", temperature=0.7, random_seed=RANDOM_SEED):
    batch = []
    cal_text = calibration_text(calibration)
    builder = persona_builder(strategy)

    for task in tasks:
        chosen_history = history_summary_for_task(task, history_length)
        agents = []
        for agent_idx in range(panel_size):
            agents.append({
                "agent_idx": agent_idx,
                "system_prompt": builder(
                    task["archetype"],
                    quirk_idx=agent_idx,
                    granularity=granularity,
                    user_history_summary=chosen_history,
                ),
                "user_prompt": (
                    f"Now evaluate the following recommendation explanation on these dimensions.\n\n"
                    f"**Recommended Movie:** {task['recommended_movie']}\n"
                    f"**Explanation:** \"{task['explanation_text']}\"\n\n"
                    f"Rate each dimension on a 1-5 Likert scale AND provide a brief textual justification (2-3 sentences).\n\n"
                    f"{cal_text}\n\n"
                    "Respond in this exact JSON format:\n"
                    "{\n"
                    "  \"utility\": {\"score\": <1-5>, \"justification\": \"<why>\"},\n"
                    "  \"trust\": {\"score\": <1-5>, \"justification\": \"<why>\"},\n"
                    "  \"persuasiveness\": {\"score\": <1-5>, \"justification\": \"<why>\"},\n"
                    "  \"cognitive_load\": {\"score\": <1-7>, \"justification\": \"<why>\"}\n"
                    "}\n"
                ),
            })

        batch.append({
            "task_id": task["task_id"],
            "archetype": task["archetype"],
            "explanation_type": task["explanation_type"],
            "recommended_movie": task["recommended_movie"],
            "explanation_text": task["explanation_text"],
            "user_history_summary": chosen_history,
            "config": {
                "strategy": strategy,
                "granularity": granularity,
                "calibration": calibration,
                "panel_size": panel_size,
                "debate_rounds": debate_rounds,
                "history_length": history_length,
                "temperature": temperature,
                "random_seed": random_seed,
            },
            "agents": agents,
        })
    return batch


def generate_compact_batch(tasks, strategy, granularity, calibration, panel_size, debate_rounds, history_length="long", temperature=0.7, random_seed=RANDOM_SEED):
    """Generate one prompt per task that role-plays the whole panel."""
    cal_text = calibration_text(calibration)
    batch = []

    for task in tasks:
        chosen_history = history_summary_for_task(task, history_length)
        persona_blocks = []
        for agent_idx in range(panel_size):
            persona_blocks.append(
                f"Agent {agent_idx + 1} persona:\n"
                f"{compact_persona_block(task['archetype'], strategy, granularity, agent_idx, chosen_history)}"
            )

        agent_schema = []
        for agent_idx in range(panel_size):
            agent_schema.append(
                "    {\n"
                f"      \"agent_idx\": {agent_idx},\n"
                "      \"utility\": {\"score\": <1-5>, \"justification\": \"...\"},\n"
                "      \"trust\": {\"score\": <1-5>, \"justification\": \"...\"},\n"
                "      \"persuasiveness\": {\"score\": <1-5>, \"justification\": \"...\"},\n"
                "      \"cognitive_load\": {\"score\": <1-7>, \"justification\": \"...\"}\n"
                "    }"
            )

        prompt = (
            f"You are simulating a focus group of {panel_size} evaluators who share the same core user archetype.\n\n"
            f"{chr(10).join(persona_blocks)}\n\n"
            f"**Recommended Movie:** {task['recommended_movie']}\n"
            f"**Explanation Type:** {task['explanation_type']}\n"
            f"**Explanation:** \"{task['explanation_text']}\"\n\n"
            f"{cal_text}\n\n"
            "For EACH agent, rate the explanation on these dimensions (1-5 Likert) with a 1-sentence justification.\n"
            "Also rate cognitive load (1-7 scale).\n\n"
            "Respond in this exact JSON format:\n"
            "{\n"
            f"  \"task_id\": \"{task['task_id']}\",\n"
            "  \"agents\": [\n"
            f"{',\n'.join(agent_schema)}\n"
            "  ]\n"
            "}\n"
        )

        batch.append({
            "task_id": task["task_id"],
            "archetype": task["archetype"],
            "explanation_type": task["explanation_type"],
            "recommended_movie": task["recommended_movie"],
            "explanation_text": task["explanation_text"],
            "user_history_summary": chosen_history,
            "config": {
                "strategy": strategy,
                "granularity": granularity,
                "calibration": calibration,
                "panel_size": panel_size,
                "debate_rounds": debate_rounds,
                "history_length": history_length,
                "temperature": temperature,
                "random_seed": random_seed,
            },
            "prompt": prompt,
        })
    return batch


def build_adversarial_tasks(adversarial_items, count=PRIMARY_ADVERSARIAL_COUNT):
    tasks = []
    for item in adversarial_items[:count]:
        tasks.append({
            "task_id": item["id"],
            "archetype": "Critical Analyst",
            "explanation_type": item["category"],
            "recommended_movie": item["movie"],
            "explanation_text": item["explanation"],
            "user_history_summary": (
                "You are a critical, analytical movie watcher who distrusts vague, contradictory, or fabricated reasoning."
            ),
        })
    return tasks


def main():
    with open(RESULTS_DIR / "evaluation_tasks.json") as handle:
        all_tasks = json.load(handle)
    with open(RESULTS_DIR / "adversarial_explanations.json") as handle:
        adversarial_items = json.load(handle)

    sampled = sample_tasks(all_tasks)
    print(f"Sampled {len(sampled)} primary tasks from {len(all_tasks)} total tasks")

    archetype_counts = Counter(task["archetype"] for task in sampled)
    explanation_counts = Counter(task["explanation_type"] for task in sampled)
    print(f"  By archetype: {dict(archetype_counts)}")
    print(f"  By explanation type: {dict(explanation_counts)}")

    compact = generate_compact_batch(
        sampled,
        strategy=PRIMARY_PERSONA_STRATEGY,
        granularity=PRIMARY_PERSONA_GRANULARITY,
        calibration=PRIMARY_CALIBRATION_LEVEL,
        panel_size=PRIMARY_PANEL_SIZE,
        debate_rounds=PRIMARY_DEBATE_ROUNDS,
        history_length="long",
    )
    with open(RESULTS_DIR / "eval_batch_compact.json", "w") as handle:
        json.dump(compact, handle, indent=2)
    print(f"Saved compact evaluation batch with {len(compact)} tasks")

    full = generate_full_batch(
        sampled,
        strategy=PRIMARY_PERSONA_STRATEGY,
        granularity=PRIMARY_PERSONA_GRANULARITY,
        calibration=PRIMARY_CALIBRATION_LEVEL,
        panel_size=PRIMARY_PANEL_SIZE,
        debate_rounds=PRIMARY_DEBATE_ROUNDS,
        history_length="long",
    )
    with open(RESULTS_DIR / "eval_batch_full.json", "w") as handle:
        json.dump(full, handle, indent=2)
    print(f"Saved full evaluation batch with {len(full)} tasks")

    adversarial_tasks = build_adversarial_tasks(adversarial_items)
    adversarial_batch = generate_compact_batch(
        adversarial_tasks,
        strategy=PRIMARY_PERSONA_STRATEGY,
        granularity=PRIMARY_PERSONA_GRANULARITY,
        calibration=PRIMARY_CALIBRATION_LEVEL,
        panel_size=PRIMARY_PANEL_SIZE,
        debate_rounds=PRIMARY_DEBATE_ROUNDS,
        history_length="long",
    )
    with open(RESULTS_DIR / "eval_batch_adversarial.json", "w") as handle:
        json.dump(adversarial_batch, handle, indent=2)
    print(f"Saved adversarial evaluation batch with {len(adversarial_batch)} tasks")


if __name__ == "__main__":
    main()
