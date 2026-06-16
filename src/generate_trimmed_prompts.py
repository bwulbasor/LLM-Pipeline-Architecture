"""Generate trimmed 19-task prompt files for manual cross-model validation."""
import json
import random
from pathlib import Path
from config import RESULTS_DIR, RANDOM_SEED

random.seed(RANDOM_SEED)

OUTPUT_DIR = Path("prompts")
OUTPUT_DIR.mkdir(exist_ok=True)

SYSTEM_PREAMBLE = """You are participating in a research study evaluating recommendation explanations.
For each task, you receive a persona description and an explanation to evaluate.
Respond with ONLY valid JSON — no commentary, no markdown fences, no extra text.
Evaluate each task independently. Use the full rating scale critically.
Most scores should fall in the 2-4 range."""


def load_tasks():
    with open(RESULTS_DIR / "eval_batch_compact.json") as f:
        main_tasks = json.load(f)
    with open(RESULTS_DIR / "eval_batch_adversarial.json") as f:
        adv_tasks = json.load(f)
    return main_tasks, adv_tasks


def select_trimmed_subset(main_tasks, adv_tasks):
    """Select 15 main (1 per archetype x explanation_type cell) + 4 adversarial (1 per flaw type)."""
    cells = {}
    for task in main_tasks:
        key = (task["archetype"], task["explanation_type"])
        if key not in cells:
            cells[key] = task

    selected_main = list(cells.values())
    selected_main.sort(key=lambda t: (t["archetype"], t["explanation_type"]))

    flaw_cells = {}
    for task in adv_tasks:
        flaw = task.get("explanation_type", "unknown")
        if flaw not in flaw_cells:
            flaw_cells[flaw] = task

    selected_adv = list(flaw_cells.values())

    return selected_main, selected_adv


def generate_file(model_key, model_name, main_tasks, adv_tasks):
    all_tasks = [(t, False) for t in main_tasks] + [(t, True) for t in adv_tasks]
    total = len(all_tasks)

    lines = []
    lines.append(f"""================================================================================
MODEL: {model_name}
CROSS-MODEL VALIDATION — {len(main_tasks)} main + {len(adv_tasks)} adversarial = {total} tasks
================================================================================

HOW TO USE:
1. Start a NEW conversation with {model_name}.
2. Paste the system preamble, then paste each task one at a time.
3. The model returns a JSON block for each task.
4. Copy each JSON response into: results/live_responses_{model_key}.jsonl
   (one JSON object per line, in task order)

SYSTEM PREAMBLE (paste this as the first message):
---
{SYSTEM_PREAMBLE}
---

""")

    for i, (task, is_adv) in enumerate(all_tasks, 1):
        adv_tag = " [ADVERSARIAL]" if is_adv else ""
        archetype = task.get("archetype", "Critical Analyst")
        exp_type = task.get("explanation_type", task.get("flaw_type", "unknown"))

        lines.append(f"""
================================================================================
TASK {i}/{total} — {task["task_id"]} [{archetype} / {exp_type}]{adv_tag}
================================================================================

""")
        lines.append(task["prompt"])
        lines.append("\n")

    outpath = OUTPUT_DIR / f"trimmed_{model_key}.txt"
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    print(f"  {outpath} — {total} tasks, {len(''.join(lines)):,} chars")


def save_task_manifest(main_tasks, adv_tasks):
    """Save which tasks were selected for the trimmed run."""
    manifest = {
        "description": "Trimmed cross-model validation subset",
        "n_main": len(main_tasks),
        "n_adversarial": len(adv_tasks),
        "main_task_ids": [t["task_id"] for t in main_tasks],
        "adversarial_task_ids": [t["task_id"] for t in adv_tasks],
        "cells_covered": [
            {"archetype": t["archetype"], "explanation_type": t["explanation_type"]}
            for t in main_tasks
        ],
    }
    outpath = RESULTS_DIR / "trimmed_manifest.json"
    with open(outpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  {outpath}")


def main():
    main_tasks, adv_tasks = load_tasks()
    print(f"Full batch: {len(main_tasks)} main + {len(adv_tasks)} adversarial")

    selected_main, selected_adv = select_trimmed_subset(main_tasks, adv_tasks)
    print(f"Trimmed to: {len(selected_main)} main + {len(selected_adv)} adversarial = {len(selected_main) + len(selected_adv)} total")
    print(f"\nCoverage:")
    for t in selected_main:
        print(f"  {t['archetype']:25s} x {t['explanation_type']:15s} -> {t['task_id']}")
    print(f"  + {len(selected_adv)} adversarial probes")

    print(f"\nGenerating prompt files:")
    for model_key, model_name in [
        ("chatgpt", "ChatGPT (GPT-4o)"),
        ("claude", "Claude 3.5 Sonnet"),
        ("llama3", "Llama-3 (via Groq/HuggingChat/Ollama)"),
    ]:
        generate_file(model_key, model_name, selected_main, selected_adv)

    print(f"\nSaving manifest:")
    save_task_manifest(selected_main, selected_adv)

    print(f"\n{'='*60}")
    print(f"19 tasks per model x 3 models = 57 total interactions")
    print(f"Estimated time: ~1 hour (or ~20 min per model)")
    print(f"")
    print(f"Files:")
    print(f"  prompts/trimmed_chatgpt.txt")
    print(f"  prompts/trimmed_claude.txt")
    print(f"  prompts/trimmed_llama3.txt")
    print(f"  results/trimmed_manifest.json")
    print(f"")
    print(f"After collecting responses, run:")
    print(f"  python src/parse_live_responses.py chatgpt")
    print(f"  python src/parse_live_responses.py claude")
    print(f"  python src/parse_live_responses.py llama3")
    print(f"  python src/parse_live_responses.py --merge")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
