"""Generate single-paste prompt files — one file = one prompt, all 19 tasks at once."""
import json
from pathlib import Path
from config import RESULTS_DIR, RANDOM_SEED

OUTPUT_DIR = Path("prompts")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_tasks():
    with open(RESULTS_DIR / "eval_batch_compact.json") as f:
        main_tasks = json.load(f)
    with open(RESULTS_DIR / "eval_batch_adversarial.json") as f:
        adv_tasks = json.load(f)
    return main_tasks, adv_tasks


def select_trimmed(main_tasks, adv_tasks):
    cells = {}
    for task in main_tasks:
        key = (task["archetype"], task["explanation_type"])
        if key not in cells:
            cells[key] = task
    selected_main = sorted(cells.values(), key=lambda t: (t["archetype"], t["explanation_type"]))

    flaw_cells = {}
    for task in adv_tasks:
        flaw = task.get("explanation_type", "unknown")
        if flaw not in flaw_cells:
            flaw_cells[flaw] = task
    selected_adv = list(flaw_cells.values())

    return selected_main, selected_adv


def build_oneshot(main_tasks, adv_tasks):
    all_tasks = main_tasks + adv_tasks

    task_blocks = []
    for i, task in enumerate(all_tasks, 1):
        is_adv = task["task_id"].startswith("ADV")
        tag = " [ADVERSARIAL PROBE]" if is_adv else ""
        archetype = task.get("archetype", "Critical Analyst")
        exp_type = task.get("explanation_type", "unknown")

        persona_prompt = task.get("prompt", "")

        block = f"""--- TASK {i}/{len(all_tasks)}: {task["task_id"]} [{archetype} / {exp_type}]{tag} ---

{persona_prompt}"""
        task_blocks.append(block)

    tasks_text = "\n\n".join(task_blocks)

    prompt = f"""You are participating in a research study evaluating recommendation explanations using persona-conditioned LLM panels.

Below are {len(all_tasks)} evaluation tasks. For EACH task, you simulate a 3-agent focus group panel. Each agent shares the same user archetype but has a different personality quirk. Each agent rates the explanation on 4 dimensions.

CRITICAL INSTRUCTIONS:
- Respond with ONLY a single JSON array containing {len(all_tasks)} objects — one per task.
- No markdown fences, no commentary, no extra text before or after the JSON.
- Use the full rating scale critically. Most scores should be 2-4, not uniformly high.
- Evaluate each task independently based on the persona described.
- The last {len(adv_tasks)} tasks are adversarial probes with deliberately flawed explanations — rate them harshly (1-2 range).

RESPONSE FORMAT — one JSON array:
[
  {{
    "task_id": "TASK-XXXX",
    "agents": [
      {{
        "agent_idx": 0,
        "utility": {{"score": <1-5>, "justification": "..."}},
        "trust": {{"score": <1-5>, "justification": "..."}},
        "persuasiveness": {{"score": <1-5>, "justification": "..."}},
        "cognitive_load": {{"score": <1-7>, "justification": "..."}}
      }},
      {{
        "agent_idx": 1,
        "utility": {{"score": <1-5>, "justification": "..."}},
        "trust": {{"score": <1-5>, "justification": "..."}},
        "persuasiveness": {{"score": <1-5>, "justification": "..."}},
        "cognitive_load": {{"score": <1-7>, "justification": "..."}}
      }},
      {{
        "agent_idx": 2,
        "utility": {{"score": <1-5>, "justification": "..."}},
        "trust": {{"score": <1-5>, "justification": "..."}},
        "persuasiveness": {{"score": <1-5>, "justification": "..."}},
        "cognitive_load": {{"score": <1-7>, "justification": "..."}}
      }}
    ]
  }},
  ... (repeat for all {len(all_tasks)} tasks)
]

HERE ARE THE {len(all_tasks)} TASKS:

{tasks_text}

NOW RESPOND WITH THE JSON ARRAY OF {len(all_tasks)} EVALUATION OBJECTS. NO OTHER TEXT."""

    return prompt


def main():
    main_tasks, adv_tasks = load_tasks()
    selected_main, selected_adv = select_trimmed(main_tasks, adv_tasks)
    total = len(selected_main) + len(selected_adv)
    print(f"Building one-shot prompts: {len(selected_main)} main + {len(selected_adv)} adversarial = {total} tasks")

    prompt = build_oneshot(selected_main, selected_adv)

    for model_key, model_name in [
        ("chatgpt", "ChatGPT (GPT-4o)"),
        ("claude", "Claude 3.5 Sonnet"),
        ("llama3", "Llama-3"),
    ]:
        outpath = OUTPUT_DIR / f"oneshot_{model_key}.txt"
        with open(outpath, "w", encoding="utf-8") as f:
            f.write(prompt)
        print(f"  {outpath} — {len(prompt):,} chars")

    print(f"\nUSAGE:")
    print(f"  1. Copy the ENTIRE contents of oneshot_chatgpt.txt")
    print(f"  2. Paste into ChatGPT as ONE message")
    print(f"  3. It responds with one JSON array of {total} objects")
    print(f"  4. Save the JSON array to: results/live_responses_chatgpt.jsonl")
    print(f"     (put each object on its own line, OR save the whole array as .json)")
    print(f"  5. Repeat for claude and llama3")
    print(f"\n  NOTE: If a model truncates the output, ask 'continue from where you stopped'")
    print(f"  and stitch the JSON together.")


if __name__ == "__main__":
    main()
