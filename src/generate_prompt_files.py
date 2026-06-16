"""Generate copy-paste prompt files for manual LLM evaluation on 3 models."""
import json
from pathlib import Path
from config import RESULTS_DIR

OUTPUT_DIR = Path("prompts")
OUTPUT_DIR.mkdir(exist_ok=True)

MODELS = ["chatgpt", "claude", "llama3"]

SYSTEM_PREAMBLE = """You are participating in a research study evaluating recommendation explanations.
For each task below, you will receive a persona description and an explanation to evaluate.
You must respond with ONLY valid JSON for each task — no commentary, no markdown fences.
Evaluate each task independently. Do not let earlier evaluations influence later ones.
Use the full rating scale critically. Most scores should fall in the 2-4 range."""

BATCH_HEADER = """================================================================================
MODEL: {model_name}
PERSONA-CONDITIONED EVALUATION — {n_main} main tasks + {n_adv} adversarial tasks
================================================================================

INSTRUCTIONS FOR THE OPERATOR:
1. Start a NEW conversation with {model_name}.
2. Paste the system preamble below first, then paste tasks one at a time
   (or in batches of ~5 if the model can handle it).
3. For each task, the model should return a JSON block.
4. Save ALL JSON responses into: results/live_responses_{model_key}.jsonl
   (one JSON object per line, in task order)

SYSTEM PREAMBLE (paste this first):
---
{preamble}
---

"""

TASK_SEPARATOR = """
================================================================================
TASK {task_num}/{total} — {task_id} [{archetype} / {explanation_type}]{adv_tag}
================================================================================

"""


def load_tasks():
    with open(RESULTS_DIR / "eval_batch_compact.json") as f:
        main_tasks = json.load(f)
    with open(RESULTS_DIR / "eval_batch_adversarial.json") as f:
        adv_tasks = json.load(f)
    return main_tasks, adv_tasks


def simplify_prompt(task, is_adversarial=False):
    """Use the pre-built prompt from the batch file."""
    return task["prompt"]


def generate_file(model_key, model_name, main_tasks, adv_tasks):
    all_tasks = [(t, False) for t in main_tasks] + [(t, True) for t in adv_tasks]
    total = len(all_tasks)

    lines = []
    lines.append(BATCH_HEADER.format(
        model_name=model_name,
        n_main=len(main_tasks),
        n_adv=len(adv_tasks),
        model_key=model_key,
        preamble=SYSTEM_PREAMBLE,
    ))

    for i, (task, is_adv) in enumerate(all_tasks, 1):
        adv_tag = " [ADVERSARIAL]" if is_adv else ""
        archetype = task.get("archetype", "Critical Analyst")
        explanation_type = task.get("explanation_type", task.get("flaw_type", "unknown"))

        lines.append(TASK_SEPARATOR.format(
            task_num=i,
            total=total,
            task_id=task["task_id"],
            archetype=archetype,
            explanation_type=explanation_type,
            adv_tag=adv_tag,
        ))
        lines.append(task["prompt"])
        lines.append("\n")

    outpath = OUTPUT_DIR / f"prompts_{model_key}.txt"
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    print(f"  {outpath} — {total} tasks, {len(''.join(lines)):,} chars")
    return outpath


def generate_response_collector():
    """Generate a script to parse collected responses into experiment format."""
    script = '''"""Parse manually collected LLM responses into experiment_results format.

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
    """Read a JSONL file of responses and structure them."""
    inpath = RESULTS_DIR / f"live_responses_{model_key}.jsonl"
    if not inpath.exists():
        print(f"ERROR: {inpath} not found. Save your responses there first.")
        return None

    responses = []
    with open(inpath, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                responses.append(obj)
            except json.JSONDecodeError as e:
                print(f"  WARNING: line {line_num} is not valid JSON: {e}")
                print(f"  Content: {line[:200]}...")

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
'''

    outpath = Path("src") / "parse_live_responses.py"
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(script)
    print(f"  {outpath} — response parser ready")


def main():
    main_tasks, adv_tasks = load_tasks()
    print(f"Loaded {len(main_tasks)} main + {len(adv_tasks)} adversarial tasks")
    print(f"Each task has a pre-built prompt with 3-agent panel instructions\n")

    model_info = [
        ("chatgpt", "ChatGPT (GPT-4o)"),
        ("claude", "Claude 3.5 Sonnet"),
        ("llama3", "Llama-3 (via Groq/HuggingChat/Ollama)"),
    ]

    print("Generating prompt files:")
    for model_key, model_name in model_info:
        generate_file(model_key, model_name, main_tasks, adv_tasks)

    print("\nGenerating response parser:")
    generate_response_collector()

    print(f"\n{'='*60}")
    print("WORKFLOW:")
    print("1. Open prompts/prompts_chatgpt.txt")
    print("   Paste tasks into ChatGPT, collect JSON responses")
    print("   Save to: results/live_responses_chatgpt.jsonl")
    print("")
    print("2. Open prompts/prompts_claude.txt")
    print("   Paste tasks into Claude, collect JSON responses")
    print("   Save to: results/live_responses_claude.jsonl")
    print("")
    print("3. Open prompts/prompts_llama3.txt")
    print("   Paste tasks into Llama-3, collect JSON responses")
    print("   Save to: results/live_responses_llama3.jsonl")
    print("")
    print("4. Parse responses:")
    print("   python src/parse_live_responses.py chatgpt")
    print("   python src/parse_live_responses.py claude")
    print("   python src/parse_live_responses.py llama3")
    print("   python src/parse_live_responses.py --merge")
    print("")
    print("5. Re-run analysis:")
    print("   python run_pipeline.py --start-at 6 --end-at 6 --force")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
