"""Step 4: Live multi-agent focus-group evaluation using a configurable LLM backend."""
import json
import os
import sys
import time

from tqdm import tqdm

from config import (
    ANTHROPIC_API_KEY,
    LLM_MODEL,
    LLM_PROVIDER,
    OPENAI_COMPAT_API_KEY,
    OPENAI_COMPAT_BASE_URL,
    OPENAI_COMPAT_REASONING,
    PRIMARY_DEBATE_ROUNDS,
    PRIMARY_PANEL_SIZE,
    RESULTS_DIR,
    TEMPERATURE,
)
from step3_personas import (
    CALIBRATION_INSTRUCTION_ONLY,
    CALIBRATION_NONE,
    CALIBRATION_WITH_EXAMPLES,
    EVALUATION_PROMPT,
    MODERATOR_PROMPT,
    build_persona_prompt_strategy_a,
    build_persona_prompt_strategy_b,
)

anthropic_client = None
openai_client = None

if LLM_PROVIDER == "anthropic":
    try:
        import anthropic

        if ANTHROPIC_API_KEY:
            anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    except ImportError:
        anthropic_client = None
else:
    try:
        from openai import OpenAI

        if OPENAI_COMPAT_API_KEY:
            openai_client = OpenAI(
                base_url=OPENAI_COMPAT_BASE_URL,
                api_key=OPENAI_COMPAT_API_KEY,
            )
    except ImportError:
        openai_client = None


def ensure_client():
    if LLM_PROVIDER == "anthropic":
        if not anthropic_client:
            raise RuntimeError("Anthropic client not initialized. Set ANTHROPIC_API_KEY and install anthropic.")
    else:
        if not openai_client:
            raise RuntimeError(
                "OpenAI-compatible client not initialized. Set OPENAI_COMPAT_API_KEY and install openai."
            )


def call_llm(system_prompt, user_prompt, temperature=TEMPERATURE, max_tokens=1024):
    """Call the configured backend and return plain text."""
    ensure_client()

    for attempt in range(3):
        try:
            if LLM_PROVIDER == "anthropic":
                response = anthropic_client.messages.create(
                    model=LLM_MODEL,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                return response.content[0].text

            extra_body = {}
            if OPENAI_COMPAT_REASONING:
                extra_body = {
                    "chat_template_kwargs": {"enable_thinking": True},
                    "reasoning_budget": min(max_tokens, 16384),
                }
            response = openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                top_p=0.95,
                max_tokens=max_tokens,
                extra_body=extra_body or None,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            if attempt < 2:
                print(f"  API error (attempt {attempt + 1}): {exc}. Retrying...")
                time.sleep(2 ** attempt)
            else:
                raise


def parse_json_response(text):
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.split("\n")
        json_lines = []
        in_block = False
        for line in lines:
            if line.startswith("```") and not in_block:
                in_block = True
                continue
            if line.startswith("```") and in_block:
                break
            if in_block:
                json_lines.append(line)
        text = "\n".join(json_lines)

    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def evaluate_single_agent(persona_prompt, task, calibration, agent_idx):
    eval_prompt = EVALUATION_PROMPT.format(
        movie_title=task["recommended_movie"],
        explanation_text=task["explanation_text"],
        calibration_instructions=calibration,
    )

    response_text = call_llm(persona_prompt, eval_prompt, max_tokens=2048)
    result = parse_json_response(response_text)
    if result is None:
        return {
            "agent_idx": agent_idx,
            "raw_response": response_text,
            "parsed": False,
            "utility": {"score": 3, "justification": "PARSE_ERROR"},
            "trust": {"score": 3, "justification": "PARSE_ERROR"},
            "persuasiveness": {"score": 3, "justification": "PARSE_ERROR"},
            "cognitive_load": {"score": 4, "justification": "PARSE_ERROR"},
        }

    result["agent_idx"] = agent_idx
    result["parsed"] = True
    return result


def run_moderator_debate(agent_evaluations, task, archetype):
    eval_summary = ""
    for i, ev in enumerate(agent_evaluations):
        eval_summary += f"\nAgent {i + 1}:\n"
        for dim in ["utility", "trust", "persuasiveness"]:
            if dim in ev and isinstance(ev[dim], dict):
                eval_summary += (
                    f"  {dim}: {ev[dim].get('score', '?')}/5 - {ev[dim].get('justification', 'N/A')}\n"
                )

    mod_prompt = MODERATOR_PROMPT.format(
        explanation_text=task["explanation_text"],
        archetype=archetype,
        n_agents=len(agent_evaluations),
        agent_evaluations=eval_summary,
    )
    response = call_llm("You are an impartial focus group moderator.", mod_prompt, max_tokens=2048)
    return parse_json_response(response) or {"focused_question": "Please reconsider your ratings."}


def run_revision_round(persona_prompt, task, original_eval, moderator_summary, calibration):
    revision_prompt = f"""The moderator of your focus group has summarized the discussion:

**Agreements:** {moderator_summary.get('agreements', 'N/A')}
**Disagreements:** {moderator_summary.get('disagreements', 'N/A')}
**Question for you:** {moderator_summary.get('focused_question', 'N/A')}

Your original ratings were:
- Utility: {original_eval.get('utility', {}).get('score', '?')}/5
- Trust: {original_eval.get('trust', {}).get('score', '?')}/5
- Persuasiveness: {original_eval.get('persuasiveness', {}).get('score', '?')}/5

Considering the moderator's summary and question, provide your REVISED ratings.
You may keep your original ratings if you still believe they are correct.

{calibration}

Respond in JSON:
{{
  "utility": {{"score": <1-5>, "justification": "<why>"}},
  "trust": {{"score": <1-5>, "justification": "<why>"}},
  "persuasiveness": {{"score": <1-5>, "justification": "<why>"}},
  "cognitive_load": {{"score": <1-7>, "justification": "<why>"}}
}}
"""
    response = call_llm(persona_prompt, revision_prompt, max_tokens=2048)
    return parse_json_response(response)


def run_focus_group(
    task,
    archetype,
    strategy="A",
    persona_granularity="thick",
    calibration_level="calibrated",
    panel_size=PRIMARY_PANEL_SIZE,
    debate_rounds=PRIMARY_DEBATE_ROUNDS,
    user_history=None,
):
    calibration_map = {
        "none": CALIBRATION_NONE,
        "instruction_only": CALIBRATION_INSTRUCTION_ONLY,
        "calibrated": CALIBRATION_WITH_EXAMPLES,
    }
    calibration = calibration_map.get(calibration_level, CALIBRATION_WITH_EXAMPLES)
    persona_builder = build_persona_prompt_strategy_a if strategy == "A" else build_persona_prompt_strategy_b

    personas = [
        persona_builder(
            archetype,
            quirk_idx=i,
            granularity=persona_granularity,
            user_history_summary=user_history,
        )
        for i in range(panel_size)
    ]

    initial_evals = [evaluate_single_agent(persona, task, calibration, i) for i, persona in enumerate(personas)]
    final_evals = initial_evals
    moderator_summary = None

    for _ in range(debate_rounds):
        moderator_summary = run_moderator_debate(initial_evals, task, archetype)
        revised_evals = []
        for i, persona in enumerate(personas):
            revised = run_revision_round(persona, task, initial_evals[i], moderator_summary, calibration)
            if revised:
                revised["agent_idx"] = i
                revised["parsed"] = True
                revised_evals.append(revised)
            else:
                revised_evals.append(initial_evals[i])
        final_evals = revised_evals
        initial_evals = revised_evals

    return {
        "task_id": task["task_id"],
        "archetype": archetype,
        "strategy": strategy,
        "persona_granularity": persona_granularity,
        "calibration_level": calibration_level,
        "panel_size": panel_size,
        "debate_rounds": debate_rounds,
        "initial_evaluations": initial_evals,
        "moderator_summary": moderator_summary,
        "final_evaluations": final_evals,
    }


def run_experiment(tasks, sample_size=None, config=None):
    if config is None:
        config = {
            "strategy": "A",
            "persona_granularity": "thick",
            "calibration_level": "calibrated",
            "panel_size": PRIMARY_PANEL_SIZE,
            "debate_rounds": PRIMARY_DEBATE_ROUNDS,
        }
    if sample_size:
        tasks = tasks[:sample_size]

    results = []
    for task in tqdm(tasks, desc="Running live focus groups"):
        try:
            results.append(run_focus_group(task=task, archetype=task["archetype"], **config))
        except Exception as exc:
            print(f"\n  Error on {task['task_id']}: {exc}")
    return results


def build_live_config_from_env():
    return {
        "strategy": os.environ.get("LIVE_STRATEGY", "A"),
        "persona_granularity": os.environ.get("LIVE_PERSONA_GRANULARITY", "thick"),
        "calibration_level": os.environ.get("LIVE_CALIBRATION_LEVEL", "calibrated"),
        "panel_size": int(os.environ.get("LIVE_PANEL_SIZE", str(PRIMARY_PANEL_SIZE))),
        "debate_rounds": int(os.environ.get("LIVE_DEBATE_ROUNDS", str(PRIMARY_DEBATE_ROUNDS))),
    }


def save_live_experiment(sample_size=10, config=None, output_name="experiment_results_live.json"):
    with open(RESULTS_DIR / "evaluation_tasks.json") as handle:
        tasks = json.load(handle)

    live_config = config or build_live_config_from_env()
    print(f"Running live experiment on {sample_size} tasks with provider={LLM_PROVIDER}, model={LLM_MODEL}")
    print(f"Live config: {live_config}")
    results = run_experiment(tasks, sample_size=sample_size, config=live_config)

    payload = {
        "provider": LLM_PROVIDER,
        "model": LLM_MODEL,
        "sample_size": sample_size,
        "config": live_config,
        "results": results,
    }
    output_path = RESULTS_DIR / output_name
    with open(output_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved {len(results)} live results to {output_path}")
    return output_path


def main():
    sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    save_live_experiment(sample_size=sample_size)


if __name__ == "__main__":
    main()
