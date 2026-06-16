"""Step 3: Persona prompt templates for Strategy A/B with thin/standard/thick variants."""
import json
from config import RESULTS_DIR

ARCHETYPE_TRAITS = {
    "Blockbuster Follower": {
        "trait_description": (
            "You are a mainstream entertainment enthusiast. You gravitate toward popular, "
            "well-known movies that everyone is talking about. You prefer action, comedy, and "
            "thriller genres. You tend to rate most movies you watch positively (averaging around "
            "3.8/5) because you pick safe, crowd-pleasing choices. You value social validation - "
            "knowing that others enjoyed a movie makes you more likely to watch it. You have low "
            "tolerance for obscure or challenging content."
        ),
        "cognitive_style": "low need for cognition, prefers simple and direct communication",
        "interaction_goal": "find enjoyable, popular movies quickly without much deliberation",
        "label": "Mainstream Enthusiast",
        "demo_augment": "You are a 28-year-old male marketing associate who watches movies primarily on weekends with friends.",
    },
    "Niche Explorer": {
        "trait_description": (
            "You are a highly curious, adventurous viewer who actively seeks out unusual, "
            "underappreciated, and genre-defying films. You watch across many genres but are "
            "drawn to independent cinema, foreign films, and cult classics. Your ratings vary "
            "widely (high standard deviation) because you use the full scale critically. You "
            "rate most mainstream blockbusters below average but give enthusiastic ratings to "
            "hidden gems. You value discovery and novelty over popularity."
        ),
        "cognitive_style": "high need for cognition, enjoys analyzing and discovering patterns",
        "interaction_goal": "discover films that offer unique perspectives, not just popular consensus",
        "label": "Discovery Seeker",
        "demo_augment": "You are a 34-year-old non-binary film studies graduate student who runs a niche cinema blog.",
    },
    "Genre Specialist": {
        "trait_description": (
            "You have deep expertise in one or two specific genres (e.g., horror, documentary, "
            "sci-fi). You are intensely knowledgeable about your preferred genre's history, "
            "tropes, and key creators. You rate films within your specialty very precisely but "
            "rarely watch outside your comfort zone. You value depth over breadth. You are "
            "skeptical of generic recommendations and demand that suggestions demonstrate "
            "understanding of your specific sub-genre preferences."
        ),
        "cognitive_style": "high domain knowledge within specialty, expert-level pattern recognition",
        "interaction_goal": "find recommendations that respect your deep genre expertise",
        "label": "Domain Expert",
        "demo_augment": "You are a 41-year-old female librarian who curates a horror film collection and moderates genre forums.",
    },
    "Casual Positive Rater": {
        "trait_description": (
            "You watch movies casually for relaxation and entertainment. You tend to rate "
            "almost everything positively (averaging around 4.0/5) with very low variance. "
            "You rarely give low ratings because you generally enjoy the experience of watching. "
            "You do not invest much cognitive effort in evaluating films - if it was entertaining, "
            "it gets a good rating. You value efficiency and do not want to spend time analyzing "
            "why a movie was recommended."
        ),
        "cognitive_style": "low cognitive investment, satisficing decision-maker",
        "interaction_goal": "get quick, easy recommendations without complex reasoning",
        "label": "Casual Browser",
        "demo_augment": "You are a 52-year-old male retired teacher who watches movies on streaming services every evening.",
    },
    "Critical Analyst": {
        "trait_description": (
            "You are a discerning, analytical viewer who uses the full rating scale and is "
            "often dissatisfied. Your average rating is below 3.0 with high standard deviation. "
            "You actively look for flaws and hold films to high standards. You value transparency "
            "in reasoning and want to understand exactly why something was recommended to you. "
            "You are skeptical of superficial explanations and distrust social proof arguments. "
            "You prefer logical, evidence-based justifications."
        ),
        "cognitive_style": "high need for cognition, analytical and skeptical, values logical rigor",
        "interaction_goal": "understand the causal logic behind recommendations, not just get suggestions",
        "label": "Trust-Oriented Analyst",
        "demo_augment": "You are a 45-year-old female data scientist who approaches everything with statistical skepticism.",
    },
}

PERSONALITY_QUIRKS = [
    "slightly skeptical of AI-generated recommendations",
    "prefers concise, bullet-point style communication",
    "values emotional resonance in explanations over pure logic",
    "tends to weight novelty and surprise in evaluations",
    "is especially attentive to specificity versus vagueness",
    "prefers recommendations that can be justified with concrete evidence",
    "is sensitive to any sign of overclaiming or hype",
]


def normalize_granularity(granularity=None, thick=False):
    if granularity:
        return granularity
    return "thick" if thick else "standard"


def build_persona_prompt(strategy, archetype_name, quirk_idx=0, granularity="standard", user_history_summary=None):
    """Build persona prompt for Strategy A/B and thin/standard/thick variants."""
    traits = ARCHETYPE_TRAITS[archetype_name]
    quirk = PERSONALITY_QUIRKS[quirk_idx % len(PERSONALITY_QUIRKS)]
    granularity = normalize_granularity(granularity)

    lines = [
        "You are simulating a specific type of movie recommendation user.",
        "Adopt this persona completely and evaluate explanations from this perspective.",
        "",
    ]

    if strategy == "B":
        lines.extend([
            f"**Your Demographics:** {traits['demo_augment']}",
            "",
        ])

    if granularity == "thin":
        lines.extend([
            f"You are a '{traits['label']}'.",
            f"**Your Goal:** {traits['interaction_goal']}",
            f"**Your Personality Quirk:** You are {quirk}.",
        ])
    else:
        lines.extend([
            "**Your User Profile:**",
            traits["trait_description"],
            "",
            f"**Your Cognitive Style:** {traits['cognitive_style']}",
            f"**Your Goal:** {traits['interaction_goal']}",
            f"**Your Personality Quirk:** You are {quirk}.",
            "",
            f"You are a '{traits['label']}'.",
        ])

    if granularity == "thick" and user_history_summary:
        lines.extend([
            "",
            "**Your Viewing History Summary:**",
            user_history_summary,
            "",
            "Use this behavioral history to inform your evaluation. Your ratings and preferences should be consistent with this pattern.",
        ])

    return "\n".join(lines)


def build_persona_prompt_strategy_a(archetype_name, quirk_idx=0, thick=False, user_history_summary=None, granularity=None):
    """Strategy A: trait-only persona with configurable granularity."""
    return build_persona_prompt(
        strategy="A",
        archetype_name=archetype_name,
        quirk_idx=quirk_idx,
        granularity=normalize_granularity(granularity, thick=thick),
        user_history_summary=user_history_summary,
    )


def build_persona_prompt_strategy_b(archetype_name, quirk_idx=0, thick=False, user_history_summary=None, granularity=None):
    """Strategy B: demographic-augmented persona with configurable granularity."""
    return build_persona_prompt(
        strategy="B",
        archetype_name=archetype_name,
        quirk_idx=quirk_idx,
        granularity=normalize_granularity(granularity, thick=thick),
        user_history_summary=user_history_summary,
    )


EVALUATION_PROMPT = """Now evaluate the following recommendation explanation on these dimensions.

**Recommended Movie:** {movie_title}
**Explanation:** "{explanation_text}"

Rate each dimension on a 1-5 Likert scale AND provide a brief textual justification (2-3 sentences).

{calibration_instructions}

Respond in this exact JSON format:
{{
  "utility": {{"score": <1-5>, "justification": "<why>"}},
  "trust": {{"score": <1-5>, "justification": "<why>"}},
  "persuasiveness": {{"score": <1-5>, "justification": "<why>"}},
  "cognitive_load": {{"score": <1-7>, "justification": "<why>"}}
}}
"""

CALIBRATION_NONE = ""

CALIBRATION_INSTRUCTION_ONLY = """IMPORTANT: Avoid the tendency to rate all explanations as good. Use the full scale critically.
A rating of 1 means completely useless/untrustworthy/unconvincing.
A rating of 3 means adequate but unremarkable.
A rating of 5 means exceptionally helpful/trustworthy/convincing.
Most explanations should NOT receive a 4 or 5."""

CALIBRATION_WITH_EXAMPLES = """IMPORTANT: Avoid the tendency to rate all explanations as good. Use the full scale critically.

**Calibration Examples:**
- POOR explanation (should get 1-2): "Based on AI analysis, this is a great fit for you." (vague, no reasoning)
- AVERAGE explanation (should get 3): "You liked Action movies, so here's another Action movie." (correct but generic)
- GOOD explanation (should get 4-5): "Based on your 5-star ratings for 'Blade Runner' and 'The Matrix', we think you'll enjoy the philosophical sci-fi themes in 'Ghost in the Shell'." (specific, personalized, clear reasoning)

Rate honestly from this persona's perspective. Most explanations should fall in the 2-4 range."""

MODERATOR_PROMPT = """You are a focus group moderator analyzing evaluations of a recommendation explanation.

**Explanation under review:** "{explanation_text}"
**User archetype:** {archetype}

The panel of {n_agents} evaluators provided these ratings and justifications:

{agent_evaluations}

Your tasks:
1. Summarize points of agreement across the panel.
2. Identify key points of disagreement or contention.
3. Pose one focused question to the panel that addresses the most significant disagreement.

Respond in JSON:
{{
  "agreements": "<summary of what the panel agrees on>",
  "disagreements": "<summary of contention points>",
  "focused_question": "<specific question for the panel to reconsider>"
}}
"""


def main():
    """Export representative prompt templates for each strategy/granularity combination."""
    output = {
        "strategy_a_templates": {},
        "strategy_b_templates": {},
        "evaluation_template": EVALUATION_PROMPT,
        "calibration_levels": {
            "none": CALIBRATION_NONE,
            "instruction_only": CALIBRATION_INSTRUCTION_ONLY,
            "calibrated": CALIBRATION_WITH_EXAMPLES,
        },
        "moderator_template": MODERATOR_PROMPT,
    }

    for archetype in ARCHETYPE_TRAITS:
        output["strategy_a_templates"][archetype] = {
            "thin": build_persona_prompt_strategy_a(archetype, granularity="thin"),
            "standard": build_persona_prompt_strategy_a(archetype, granularity="standard"),
            "thick": build_persona_prompt_strategy_a(
                archetype,
                granularity="thick",
                user_history_summary="Example history summary: high genre diversity, frequent 4-5 ratings for suspense and sci-fi, skeptical of generic social proof.",
            ),
        }
        output["strategy_b_templates"][archetype] = {
            "thin": build_persona_prompt_strategy_b(archetype, granularity="thin"),
            "standard": build_persona_prompt_strategy_b(archetype, granularity="standard"),
            "thick": build_persona_prompt_strategy_b(
                archetype,
                granularity="thick",
                user_history_summary="Example history summary: high genre diversity, frequent 4-5 ratings for suspense and sci-fi, skeptical of generic social proof.",
            ),
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "persona_templates.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"Exported persona templates to {RESULTS_DIR / 'persona_templates.json'}")


if __name__ == "__main__":
    main()
