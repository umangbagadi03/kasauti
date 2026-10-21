import os
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from state import ACHISState


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.3
)


VERDICT_TONE = {
    "Safe":     "positive and reassuring",
    "Moderate": "cautionary but balanced",
    "Risky":    "clear and direct about the health concern",
}


def explanation_agent(state: ACHISState) -> ACHISState:
    verdict     = state.get("final_verdict", "Moderate")
    confidence  = state.get("verdict_confidence", 0.7)
    profile     = state.get("user_profile", {})
    signals     = state.get("risk_signals", [])
    chain       = state.get("reasoning_chain", [])
    allergens   = state.get("allergen_flags", [])
    nutrition   = state.get("nutrition_facts", {})
    guard_pass  = state.get("guardrail_passed", True)


    # Critical Alerts
    critical_alerts = []

    if state.get("banned_status"):
        critical_alerts.append("Contains banned ingredient")

    if allergens:
        critical_alerts.append(f"Contains allergens: {', '.join(allergens)}")

    critical_text = "\n".join(critical_alerts) if critical_alerts else "None"


    # Smart Signal Ranking
    sorted_signals = sorted(
        signals,
        key=lambda x: (
            {"high": 0, "moderate": 1, "low": 2}.get(x.get("risk"), 2),
            -x.get("confidence", 0)
        )
    )[:4]


    top_signals = [
        f"- {s.get('source','unknown')}: {s.get('reason','no reason')}"
        for s in sorted_signals
    ]

    signals_text = "\n".join(top_signals) if top_signals else "No major risk signals."
    reasoning_text = "\n".join(f"- {r}" for r in chain[:4]) if chain else "Not available."

    condition = profile.get("condition", "general user")
    goal      = profile.get("goal", "healthy eating")
    tone      = VERDICT_TONE.get(verdict, "balanced")

    reliability = "High" if guard_pass else "Low — limited evidence"


    # PROMPT
    prompt = f"""
You are a friendly food safety advisor.

VERDICT: {verdict} (confidence: {confidence:.0%})
RELIABILITY: {reliability}

USER: condition={condition}, goal={goal}

CRITICAL ALERTS:
{critical_text}

TOP RISK SIGNALS:
{signals_text}

REASONING:
{reasoning_text}

NUTRITION:
{nutrition}

INSTRUCTIONS:
- Write in a {tone} tone
- Mention actual ingredients and numbers
- Keep explanation concise (2-3 sentences)
- Subtly reflect confidence level in wording

Respond EXACTLY in this format:

EXPLANATION:
<text>

RECOMMENDATIONS:
- <tip 1>
- <tip 2>
- <tip 3>

MARKETING_FLAGS:
- <flag or "None detected">
"""

    # LLM CALL
    
    try:
        response = llm.invoke(prompt)
        explanation, recommendations, marketing_flags = _parse_response(response.content)

    except Exception:
        explanation, recommendations, marketing_flags = _fallback(state)

    return {
        **state,
        "explanation": explanation,
        "recommendations": recommendations,
        "marketing_flags": marketing_flags,
    }


# PARSER

def _parse_response(text: str):
    explanation = ""
    recommendations: List[str] = []
    marketing_flags: List[str] = []
    section = None

    for line in text.strip().splitlines():
        line = line.strip()

        if line.startswith("EXPLANATION:"):
            section = "explanation"
        elif line.startswith("RECOMMENDATIONS:"):
            section = "recommendations"
        elif line.startswith("MARKETING_FLAGS:"):
            section = "marketing"

        elif section == "explanation" and line:
            explanation += (" " + line) if explanation else line

        elif section == "recommendations" and line.startswith("- "):
            recommendations.append(line[2:])

        elif section == "marketing" and line.startswith("- "):
            flag = line[2:]
            if flag.lower() != "none detected":
                marketing_flags.append(flag)

    return explanation.strip(), recommendations, marketing_flags


# FALLBACK (CRITICAL)
def _fallback(state: ACHISState):
    verdict   = state.get("final_verdict", "Moderate")
    signals   = state.get("risk_signals", [])
    allergens = state.get("allergen_flags", [])
    profile   = state.get("user_profile", {})
    nutrition = state.get("nutrition_facts", {})

    # Explanation
    high_signals = [s for s in signals if s.get("risk") == "high"]

    if allergens:
        explanation = f"This product contains allergens: {', '.join(allergens)} and is rated {verdict}."
    elif high_signals:
        names = ", ".join(s["source"] for s in high_signals[:2])
        explanation = f"This product is rated {verdict} due to {names}, which may pose health risks."
    else:
        explanation = f"This product is rated {verdict} based on its ingredient and nutrition profile."

    # Recommendations
    recs = []
    condition = profile.get("condition", "")
    sugar = nutrition.get("sugar_g", 0)
    sodium = nutrition.get("sodium_mg", 0)

    if condition == "diabetic" and sugar > 10:
        recs.append(f"Limit intake — contains {sugar}g sugar.")
    if sodium > 400:
        recs.append(f"High sodium ({sodium}mg). Choose lower-sodium alternatives.")
    if state.get("ultra_processed_score", 0) > 0.5:
        recs.append("Prefer whole-food options with fewer additives.")
    if not recs:
        recs.append("Consume in moderation.")

    # Marketing flags
    flags = []
    ingredients_text = " ".join(
        str(i) for i in state.get("raw_ingredients", []) if i
    ).lower()

    if "natural" in ingredients_text and signals:
        flags.append("‘Natural’ claim may be misleading due to additives")

    if "high protein" in ingredients_text and nutrition.get("protein_g", 0) < 5:
        flags.append("Protein claim appears misleading")

    return explanation, recs, flags
