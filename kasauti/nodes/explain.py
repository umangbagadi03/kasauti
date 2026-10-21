"""Explain the (already decided) verdict in plain language — LLM if available, template otherwise."""

import re

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from ..llm import LLMError
from ..models import UserProfile
from ..state import GraphState
from . import llm_from
from .analyze import RANK

SYSTEM = """You explain food-label analysis results to shoppers in plain, friendly language.
Hard rules:
- The verdict is final. Never contradict, soften or upgrade it.
- Use ONLY the facts provided. Do not add health claims, numbers or ingredients that are not listed.
- Mention the 1-3 most important concrete reasons (ingredient names, numbers).
- Tips must follow from the listed reasons only — never mention claims, nutrients or risks that are not listed.
- No diagnosis or medical advice. If a hard stop involves a medical condition, suggest checking with their doctor.
- Write in the requested language."""


class _Explanation(BaseModel):
    explanation: str = Field(description="2-4 sentences")
    tips: list[str] = Field(description="2-3 short, practical tips")


def explain_node(state: GraphState, config: RunnableConfig) -> dict:
    verdict = state.get("verdict", "Unknown")
    if verdict == "Unknown":
        return {
            "explanation": "We couldn't read enough of this label to give a verdict.",
            "tips": ["Retake the photo in good light with the ingredient list and nutrition table fully visible.",
                     "Or type the ingredient list, or scan the barcode."],
        }

    llm = llm_from(config)
    if llm:
        try:
            out = llm.structured(_Explanation, SYSTEM, _facts(state))
            tips = [t.strip() for t in out.tips if _is_real_text(t)][:3]
            return {"explanation": out.explanation.strip(), "tips": tips or _template(state)["tips"]}
        except LLMError as e:
            return {**_template(state), "warnings": [f"Explanation fell back to template: {e}"]}
    return _template(state)


def _is_real_text(s: str) -> bool:
    """Drop fragments small models leak, like ']}assistant' — keep anything with 3+ real words."""
    return len(re.findall(r"[A-Za-z\u0900-\u0DFF]{2,}", s)) >= 3


def _top(state: GraphState, n: int = 8):
    return sorted(
        (s for s in state.get("signals", []) if s.kind != "claim"),
        key=lambda s: RANK[s.level], reverse=True,
    )[:n]


def _facts(state: GraphState) -> str:
    label = state["label"]
    profile = state.get("profile") or UserProfile()
    lines = [
        f"Product: {label.product_name or 'unknown'}{f' by {label.brand}' if label.brand else ''}",
        f"Verdict: {state['verdict']} (score {state['score']}/10, confidence {state['confidence']:.0%})",
        f"User conditions: {', '.join(profile.conditions) or 'none'}; allergies: {', '.join(profile.allergies) or 'none'}",
        f"Write in: {profile.language}",
        "Reasons (most serious first):",
        *[f"- [{s.level}] {s.title}: {s.detail}" for s in _top(state)],
    ]
    if pos := state.get("positives"):
        lines += ["Positives:", *[f"- {p}" for p in pos]]
    if claims := [s for s in state.get("signals", []) if s.kind == "claim"]:
        lines += ["Misleading pack claims:", *[f"- {s.title}: {s.detail}" for s in claims]]
    if unknown := state.get("unknown"):
        lines.append(f"Ingredients we could not identify: {', '.join(unknown[:6])}")
    return "\n".join(lines)


def _template(state: GraphState) -> dict:
    verdict = state["verdict"]
    top = _top(state, 3)
    if not top:
        text = "Nothing in the ingredients or nutrition table raised a concern."
    else:
        lead = "Minor notes" if verdict == "Safe" else "Main reasons"
        text = f"{lead}: " + "; ".join(f"{s.title.lower()}" for s in top) + "."
    if pos := state.get("positives"):
        text += " On the plus side: " + "; ".join(p.lower() for p in pos) + "."
    opener = {"Safe": "Looks like a reasonable choice.", "Moderate": "Fine occasionally, in moderation.",
              "Risky": "Best avoided or kept rare."}[verdict]
    tips = []
    for s in _top(state, 8):
        if s.level == "avoid":
            tips.append(f"Avoid: {s.title}.")
        elif s.evidence == "sugar_g":
            tips.append("Look for a version with under 5g sugar per 100g.")
        elif s.evidence == "sodium_mg":
            tips.append("Pick a lower-salt option (under 120mg sodium per 100g is 'low').")
        elif s.kind == "processing":
            tips.append("Prefer products with a short ingredient list of foods you recognise.")
    claims = [s for s in state.get("signals", []) if s.kind == "claim"]
    if claims:
        tips.append(f"Don't rely on the pack claim — {claims[0].detail[0].lower() + claims[0].detail[1:]}")
    return {"explanation": f"{opener} {text}", "tips": list(dict.fromkeys(tips))[:3] or ["Enjoy as part of a varied diet."]}
