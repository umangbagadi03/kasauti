"""Turn signals into a verdict. Pure arithmetic — the LLM has no vote here."""

from ..state import GraphState

POINTS = {"low": 0.5, "moderate": 1.5, "high": 3.0}
SAFE_BELOW = 3.0
RISKY_FROM = 6.5


def score_node(state: GraphState) -> dict:
    label = state.get("label")
    signals = state.get("signals", [])
    has_data = bool(label and (label.ingredients or label.nutrition.model_dump(exclude_none=True)))
    if not has_data:
        return {"verdict": "Unknown", "score": 0.0, "confidence": 0.0}

    # Misleading claims are reported, but they're about honesty, not health — no points.
    raw = sum(POINTS.get(s.level, 0) for s in signals if s.kind != "claim")
    raw -= len(state.get("positives", []))
    score = round(max(0.0, min(raw, 10.0)), 1)

    hard_stop = any(s.level == "avoid" for s in signals)
    if hard_stop or score >= RISKY_FROM:
        verdict = "Risky"
    elif score >= SAFE_BELOW:
        verdict = "Moderate"
    else:
        verdict = "Safe"

    # Confidence reflects how complete and readable the input was.
    confidence = label.legibility
    n_items = len(label.ingredients)
    if n_items:
        known = 1 - len(state.get("unknown", [])) / n_items
        confidence *= 0.6 + 0.4 * max(known, 0)
    else:
        confidence *= 0.6
    if not label.nutrition.model_dump(exclude_none=True):
        confidence *= 0.75
    if hard_stop:  # an allergen/ban hit is certain as long as we read the label right
        confidence = max(confidence, 0.9 * label.legibility)

    return {"verdict": verdict, "score": score, "confidence": round(confidence, 2)}
