from state import ACHISState

RISK_WEIGHTS = {"high": 3.0, "moderate": 1.5, "low": 0.5}
LLM_VERDICT_MAP = {"Safe": 0, "Moderate": 5, "Risky": 10}

WEIGHTS = {
    "risk_signals":       0.45,   # ↑ more trust
    "ultra_processed":    0.20,   # ↓ reduce dominance
    "llm_verdict":        0.25,   # ↓ safer
}

THRESHOLDS = {
    "Safe":     (0.0, 3.5),
    "Moderate": (3.5, 6.5),
    "Risky":    (6.5, 10.0),
}

def decision_agent(state: ACHISState) -> ACHISState:
    trace = state.get("reasoning_trace", [])

    # CRITICAL OVERRIDES
    if state.get("banned_status"):
        return {
            **state,
            "final_verdict": "Risky",
            "verdict_confidence": 1.0,
            "decision_notes": "Banned ingredient override",
            "reasoning_trace": trace + [{"agent": "decision", "msg": "Banned override"}]
        }
    if state.get("allergen_flags"):
        return {
            **state,
            "final_verdict": "Risky",
            "verdict_confidence": state.get("confidence_score", 0.8),
            "decision_notes": "Allergen override applied",
            "reasoning_trace": trace + [{"agent": "decision", "msg": "Allergen override"}]
        }
    
    # RISK SIGNAL SCORE (CONFIDENCE-AWARE)
    raw_risk = sum(
        RISK_WEIGHTS.get(s.get("risk", "low"), 0.5) * s.get("confidence", 0.5)
        for s in state.get("risk_signals", [])
    )

    # NORMALIZE NON-LINARITY
    risk_score = min(raw_risk ** 0.8, 10.0)

    # ULTRA-PROCESSED IMPACT
    ultra_score = state.get("ultra_processed_score", 0.0) * 10

    # LLM verdict
    llm_score = LLM_VERDICT_MAP.get(state.get("verdict_draft", "Moderate"), 5)

    # COMPOSITE
    composite = (
        risk_score  * WEIGHTS["risk_signals"] +
        ultra_score * WEIGHTS["ultra_processed"] +
        llm_score   * WEIGHTS["llm_verdict"]
    )

    # PERSONALIZATION
    profile = state.get("user_profile", {})
    nutrition = state.get("nutrition_facts", {})

    if profile.get("condition") == "diabetic":
        if nutrition.get("sugar_g", 0) > 10:
            composite += 1.5
    
    # CLAMP
    composite = min(composite, 10.0)

    # THRESHOLD CLASSIFICATION
    final_verdict = "Moderate"
    for verdict, (low, high) in THRESHOLDS.items():
        if low <= composite < high:
            final_verdict = verdict
            break

    # STABILITY SCORING
    llm_agrees = (
        (final_verdict == "Safe" and llm_score <= 2) or
        (final_verdict == "Moderate" and 3 <= llm_score <= 7) or
        (final_verdict == "Risky" and llm_score >= 8)
    )

    stability = 1.0 if llm_agrees else 0.75

    confidence = round(state.get("confidence_score", 0.8) * stability, 2)

    # TRACE
    trace.append({
        "agent": "decision",
        "msg": f"composite={round(composite,2)}, verdict={final_verdict}, stability={stability}"
    })

    return {
        **state,
        "final_verdict": final_verdict,
        "verdict_confidence": confidence,
        "composite_score": round(composite, 2),
        "decision_notes": "Composite multi-signal decision",
        "reasoning_trace": trace
    }




