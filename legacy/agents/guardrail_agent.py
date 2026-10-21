from state import ACHISState

VALID_VERDICTS = {"Safe", "Moderate", "Risky"}

def guardrail_agent(state: ACHISState) -> ACHISState:
    flags = []
    hallucination_flags = []
    trace = state.get("reasoning_trace", [])

    chain = state.get("reasoning_chain", [])
    evidence = state.get("external_evidence", [])
    signals = state.get("risk_signals", [])

    verdict = state.get("verdict_draft", "Moderate")
    confidence = state.get("confidence_score", 0.7)

    # VERDICT VALIDATION
    if verdict not in VALID_VERDICTS:
        hallucination_flags.append(f"Invalid verdict: {verdict}")
        verdict = "Moderate"
        confidence *= 0.6

    # WEAK REASONING DETECTION
    if len(chain) < 2:
        flags.append("reasoning_too_short")
        confidence *= 0.8

    # MISSING EVIDENCE FOR RISKY
    if verdict == "Risky" and not evidence and not signals:
        flags.append("risky_without_evidence")
        confidence *= 0.75
    
    # SIGNAL INCONSISTENCY
    if state.get("ultra_processed_score", 0) > 0.6 and len(signals) == 0:
        flags.append("processed_score_mismatch")
        confidence *= 0.8
    
    # OCR QUALITY IMPACT
    ocr_conf = state.get("ocr_confidence", 1.0)
    if ocr_conf < 0.6:
        flags.append("low_ocr_confidence")
        confidence *= 0.7
    
    # no signal at all
    if len(signals) == 0:
        flags.append("no_risk_signals")
        confidence *= 0.75

    # CLAMP CONFIDENCE
    confidence = max(0.2, min(confidence, 1.0))

    # PASS CRITERIA
    passed = confidence >= 0.6

    #TRACE LOGGING
    trace.append({
        "agent": "guardrail",
        "msg": f"confidence={round(confidence,2)}, flags={flags}, hallucinations={hallucination_flags}"
    })

    return {
        **state,
        "verdict_draft": verdict,
        "confidence_score": round(confidence, 2),
        "guardrail_passed": passed,
        "low_evidence_flags": flags,
        "hallucination_flags": hallucination_flags,
        "reasoning_trace": trace
    }
