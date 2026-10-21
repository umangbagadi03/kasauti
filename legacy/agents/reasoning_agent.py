from typing import List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from state import ACHISState


# STRUCTURED OUTPUT SCHEMA
class ReasoningSchema(BaseModel):
    verdict: str = Field(description="Strictly one of: Safe, Moderate, Risky")
    steps: List[str] = Field(description="Max 5 concise reasoning steps")
    primary_risk_factor: str = Field(description="Main reason for the verdict")
    confidence: float = Field(description="0.0 to 1.0 confidence score")

# AGENT
def reasoning_agent(state: ACHISState) -> ACHISState:

    trace = state.get("reasoning_trace", [])

    # HARD OVERRIDE 
    if state.get("banned_status"):
        return {
            **state,
            "verdict_draft": "Risky",
            "reasoning_chain": ["Product contains a banned ingredient"],
            "confidence_score": 1.0,
            "reasoning_trace": trace + [
                {"agent": "reasoning", "msg": "Bypassed LLM due to banned ingredient"}
            ]
        }

    # INIT LLM (STRUCTURED)
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2  # reduce randomness
    )

    structured_llm = llm.with_structured_output(ReasoningSchema)

    # SELECT TOP SIGNALS (NOISE REDUCTION)
    signals = state.get("risk_signals", [])
    top_signals = sorted(
        signals,
        key=lambda x: (x.get("risk") == "high", x.get("confidence", 0)),
        reverse=True
    )[:5]

    # BUILD CONTEXT
    prompt = f"""
Act as a Clinical Nutritionist and Food Safety Expert.

STRICT RULES:
- Verdict MUST be exactly one of: Safe, Moderate, Risky
- Prioritize HIGH risk signals over moderate
- Consider USER PROFILE strongly (e.g., diabetic → sugar is critical)
- If evidence is weak → lower confidence
- Do NOT hallucinate unknown health claims

PRODUCT DATA:
- Ingredients: {state.get('decoded_ingredients')}
- Top Risk Signals: {top_signals}
- Nutrition: {state.get('nutrition_facts')}
- Ultra Processed Score: {state.get('ultra_processed_score')}
- External Evidence: {state.get('external_evidence', [])}

USER CONTEXT:
- Profile: {state.get('user_profile')}

TASK:
1. Decide overall health risk
2. Explain reasoning in max 5 steps
3. Identify primary risk factor
4. Provide confidence score
"""

    # INVOKE MODEL
    try:
        result = structured_llm.invoke(prompt)

        # NORMALIZE OUTPUT
        verdict = result.verdict.capitalize()
        if verdict not in {"Safe", "Moderate", "Risky"}:
            verdict = "Moderate"

        confidence = max(0.0, min(result.confidence, 1.0))

        # UPDATE STATE
        return {
            **state,
            "verdict_draft": verdict,
            "reasoning_chain": result.steps[:5],
            "confidence_score": round(confidence, 2),
            "reasoning_trace": trace + [
                {
                    "agent": "reasoning",
                    "msg": f"{verdict} based on {result.primary_risk_factor}"
                }
            ]
        }

    except Exception as e:
        # FALLBACK (VERY IMPORTANT)
        return {
            **state,
            "verdict_draft": "Moderate",
            "reasoning_chain": ["Fallback: reasoning failed, using safe default"],
            "confidence_score": 0.3,
            "errors": state.get("errors", []) + [f"Reasoning agent error: {str(e)}"],
            "reasoning_trace": trace + [
                {"agent": "reasoning", "msg": "Fallback triggered"}
            ]
        }
