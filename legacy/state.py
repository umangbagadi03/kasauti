from typing import TypedDict, Optional, List, Dict, Any
from datetime import datetime

# FIXED: Changed TypeDict to TypedDict here
class ACHISState(TypedDict, total=False):
    
    # -------------------------
    # 🔹 METADATA
    # -------------------------
    request_id: str
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]

    # -------------------------
    # 🔹 INPUT
    # -------------------------
    raw_image_path: Optional[str]
    raw_text_input: Optional[str]
    barcode: Optional[str]
    user_profile: dict  # {"condition": "diabetic", "goal": "weight_loss"}

    # -------------------------
    # 🔹 OCR AGENT OUTPUT
    # -------------------------
    raw_ingredients: List[str]
    nutrition_facts: Dict[str, Any] 
    ocr_confidence: Optional[float]

    # -------------------------
    # 🔹 INGREDIENT AGENT OUTPUT
    # -------------------------
    # FIXED: Renamed to decoded_ingredients to match your agent code
    decoded_ingredients: List[dict] 
    allergen_flags: List[str]
    unknown_ingredients: List[str]

    # -------------------------
    # 🔹 RISK AGENT OUTPUT
    # -------------------------
    risk_signals: List[dict] 
    ultra_processed_score: float 
    # FIXED: Spelling of cumulative
    cumulative_risk_score: float

    # -------------------------
    # 🔹 KNOWLEDGE AGENT OUTPUT
    # -------------------------
    external_evidence: List[dict]
    banned_status: Optional[str] 

    # -------------------------
    # 🔹 REASONING AGENT OUTPUT
    # -------------------------
    verdict_draft: str
    reasoning_chain: List[str]
    reasoning_trace: List[dict] # Added trace to match your agents

    # -------------------------
    # 🔹 GUARDRAILS
    # -------------------------
    confidence_score: float
    # FIXED: Spelling of guardrail
    guardrail_passed: bool
    low_evidence_flags: List[str]
    hallucination_flags: List[str]  

    # -------------------------
    # 🔹 DECISION OUTPUT
    # -------------------------
    final_verdict: str 
    verdict_confidence: float
    decision_factors: List[str]  
    composite_score: float # Added to match your decision agent

    # -------------------------
    # 🔹 EXPLANATION OUTPUT
    # -------------------------
    explanation: str
    recommendations: List[str]  
    marketing_flags: List[str]