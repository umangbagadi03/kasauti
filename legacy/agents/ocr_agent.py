import pytesseract
from PIL import Image
from typing import Tuple
import re
import time
import os
import json

from langchain_google_genai import ChatGoogleGenerativeAI
from state import ACHISState


# -------------------------
# 🔹 LLM (DEFINED CORRECTLY)
# -------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1
)


# -------------------------
# 🔹 MAIN AGENT
# -------------------------
def ocr_agent(state: ACHISState) -> ACHISState:
    start_time = time.time()

    try:
        # -------------------------
        # INPUT HANDLING
        # -------------------------
        if state.get("raw_text_input"):
            raw_text = state["raw_text_input"]
            ocr_conf = 1.0
        else:
            raw_text, ocr_conf = _extract_text_from_image(state["raw_image_path"])

        # -------------------------
        # CLEAN TEXT (SAFE)
        # -------------------------
        cleaned_text = _clean_text(raw_text)

        # -------------------------
        # REGEX PARSING
        # -------------------------
        ingredients = _parse_ingredients(cleaned_text)
        nutrition = _parse_nutrition(cleaned_text)

        parsing_source = "regex"

        # -------------------------
        # 🔥 SMART LLM FALLBACK (1 CALL MAX)
        # -------------------------
        if (len(ingredients) < 2) or (len(nutrition) < 2):
            llm_data = _llm_parse(cleaned_text)
            parsing_source = "llm"

            if not ingredients:
                ingredients = llm_data.get("ingredients", [])

            if not nutrition:
                nutrition = llm_data.get("nutrition", {})

        # -------------------------
        # 🔹 CONFIDENCE SIGNAL
        # -------------------------
        parse_confidence = 0.9 if parsing_source == "regex" else 0.6

        # -------------------------
        # WARNINGS
        # -------------------------
        warnings = state.get("warnings", [])

        if not ingredients:
            warnings.append("No ingredients detected")

        if not nutrition:
            warnings.append("No nutrition facts detected")

        # -------------------------
        # TIMING
        # -------------------------
        timings = state.get("agent_timings", {})
        timings["ocr_agent"] = round(time.time() - start_time, 3)

        return {
            **state,
            "raw_ingredients": ingredients,
            "nutrition_facts": nutrition,
            "ocr_confidence": ocr_conf,
            "parsing_source": parsing_source,
            "parse_confidence": parse_confidence,
            "warnings": warnings,
            "agent_timings": timings,
        }

    except Exception as e:
        errors = state.get("errors", [])
        errors.append(f"OCR Agent error: {str(e)}")

        return {
            **state,
            "errors": errors,
        }


# IMAGE OCR
def _extract_text_from_image(path: str) -> Tuple[str, float]:
    img = Image.open(path)
    img = img.convert("L")  # grayscale

    text = pytesseract.image_to_string(img)
    confidence = 0.8 if text.strip() else 0.0

    return text, confidence


# -------------------------
# 🔹 TEXT CLEANING (SAFE VERSION)
# -------------------------
def _clean_text(text: str) -> str:
    text = text.lower()

    # SAFE OCR FIXES ONLY
    replacements = {
        "0": "o",
        "1": "l",
        "@": "a",
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    # COMMON WORD FIXES
    word_fixes = {
        "s0dium": "sodium",
        "sodlum": "sodium",
        "proteln": "protein",
        "ingredlents": "ingredients",
        "ingrediens": "ingredients",
        "sugor": "sugar",
    }

    for wrong, correct in word_fixes.items():
        text = text.replace(wrong, correct)

    # REMOVE NOISE
    text = re.sub(r"[^a-z0-9\s:.,()%\-]", " ", text)

    # FIX BROKEN NUMBERS
    text = re.sub(r"(\d)\s+(\d)", r"\1\2", text)

    # NORMALIZE SPACES
    text = re.sub(r"\s+", " ", text).strip()

    return text


# -------------------------
# 🔹 LLM PARSER
# -------------------------
def _llm_parse(text: str):
    prompt = f"""
Extract structured food label data.

Return JSON ONLY:
{{
  "ingredients": ["..."],
  "nutrition": {{
    "sugar_g": number,
    "sodium_mg": number,
    "fat_g": number,
    "protein_g": number
  }}
}}

TEXT:
{text}
"""
    try:
        res = llm.invoke(prompt)
        return json.loads(res.content)
    except:
        return {"ingredients": [], "nutrition": {}}


# -------------------------
# 🔹 INGREDIENT PARSER
# -------------------------
def _parse_ingredients(text: str) -> list:
    patterns = [
        r"ingredients[:\-\s]+(.+?)(nutrition|$)",
        r"contains[:\-\s]+(.+?)(nutrition|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            raw = match.group(1).strip()

            return [
                i.strip().rstrip(".")
                for i in re.split(r"[,;]", raw)
                if len(i.strip()) > 2
            ]

    return []


# -------------------------
# 🔹 NUTRITION PARSER
# -------------------------
def _parse_nutrition(text: str) -> dict:
    facts = {}

    patterns = {
        "calories":    r"calories[:\s]*([\d.]+)",
        "sugar_g":     r"sugar[s]?[:\s]*([\d.]+)\s*g",
        "sodium_mg":   r"sodium[:\s]*([\d.]+)\s*mg",
        "fat_g":       r"(total fat|fat)[:\s]*([\d.]+)\s*g",
        "protein_g":   r"protein[:\s]*([\d.]+)\s*g",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.groups()[-1]
            try:
                facts[key] = float(value)
            except:
                continue

    return facts