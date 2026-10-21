import dataclasses

import pytest

import kasauti.llm as llm_mod
from kasauti import UserProfile, analyze
from kasauti.llm import LLMError, get_llm
from kasauti.llm.base import _loads_lenient
from kasauti.nodes.extract import parse_label_text

from .conftest import FakeLLM


# ---------------------------------------------------------------- offline (rules only)

def test_offline_parser_reads_numbers(sample):
    label = parse_label_text(sample("biscuit"))
    assert label.nutrition.sugar_g == 31 and label.nutrition.sodium_mg == 290
    assert label.nutrition.saturated_fat_g == 9.4 and label.nutrition.fat_g == 19
    assert label.ingredients[0] == "refined wheat flour (maida)"


def test_biscuit_is_risky(sample):
    r = analyze(text=sample("biscuit"), use_llm=False)
    assert r.verdict == "Risky" and r.ultra_processed


def test_oats_are_safe(sample):
    r = analyze(text=sample("oats"), use_llm=False)
    assert r.verdict == "Safe", r.signals
    assert r.positives


def test_namkeen_flags_salt_and_fat(sample):
    r = analyze(text=sample("namkeen"), use_llm=False)
    evidence = {s.evidence for s in r.signals}
    assert {"sodium_mg", "saturated_fat_g"} <= evidence
    assert r.verdict == "Risky"


def test_allergen_only_matters_if_user_is_allergic(sample):
    plain = analyze(text=sample("oats"), use_llm=False)
    assert not any(s.level == "avoid" for s in plain.signals)
    celiac = analyze(text=sample("oats"), use_llm=False, profile=UserProfile(conditions=["celiac"]))
    assert celiac.verdict == "Risky" and any(s.kind == "allergen" for s in celiac.signals)


def test_trace_allergen_is_high_not_avoid(sample):
    r = analyze(text=sample("namkeen"), use_llm=False, profile=UserProfile(allergies=["peanuts"]))
    lvl = [s.level for s in r.signals if s.kind == "allergen"]
    assert lvl == ["high"]


def test_region_rule_is_a_hard_stop():
    text = "Ingredients: Wheat Flour, Water, Yeast, Salt, Potassium Bromate."
    india = analyze(text=text, use_llm=False, profile=UserProfile(region="IN"))
    us = analyze(text=text, use_llm=False, profile=UserProfile(region="US"))
    assert india.verdict == "Risky" and any(s.level == "avoid" for s in india.signals)
    assert not any(s.level == "avoid" for s in us.signals)


def test_pku_and_aspartame():
    text = "Ingredients: Carbonated Water, Acidity Regulator (338), Sweetener (951), Caffeine."
    r = analyze(text=text, use_llm=False, profile=UserProfile(conditions=["pku"]))
    assert r.verdict == "Risky"


def test_empty_input_is_unknown():
    r = analyze(text="hello world", use_llm=False)
    assert r.verdict == "Unknown"


# ---------------------------------------------------------------- with a (fake) LLM

EXTRACTION = {
    "is_food_label": True, "product_name": "Masala Bhujia", "brand": "Acme",
    "ingredients": ["Gram Flour (Besan)", "Kokum Extract", "Palmolein", "Salt", "Spices"],
    "nutrition": {"sugar_g": 1.5, "fat_g": 38, "saturated_fat_g": 16, "sodium_mg": 820},
    "claims": ["Baked not fried"], "allergen_statements": [], "legibility": 0.95,
}
BATCH = {"items": [{"original": "Kokum Extract", "name": "Kokum extract", "category": "whole_food",
                    "risk": "none", "note": "Fruit extract."}]}
EXPLANATION = {"explanation": "High in salt and fat.", "tips": ["Eat a small portion."]}


def test_image_flow_uses_llm_for_extract_enrich_explain():
    fake = FakeLLM({"Extraction": EXTRACTION, "_Batch": BATCH, "_Explanation": EXPLANATION})
    png = _tiny_png()
    r = analyze(image=png, llm=fake)
    assert fake.calls == ["Extraction", "_Batch", "_Explanation"]
    assert r.unknown_ingredients == []
    assert any(m.origin == "ai" for m in r.ingredients)
    assert r.explanation == "High in salt and fat."
    assert r.verdict == "Risky"
    assert any(s.kind == "claim" for s in r.signals)  # "baked not fried" but 38 g fat


def test_llm_cannot_raise_risk_above_moderate():
    batch = {"items": [{"original": "Kokum Extract", "name": "x", "category": "other", "risk": "high", "note": "scary"}]}
    fake = FakeLLM({"Extraction": EXTRACTION, "_Batch": batch, "_Explanation": EXPLANATION})
    r = analyze(image=_tiny_png(), llm=fake)
    ai = [s for s in r.signals if "AI-classified" in s.title]
    assert ai and all(s.level == "moderate" for s in ai)


def test_explanation_falls_back_when_llm_fails(sample):
    fake = FakeLLM({"Extraction": EXTRACTION, "_Batch": BATCH}, fail={"_Explanation"})
    r = analyze(image=_tiny_png(), llm=fake)
    assert r.explanation and any("template" in w for w in r.warnings)


def test_text_extraction_falls_back_to_parser(sample):
    fake = FakeLLM({"_Explanation": EXPLANATION}, fail={"Extraction", "_Batch"})
    r = analyze(text=sample("biscuit"), llm=fake)
    assert r.verdict == "Risky" and any("basic parser" in w for w in r.warnings)


def test_structured_repairs_invalid_json_once():
    class Flaky(FakeLLM):
        def _json(self, system, prompt, schema, image):
            self.calls.append("x")
            return "not json" if len(self.calls) == 1 else '```json\n{"explanation": "ok", "tips": []}\n```'

    from kasauti.nodes.explain import _Explanation

    assert Flaky().structured(_Explanation, "s", "p").explanation == "ok"


def test_lenient_json():
    assert _loads_lenient('Sure! {"a": 1} hope that helps') == {"a": 1}


# ---------------------------------------------------------------- provider factory

def test_provider_factory(monkeypatch):
    monkeypatch.setattr(llm_mod, "settings", dataclasses.replace(llm_mod.settings, anthropic_api_key=None, openai_api_key=None))
    llm_mod._clients.clear()
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        get_llm("anthropic")
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        get_llm("openai")
    with pytest.raises(LLMError, match="Unknown provider"):
        get_llm("mistral")
    ollama = get_llm("ollama", "llama3.2-vision")  # no key needed
    assert ollama.provider == "ollama" and ollama.model == "llama3.2-vision"


def test_provider_clients_build_with_keys(monkeypatch):
    monkeypatch.setattr(llm_mod, "settings", dataclasses.replace(
        llm_mod.settings, anthropic_api_key="k", openai_api_key="k", gemini_api_key="k"))
    llm_mod._clients.clear()
    for p in ("anthropic", "gemini", "openai", "ollama"):
        assert get_llm(p).provider == p


def _tiny_png() -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, format="PNG")
    return buf.getvalue()


def test_llm_gaps_in_text_input_are_filled_by_parser(sample):
    extraction = {**EXTRACTION, "nutrition": {"fat_g": 19}}  # model "forgot" sugar
    fake = FakeLLM({"Extraction": extraction, "_Batch": BATCH, "_Explanation": EXPLANATION})
    r = analyze(text=sample("biscuit"), llm=fake)
    assert r.label.nutrition.sugar_g == 31 and r.label.nutrition.fat_g == 19
