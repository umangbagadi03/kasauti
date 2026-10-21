import json

from fastapi.testclient import TestClient

from kasauti.api import app

client = TestClient(app)


def test_health_and_providers():
    assert client.get("/health").json()["status"] == "ok"
    assert set(client.get("/providers").json()) == {"anthropic", "gemini", "openai", "ollama"}


def test_analyze_text_offline(sample):
    r = client.post("/analyze", data={
        "text": sample("biscuit"), "use_llm": "false",
        "profile": json.dumps({"conditions": ["diabetes"], "allergies": ["milk"]}),
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["verdict"] == "Risky" and body["allergens"]["milk"] == "contains"


def test_bad_profile_is_422():
    r = client.post("/analyze", data={"text": "Ingredients: sugar", "use_llm": "false", "profile": "{bad"})
    assert r.status_code == 422


def test_no_input_is_422():
    assert client.post("/analyze", data={"use_llm": "false"}).status_code == 422


def test_llm_is_on_unless_explicitly_disabled(monkeypatch):
    import kasauti.api as api

    seen = []

    def fake_analyze(**kw):
        seen.append(kw["use_llm"])
        raise ValueError("stop here")

    monkeypatch.setattr(api, "analyze", fake_analyze)
    for extra in ({}, {"use_llm": "false"}, {"use_llm": "true"}):
        client.post("/analyze", data={"text": "x", **extra})
    assert seen == [True, False, True]
