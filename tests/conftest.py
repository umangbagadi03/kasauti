import os
from pathlib import Path

os.environ["KASAUTI_CACHE_PATH"] = "off"  # before kasauti.config is imported

import pytest

from kasauti.llm.base import LLMClient, LLMError

SAMPLES = Path(__file__).parent.parent / "samples"


class FakeLLM(LLMClient):
    """Returns canned JSON per requested schema; records calls."""

    provider = "fake"
    model = "fake-1"

    def __init__(self, responses: dict[str, dict] | None = None, fail: set[str] | None = None):
        self.responses = responses or {}
        self.fail = fail or set()
        self.calls: list[str] = []

    def _text(self, system, prompt, image):
        return "ok"

    def _json(self, system, prompt, schema, image):
        name = schema.get("title", "")
        self.calls.append(name)
        if name in self.fail:
            raise LLMError("boom")
        return self.responses[name]


@pytest.fixture
def sample():
    return lambda name: (SAMPLES / f"{name}.txt").read_text()
