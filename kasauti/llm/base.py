"""Provider-agnostic LLM interface.

Every provider implements two primitives: plain text, and JSON that must
validate against a pydantic model. Images are optional on both.
"""

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


_AUTH_HINTS = ("api key not valid", "invalid api key", "invalid x-api-key", "incorrect api key",
               "authentication", "unauthorized", "401", "permission_denied", "api_key_invalid")
_KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY"}


def _describe(provider: str, model: str, e: Exception) -> str:
    msg = str(e)
    low = msg.lower()
    if any(h in low for h in _AUTH_HINTS):
        return f"{provider}: authentication failed — check {_KEY_ENV.get(provider, 'the API key')}"
    if "connection" in low or "connect" in low:
        return f"{provider}: cannot connect{' — is `ollama serve` running?' if provider == 'ollama' else ''}"
    if "not found" in low or "404" in low:
        return f"{provider}: model '{model}' not found{' — run `ollama pull " + model + "`' if provider == 'ollama' else ''}"
    return f"{provider}/{model}: {msg[:200]}"


@dataclass
class Image:
    data: bytes
    mime_type: str = "image/jpeg"


class LLMClient(ABC):
    provider: str
    model: str
    _fatal: tuple[str, float] | None = None  # auth/connection failure: fail fast for a minute

    @abstractmethod
    def _text(self, system: str, prompt: str, image: Image | None) -> str: ...

    @abstractmethod
    def _json(self, system: str, prompt: str, schema: dict, image: Image | None) -> str | dict:
        """Return a JSON string or an already-parsed dict."""

    def _fail(self, e: Exception) -> LLMError:
        msg = _describe(self.provider, self.model, e)
        if "authentication" in msg or "cannot connect" in msg:
            self._fatal = (msg, time.monotonic())
        return LLMError(msg)

    def _check_fatal(self) -> None:
        if self._fatal and time.monotonic() - self._fatal[1] < 60:
            raise LLMError(self._fatal[0])

    def text(self, system: str, prompt: str, image: Image | None = None) -> str:
        self._check_fatal()
        try:
            return self._text(system, prompt, image).strip()
        except LLMError:
            raise
        except Exception as e:  # SDK / network errors → one error type for callers
            raise self._fail(e) from e

    def structured(self, model_cls: type[T], system: str, prompt: str, image: Image | None = None) -> T:
        """Ask for JSON matching `model_cls`; one repair attempt on invalid output."""
        self._check_fatal()
        schema = inline_refs(model_cls.model_json_schema())
        # Small models copy a raw JSON schema back verbatim; an example shape works far better.
        full_prompt = (
            f"{prompt}\n\nRespond with ONLY a JSON object of this shape (fill in real values):\n"
            f"{json.dumps(skeleton(schema), ensure_ascii=False)}"
        )
        last_err: Exception | None = None
        for _ in range(2):
            try:
                raw = self._json(system, full_prompt, schema, image)
                data = raw if isinstance(raw, dict) else _loads_lenient(raw)
                if "$defs" in data or ("properties" in data and data.get("type") == "object"):
                    raise ValueError("you returned the schema instead of the data")
                return model_cls.model_validate(data)
            except (ValidationError, ValueError) as e:
                last_err = e
                full_prompt += f"\n\nYour previous answer was invalid ({str(e)[:300]}). Return corrected JSON only."
            except LLMError:
                raise
            except Exception as e:
                raise self._fail(e) from e
        raise LLMError(f"{self.provider}/{self.model} returned invalid JSON: {last_err}")


def _loads_lenient(text: str) -> dict:
    """Parse JSON that may be wrapped in ``` fences or surrounded by prose."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object found in model output")
        return json.loads(text[start : end + 1])


def inline_refs(schema: dict) -> dict:
    """Resolve local $refs and drop $defs — not every provider supports them."""
    defs = schema.get("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(defs[node["$ref"].split("/")[-1]])
            return {k: walk(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


def skeleton(schema: dict):
    """A compact example value for a JSON schema: {"name": "<string>", "items": [ ... ]}."""
    if "anyOf" in schema:
        non_null = [s for s in schema["anyOf"] if s.get("type") != "null"]
        return skeleton(non_null[0]) if non_null else None
    if "enum" in schema:
        return "|".join(map(str, schema["enum"]))
    t = schema.get("type")
    if t == "object":
        return {k: skeleton(v) for k, v in schema.get("properties", {}).items()}
    if t == "array":
        return [skeleton(schema.get("items", {}))]
    if t == "string":
        return f"<{schema.get('description') or 'string'}>"
    return {"number": 0.0, "integer": 0, "boolean": True}.get(t)
