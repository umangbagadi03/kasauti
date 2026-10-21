"""OpenAI, and anything speaking the OpenAI Chat Completions API (Ollama included)."""

import base64

from .base import Image, LLMClient, LLMError


class OpenAICompatibleClient(LLMClient):
    def __init__(self, provider: str, model: str, api_key: str | None, base_url: str | None, timeout: float):
        if provider == "openai" and not api_key:
            raise LLMError("OPENAI_API_KEY is not set")
        import openai

        self.provider = provider
        self.model = model
        self._client = openai.OpenAI(api_key=api_key or "unused", base_url=base_url, timeout=timeout, max_retries=2)

    def _messages(self, system: str, prompt: str, image: Image | None) -> list[dict]:
        content: list[dict] = [{"type": "text", "text": prompt}]
        if image:
            url = f"data:{image.mime_type};base64,{base64.b64encode(image.data).decode()}"
            content.insert(0, {"type": "image_url", "image_url": {"url": url}})
        return [{"role": "system", "content": system}, {"role": "user", "content": content}]

    def _call(self, system, prompt, image, json_mode: bool, schema: dict | None = None) -> str:
        # No temperature / max_tokens: newer OpenAI reasoning models reject non-defaults.
        kwargs = {}
        if schema is not None:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {"name": "result", "schema": schema, "strict": False}}
        elif json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self._client.chat.completions.create(
                model=self.model, messages=self._messages(system, prompt, image), **kwargs
            )
        except Exception as e:
            if schema is None or not any(w in str(e).lower() for w in ("schema", "response_format")):
                raise
            return self._call(system, prompt, image, json_mode, None)  # schema rejected: plain JSON mode
        text = resp.choices[0].message.content
        if not text:
            raise LLMError(f"{self.provider}/{self.model}: empty response")
        return text

    def _text(self, system, prompt, image):
        return self._call(system, prompt, image, json_mode=False)

    def _json(self, system, prompt, schema, image):
        return self._call(system, prompt, image, json_mode=True, schema=schema)
