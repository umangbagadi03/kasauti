from .base import Image, LLMClient, LLMError


class GeminiClient(LLMClient):
    provider = "gemini"

    def __init__(self, model: str, api_key: str | None, timeout: float):
        if not api_key:
            raise LLMError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set")
        from google import genai
        from google.genai import types

        self.model = model
        self._types = types
        self._client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout * 1000))
        )

    def _call(self, system, prompt, image, json_mode: bool, schema: dict | None = None) -> str:
        t = self._types
        contents: list = []
        if image:
            contents.append(t.Part.from_bytes(data=image.data, mime_type=image.mime_type))
        contents.append(prompt)
        config = t.GenerateContentConfig(
            system_instruction=system,
            temperature=0.1,
            response_mime_type="application/json" if json_mode else None,
            response_json_schema=schema,
        )
        try:
            resp = self._client.models.generate_content(model=self.model, contents=contents, config=config)
        except Exception as e:
            if schema is None or "schema" not in str(e).lower():
                raise
            return self._call(system, prompt, image, json_mode, None)  # schema rejected: plain JSON mode
        if not resp.text:
            raise LLMError(f"gemini/{self.model}: empty response (possibly blocked)")
        return resp.text

    def _text(self, system, prompt, image):
        return self._call(system, prompt, image, json_mode=False)

    def _json(self, system, prompt, schema, image):
        return self._call(system, prompt, image, json_mode=True, schema=schema)
