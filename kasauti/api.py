"""HTTP API.  Run:  uvicorn kasauti.api:app --reload"""

import json
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import ValidationError

from . import __version__
from .config import settings
from .llm import LLMError, available_providers
from .models import AnalysisResult, UserProfile
from .pipeline import analyze

MAX_UPLOAD = 10 * 1024 * 1024
STATIC = Path(__file__).parent / "static"
SAMPLES = Path(__file__).parent.parent / "samples"

app = FastAPI(title="Kasauti", version=__version__, description="Food label analysis with pluggable LLM providers.")


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(STATIC / "index.html")


@app.get("/samples/{name}", include_in_schema=False, response_class=PlainTextResponse)
def sample(name: str):
    path = SAMPLES / f"{name}.txt"
    if not name.isalnum() or not path.is_file():
        raise HTTPException(404, "No such sample")
    return path.read_text()


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__, "default_provider": settings.provider}


@app.get("/providers")
def providers():
    return available_providers()


@app.post("/analyze", response_model=AnalysisResult)
def analyze_endpoint(
    image: UploadFile | None = File(None, description="Photo of the label (ingredients + nutrition table)"),
    text: str | None = Form(None, description="Label text, if typed/pasted instead of a photo"),
    barcode: str | None = Form(None, description="EAN/UPC barcode — looked up on Open Food Facts first"),
    profile: str | None = Form(None, description='JSON, e.g. {"conditions":["diabetes"],"allergies":["peanut"],"region":"IN"}'),
    provider: str | None = Form(None, description="anthropic | gemini | openai | ollama"),
    model: str | None = Form(None),
    # str, not bool: FastAPI reads a *missing* bool form field as False (HTML checkbox rules),
    # which would silently switch callers who omit it to offline mode.
    use_llm: str | None = Form(None, description="false = offline rules only (default: true)"),
):
    # Plain `def`: FastAPI runs it in a worker thread, so blocking LLM calls don't stall the server.
    data = None
    if image is not None and image.filename:
        data = image.file.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, "Image larger than 10 MB")
    try:
        user = UserProfile.model_validate(json.loads(profile)) if profile else UserProfile(region=settings.default_region)
    except (json.JSONDecodeError, ValidationError) as e:
        raise HTTPException(422, f"Invalid profile: {e}")

    try:
        return analyze(
            image=data, text=text, barcode=barcode, profile=user,
            provider=provider, model=model,
            use_llm=(use_llm or "true").strip().lower() not in ("false", "0", "off", "no"),
        )
    except ValueError as e:
        raise HTTPException(422, str(e))
    except LLMError as e:
        raise HTTPException(502, str(e))
