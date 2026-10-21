import os
from dotenv import load_dotenv

load_dotenv()
class Config:
    LLM_API_KEY = os.getenv('LLM_API_KEY')
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
    CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"

    if not LLM_API_KEY:
        raise ValueError("Missing critical environment variable: LLM_API_KEY")
    
    if not isinstance(CACHE_ENABLED, bool):
        raise ValueError("CACHE_ENABLED must be a boolean value (true/false).")                         

