"""Kasauti — food label analysis with deterministic scoring and pluggable LLMs."""

from .models import AnalysisResult, UserProfile
from .pipeline import analyze

__all__ = ["analyze", "AnalysisResult", "UserProfile"]
__version__ = "0.2.0"
