"""
Configuration for the Deep Research Agent.
Inspired by langchain-ai/open_deep_research configuration.py.
"""

import os
from typing import Optional

from pydantic import BaseModel, Field


class Configuration(BaseModel):
    """Main configuration for the Deep Research Agent."""

    # ── LLM Models ──────────────────────────────────────────
    clarify_model: str = Field(
        default="meta-llama/llama-4-scout-17b-16e-instruct",
        description="Model for conversational clarification decisions.",
    )
    planner_model: str = Field(
        default="meta-llama/llama-4-scout-17b-16e-instruct",
        description="Model for generating search plans.",
    )
    researcher_model: str = Field(
        default="meta-llama/llama-4-scout-17b-16e-instruct",
        description="Model for analysing search results.",
    )
    filter_model: str = Field(
        default="meta-llama/llama-4-scout-17b-16e-instruct",
        description="Model for filtering events by Argentina relevance.",
    )
    report_model: str = Field(
        default="meta-llama/llama-4-scout-17b-16e-instruct",
        description="Model for generating the final structured report.",
    )
    synthesis_model: str = Field(
        default="meta-llama/llama-4-scout-17b-16e-instruct",
        description="Model for synthesizing large search results.",
    )

    # ── Context Config ──────────────────────────────────────
    context_synthesis_threshold: int = Field(
        default=25000,
        description="Threshold (chars) to trigger context synthesis to avoid LLM overflow.",
    )

    # ── Search Config ───────────────────────────────────────
    search_max_results: int = Field(
        default=8,
        description="Max results per search query.",
    )

    # ── Research Config ─────────────────────────────────────
    max_queries_per_category: int = Field(
        default=3,
        description="Max search queries to generate per category.",
    )

    @classmethod
    def from_env(cls) -> "Configuration":
        """Build configuration, allowing env-var overrides."""
        overrides = {}
        for field_name in cls.model_fields:
            env_val = os.environ.get(field_name.upper())
            if env_val is not None:
                overrides[field_name] = env_val
        return cls(**overrides)
