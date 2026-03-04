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
    planner_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Model for generating search plans.",
    )
    researcher_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Model for analysing search results.",
    )
    filter_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Model for filtering events by Argentina relevance.",
    )
    report_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Model for generating the final structured report.",
    )

    # ── Search Config ───────────────────────────────────────
    tavily_max_results: int = Field(
        default=8,
        description="Max results per Tavily search query.",
    )
    tavily_search_depth: str = Field(
        default="advanced",
        description="Tavily search depth: 'basic' or 'advanced'.",
    )
    search_max_results: int = Field(
        default=8,
        description="Max results per DDG search query (alias for tavily_max_results).",
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
