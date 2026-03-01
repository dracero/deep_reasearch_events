"""
Configuration for the Travel Agent.
"""

import os
from typing import Optional

from pydantic import BaseModel, Field


class Configuration(BaseModel):
    """Main configuration for the Travel Agent."""

    # ── LLM Models ──────────────────────────────────────────
    # NOTE: We deliberately use DIFFERENT models per node-role to distribute
    # load across separate Groq token/request pools and avoid 429s.
    planner_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Model for generating multi-segment route plans. Needs high reasoning.",
    )
    clarify_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Model for generating the pre-search clarifying question. Tiny task, fast model.",
    )
    researcher_model: str = Field(
        default="llama-3.3-70b-versatile",    # High RPM limit — fires in parallel for each segment
        description="Model for analysing search results and extracting routes per segment.",
    )
    ranker_model: str = Field(
        default="llama-3.3-70b-versatile",   # Smarter model for final ranking logic
        description="Model for ranking and optimizing routes.",
    )
    report_model: str = Field(
        default="llama-3.3-70b-versatile",      # Fast model, just formats JSON output
        description="Model for generating the final itinerary JSON.",
    )

    # ── Search Config ───────────────────────────────────────
    tavily_max_results: int = Field(
        default=3,                            # Reduced from 5 to cut latency per segment
        description="Max results per Tavily search query.",
    )
    tavily_search_depth: str = Field(
        default="basic",                      # basic is faster and sufficient for price estimation
        description="Tavily search depth: 'basic' or 'advanced'.",
    )

    # ── Research Config ─────────────────────────────────────
    max_queries_per_type: int = Field(
        default=2,
        description="Max search queries to generate per search type.",
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

