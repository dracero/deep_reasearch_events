"""
Configuration for the Travel Agent.
"""

import os
from typing import Optional

from pydantic import BaseModel, Field


class Configuration(BaseModel):
    """Main configuration for the Explainer Agent."""

    # ── LLM Models ──────────────────────────────────────────
    answer_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Model for generating the final explanation based on scraped content.",
    )
    clarify_model: str = Field(
        default="llama-3.1-8b-instant",
        description="Model for generating the pre-search clarifying question. Tiny task, fast model.",
    )

    # ── Scraping Config ───────────────────────────────────────
    max_content_length: int = Field(
        default=40000,
        description="Max characters to keep from the scraped webpage content.",
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

