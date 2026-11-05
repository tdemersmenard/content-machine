"""Schémas de la bibliothèque de patterns.

Ce que Claude remplit n'a ni champ optionnel ni union, comme le schéma d'analyse.
"""

import re
from datetime import datetime
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.analysis.schemas import Evidence
from app.db.models import PatternType

_PLACEHOLDER_RE = re.compile(r"\[[^\[\]]+\]")

PatternRole = Literal["opening", "development", "climax", "closing", "throughout"]


class PatternCandidate(BaseModel):
    type: PatternType
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1)
    template: str = Field(min_length=1)
    when_to_use: str = Field(min_length=1)
    when_not_to_use: str = Field(min_length=1)
    role: PatternRole
    evidence: list[Evidence] = Field(min_length=1)

    @model_validator(mode="after")
    def _template_has_placeholders(self) -> Self:
        if not _PLACEHOLDER_RE.search(self.template):
            raise ValueError("Le template doit contenir au moins un trou entre crochets.")
        return self

    def embedding_text(self) -> str:
        return embedding_text(self.name, self.description, self.template)


def embedding_text(name: str, description: str, template: str) -> str:
    return f"{name}\n{description}\n{template}"


class ExtractedPatterns(BaseModel):
    candidates: list[PatternCandidate]


class MergeDecision(BaseModel):
    same_mechanism: bool
    reasoning: str
    merged_description: str  # Chaîne vide si les mécanismes diffèrent.

    @model_validator(mode="after")
    def _merge_needs_description(self) -> Self:
        if self.same_mechanism and not self.merged_description.strip():
            raise ValueError("Une fusion exige une description enrichie.")
        return self


class PatternOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    type: PatternType
    name: str
    description: str
    template: str
    when_to_use: str
    when_not_to_use: str
    examples: list[dict[str, Any]]
    score_alpha: float
    score_beta: float
    score_mean: float
    example_count: int
    times_used: int
    created_at: datetime
    updated_at: datetime


class PatternVideoOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    video_id: int
    role: str
    evidence: str | None


class PatternDetail(PatternOut):
    videos: list[PatternVideoOut]


class PatternSearchHit(PatternOut):
    similarity: float
