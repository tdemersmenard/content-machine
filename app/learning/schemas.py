"""Schémas de l'apprentissage."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Horizon = Literal[24, 72, 168]
CauseCategory = Literal["idea_angle", "hook", "execution", "structure", "external"]


class PerformanceInput(BaseModel):
    """Un relevé. Une stat inconnue reste None, elle n'est jamais estimée."""

    horizon_h: Horizon
    measured_at: datetime | None = None
    video_url: str | None = None
    views: int | None = Field(default=None, ge=0)
    likes: int | None = Field(default=None, ge=0)
    avg_watch_time_s: float | None = Field(default=None, ge=0)
    retention_3s: float | None = Field(default=None, ge=0, le=1)
    retention_50: float | None = Field(default=None, ge=0, le=1)
    completion_rate: float | None = Field(default=None, ge=0, le=1)
    shares: int | None = Field(default=None, ge=0)
    saves: int | None = Field(default=None, ge=0)
    comments: int | None = Field(default=None, ge=0)
    follows: int | None = Field(default=None, ge=0)
    notes: str | None = None


class Cause(BaseModel):
    category: CauseCategory
    weight_percent: int = Field(ge=0, le=100)
    explanation: str
    evidence: str


class PostMortemResult(BaseModel):
    summary: str
    causes: list[Cause]
    script_vs_actual: str
    keep: list[str]
    change: list[str]
    prediction_review: str


class LessonDraft(BaseModel):
    statement: str = Field(min_length=1)
    scope: Literal["global", "niche"]
    evidence_video_ids: list[int]
    confidence: float = Field(ge=0, le=1)
    avoid_pattern_ids: list[int]


class Contradiction(BaseModel):
    lesson_id: int
    reason: str


class LessonSynthesis(BaseModel):
    lessons: list[LessonDraft]
    contradicted: list[Contradiction]
