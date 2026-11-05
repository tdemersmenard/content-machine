"""Modèle de données de ContentMachine."""

from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    CheckConstraint,
    Enum,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, UTCDateTime, utcnow


class ScriptStatus(StrEnum):
    DRAFT = "draft"
    APPROVED = "approved"
    PUBLISHED = "published"


class LessonScope(StrEnum):
    GLOBAL = "global"
    NICHE = "niche"


class LLMCallStatus(StrEnum):
    OK = "ok"
    INVALID_OUTPUT = "invalid_output"
    REFUSAL = "refusal"
    TRUNCATED = "truncated"
    ERROR = "error"


def _enum(enum_cls: type[StrEnum], name: str) -> Enum:
    """Enum stocké en texte avec une contrainte CHECK sur les valeurs."""
    return Enum(
        enum_cls,
        name=name,
        native_enum=False,
        create_constraint=True,
        length=20,
        values_callable=lambda cls: [member.value for member in cls],
    )


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_url: Mapped[str | None] = mapped_column(String(2048), unique=True)
    platform: Mapped[str | None] = mapped_column(String(32))
    creator: Mapped[str | None] = mapped_column(String(255))
    niche_source: Mapped[str | None] = mapped_column(String(255))
    duration_s: Mapped[float | None] = mapped_column(Float)
    posted_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    views: Mapped[int | None] = mapped_column(BigInteger)
    likes: Mapped[int | None] = mapped_column(BigInteger)
    comments: Mapped[int | None] = mapped_column(BigInteger)
    shares: Mapped[int | None] = mapped_column(BigInteger)
    saves: Mapped[int | None] = mapped_column(BigInteger)
    is_own_content: Mapped[bool] = mapped_column(default=False)
    file_path: Mapped[str | None] = mapped_column(String(1024))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class Transcript(Base):
    __tablename__ = "transcripts"

    video_id: Mapped[int] = mapped_column(
        ForeignKey("videos.id", ondelete="CASCADE"), primary_key=True
    )
    words: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    full_text: Mapped[str] = mapped_column(Text, default="")
    language: Mapped[str | None] = mapped_column(String(16))


class Shot(Base):
    __tablename__ = "shots"
    __table_args__ = (UniqueConstraint("video_id", "index", name="uq_shots_video_id_index"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    index: Mapped[int] = mapped_column(Integer)
    start_s: Mapped[float] = mapped_column(Float)
    end_s: Mapped[float] = mapped_column(Float)
    frame_path: Mapped[str | None] = mapped_column(String(1024))
    on_screen_text: Mapped[str | None] = mapped_column(Text)
    visual_description: Mapped[str | None] = mapped_column(Text)


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(primary_key=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    prompt_version: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(64))
    json_result: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class Pattern(Base):
    __tablename__ = "patterns"

    id: Mapped[int] = mapped_column(primary_key=True)
    type: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text, default="")
    template: Mapped[str] = mapped_column(Text, default="")
    examples: Mapped[list[Any]] = mapped_column(JSON, default=list)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary)
    # Paramètres d'une loi Beta : 1/1 est l'a priori neutre.
    score_alpha: Mapped[float] = mapped_column(Float, default=1.0)
    score_beta: Mapped[float] = mapped_column(Float, default=1.0)
    times_used: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)


class VideoPattern(Base):
    __tablename__ = "video_patterns"

    video_id: Mapped[int] = mapped_column(
        ForeignKey("videos.id", ondelete="CASCADE"), primary_key=True
    )
    pattern_id: Mapped[int] = mapped_column(
        ForeignKey("patterns.id", ondelete="CASCADE"), primary_key=True
    )
    role: Mapped[str] = mapped_column(String(64), primary_key=True)
    evidence: Mapped[str | None] = mapped_column(Text)


class Niche(Base):
    __tablename__ = "niches"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True)
    audience: Mapped[str] = mapped_column(Text, default="")
    offer: Mapped[str] = mapped_column(Text, default="")
    tone: Mapped[str] = mapped_column(Text, default="")
    banned_words: Mapped[list[str]] = mapped_column(JSON, default=list)
    examples: Mapped[list[Any]] = mapped_column(JSON, default=list)
    constraints: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class Script(Base):
    __tablename__ = "scripts"

    id: Mapped[int] = mapped_column(primary_key=True)
    niche_id: Mapped[int] = mapped_column(ForeignKey("niches.id"), index=True)
    source_video_id: Mapped[int | None] = mapped_column(
        ForeignKey("videos.id", ondelete="SET NULL")
    )
    pattern_ids: Mapped[list[int]] = mapped_column(JSON, default=list)
    prompt_version: Mapped[str] = mapped_column(String(64))
    content: Mapped[dict[str, Any]] = mapped_column(JSON)
    status: Mapped[ScriptStatus] = mapped_column(
        _enum(ScriptStatus, "script_status"), default=ScriptStatus.DRAFT
    )
    published_video_id: Mapped[int | None] = mapped_column(
        ForeignKey("videos.id", ondelete="SET NULL")
    )
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class Performance(Base):
    """Relevé de stats d'une vidéo publiée à un instant donné."""

    __tablename__ = "performance"
    __table_args__ = (
        UniqueConstraint("video_id", "measured_at", name="uq_performance_video_id_measured_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    script_id: Mapped[int | None] = mapped_column(
        ForeignKey("scripts.id", ondelete="SET NULL"), index=True
    )
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    measured_at: Mapped[datetime] = mapped_column(UTCDateTime)
    hours_since_post: Mapped[float | None] = mapped_column(Float)
    views: Mapped[int | None] = mapped_column(BigInteger)
    avg_watch_time_s: Mapped[float | None] = mapped_column(Float)
    retention_3s: Mapped[float | None] = mapped_column(Float)
    retention_50: Mapped[float | None] = mapped_column(Float)
    completion_rate: Mapped[float | None] = mapped_column(Float)
    shares: Mapped[int | None] = mapped_column(BigInteger)
    saves: Mapped[int | None] = mapped_column(BigInteger)
    comments: Mapped[int | None] = mapped_column(BigInteger)
    follows: Mapped[int | None] = mapped_column(BigInteger)
    notes: Mapped[str | None] = mapped_column(Text)


class Lesson(Base):
    __tablename__ = "lessons"
    __table_args__ = (
        CheckConstraint(
            "scope != 'niche' OR niche_id IS NOT NULL",
            name="niche_scope_requires_niche_id",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    scope: Mapped[LessonScope] = mapped_column(_enum(LessonScope, "lesson_scope"))
    niche_id: Mapped[int | None] = mapped_column(
        ForeignKey("niches.id", ondelete="CASCADE"), index=True
    )
    statement: Mapped[str] = mapped_column(Text)
    evidence: Mapped[list[Any]] = mapped_column(JSON, default=list)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class LLMCall(Base):
    """Une ligne par tentative d'appel à Claude."""

    __tablename__ = "llm_calls"

    id: Mapped[int] = mapped_column(primary_key=True)
    prompt_version: Mapped[str] = mapped_column(String(64), index=True)
    prompt_hash: Mapped[str] = mapped_column(String(16))
    model_requested: Mapped[str] = mapped_column(String(64))
    model_served: Mapped[str | None] = mapped_column(String(64))
    request_id: Mapped[str | None] = mapped_column(String(128))
    attempt: Mapped[int] = mapped_column(Integer)
    status: Mapped[LLMCallStatus] = mapped_column(_enum(LLMCallStatus, "llm_call_status"))
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cache_read_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cache_write_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float | None] = mapped_column(Float)
    latency_ms: Mapped[int] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(Text)
    raw_output: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)
