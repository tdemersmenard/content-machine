"""Objets de données de l'ingestion."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class Word(BaseModel):
    word: str
    start: float
    end: float
    probability: float | None = None


class Transcription(BaseModel):
    words: list[Word] = []
    full_text: str = ""
    language: str | None = None


class WordRateBucket(BaseModel):
    start_s: float
    end_s: float
    words: int
    words_per_second: float


class Silence(BaseModel):
    start_s: float
    end_s: float
    duration_s: float


class PromiseMoment(BaseModel):
    text: str
    start_s: float
    end_s: float


class Signals(BaseModel):
    words_per_second: float
    words_per_second_by_3s: list[WordRateBucket]
    shot_count: int
    avg_shot_duration_s: float
    cuts_first_3s: int
    silences: list[Silence]
    promise: PromiseMoment | None


class VideoOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    source_url: str | None
    platform: str | None
    creator: str | None
    creator_followers: int | None
    niche_source: str | None
    duration_s: float | None
    posted_at: datetime | None
    views: int | None
    likes: int | None
    comments: int | None
    shares: int | None
    saves: int | None
    description: str | None
    hashtags: list[str]
    sound: str | None
    is_own_content: bool
    file_path: str | None
    file_hash: str | None
    ingested_at: datetime | None


class TranscriptOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    words: list[dict[str, Any]]
    full_text: str
    language: str | None


class ShotOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    index: int
    start_s: float
    end_s: float
    frame_path: str | None
    on_screen_text: str | None
    visual_description: str | None


class FrameOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    t_s: float
    reason: str
    shot_index: int
    frame_path: str
    on_screen_text: str | None
    visual_description: str | None
    framing: str | None
    face_present: bool | None
    camera_motion: str | None
    graphic_elements: list[str] | None


class VideoBundle(BaseModel):
    """Tout ce que l'ingestion sait d'une vidéo."""

    video: VideoOut
    transcript: TranscriptOut | None
    shots: list[ShotOut]
    frames: list[FrameOut]
    signals: Signals | None
    timeline_path: str | None
