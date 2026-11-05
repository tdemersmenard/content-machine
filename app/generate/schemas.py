"""Schémas de la génération de scripts.

Ce que Claude remplit n'a ni champ optionnel ni union.
"""

from typing import Literal

from pydantic import BaseModel, Field

HookType = Literal[
    "contrarian",
    "result_first",
    "question",
    "curiosity_gap",
    "visual_pattern_interrupt",
    "identification",
    "fear_loss",
    "shock_number",
    "in_medias_res",
    "demonstration",
    "other",
]
BeatRole = Literal["setup", "tension", "proof", "twist", "payoff", "cta", "transition", "other"]


class SkeletonBeat(BaseModel):
    role: BeatRole
    mechanism: str = Field(min_length=1)
    start_pct: int = Field(ge=0, le=100)
    end_pct: int = Field(ge=0, le=100)


class Skeleton(BaseModel):
    beats: list[SkeletonBeat]


class TransposedBeat(BaseModel):
    beat_index: int
    pain: str
    belief: str
    proof: str
    example: str


class Variant(BaseModel):
    angle: str = Field(min_length=1)
    belief_attacked: str = Field(min_length=1)
    topic_ids: list[str]
    beats: list[TransposedBeat]


class Transposition(BaseModel):
    variants: list[Variant]


class Idea(BaseModel):
    combination_index: int
    topic_id: str
    angle: str = Field(min_length=1)
    belief_attacked: str = Field(min_length=1)
    why: str = Field(min_length=1)
    potential: int = Field(ge=1, le=10)


class IdeaBatch(BaseModel):
    ideas: list[Idea]


class HookOption(BaseModel):
    spoken: str = Field(min_length=1)
    on_screen_text: str
    visual: str
    hook_type: HookType


class ScriptBeat(BaseModel):
    role: BeatRole
    target_start_s: float = Field(ge=0)
    target_end_s: float = Field(ge=0)
    spoken: str = Field(min_length=1)
    on_screen_text: str
    visual: str
    performance_note: str


class OpenLoop(BaseModel):
    description: str
    opened_beat: int
    closed_beat: int


class Rationale(BaseModel):
    choice: str
    reason: str
    source: Literal["source_analysis", "lesson", "pattern", "profile"]


class ScriptDraft(BaseModel):
    """Ce que Claude écrit. Les beats viennent après le hook choisi."""

    internal_title: str = Field(min_length=1)
    angle: str = Field(min_length=1)
    belief_attacked: str = Field(min_length=1)
    hooks: list[HookOption]
    beats: list[ScriptBeat]
    open_loops: list[OpenLoop]
    ending: str
    cta: str
    caption: str
    hashtags: list[str]
    pattern_ids: list[int]
    rationale: list[Rationale]


class SwipePoint(BaseModel):
    quote: str
    reason: str


class Critique(BaseModel):
    hook_score: int = Field(ge=0, le=10)
    swipe_points: list[SwipePoint]
    fixes: list[str]


class PredictedScore(BaseModel):
    """Estimation sur 100, comparable plus tard aux stats réelles.

    Formule simple et non calibrée : 60 % note du hook, 40 % moyenne Beta des patterns.
    """

    value: float
    hook_score: int
    pattern_mean: float


class ScriptContent(ScriptDraft):
    """Contenu de scripts.content."""

    mode: Literal["adapt", "generate"]
    topic_ids: list[str]
    estimated_duration_s: float
    words_per_second: float
    duration_source: Literal["voice_profile", "source_video", "default"]
    to_verify: list[str]
    critiques: list[Critique]  # Historique complet, une critique par version du script.
    final_critique: Critique  # Celle de la version retenue, la mieux notée.
    predicted_score: PredictedScore
    prompt_versions: dict[str, str]
