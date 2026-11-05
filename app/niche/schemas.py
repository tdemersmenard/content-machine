"""Schémas du profil de niche.

Rien n'est inventé : une information que l'entrevue n'a pas donnée reste vide (chaîne vide ou
liste vide). Ce que Claude remplit n'a ni champ optionnel ni union.
"""

from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

InterviewSection = Literal["identity", "avatar", "beliefs", "voice", "filming", "references"]
INTERVIEW_SECTIONS: tuple[str, ...] = (
    "identity",
    "avatar",
    "beliefs",
    "voice",
    "filming",
    "references",
)


class InterviewTurn(BaseModel):
    covered_sections: list[InterviewSection]
    done: bool
    next_question: str  # Chaîne vide si done.

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.done and set(self.covered_sections) != set(INTERVIEW_SECTIONS):
            raise ValueError("L'entrevue ne peut finir que si les 6 sections sont couvertes.")
        if not self.done and not self.next_question.strip():
            raise ValueError("Il faut une prochaine question tant que l'entrevue continue.")
        return self


class Identity(BaseModel):
    who_i_am: str
    offer: str
    goal: Literal["followers", "leads", "sales", "authority", "unknown"]
    goal_details: str


class Avatar(BaseModel):
    situation: str
    frustrations: list[str]
    desires: list[str]
    vocabulary: list[str]
    objections: list[str]
    already_tried: list[str]


class Voice(BaseModel):
    tone: str
    expressions: list[str]
    never_say: list[str]
    language: str
    speech_register: str


class Filming(BaseModel):
    on_camera: Literal["yes", "no", "mixed", "unknown"]
    equipment: str
    time_available: str
    location: str
    other_constraints: list[str]


class Reference(BaseModel):
    name: str
    why: str


class References(BaseModel):
    competitors: list[Reference]
    inspirations: list[Reference]


class ProfileDraft(BaseModel):
    """Ce que Claude tire de l'entrevue."""

    niche_name: str = Field(min_length=1, max_length=120)
    identity: Identity
    avatar: Avatar
    dominant_beliefs: list[str]
    voice: Voice
    filming: Filming
    references: References


class VoiceTraits(BaseModel):
    verbal_tics: list[str]
    signature_expressions: list[str]
    style_summary: str


class VoiceProfile(BaseModel):
    """Style mesuré sur les vidéos de l'utilisateur."""

    video_ids: list[int]
    avg_sentence_words: float
    median_sentence_words: float
    short_sentence_ratio: float
    words_per_second: float
    silences_per_minute: float
    verbal_tics: list[str]
    signature_expressions: list[str]
    style_summary: str
    prompt_version: str


class NicheProfile(ProfileDraft):
    # Faits vécus confirmés par le créateur : la seule matière pour parler en son nom.
    creator_facts: list[str] = []
    voice_profile: VoiceProfile | None = None


class TopicIdea(BaseModel):
    title: str = Field(min_length=1)
    pain: str = Field(min_length=1)
    intensity: int = Field(ge=1, le=10)
    rationale: str = Field(min_length=1)
    # "stated" : dit dans le profil. "inferred" : hypothèse, à valider auprès de l'audience.
    grounding: Literal["stated", "inferred"]


class TopicBatch(BaseModel):
    topics: list[TopicIdea]
