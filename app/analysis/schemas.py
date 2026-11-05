"""Schéma de l'analyse virale.

Deux règles de conception, vérifiées par tests/test_analysis_schema.py :

- Toute affirmation est un `Claim`, qui porte au moins une preuve horodatée.
- Aucun champ de ce que Claude remplit n'est optionnel ni nullable. Une absence s'exprime par une
  liste vide, ce qui permet de retirer une affirmation sans preuve sans casser le schéma.

Ce schéma dépasse la limite de grammaire de l'API : il est envoyé dans le prompt
(`constrained=False`) et validé ici par Pydantic.

Un champ `Claim` seul marque l'élément principal d'un objet de liste : s'il tombe, l'objet entier
est retiré. Les objets hors liste ne contiennent donc que des `list[Claim]`.
"""

from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

EvidenceKind = Literal["quote", "on_screen_text", "frame", "signal", "metadata"]


class Evidence(BaseModel):
    kind: EvidenceKind
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    excerpt: str = Field(min_length=1)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end_s < self.start_s:
            raise ValueError("end_s doit être supérieur ou égal à start_s")
        return self


class Claim(BaseModel):
    statement: str = Field(min_length=1)
    evidence: list[Evidence] = Field(min_length=1)


class Hook(BaseModel):
    spoken_text: list[Claim]
    on_screen_text: list[Claim]
    visual: list[Claim]
    hook_type: Literal[
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
    hook_type_other: str  # Description si hook_type vaut "other", sinon chaîne vide.
    hook_type_justification: list[Claim]
    psychological_mechanism: list[Claim]
    alignment: Literal["same_message", "complementary", "contradictory"]
    alignment_explanation: list[Claim]
    score: int = Field(ge=0, le=10)
    score_justification: list[Claim]


class AudienceBelief(BaseModel):
    stance: Literal["attacks", "confirms"]
    claim: Claim


class AlternativeAngle(BaseModel):
    angle: str  # Hypothèse du créateur alternatif, pas un fait sur la vidéo.
    comparison: Literal["original_better", "original_worse", "equivalent"]
    claim: Claim


class Angle(BaseModel):
    raw_topic: list[Claim]
    chosen_angle: list[Claim]
    audience_beliefs: list[AudienceBelief]
    implicit_promise: list[Claim]
    promise_payoff: list[Claim]  # Vide si la promesse n'est jamais payée.
    alternatives: list[AlternativeAngle]


class Beat(BaseModel):
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    role: Literal[
        "hook",
        "setup",
        "tension",
        "proof",
        "twist",
        "payoff",
        "cta",
        "transition",
        "other",
    ]
    claim: Claim


class Structure(BaseModel):
    beats: list[Beat]
    framework: Literal[
        "pas",
        "before_after",
        "list",
        "story_loop",
        "tutorial",
        "myth_busting",
        "demonstration",
        "hybrid",
    ]
    framework_justification: list[Claim]


class OpenLoop(BaseModel):
    opened: Claim
    closed: list[Claim]  # Vide si la boucle ne se ferme jamais.


class DropRisk(BaseModel):
    risk: Claim
    compensation: list[Claim]


class Retention(BaseModel):
    open_loops: list[OpenLoop]
    rehooks: list[Claim]
    drop_risks: list[DropRisk]
    rhythm: list[Claim]
    ending_type: Literal["replay_loop", "payoff", "cta", "abrupt", "mixed"]
    ending_effect: list[Claim]


class CommentTrigger(BaseModel):
    trigger: Literal["debate", "identification", "open_question", "deliberate_mistake", "other"]
    claim: Claim


class ShareTrigger(BaseModel):
    trigger: Literal["thats_so_you", "utility", "status", "outrage", "other"]
    claim: Claim


class Engagement(BaseModel):
    comments: list[CommentTrigger]
    shares: list[ShareTrigger]
    saves: list[Claim]


class EmotionPoint(BaseModel):
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    emotion: str
    intensity: int = Field(ge=0, le=10)
    claim: Claim


class CreatorElement(BaseModel):
    source: Literal["charisma", "existing_audience", "news_timing", "trending_sound", "other"]
    claim: Claim


class Transferability(BaseModel):
    format_elements: list[Claim]
    creator_elements: list[CreatorElement]


class VerdictReason(BaseModel):
    rank: int = Field(ge=1)
    weight_percent: int = Field(ge=0, le=100)
    confidence: Literal["low", "medium", "high"]
    claim: Claim


class Verdict(BaseModel):
    reasons: list[VerdictReason]


class PerformanceContext(BaseModel):
    video_kind: Literal["utility", "entertainment", "debate", "mixed", "unknown"]
    interpretation: list[Claim]


class ViralAnalysis(BaseModel):
    hook: Hook
    angle: Angle
    structure: Structure
    retention: Retention
    engagement: Engagement
    emotion_curve: list[EmotionPoint]
    transferability: Transferability
    verdict: Verdict
    performance_context: PerformanceContext


class DraftAnalysis(ViralAnalysis):
    """Sortie de la passe 1, avant que la vérification puisse retirer des éléments."""

    @model_validator(mode="after")
    def _complete(self) -> Self:
        if len(self.angle.alternatives) != 3:
            raise ValueError("Il faut exactement 3 angles alternatifs.")
        if sorted(reason.rank for reason in self.verdict.reasons) != [1, 2, 3]:
            raise ValueError("Il faut exactement 3 raisons de succès, de rang 1 à 3.")
        if not self.structure.beats:
            raise ValueError("Il faut au moins un beat.")
        return self


class Correction(BaseModel):
    path: str
    change: str


class Removal(BaseModel):
    path: str
    statement: str
    reason: str


class VerifiedAnalysis(BaseModel):
    """Sortie de la passe 2."""

    analysis: ViralAnalysis
    corrections: list[Correction]
    removals: list[Removal]


class PerformanceMetrics(BaseModel):
    """Ratios calculés par le code. None si une stat manque."""

    views_per_follower: float | None
    share_rate: float | None
    save_rate: float | None
    comment_rate: float | None
    like_rate: float | None


class Verification(BaseModel):
    prompt_version: str
    model: str
    llm_corrections: list[Correction]
    llm_removals: list[Removal]
    code_corrections: list[Correction]
    code_removals: list[Removal]


class AnalysisDocument(BaseModel):
    """Contenu de analyses.json_result."""

    analysis: ViralAnalysis
    metrics: PerformanceMetrics
    verification: Verification
