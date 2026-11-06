"""Génération de scripts : adaptation d'une vidéo, ou idées par Thompson sampling."""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Self

import anthropic
import numpy as np
from pydantic import model_validator
from sqlalchemy import or_, select, update
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.base import utcnow
from app.db.models import (
    Analysis,
    Lesson,
    LessonScope,
    Niche,
    Pattern,
    PatternType,
    Performance,
    Script,
    ScriptStatus,
    Transcript,
    Video,
    VideoPattern,
)
from app.generate.guards import (
    MAX_COPIED_WORDS,
    ScriptGuard,
    copied_runs,
    flag_script,
    ngrams,
    numbers_in,
    spoken_texts,
    verification_notes,
)
from app.generate.novelty import NoveltyFilter, Repeat
from app.generate.report import render_script
from app.generate.schemas import (
    Critique,
    IdeaBatch,
    PredictedScore,
    ScriptContent,
    ScriptDraft,
    Skeleton,
    Transposition,
)
from app.llm.client import ClaudeClient
from app.llm.errors import ClaudeError
from app.niche.profile import deep_merge, load_profile
from app.niche.schemas import NicheProfile
from app.niche.topics import set_topic_status
from app.niche.voice import contains_phrase
from app.patterns.embeddings import Embedder
from app.patterns.search import niche_scores, score_mean

logger = logging.getLogger(__name__)

TARGET_HOOK_SCORE = 8
MAX_REWRITES = 2
DEFAULT_WORDS_PER_SECOND = 3.0
DEFAULT_TARGET_DURATION_S = 60.0
SKELETON_MAX_RUN = 4  # Le squelette ne garde aucune suite de 4 mots de la source.
ACTIVE_TOPIC_LIMIT = 30


class GenerationError(Exception):
    """La génération ne peut pas démarrer."""


@dataclass
class GenerationResult:
    scripts: list[Script] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    repeats: list[Repeat] = field(default_factory=list)  # Idées écartées car déjà produites.


@dataclass
class Brief:
    """Tout ce qu'il faut pour écrire un script."""

    mode: str
    niche: Niche
    profile: NicheProfile
    pattern_ids: list[int]
    topic_ids: list[str]
    content: dict[str, Any]
    guard: ScriptGuard
    words_per_second: float
    duration_source: str
    source_video_id: int | None = None
    experiment: bool = False
    core_idea: str = ""


def _model() -> str:
    return get_settings().model_generation


def _niche_with_profile(session: Session, niche_id: int) -> tuple[Niche, NicheProfile]:
    niche = session.get(Niche, niche_id)
    if niche is None:
        raise LookupError(f"Niche {niche_id} inconnue.")
    if niche.profile is None:
        raise GenerationError(f"La niche {niche_id} n'a pas de profil.")
    return niche, load_profile(niche)


def _profile_json(profile: NicheProfile) -> str:
    return profile.model_dump_json()


def _pattern_cards(patterns: list[Pattern]) -> list[dict[str, Any]]:
    return [
        {
            "id": item.id,
            "type": item.type,
            "name": item.name,
            "description": item.description,
            "template": item.template,
            "when_to_use": item.when_to_use,
            "when_not_to_use": item.when_not_to_use,
        }
        for item in patterns
    ]


def _active_topics(niche: Niche) -> list[dict[str, Any]]:
    active = [item for item in niche.topics if item["status"] == "new"]
    return active[:ACTIVE_TOPIC_LIMIT]


def active_lessons(session: Session, niche_id: int) -> list[Lesson]:
    return list(
        session.scalars(
            select(Lesson).where(
                Lesson.active.is_(True),
                or_(Lesson.scope == LessonScope.GLOBAL, Lesson.niche_id == niche_id),
            )
        )
    )


def avoided_pattern_ids(lessons: list[Lesson]) -> set[int]:
    """Une leçon écarte un pattern avec une preuve {"pattern_id": 3, "effect": "avoid"}."""
    return {
        item["pattern_id"]
        for lesson in lessons
        for item in lesson.evidence
        if isinstance(item, dict) and item.get("effect") == "avoid" and "pattern_id" in item
    }


def _lesson_texts(lessons: list[Lesson]) -> list[str]:
    return [lesson.statement for lesson in lessons]


def _voice_speed(profile: NicheProfile, fallback: float | None) -> tuple[float, str]:
    if profile.voice_profile is not None:
        return profile.voice_profile.words_per_second, "voice_profile"
    if fallback:
        return fallback, "source_video"
    return DEFAULT_WORDS_PER_SECOND, "default"


def _allowed_numbers(profile: NicheProfile, topics: list[dict[str, Any]]) -> set[str]:
    """Seuls les chiffres du profil et des sujets retenus sont considérés comme vérifiés."""
    data = profile.model_dump(mode="json", exclude={"voice_profile"})
    return numbers_in(json.dumps([data, topics], ensure_ascii=False))


def _apply_timing(draft: ScriptDraft, words_per_second: float) -> float:
    """Recalcule les temps cibles à partir du débit. Renvoie la durée totale."""
    elapsed = len(draft.hooks[0].spoken.split()) / words_per_second
    for beat in draft.beats:
        beat.target_start_s = round(elapsed, 1)
        elapsed += len(beat.spoken.split()) / words_per_second
        beat.target_end_s = round(elapsed, 1)
    return round(elapsed, 1)


def _predict(hook_score: int, patterns: list[Pattern]) -> PredictedScore:
    pattern_mean = mean(score_mean(item) for item in patterns) if patterns else 0.5
    return PredictedScore(
        value=round(hook_score * 10 * 0.6 + pattern_mean * 100 * 0.4, 1),
        hook_score=hook_score,
        pattern_mean=round(pattern_mean, 3),
    )


def _critique(claude: ClaudeClient, draft: ScriptDraft, brief: Brief) -> Critique:
    critique = claude.call_json(
        prompt="script_critic",
        version="v1",
        schema=Critique,
        model=_model(),
        variables={
            "avatar": brief.profile.avatar.model_dump_json(),
            "script": draft.model_dump_json(),
            "lessons": json.dumps(brief.content.get("lessons", []), ensure_ascii=False),
        },
        effort="medium",
    ).data
    texts = [draft.model_dump_json()]
    critique.swipe_points = [
        item for item in critique.swipe_points if contains_phrase(texts, item.quote)
    ]
    return critique


def _write(claude: ClaudeClient, brief: Brief) -> ScriptDraft:
    return claude.call_json(
        prompt="script_write",
        version="v1",
        schema=brief.guard.schema(),
        model=_model(),
        variables={"brief": json.dumps(brief.content, ensure_ascii=False)},
        effort="high",
    ).data


def _rewrite(
    claude: ClaudeClient, brief: Brief, draft: ScriptDraft, critique: Critique
) -> ScriptDraft:
    return claude.call_json(
        prompt="script_rewrite",
        version="v1",
        schema=brief.guard.schema(),
        model=_model(),
        variables={
            "brief": json.dumps(brief.content, ensure_ascii=False),
            "script": draft.model_dump_json(),
            "critique": critique.model_dump_json(),
        },
        effort="high",
    ).data


def _polish(
    claude: ClaudeClient, brief: Brief, draft: ScriptDraft
) -> tuple[ScriptDraft, Critique, list[Critique]]:
    """Réécrit tant que le hook est sous 8, 2 fois au plus, et garde la meilleure version."""
    critique = _critique(claude, draft, brief)
    critiques = [critique]
    best, best_critique = draft, critique

    for _ in range(MAX_REWRITES):
        if best_critique.hook_score >= TARGET_HOOK_SCORE:
            break
        try:
            candidate = _rewrite(claude, brief, best, best_critique)
        except ClaudeError as exc:
            logger.warning("Réécriture abandonnée, la version précédente reste : %s", exc)
            break
        candidate_critique = _critique(claude, candidate, brief)
        critiques.append(candidate_critique)
        if candidate_critique.hook_score >= best_critique.hook_score:
            best, best_critique = candidate, candidate_critique

    return best, best_critique, critiques


def script_path(script_id: int) -> Path:
    folder = get_settings().data_dir.resolve() / "scripts"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"script_{script_id}.md"


def _produce(session: Session, claude: ClaudeClient, brief: Brief) -> Script:
    draft, final_critique, critiques = _polish(claude, brief, _write(claude, brief))

    topics = [item for item in brief.niche.topics if item["id"] in brief.topic_ids]
    flag_script(draft, _allowed_numbers(brief.profile, topics))
    duration = _apply_timing(draft, brief.words_per_second)
    patterns = list(session.scalars(select(Pattern).where(Pattern.id.in_(draft.pattern_ids))))

    content = ScriptContent(
        **ScriptDraft.model_validate(draft.model_dump()).model_dump(),
        mode=brief.mode,
        experiment=brief.experiment,
        core_idea=brief.core_idea,
        topic_ids=brief.topic_ids,
        estimated_duration_s=duration,
        words_per_second=brief.words_per_second,
        duration_source=brief.duration_source,
        to_verify=verification_notes(spoken_texts(draft)),
        critiques=critiques,
        final_critique=final_critique,
        predicted_score=_predict(final_critique.hook_score, patterns),
        prompt_versions={"write": "script_write/v1", "critic": "script_critic/v1"},
    )
    script = Script(
        niche_id=brief.niche.id,
        source_video_id=brief.source_video_id,
        pattern_ids=content.pattern_ids,
        prompt_version="script_write/v1",
        content=content.model_dump(mode="json"),
        predicted_score=content.predicted_score.value,
    )
    session.add(script)
    session.commit()
    script_path(script.id).write_text(render_script(content), encoding="utf-8")
    return script


def _produce_all(session: Session, claude: ClaudeClient, briefs: list[Brief]) -> GenerationResult:
    """Un script qui échoue n'annule pas ceux qui sont déjà payés."""
    result = GenerationResult()
    for brief in briefs:
        try:
            result.scripts.append(_produce(session, claude, brief))
        except (ClaudeError, anthropic.APIError) as exc:
            session.rollback()
            logger.warning("Script abandonné : %s", exc)
            result.failures.append(str(exc))
    if not result.scripts and result.failures:
        raise GenerationError(f"Aucun script produit : {result.failures[0]}")
    return result


def _skeleton_schema(source: set[tuple[str, ...]]) -> type[Skeleton]:
    class CleanSkeleton(Skeleton):
        @model_validator(mode="after")
        def _abstract(self) -> Self:
            if not self.beats:
                raise ValueError("Il faut au moins un beat.")
            for beat in self.beats:
                copied = copied_runs(beat.mechanism, source, SKELETON_MAX_RUN)
                if copied:
                    raise ValueError(f"Mots repris de la vidéo source : « {copied[0]} »")
                if beat.end_pct < beat.start_pct:
                    raise ValueError("end_pct doit être supérieur ou égal à start_pct.")
            return self

    return CleanSkeleton


def _transposition_schema(count: int, beat_count: int, topic_ids: set[str]) -> type[Transposition]:
    class ExactTransposition(Transposition):
        @model_validator(mode="after")
        def _complete(self) -> Self:
            if len(self.variants) != count:
                raise ValueError(f"Il faut exactement {count} variantes.")
            for variant in self.variants:
                indexes = sorted(beat.beat_index for beat in variant.beats)
                if indexes != list(range(beat_count)):
                    raise ValueError(f"Il faut un équivalent par beat, de 0 à {beat_count - 1}.")
                unknown = set(variant.topic_ids) - topic_ids
                if unknown:
                    raise ValueError(f"Sujets inconnus : {sorted(unknown)}")
            return self

    return ExactTransposition


def adapt(
    session: Session,
    video_id: int,
    niche_id: int,
    claude: ClaudeClient,
    n_variantes: int = 3,
    embedder: Embedder | None = None,
) -> GenerationResult:
    """Adapte une vidéo analysée à une niche : squelette, transposition, écriture."""
    niche, profile = _niche_with_profile(session, niche_id)
    novelty = NoveltyFilter(session, niche_id, embedder)
    video = session.get(Video, video_id)
    if video is None:
        raise LookupError(f"Vidéo {video_id} inconnue.")
    while video.duplicate_of_id is not None:
        video = session.get(Video, video.duplicate_of_id)

    analysis = session.scalar(
        select(Analysis).where(Analysis.video_id == video.id).order_by(Analysis.id.desc()).limit(1)
    )
    if analysis is None:
        raise GenerationError(f"La vidéo {video.id} n'a pas d'analyse : lance cm analyze.")

    patterns = list(
        session.scalars(
            select(Pattern)
            .join(VideoPattern, VideoPattern.pattern_id == Pattern.id)
            .where(VideoPattern.video_id == video.id)
            .distinct()
            .order_by(Pattern.id)
        )
    )
    transcript = session.get(Transcript, video.id)
    source_text = transcript.full_text if transcript else ""

    skeleton = claude.call_json(
        prompt="script_skeleton",
        version="v1",
        schema=_skeleton_schema(ngrams(source_text, SKELETON_MAX_RUN)),
        model=_model(),
        variables={
            "analysis": json.dumps(analysis.json_result["analysis"], ensure_ascii=False),
            "patterns": json.dumps(_pattern_cards(patterns), ensure_ascii=False),
        },
        effort="medium",
    ).data

    topics = _active_topics(niche)
    lessons = active_lessons(session, niche_id)
    transposition = claude.call_json(
        prompt="script_transpose",
        version="v2",
        schema=_transposition_schema(
            n_variantes, len(skeleton.beats), {item["id"] for item in topics}
        ),
        model=_model(),
        variables={
            "count": n_variantes,
            "skeleton": skeleton.model_dump_json(),
            "profile": _profile_json(profile),
            "topics": json.dumps(topics, ensure_ascii=False),
            "lessons": json.dumps(_lesson_texts(lessons), ensure_ascii=False),
            "past_ideas": json.dumps(novelty.past, ensure_ascii=False),
        },
        effort="high",
    ).data

    verdicts = novelty.check([item.core_idea for item in transposition.variants])
    variants = [
        item
        for item, repeat in zip(transposition.variants, verdicts, strict=True)
        if repeat is None
    ]
    repeats = [item for item in verdicts if item is not None]
    if not variants:
        raise GenerationError("Toutes les variantes répètent des idées déjà produites.")

    fallback = (video.signals or {}).get("words_per_second")
    speed, speed_source = _voice_speed(profile, fallback)
    guard = ScriptGuard(
        source_runs=ngrams(source_text, MAX_COPIED_WORDS + 1),
        never_say=tuple(profile.voice.never_say),
        allowed_pattern_ids=frozenset(item.id for item in patterns),
    )
    briefs = [
        Brief(
            mode="adapt",
            niche=niche,
            profile=profile,
            pattern_ids=[item.id for item in patterns],
            topic_ids=variant.topic_ids,
            guard=guard,
            words_per_second=speed,
            duration_source=speed_source,
            source_video_id=video.id,
            core_idea=variant.core_idea,
            content={
                "profile": profile.model_dump(mode="json"),
                "skeleton": skeleton.model_dump(),
                "variant": variant.model_dump(),
                "patterns": _pattern_cards(patterns),
                "lessons": _lesson_texts(lessons),
                "target_duration_s": video.duration_s,
                "words_per_second": speed,
            },
        )
        for variant in variants
    ]
    result = _produce_all(session, claude, briefs)
    result.repeats = repeats
    return result


def _by_type(patterns: list[Pattern]) -> list[list[Pattern]]:
    grouped: dict[str, list[Pattern]] = defaultdict(list)
    for item in patterns:
        grouped[item.type].append(item)
    return [grouped[kind.value] for kind in PatternType if grouped.get(kind.value)]


def _beta(item: Pattern, scores: dict[int, tuple[float, float]] | None) -> tuple[float, float]:
    return (scores or {}).get(item.id, (item.score_alpha, item.score_beta))


def thompson_combination(
    patterns: list[Pattern],
    rng: np.random.Generator,
    scores: dict[int, tuple[float, float]] | None = None,
) -> list[Pattern]:
    """Tire une valeur dans la Beta de chaque pattern et garde le meilleur de chaque type.

    Un pattern peu testé a une loi large : il garde ses chances d'être essayé. `scores` donne
    les scores de la niche, qui remplacent le score global des patterns qui en ont un.
    """
    chosen = []
    for candidates in _by_type(patterns):
        draws = [rng.beta(*_beta(item, scores)) for item in candidates]
        chosen.append(candidates[int(np.argmax(draws))])
    return chosen


def exploration_combination(
    patterns: list[Pattern],
    rng: np.random.Generator,
    scores: dict[int, tuple[float, float]] | None = None,
) -> list[Pattern]:
    """Le pattern le moins testé de chaque type, pour que la machine ne tourne pas en rond."""
    return [
        min(candidates, key=lambda item: (sum(_beta(item, scores)), rng.random()))
        for candidates in _by_type(patterns)
    ]


def exploration_slots(count: int, rate: float, rng: np.random.Generator) -> int:
    """Arrondi aléatoire : en moyenne `rate` des places, même sur de petits lots."""
    return int(count * rate + rng.random())


def _ideas_schema(
    count: int,
    combinations: int,
    topic_ids: set[str],
    first_test: int | None = None,
    required_tests: int = 0,
) -> type[IdeaBatch]:
    class ExactIdeas(IdeaBatch):
        @model_validator(mode="after")
        def _complete(self) -> Self:
            if len(self.ideas) != count:
                raise ValueError(f"Il faut exactement {count} idées.")
            for idea in self.ideas:
                if not 0 <= idea.combination_index < combinations:
                    raise ValueError(f"Combinaison inconnue : {idea.combination_index}")
                if idea.topic_id not in topic_ids:
                    raise ValueError(f"Sujet inconnu : {idea.topic_id}")
            if required_tests:
                tests = sum(idea.combination_index >= first_test for idea in self.ideas)
                if tests < required_tests:
                    raise ValueError(
                        f"Il faut au moins {required_tests} idée(s) sur les combinaisons "
                        f"à partir de l'index {first_test}."
                    )
            return self

    return ExactIdeas


def generate(
    session: Session,
    niche_id: int,
    claude: ClaudeClient,
    objectif: str | None = None,
    n: int = 5,
    develop: int = 3,
    seed: int | None = None,
    exploration: int | None = None,
    embedder: Embedder | None = None,
) -> GenerationResult:
    """Propose `n` idées puis développe les `develop` meilleures en scripts.

    `exploration` impose un nombre exact de scripts d'exploration. Sinon, le nombre suit
    LEARNING_EXPLORATION_RATE, avec un arrondi aléatoire.

    Une idée qui répète un script déjà produit est écartée. S'il n'en reste pas assez, Claude en
    propose d'autres une seule fois, et il peut donc y avoir moins de `develop` scripts.
    """
    niche, profile = _niche_with_profile(session, niche_id)
    novelty = NoveltyFilter(session, niche_id, embedder)
    lessons = active_lessons(session, niche_id)
    avoided = avoided_pattern_ids(lessons)
    patterns = [item for item in session.scalars(select(Pattern)).all() if item.id not in avoided]
    if not patterns:
        raise GenerationError("La bibliothèque de patterns est vide : analyse d'abord des vidéos.")
    topics = _active_topics(niche)
    if not topics:
        raise GenerationError("La banque de sujets est vide : lance cm niche topics --refresh.")

    goal = objectif or profile.identity.goal_details or profile.identity.goal
    rng = np.random.default_rng(seed)
    scores = niche_scores(session, niche_id)
    rate = get_settings().learning_exploration_rate
    tests = exploration if exploration is not None else exploration_slots(n, rate, rng)
    combinations = [
        *[thompson_combination(patterns, rng, scores) for _ in range(n - tests)],
        *[exploration_combination(patterns, rng, scores) for _ in range(tests)],
    ]
    first_test = n - tests

    def propose(past: list[str]) -> list[Any]:
        return claude.call_json(
            prompt="script_ideas",
            version="v2",
            schema=_ideas_schema(
                n,
                len(combinations),
                {item["id"] for item in topics},
                first_test,
                exploration or 0,
            ),
            model=_model(),
            variables={
                "count": n,
                "objective": goal,
                "profile": _profile_json(profile),
                "topics": json.dumps(topics, ensure_ascii=False),
                "combinations": json.dumps(
                    [_pattern_cards(items) for items in combinations], ensure_ascii=False
                ),
                "lessons": json.dumps(_lesson_texts(lessons), ensure_ascii=False),
                "past_ideas": json.dumps(past, ensure_ascii=False),
            },
            effort="high",
        ).data.ideas

    def fresh_only(batch: list[Any]) -> tuple[list[Any], list[Repeat]]:
        verdicts = novelty.check([item.core_idea for item in batch])
        kept = [item for item, repeat in zip(batch, verdicts, strict=True) if repeat is None]
        return kept, [item for item in verdicts if item is not None]

    ideas, repeats = fresh_only(propose(novelty.past))
    if len(ideas) < develop:
        more, rejected = fresh_only(propose([*novelty.past, *[item.text for item in repeats]]))
        ideas += more
        repeats += rejected
    if not ideas:
        raise GenerationError("Toutes les idées proposées répètent des scripts déjà produits.")

    ranked = sorted(ideas, key=lambda item: item.potential, reverse=True)
    # Sans place réservée, une idée de test perdrait presque toujours face aux valeurs sûres.
    reserved = [item for item in ranked if item.combination_index >= first_test]
    slots = exploration if exploration is not None else exploration_slots(develop, rate, rng)
    reserved = reserved[:slots]
    best = [*reserved, *[item for item in ranked if item not in reserved]][:develop]
    speed, speed_source = _voice_speed(profile, None)
    never_say = tuple(profile.voice.never_say)

    briefs = []
    for idea in best:
        chosen = combinations[idea.combination_index]
        examples = " ".join(example["excerpt"] for item in chosen for example in item.examples)
        briefs.append(
            Brief(
                mode="generate",
                niche=niche,
                profile=profile,
                pattern_ids=[item.id for item in chosen],
                topic_ids=[idea.topic_id],
                guard=ScriptGuard(
                    source_runs=ngrams(examples, MAX_COPIED_WORDS + 1),
                    never_say=never_say,
                    allowed_pattern_ids=frozenset(item.id for item in chosen),
                ),
                words_per_second=speed,
                duration_source=speed_source,
                experiment=idea.combination_index >= first_test,
                core_idea=idea.core_idea,
                content={
                    "profile": profile.model_dump(mode="json"),
                    "idea": idea.model_dump(),
                    "objective": goal,
                    "topic": next(item for item in topics if item["id"] == idea.topic_id),
                    "patterns": _pattern_cards(chosen),
                    "lessons": _lesson_texts(lessons),
                    "target_duration_s": DEFAULT_TARGET_DURATION_S,
                    "words_per_second": speed,
                },
            )
        )
    result = _produce_all(session, claude, briefs)
    result.repeats = repeats
    return result


def load_content(script: Script) -> ScriptContent:
    return ScriptContent.model_validate(script.content)


def update_script(
    session: Session,
    script: Script,
    status: ScriptStatus | None = None,
    changes: dict[str, Any] | None = None,
    published_video_id: int | None = None,
    published_at: datetime | None = None,
) -> Script:
    """Édite un script. Lève une ValidationError si le contenu devient invalide.

    Les garde-fous ne s'appliquent qu'à ce que Claude écrit : tes éditions restent les tiennes.
    À la première validation, les sujets passent à « used » et le compteur des patterns monte.
    Publier fixe `published_at`, d'où partent les rappels de stats.
    """
    if published_at is not None:
        script.published_at = published_at
    elif status is ScriptStatus.PUBLISHED and script.published_at is None:
        script.published_at = utcnow()

    if changes:
        content = ScriptContent.model_validate(deep_merge(script.content, changes))
        content.estimated_duration_s = _apply_timing(content, content.words_per_second)
        script.content = content.model_dump(mode="json")

    if published_video_id is not None:
        if session.get(Video, published_video_id) is None:
            raise LookupError(f"Vidéo {published_video_id} inconnue.")
        script.published_video_id = published_video_id
        # La vidéo réellement ingérée remplace sa fiche provisoire dans les relevés existants.
        session.execute(
            update(Performance)
            .where(Performance.script_id == script.id)
            .values(video_id=published_video_id)
        )

    if status is not None:
        first_validation = script.status is ScriptStatus.DRAFT and status is not ScriptStatus.DRAFT
        script.status = status
        if first_validation:
            for pattern in session.scalars(
                select(Pattern).where(Pattern.id.in_(script.pattern_ids))
            ):
                pattern.times_used += 1
            session.commit()
            niche = session.get(Niche, script.niche_id)
            known = {item["id"] for item in niche.topics}
            for topic_id in script.content["topic_ids"]:
                if topic_id in known:
                    set_topic_status(session, niche, topic_id, "used")

    session.commit()
    script_path(script.id).write_text(render_script(load_content(script)), encoding="utf-8")
    return script
