"""Les outils de l'agent : les modules existants, exposés à la boucle d'outils.

Aucun outil ne publie sur un réseau : la liste de `TOOLS` est fermée, et un test le vérifie.
Les descriptions envoyées à Claude sont dans app/prompts/agent_tools_v1.md.
"""

import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import anthropic
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.budget import (
    TOKENS_ADAPT_BASE,
    TOKENS_ANALYSIS,
    TOKENS_IDEAS,
    TOKENS_INGEST,
    TOKENS_POST_MORTEM,
    TOKENS_SCRIPT,
    TOKENS_SYNTHESIS,
    BudgetExceeded,
    check_budget,
    paid_work,
)
from app.agent.journal import logged_action
from app.analysis.service import CURRENT_PROMPT_VERSION, AnalysisError, analyze
from app.db.models import (
    ActionStatus,
    Analysis,
    IngestJob,
    JobStatus,
    Niche,
    Pattern,
    PatternType,
    Performance,
    PostMortem,
    Script,
    Video,
    VideoPattern,
)
from app.generate.service import (
    GenerationError,
    GenerationResult,
    active_lessons,
    adapt,
    generate,
    load_content,
    script_path,
)
from app.ingest.jobs import canonical, is_url, run_job, submit
from app.ingest.pipeline import IngestDeps, IngestError
from app.learning.lessons import maybe_synthesize
from app.learning.postmortem import MIN_HORIZON_H, run_post_mortem
from app.learning.report import build_report
from app.llm.client import ClaudeClient
from app.llm.errors import ClaudeError
from app.llm.prompts import load_prompt
from app.patterns.embeddings import Embedder
from app.patterns.search import score_mean, search_patterns, top_patterns

logger = logging.getLogger(__name__)

MAX_RESULT_CHARS = 20_000
TOOLS_PROMPT = ("agent_tools", "v1")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
JOB_POLL_S = 2.0
JOB_TIMEOUT_S = 1800.0
EXPECTED = (
    LookupError,
    AnalysisError,
    GenerationError,
    IngestError,
    ClaudeError,
    anthropic.APIError,
)


class ToolError(Exception):
    """L'outil ne peut pas répondre : le message est renvoyé à Claude."""


@dataclass
class ToolContext:
    session: Session
    niche: Niche
    claude: ClaudeClient
    source: str = "chat"
    session_id: int | None = None
    embedder: Embedder | None = None
    ingest_deps: IngestDeps | None = None


@dataclass(frozen=True)
class ToolOutcome:
    content: str
    is_error: bool = False


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IngestInput(Input):
    source: str = Field(description="URL de la vidéo, ou chemin d'un fichier vidéo local.")
    is_own_content: bool = Field(default=False, description="Vrai si c'est une vidéo du créateur.")


class AnalyzeInput(Input):
    video_id: int
    force: bool = Field(default=False, description="Refait une analyse existante (payant).")


class SearchInput(Input):
    query: str
    type: PatternType | None = None
    k: int = Field(default=10, ge=1, le=30)


class TopInput(Input):
    type: PatternType | None = None
    limit: int = Field(default=10, ge=1, le=50)


class NoInput(Input):
    pass


class AdaptInput(Input):
    video_id: int
    n_variantes: int = Field(default=1, ge=1, le=5)


class GenerateInput(Input):
    count: int = Field(default=3, ge=1, le=5, description="Nombre de scripts à écrire.")
    objectif: str | None = Field(default=None, description="Objectif, si différent du profil.")


class ScriptRef(Input):
    script_id: int | None = Field(
        default=None, description="Sans id : le dernier script qui a des stats."
    )


class PostMortemInput(Input):
    script_id: int


def wait_for_job(session: Session, job_id: int) -> IngestJob:
    """L'API a un worker : il peut avoir pris le job avant nous."""
    deadline = time.monotonic() + JOB_TIMEOUT_S
    while True:
        session.commit()  # Termine la transaction de lecture pour voir l'état réel.
        job = session.get(IngestJob, job_id)
        session.refresh(job)
        if job.status in (JobStatus.DONE, JobStatus.FAILED):
            return job
        if time.monotonic() > deadline:
            raise IngestError(f"Job {job_id} toujours en cours après {JOB_TIMEOUT_S:.0f} s.")
        time.sleep(JOB_POLL_S)


def _ingest(ctx: ToolContext, data: IngestInput) -> dict[str, Any]:
    if not is_url(data.source) and Path(data.source).suffix.lower() not in VIDEO_EXTENSIONS:
        raise ToolError("La source doit être une URL ou un fichier vidéo.")

    submission = submit(
        ctx.session,
        data.source,
        is_own_content=data.is_own_content,
        niche_source=ctx.niche.name,
    )
    if not submission.deduplicated:
        run_job(submission.job_id, ctx.ingest_deps)
        job = wait_for_job(ctx.session, submission.job_id)
        if job.status is JobStatus.FAILED:
            raise ToolError(f"Ingestion échouée à l'étape {job.stage} : {job.error}")

    video = canonical(ctx.session, ctx.session.get(Video, submission.video_id))
    return {
        "video_id": video.id,
        "already_known": submission.deduplicated,
        "creator": video.creator,
        "platform": video.platform,
        "duration_s": video.duration_s,
        "views": video.views,
        "description": video.description,
    }


def _statements(claims: list[dict[str, Any]]) -> list[str]:
    return [item["statement"] for item in claims]


def analysis_digest(session: Session, row: Analysis) -> dict[str, Any]:
    """L'essentiel d'une analyse, assez court pour en comparer plusieurs."""
    data = row.json_result["analysis"]
    patterns = session.execute(
        select(Pattern.id, Pattern.type, Pattern.name)
        .join(VideoPattern, VideoPattern.pattern_id == Pattern.id)
        .where(VideoPattern.video_id == row.video_id)
        .distinct()
        .order_by(Pattern.id)
    ).all()
    return {
        "video_id": row.video_id,
        "analysis_id": row.id,
        "hook": {
            "type": data["hook"]["hook_type"],
            "score": data["hook"]["score"],
            "spoken": _statements(data["hook"]["spoken_text"]),
            "mechanism": _statements(data["hook"]["psychological_mechanism"]),
        },
        "angle": _statements(data["angle"]["chosen_angle"]),
        "promise": _statements(data["angle"]["implicit_promise"]),
        "framework": data["structure"]["framework"],
        "beats": [item["role"] for item in data["structure"]["beats"]],
        "ending": data["retention"]["ending_type"],
        "why_it_worked": [
            {
                "rank": item["rank"],
                "weight_percent": item["weight_percent"],
                "confidence": item["confidence"],
                "reason": item["claim"]["statement"],
            }
            for item in sorted(data["verdict"]["reasons"], key=lambda reason: reason["rank"])
        ],
        "transferable": _statements(data["transferability"]["format_elements"]),
        "creator_specific": [
            item["claim"]["statement"] for item in data["transferability"]["creator_elements"]
        ],
        "patterns": [{"id": item.id, "type": item.type, "name": item.name} for item in patterns],
    }


def _already_analyzed(session: Session, video_id: int) -> bool:
    return (
        session.scalar(
            select(Analysis.id).where(
                Analysis.video_id == video_id,
                Analysis.prompt_version == CURRENT_PROMPT_VERSION,
            )
        )
        is not None
    )


def _analyze(ctx: ToolContext, data: AnalyzeInput) -> dict[str, Any]:
    with paid_work:
        row = analyze(
            ctx.session, data.video_id, ctx.claude, force=data.force, embedder=ctx.embedder
        )
    return analysis_digest(ctx.session, row)


def _pattern_out(item: Any, score: float, **extra: Any) -> dict[str, Any]:
    return {
        "id": item.id,
        "type": item.type,
        "name": item.name,
        "description": item.description,
        "template": item.template,
        "score": round(score, 3),
        "tested": item.score_alpha + item.score_beta > 2,
        **extra,
    }


def _search_patterns(ctx: ToolContext, data: SearchInput) -> dict[str, Any]:
    hits = search_patterns(ctx.session, data.query, data.type, data.k, ctx.embedder)
    return {
        "patterns": [
            _pattern_out(item, item.score_mean, similarity=round(item.similarity, 3))
            for item in hits
        ]
    }


def _top_patterns(ctx: ToolContext, data: TopInput) -> dict[str, Any]:
    patterns = top_patterns(ctx.session, data.type, ctx.niche.id, data.limit)
    return {"patterns": [_pattern_out(item, score_mean(item)) for item in patterns]}


def _get_niche(ctx: ToolContext, _: NoInput) -> dict[str, Any]:
    niche = ctx.niche
    return {
        "id": niche.id,
        "name": niche.name,
        "version": niche.version,
        "avatar": niche.avatar_markdown,
        "creator_facts": niche.profile.get("creator_facts", []),
        "topics": [
            {key: item[key] for key in ("id", "title", "intensity", "grounding")}
            for item in niche.topics
            if item["status"] == "new"
        ][:30],
    }


def _scripts_out(result: GenerationResult) -> dict[str, Any]:
    scripts = []
    for script in result.scripts:
        content = load_content(script)
        scripts.append(
            {
                "script_id": script.id,
                "title": content.internal_title,
                "angle": content.angle,
                "hook": content.hooks[0].spoken,
                "hook_score": content.final_critique.hook_score,
                "duration_s": content.estimated_duration_s,
                "experiment": content.experiment,
                "pattern_ids": content.pattern_ids,
                "to_verify": content.to_verify,
                "file": str(script_path(script.id)),
            }
        )
    return {
        "scripts": scripts,
        "failures": result.failures,
        "repeated_ideas_dropped": [
            {"idea": item.text, "script_id": item.script_id, "similarity": item.similarity}
            for item in result.repeats
        ],
    }


def _adapt(ctx: ToolContext, data: AdaptInput) -> dict[str, Any]:
    return _scripts_out(
        adapt(
            ctx.session,
            data.video_id,
            ctx.niche.id,
            ctx.claude,
            data.n_variantes,
            embedder=ctx.embedder,
        )
    )


def _generate(ctx: ToolContext, data: GenerateInput) -> dict[str, Any]:
    return _scripts_out(
        generate(
            ctx.session,
            ctx.niche.id,
            ctx.claude,
            data.objectif,
            n=data.count + 2,  # Une marge pour les idées écartées.
            develop=data.count,
            embedder=ctx.embedder,
        )
    )


def _niche_script(ctx: ToolContext, script_id: int) -> Script:
    script = ctx.session.get(Script, script_id)
    if script is None or script.niche_id != ctx.niche.id:
        raise ToolError(f"Script {script_id} inconnu dans cette niche.")
    return script


def _reading(row: Performance) -> dict[str, Any]:
    return {
        "horizon_h": row.horizon_h,
        "measured_at": row.measured_at.isoformat(),
        "video_id": row.video_id,
        "views": row.views,
        "likes": row.likes,
        "retention_3s": row.retention_3s,
        "retention_50": row.retention_50,
        "completion_rate": row.completion_rate,
        "avg_watch_time_s": row.avg_watch_time_s,
        "shares": row.shares,
        "saves": row.saves,
        "comments": row.comments,
        "follows": row.follows,
        "notes": row.notes,
        "composite": row.composite,
        "label": row.label,
        "indexes": row.indexes,
        "baseline_size": row.baseline_size,
    }


def _get_performance(ctx: ToolContext, data: ScriptRef) -> dict[str, Any]:
    if data.script_id is not None:
        script = _niche_script(ctx, data.script_id)
    else:
        script = ctx.session.scalar(
            select(Script)
            .join(Performance, Performance.script_id == Script.id)
            .where(Script.niche_id == ctx.niche.id)
            .order_by(Performance.measured_at.desc())
            .limit(1)
        )
        if script is None:
            return {"message": "Aucune stat enregistrée : aucune vidéo n'a encore de relevé."}

    readings = ctx.session.scalars(
        select(Performance)
        .where(Performance.script_id == script.id)
        .order_by(Performance.horizon_h)
    )
    post_mortems = ctx.session.scalars(
        select(PostMortem).where(PostMortem.script_id == script.id).order_by(PostMortem.horizon_h)
    )
    content = load_content(script)
    return {
        "script_id": script.id,
        "title": content.internal_title,
        "angle": content.angle,
        "hook": content.hooks[0].spoken,
        "status": script.status.value,
        "published_video_id": script.published_video_id,
        "pattern_ids": script.pattern_ids,
        "predicted_score": script.predicted_score,
        "readings": [_reading(item) for item in readings],
        "post_mortems": [
            {"horizon_h": item.horizon_h, "content": item.content} for item in post_mortems
        ],
    }


def _run_post_mortem(ctx: ToolContext, data: PostMortemInput) -> dict[str, Any]:
    script = _niche_script(ctx, data.script_id)
    row = ctx.session.scalar(
        select(Performance)
        .where(Performance.script_id == script.id, Performance.composite.is_not(None))
        .order_by(Performance.horizon_h.desc())
        .limit(1)
    )
    if row is None:
        raise ToolError(
            "Ce script n'a aucun relevé noté. Il faut au moins 3 relevés précédents au même "
            "horizon pour calculer une baseline."
        )
    result = run_post_mortem(ctx.session, script, row, ctx.claude)
    if result is None:
        raise ToolError(f"Pas de post-mortem avant {MIN_HORIZON_H} h : il est trop tôt.")
    return {"script_id": script.id, "horizon_h": result.horizon_h, "content": result.content}


def _synthesize_lessons(ctx: ToolContext, _: NoInput) -> dict[str, Any]:
    if not maybe_synthesize(ctx.session, ctx.niche, ctx.claude, force=True):
        raise ToolError("Aucun post-mortem : il n'y a rien à synthétiser.")
    return {
        "active_lessons": [
            {
                "id": item.id,
                "statement": item.statement,
                "confidence": item.confidence,
                "evidence": item.evidence,
            }
            for item in active_lessons(ctx.session, ctx.niche.id)
        ]
    }


def _learning_report(ctx: ToolContext, _: NoInput) -> dict[str, Any]:
    return build_report(ctx.session, ctx.niche.id)


def _analysis_tokens(ctx: ToolContext, data: AnalyzeInput) -> int:
    return (
        0 if _already_analyzed(ctx.session, data.video_id) and not data.force else TOKENS_ANALYSIS
    )


@dataclass(frozen=True)
class Tool:
    name: str
    input_model: type[Input]
    run: Callable[[ToolContext, Any], dict[str, Any]]
    tokens: Callable[[ToolContext, Any], int] = lambda ctx, data: 0


TOOLS: dict[str, Tool] = {
    tool.name: tool
    for tool in (
        Tool("ingest", IngestInput, _ingest, lambda ctx, data: TOKENS_INGEST),
        Tool("analyze", AnalyzeInput, _analyze, _analysis_tokens),
        Tool("search_patterns", SearchInput, _search_patterns),
        Tool("top_patterns", TopInput, _top_patterns),
        Tool("get_niche", NoInput, _get_niche),
        Tool(
            "adapt",
            AdaptInput,
            _adapt,
            lambda ctx, data: TOKENS_ADAPT_BASE + data.n_variantes * TOKENS_SCRIPT,
        ),
        Tool(
            "generate",
            GenerateInput,
            _generate,
            lambda ctx, data: TOKENS_IDEAS + data.count * TOKENS_SCRIPT,
        ),
        Tool("get_performance", ScriptRef, _get_performance),
        Tool(
            "run_post_mortem",
            PostMortemInput,
            _run_post_mortem,
            lambda ctx, data: TOKENS_POST_MORTEM,
        ),
        Tool(
            "synthesize_lessons",
            NoInput,
            _synthesize_lessons,
            lambda ctx, data: TOKENS_SYNTHESIS,
        ),
        Tool("learning_report", NoInput, _learning_report),
    )
}


@lru_cache
def tool_descriptions() -> dict[str, str]:
    """Chaque outil est une section `## nom` du prompt versionné."""
    name, version = TOOLS_PROMPT
    sections = re.split(r"^## ", load_prompt(name, version).system, flags=re.MULTILINE)[1:]
    return {
        section.split("\n", 1)[0].strip(): section.split("\n", 1)[1].strip() for section in sections
    }


def api_tools() -> list[dict[str, Any]]:
    descriptions = tool_descriptions()
    return [
        {
            "name": tool.name,
            "description": descriptions[tool.name],
            "input_schema": tool.input_model.model_json_schema(),
            "eager_input_streaming": True,
        }
        for tool in TOOLS.values()
    ]


def _encode(result: dict[str, Any]) -> str:
    text = json.dumps(result, ensure_ascii=False, default=str)
    if len(text) > MAX_RESULT_CHARS:
        return f"{text[:MAX_RESULT_CHARS]} … [résultat tronqué]"
    return text


def run_tool(ctx: ToolContext, name: str, raw_input: dict[str, Any]) -> ToolOutcome:
    """Exécute un outil. Une erreur revient à Claude, elle n'interrompt jamais la conversation."""
    with logged_action(ctx.session, ctx.source, name, raw_input, ctx.session_id) as action:
        tool = TOOLS.get(name)
        if tool is None:
            action.status = ActionStatus.REFUSED
            action.summary = "Outil inconnu."
            return ToolOutcome(f"Outil inconnu : {name}", is_error=True)

        try:
            data = tool.input_model.model_validate(raw_input)
            estimate = tool.tokens(ctx, data)
            if estimate:  # Un outil gratuit reste utilisable une fois le budget atteint.
                check_budget(ctx.session, estimate)
            result = tool.run(ctx, data)
        except ValidationError as exc:
            action.status = ActionStatus.REFUSED
            action.summary = f"Entrée invalide : {exc}"
            return ToolOutcome(f"Entrée invalide : {exc}", is_error=True)
        except BudgetExceeded as exc:
            action.status = ActionStatus.REFUSED
            action.summary = str(exc)
            return ToolOutcome(str(exc), is_error=True)
        except (ToolError, *EXPECTED) as exc:
            ctx.session.rollback()
            action.status = ActionStatus.ERROR
            action.summary = str(exc)
            return ToolOutcome(str(exc), is_error=True)
        except Exception as exc:
            logger.exception("Erreur inattendue de l'outil %s", name)
            ctx.session.rollback()
            action.status = ActionStatus.ERROR
            action.summary = f"{type(exc).__name__}: {exc}"
            return ToolOutcome(f"Erreur interne de l'outil {name} : {exc}", is_error=True)

        content = _encode(result)
        action.summary = content
        return ToolOutcome(content)
