"""Routines planifiées. Toutes sont sûres à relancer : ce qui est fait n'est jamais repayé.

Une routine décide elle-même si elle est due, d'après la base. Si l'ordinateur était éteint à
l'heure prévue, elle tourne au prochain démarrage de l'API.
"""

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any

import anthropic
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.budget import (
    TOKENS_ANALYSIS,
    TOKENS_IDEAS,
    TOKENS_INGEST,
    TOKENS_POST_MORTEM,
    TOKENS_SCRIPT,
    TOKENS_WEEKLY_REPORT,
    can_spend,
    local_now,
    local_zone,
    paid_work,
)
from app.agent.journal import logged_action, notify
from app.agent.tools import VIDEO_EXTENSIONS, wait_for_job
from app.analysis.service import AnalysisError, analyze, pending_video_ids
from app.config import get_settings
from app.db.base import utcnow
from app.db.models import (
    ActionStatus,
    AgentAction,
    IngestJob,
    JobStatus,
    Lesson,
    Niche,
    Notification,
    Performance,
    PostMortem,
    Script,
    Video,
    WeeklyReport,
)
from app.db.session import get_session_factory
from app.generate.service import GenerationError, generate, load_content
from app.ingest.jobs import canonical, run_job, run_next, submit
from app.ingest.pipeline import IngestDeps, IngestError
from app.learning.postmortem import run_post_mortem
from app.learning.report import build_report
from app.learning.service import scored_results
from app.llm.client import ClaudeClient, get_claude
from app.llm.errors import ClaudeError
from app.patterns.embeddings import Embedder

logger = logging.getLogger(__name__)

HORIZONS_H = (24, 72, 168)
WEEK = timedelta(days=7)
DAILY_ACTION = "routine_daily"
EXPLORATION_SCRIPTS = 1
EXTRA_IDEAS = 2  # Une marge pour les idées écartées parce que déjà produites.
PROMPT = ("weekly_report", "v1")
CLAUDE_FAILURES = (ClaudeError, anthropic.APIError)


@dataclass
class Services:
    """Ce dont les routines ont besoin. Les tests y mettent leurs doubles."""

    claude: ClaudeClient | None = None
    deps: IngestDeps | None = None
    embedder: Embedder | None = None

    def __post_init__(self) -> None:
        self.claude = self.claude or get_claude()


def _niches(session: Session) -> list[Niche]:
    return list(session.scalars(select(Niche).where(Niche.profile.is_not(None)).order_by(Niche.id)))


def _budget_notice(session: Session, what: str) -> None:
    """Une seule alerte non lue à la fois : une routine qui repasse n'en empile pas d'autres."""
    unread = session.scalar(
        select(Notification.id).where(Notification.kind == "budget", Notification.read_at.is_(None))
    )
    if unread is None:
        notify(
            session,
            "budget",
            "Budget du jour atteint",
            f"{what} attendra le prochain budget. AGENT_DAILY_TOKEN_BUDGET permet de l'augmenter.",
        )


def analyze_pending(session: Session, services: Services) -> dict[str, list[int]]:
    """Analyse ce qui attend. Une vidéo en échec n'empêche pas les suivantes."""
    done: list[int] = []
    failed: list[int] = []
    for video_id in pending_video_ids(session):
        if not can_spend(session, TOKENS_ANALYSIS):
            _budget_notice(session, "L'analyse des vidéos en attente")
            break
        try:
            with logged_action(session, "routine", "analyze", {"video_id": video_id}):
                analyze(session, video_id, services.claude, embedder=services.embedder)
            done.append(video_id)
        except (LookupError, AnalysisError, *CLAUDE_FAILURES) as exc:
            logger.warning("Analyse de la vidéo %d échouée : %s", video_id, exc)
            failed.append(video_id)
    return {"analyzed": done, "failed": failed}


def daily_due(session: Session, now: datetime | None = None) -> bool:
    local = local_now(now)
    start = datetime.combine(
        local.date(), time(get_settings().agent_daily_hour), tzinfo=local_zone()
    )
    if local < start:
        return False
    last = session.scalar(
        select(AgentAction.id).where(
            AgentAction.action == DAILY_ACTION,
            AgentAction.status == ActionStatus.OK,
            AgentAction.created_at >= start,
        )
    )
    return last is None


def run_daily(services: Services | None = None, force: bool = False) -> dict[str, Any] | None:
    """Traite la file d'ingestion, puis analyse ce qui est en attente."""
    with get_session_factory()() as session:
        if not force and not daily_due(session):
            return None
        services = services or Services()
        with paid_work, logged_action(session, "routine", DAILY_ACTION) as action:
            ingested = 0
            while can_spend(session, TOKENS_INGEST) and run_next(services.deps):
                ingested += 1
            session.commit()  # Les jobs ont leurs propres sessions : relit l'état réel.
            result = {"ingested": ingested, **analyze_pending(session, services)}
            action.summary = json.dumps(result)
        return result


def run_reminders(now: datetime | None = None) -> int:
    """Rappels de stats à 24 h, 72 h et 7 jours. Un relevé saisi ferme son rappel."""
    now = now or utcnow()
    created = 0
    with get_session_factory()() as session:
        scripts = session.scalars(select(Script).where(Script.published_at.is_not(None))).all()
        for script in scripts:
            title = script.content["internal_title"]
            for horizon in HORIZONS_H:
                if now < script.published_at + timedelta(hours=horizon):
                    continue
                entered = session.scalar(
                    select(Performance.id).where(
                        Performance.script_id == script.id, Performance.horizon_h == horizon
                    )
                )
                reminder = session.scalar(
                    select(Notification).where(
                        Notification.kind == "stats_reminder",
                        Notification.script_id == script.id,
                        Notification.horizon_h == horizon,
                    )
                )
                if entered is not None:
                    if reminder is not None and reminder.read_at is None:
                        reminder.read_at = now
                        session.commit()
                elif reminder is None:
                    label = "7 jours" if horizon == 168 else f"{horizon} h"
                    with logged_action(
                        session,
                        "routine",
                        "stats_reminder",
                        {"script_id": script.id, "horizon_h": horizon},
                    ):
                        notify(
                            session,
                            "stats_reminder",
                            f"Stats à {label} : {title}",
                            f"Entre les stats du script {script.id} avec cm stats {script.id}, "
                            "ou sur /forms/performance.",
                            niche_id=script.niche_id,
                            script_id=script.id,
                            horizon_h=horizon,
                        )
                    created += 1
    return created


class Learned(BaseModel):
    statement: str
    video_ids: list[int]


class WeeklyLearnings(BaseModel):
    headline: str
    learned: list[Learned] = Field(max_length=5)
    open_questions: list[str]
    focus: str


def week_start(now: datetime | None = None) -> datetime:
    """Le lundi de la semaine, à l'heure de la routine, en heure locale."""
    local = local_now(now)
    monday = local.date() - timedelta(days=local.weekday())
    return datetime.combine(monday, time(get_settings().agent_weekly_hour), tzinfo=local_zone())


def _week_key(now: datetime | None = None) -> str:
    start = week_start(now)
    # Un dimanche ou un lundi avant l'heure, la semaine en cours reste la précédente.
    if local_now(now) < start:
        start -= WEEK
    return start.date().isoformat()


def weekly_due(session: Session, niche: Niche, now: datetime | None = None) -> bool:
    if local_now(now) < week_start(now):
        return False
    return (
        session.scalar(
            select(WeeklyReport.id).where(
                WeeklyReport.niche_id == niche.id, WeeklyReport.week_start == _week_key(now)
            )
        )
        is None
    )


def _catch_up_post_mortems(session: Session, niche: Niche, services: Services) -> None:
    """L'import CSV ne lance pas les post-mortems : le rapport de la semaine en a besoin."""
    for script, row in scored_results(session):
        known = session.scalar(
            select(PostMortem.id).where(
                PostMortem.script_id == script.id, PostMortem.horizon_h == row.horizon_h
            )
        )
        if script.niche_id != niche.id or known is not None:
            continue
        if not can_spend(session, TOKENS_POST_MORTEM):
            _budget_notice(session, "Le rattrapage des post-mortems")
            return
        try:
            run_post_mortem(session, script, row, services.claude)
        except CLAUDE_FAILURES as exc:
            session.rollback()
            logger.warning("Post-mortem du script %d échoué : %s", script.id, exc)


def _week_data(session: Session, niche: Niche, since: datetime) -> dict[str, Any]:
    readings = session.execute(
        select(Performance, Script)
        .join(Script, Script.id == Performance.script_id)
        .where(Script.niche_id == niche.id, Performance.measured_at >= since)
        .order_by(Performance.measured_at)
    ).all()
    post_mortems = session.scalars(
        select(PostMortem)
        .join(Script, Script.id == PostMortem.script_id)
        .where(Script.niche_id == niche.id, PostMortem.created_at >= since)
    ).all()
    lessons = session.scalars(
        select(Lesson).where(
            (Lesson.created_at >= since) | (Lesson.deactivated_at >= since),
            (Lesson.niche_id == niche.id) | (Lesson.niche_id.is_(None)),
        )
    ).all()
    return {
        "readings": [
            {
                "video_id": row.video_id,
                "script_id": script.id,
                "title": script.content["internal_title"],
                "experiment": script.content.get("experiment", False),
                "horizon_h": row.horizon_h,
                "views": row.views,
                "composite": row.composite,
                "label": row.label,
                "indexes": row.indexes,
            }
            for row, script in readings
        ],
        "post_mortems": [
            {"video_id": item.video_id, "horizon_h": item.horizon_h, "content": item.content}
            for item in post_mortems
        ],
        "lessons": [
            {
                "id": item.id,
                "statement": item.statement,
                "active": item.active,
                "deactivated_reason": item.deactivated_reason,
            }
            for item in lessons
        ],
    }


def _learnings(
    session: Session, niche: Niche, services: Services, key: str, since: datetime
) -> dict[str, Any] | None:
    """None s'il n'y a rien de mesuré : pas d'appel à Claude pour ne rien dire."""
    data = _week_data(session, niche, since)
    if not data["readings"]:
        return None
    name, version = PROMPT
    result = services.claude.call_json(
        prompt=name,
        version=version,
        schema=WeeklyLearnings,
        model=get_settings().model_analysis,
        variables={
            "week_start": key,
            "readings": json.dumps(data["readings"], ensure_ascii=False),
            "post_mortems": json.dumps(data["post_mortems"], ensure_ascii=False),
            "lessons": json.dumps(data["lessons"], ensure_ascii=False),
            "report": json.dumps(build_report(session, niche.id), ensure_ascii=False),
        },
        effort="medium",
    ).data
    week_videos = {item["video_id"] for item in data["readings"]}
    # Comme pour les leçons, une affirmation sans vidéo réelle à l'appui est retirée.
    result.learned = [
        Learned(statement=item.statement, video_ids=sorted(set(item.video_ids) & week_videos))
        for item in result.learned
        if set(item.video_ids) & week_videos
    ]
    return result.model_dump()


@dataclass
class Plan:
    script_ids: list[int] = field(default_factory=list)
    error: str | None = None
    notes: list[str] = field(default_factory=list)


def _content_plan(session: Session, niche: Niche, services: Services) -> Plan:
    count = get_settings().agent_weekly_scripts
    if count <= 0:
        return Plan()
    if not can_spend(session, TOKENS_IDEAS + count * TOKENS_SCRIPT):
        _budget_notice(session, "Le plan de contenu de la semaine")
        return Plan(error="Budget du jour insuffisant pour le plan de la semaine.")
    try:
        result = generate(
            session,
            niche.id,
            services.claude,
            n=count + EXTRA_IDEAS,
            develop=count,
            exploration=min(EXPLORATION_SCRIPTS, count),
            embedder=services.embedder,
        )
    except (LookupError, GenerationError, *CLAUDE_FAILURES) as exc:
        session.rollback()
        return Plan(error=str(exc))

    notes = [f"Script abandonné : {item}" for item in result.failures]
    if result.repeats:
        notes.append(f"{len(result.repeats)} idée(s) écartée(s) car déjà produite(s).")
    if len(result.scripts) < count:
        notes.append(f"{len(result.scripts)} script(s) sur {count} prévus.")
    return Plan([item.id for item in result.scripts], notes=notes)


def _render_week(session: Session, key: str, learnings: dict[str, Any] | None, plan: Plan) -> str:
    lines = [f"# Ce que j'ai appris cette semaine ({key})", ""]
    if learnings is None:
        lines += ["Aucune stat enregistrée cette semaine : rien de nouveau à apprendre.", ""]
    else:
        lines += [learnings["headline"], ""]
        for item in learnings["learned"]:
            videos = " ".join(f"[vidéo {video_id}]" for video_id in item["video_ids"])
            lines.append(f"- {item['statement']} {videos}")
        if learnings["open_questions"]:
            lines += ["", "**Encore inconnu :**", ""]
            lines += [f"- {item}" for item in learnings["open_questions"]]
        lines += ["", f"**À faire différemment :** {learnings['focus']}", ""]

    lines += ["# Plan de contenu de la semaine", ""]
    if plan.error:
        lines += [f"Plan non généré : {plan.error}", ""]
    elif not plan.script_ids:
        lines += ["Aucun script prévu (AGENT_WEEKLY_SCRIPTS=0).", ""]
    for script_id in plan.script_ids:
        script = session.get(Script, script_id)
        content = load_content(script)
        tag = " (exploration)" if content.experiment else ""
        check = f", {len(content.to_verify)} à vérifier" if content.to_verify else ""
        lines.append(
            f"- [script {script_id}]{tag} {content.internal_title} : "
            f"« {content.hooks[0].spoken} » (hook {content.final_critique.hook_score}/10{check})"
        )
    lines += [f"- {item}" for item in plan.notes]
    return "\n".join(lines).strip() + "\n"


def _weekly_for(
    session: Session, niche: Niche, services: Services, now: datetime | None
) -> WeeklyReport:
    key = _week_key(now)
    row = session.scalar(
        select(WeeklyReport).where(
            WeeklyReport.niche_id == niche.id, WeeklyReport.week_start == key
        )
    )
    if row is None:
        row = WeeklyReport(niche_id=niche.id, week_start=key)
        session.add(row)
        session.commit()

    since = (now or utcnow()) - WEEK
    _catch_up_post_mortems(session, niche, services)
    learnings = None
    if can_spend(session, TOKENS_WEEKLY_REPORT):
        try:
            learnings = _learnings(session, niche, services, key, since)
        except CLAUDE_FAILURES as exc:
            logger.warning("Rapport de la semaine non généré : %s", exc)
    else:
        _budget_notice(session, "Le rapport de la semaine")

    plan = _content_plan(session, niche, services)
    row.learnings = learnings
    row.script_ids = plan.script_ids
    row.plan_error = plan.error
    row.markdown = _render_week(session, key, learnings, plan)
    session.commit()

    folder = get_settings().data_dir.resolve() / "reports"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"week_{key}_niche_{niche.id}.md").write_text(row.markdown, encoding="utf-8")
    notify(
        session,
        "weekly_report",
        f"Rapport de la semaine du {key}",
        row.markdown,
        niche_id=niche.id,
    )
    return row


def run_weekly(
    services: Services | None = None,
    force: bool = False,
    now: datetime | None = None,
) -> list[int]:
    """Rapport et plan de chaque niche. Renvoie les ids des rapports créés."""
    created: list[int] = []
    with get_session_factory()() as session:
        due = [item for item in _niches(session) if force or weekly_due(session, item, now)]
        if not due:
            return created
        services = services or Services()
        for niche in due:
            with (
                paid_work,
                logged_action(
                    session, "routine", "routine_weekly", {"niche_id": niche.id}
                ) as action,
            ):
                row = _weekly_for(session, niche, services, now)
                action.summary = f"Rapport {row.id}, scripts {row.script_ids}"
                created.append(row.id)
    return created


@dataclass
class Inbox:
    """Dossier surveillé. Un fichier n'est pris que si sa taille n'a pas bougé depuis le dernier
    passage : une copie en cours ne doit pas être ingérée à moitié.
    """

    sizes: dict[Path, int] = field(default_factory=dict)

    def _settled(self, path: Path) -> bool:
        size = path.stat().st_size
        previous = self.sizes.get(path)
        self.sizes[path] = size
        return previous == size and size > 0

    def scan(self, services: Services | None = None, wait_for_copy: bool = True) -> list[int]:
        """Traite ce qui est prêt et renvoie les ids des vidéos analysées."""
        if not paid_work.acquire(blocking=False):
            return []  # Une autre routine travaille : le prochain passage reprendra.
        try:
            return self._scan(services, wait_for_copy)
        finally:
            paid_work.release()

    def _scan(self, services: Services | None, wait_for_copy: bool) -> list[int]:
        root = inbox_root()
        files = [(path, False) for path in _videos_in(root)]
        files += [(path, True) for path in _videos_in(root / "mine")]
        ready = [(path, own) for path, own in files if self._ready(path, wait_for_copy)]
        if not ready:
            return []

        services = services or Services()
        done: list[int] = []
        with get_session_factory()() as session:
            for path, own in ready:
                if not can_spend(session, TOKENS_INGEST + TOKENS_ANALYSIS):
                    _budget_notice(session, "Le dossier inbox")
                    break  # Les fichiers restent dans le dossier.
                video_id = _process_file(session, path, own, services)
                self.sizes.pop(path, None)
                if video_id is not None:
                    done.append(video_id)
        return done

    def _ready(self, path: Path, wait_for_copy: bool) -> bool:
        return not wait_for_copy or self._settled(path)


def inbox_root() -> Path:
    root = get_settings().data_dir.resolve() / "inbox"
    (root / "mine").mkdir(parents=True, exist_ok=True)
    return root


def _videos_in(folder: Path) -> list[Path]:
    return sorted(
        item
        for item in folder.iterdir()
        if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS
    )


def _file_away(path: Path, folder: str) -> None:
    """Sort le fichier de l'inbox. S'il est verrouillé, le hash évitera de le repayer."""
    target = inbox_root() / folder
    target.mkdir(exist_ok=True)
    destination = target / path.name
    if destination.exists():
        destination = target / f"{path.stem}_{utcnow():%Y%m%d%H%M%S}{path.suffix}"
    try:
        shutil.move(path, destination)
    except OSError as exc:
        logger.warning("Impossible de déplacer %s : %s", path, exc)


def _default_niche(session: Session) -> Niche | None:
    niches = _niches(session)
    return niches[0] if len(niches) == 1 else None


def _process_file(session: Session, path: Path, own: bool, services: Services) -> int | None:
    """Ingère et analyse un fichier. Une analyse ratée sera reprise par la routine quotidienne."""
    niche = _default_niche(session)
    with logged_action(
        session, "routine", "inbox", {"file": path.name, "is_own_content": own}
    ) as action:
        try:
            submission = submit(
                session,
                str(path),
                is_own_content=own,
                niche_source=niche.name if niche else None,
            )
            if not submission.deduplicated:
                run_job(submission.job_id, services.deps)
                job: IngestJob = wait_for_job(session, submission.job_id)
                if job.status is JobStatus.FAILED:
                    raise IngestError(f"étape {job.stage} : {job.error}")
        except IngestError as exc:
            action.status = ActionStatus.ERROR
            action.summary = str(exc)
            _file_away(path, "failed")
            notify(session, "inbox_failed", f"Ingestion échouée : {path.name}", str(exc))
            return None

        video = canonical(session, session.get(Video, submission.video_id))
        video_id = video.id
        _file_away(path, "done")
        try:
            analyze(session, video_id, services.claude, embedder=services.embedder)
        except (LookupError, AnalysisError, *CLAUDE_FAILURES) as exc:
            session.rollback()
            action.status = ActionStatus.ERROR
            action.summary = f"Vidéo {video_id} ingérée, analyse échouée : {exc}"
            notify(session, "inbox_failed", f"Analyse échouée : {path.name}", str(exc))
            return None

        action.summary = f"Vidéo {video_id} ingérée et analysée."
        body = f"Vidéo {video_id}."
        if own:
            body += f" Relie-la à son script : cm publish <script_id> --video {video_id}."
        notify(session, "inbox_done", f"Analysée : {path.name}", body)
        return video_id
