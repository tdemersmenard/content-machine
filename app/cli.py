"""Commande `cm`."""

import json
import logging
import socket
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import anthropic
import typer
import uvicorn
from alembic.config import Config
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from alembic import command
from app.agent.budget import budget_status
from app.agent.chat import AgentError, Step, chat, start_session
from app.agent.routines import (
    Inbox,
    Services,
    run_daily,
    run_reminders,
    run_weekly,
    weekly_due,
)
from app.analysis.service import AnalysisError, analyze, pending_video_ids, report_path
from app.config import get_settings
from app.db.base import utcnow
from app.db.models import (
    AgentAction,
    AgentSession,
    IngestJob,
    JobStatus,
    Niche,
    NicheInterview,
    Notification,
    Pattern,
    PatternType,
    Script,
    ScriptStatus,
    Video,
)
from app.db.session import get_session_factory
from app.generate.service import (
    GenerationError,
    GenerationResult,
    adapt,
    generate,
    load_content,
    script_path,
    update_script,
)
from app.ingest.jobs import Submission, Worker, canonical, drain, latest_job, run_job, submit
from app.ingest.pipeline import IngestDeps, IngestError
from app.learning.imports import ImportError_, import_csv, parse_count, parse_rate, parse_seconds
from app.learning.lessons import maybe_synthesize
from app.learning.report import build_report
from app.learning.schemas import PerformanceInput
from app.learning.service import record_performance, run_learning, scored_results
from app.llm.client import ClaudeClient, get_claude
from app.llm.errors import ClaudeError
from app.niche.interview import (
    InterviewError,
    answer_interview,
    current_question,
    finish_interview,
    start_interview,
)
from app.niche.topics import refresh_topics
from app.niche.voice import refresh_voice
from app.patterns.search import score_mean, to_detail, top_patterns

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}

cli = typer.Typer(help="ContentMachine", no_args_is_help=True)
patterns_cli = typer.Typer(help="Bibliothèque de patterns.", no_args_is_help=True)
cli.add_typer(patterns_cli, name="patterns")
niche_cli = typer.Typer(help="Profils de niche.", no_args_is_help=True)
cli.add_typer(niche_cli, name="niche")

Own = Annotated[bool, typer.Option("--own", help="C'est une de mes vidéos.")]
NicheOption = Annotated[str | None, typer.Option("--niche", help="Niche d'origine de la vidéo.")]
NoWait = Annotated[bool, typer.Option("--no-wait", help="Met en file sans traiter.")]


def default_deps() -> IngestDeps:
    return IngestDeps()


def default_claude() -> ClaudeClient:
    return get_claude()


def _report(submission: Submission) -> bool:
    """Affiche le résultat d'une soumission. Renvoie False si elle a échoué."""
    with get_session_factory()() as session:
        job = session.get(IngestJob, submission.job_id) if submission.job_id else None
        video_id = job.video_id if job else submission.video_id
        video = canonical(session, session.get(Video, video_id))

        if submission.deduplicated:
            typer.echo(f"Vidéo {video.id} : déjà ingérée, rien à faire.")
        elif job is None or job.status is JobStatus.DONE:
            typer.echo(f"Vidéo {video.id} : ingérée. Timeline : {video.timeline_path}")
        elif job.status is JobStatus.FAILED:
            typer.echo(f"Vidéo {video.id} : échec à l'étape {job.stage}. {job.error}", err=True)
            return False
        else:
            typer.echo(f"Vidéo {video.id} : {job.status.value} (job {job.id}).")
    return True


def _submit(source: str, own: bool, niche: str | None) -> Submission:
    with get_session_factory()() as session:
        try:
            return submit(session, source, is_own_content=own, niche_source=niche)
        except IngestError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(1) from exc


@cli.command("ingest")
def ingest_command(
    source: Annotated[str, typer.Argument(help="Fichier vidéo ou URL.")],
    own: Own = False,
    niche: NicheOption = None,
    no_wait: NoWait = False,
) -> None:
    """Ingère une vidéo."""
    submission = _submit(source, own, niche)
    if not no_wait and not submission.deduplicated:
        run_job(submission.job_id, default_deps())
    if not _report(submission):
        raise typer.Exit(1)


@cli.command("ingest-folder")
def ingest_folder_command(
    folder: Annotated[Path, typer.Argument(exists=True, file_okay=False, help="Dossier.")],
    own: Own = False,
    niche: NicheOption = None,
    no_wait: NoWait = False,
) -> None:
    """Ingère toutes les vidéos d'un dossier (sans ses sous-dossiers)."""
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)
    if not files:
        typer.echo(f"Aucune vidéo dans {folder}.", err=True)
        raise typer.Exit(1)

    submissions = [_submit(str(path), own, niche) for path in files]
    if not no_wait:
        drain(default_deps())
    results = [_report(submission) for submission in submissions]
    typer.echo(f"{sum(results)}/{len(results)} vidéos traitées sans erreur.")
    if not all(results):
        raise typer.Exit(1)


def _analyze_one(video_id: int, force: bool) -> bool:
    with get_session_factory()() as session:
        try:
            row = analyze(session, video_id, default_claude(), force=force)
        except (LookupError, AnalysisError, ClaudeError, anthropic.APIError) as exc:
            typer.echo(f"Vidéo {video_id} : échec de l'analyse. {exc}", err=True)
            return False
        path = report_path(session.get(Video, row.video_id), row.id)
        typer.echo(f"Vidéo {video_id} : analyse {row.id}. Rapport : {path}")
        if row.patterns_extracted_at is None:
            typer.echo("Patterns non extraits : relance la commande pour réessayer.", err=True)
    return True


@cli.command("analyze")
def analyze_command(
    video_id: Annotated[int | None, typer.Argument(help="Id de la vidéo.")] = None,
    all_pending: Annotated[
        bool, typer.Option("--all-pending", help="Toutes les vidéos non analysées.")
    ] = False,
    force: Annotated[bool, typer.Option("--force", help="Refait une analyse existante.")] = False,
) -> None:
    """Analyse une vidéo ingérée, ou toutes celles en attente."""
    if (video_id is not None) == all_pending:
        typer.echo("Indique soit un id de vidéo, soit --all-pending.", err=True)
        raise typer.Exit(2)

    if video_id is not None:
        ids = [video_id]
    else:
        with get_session_factory()() as session:
            ids = pending_video_ids(session)
        if not ids:
            typer.echo("Aucune vidéo en attente d'analyse.")
            return

    results = [_analyze_one(item, force) for item in ids]
    if all_pending:
        typer.echo(f"{sum(results)}/{len(results)} vidéos analysées.")
    if not all(results):
        raise typer.Exit(1)


@patterns_cli.command("list")
def patterns_list_command(
    type: Annotated[PatternType | None, typer.Option("--type", help="Type de pattern.")] = None,
    niche_id: Annotated[int | None, typer.Option("--niche-id", help="Id de la niche.")] = None,
    limit: Annotated[int, typer.Option("--limit", help="Nombre maximal.")] = 50,
) -> None:
    """Liste les patterns, du mieux noté au moins bien noté."""
    with get_session_factory()() as session:
        try:
            patterns = top_patterns(session, type, niche_id, limit)
        except LookupError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(1) from exc
        if not patterns:
            typer.echo("Aucun pattern.")
        for item in patterns:
            typer.echo(
                f"#{item.id} [{item.type}] {item.name} "
                f"(score {score_mean(item):.2f}, {len(item.examples)} exemple(s))"
            )


@patterns_cli.command("show")
def patterns_show_command(pattern_id: int) -> None:
    """Affiche un pattern."""
    with get_session_factory()() as session:
        pattern = session.get(Pattern, pattern_id)
        if pattern is None:
            typer.echo("Pattern inconnu.", err=True)
            raise typer.Exit(1)
        detail = to_detail(session, pattern)

    typer.echo(f"#{detail.id} [{detail.type.value}] {detail.name}")
    typer.echo(f"\n{detail.description}")
    typer.echo(f"\nTemplate : {detail.template}")
    typer.echo(f"Quand l'utiliser : {detail.when_to_use}")
    typer.echo(f"Quand l'éviter : {detail.when_not_to_use}")
    typer.echo(f"Score : {detail.score_mean:.2f} (Beta {detail.score_alpha}/{detail.score_beta})")
    typer.echo(f"\nExemples ({detail.example_count}) :")
    for example in detail.examples:
        typer.echo(
            f"- Vidéo {example['video_id']} à {example['timestamp']} s : {example['excerpt']}"
        )


def _load_niche(session: Session, niche_id: int) -> Niche:
    niche = session.get(Niche, niche_id)
    if niche is None or niche.profile is None:
        typer.echo("Niche inconnue ou sans profil.", err=True)
        raise typer.Exit(1)
    return niche


def _claude_failure(exc: Exception) -> typer.Exit:
    typer.echo(f"Échec de l'appel à Claude : {exc}", err=True)
    return typer.Exit(1)


@niche_cli.command("new")
def niche_new_command(
    name: Annotated[str | None, typer.Option("--name", help="Nom de la niche.")] = None,
    resume: Annotated[int | None, typer.Option("--resume", help="Id de l'entrevue.")] = None,
) -> None:
    """Crée une niche par entrevue. Tape /fin pour terminer avant la fin."""
    claude = default_claude()
    with get_session_factory()() as session:
        try:
            if resume is None:
                interview = start_interview(session, claude, name)
            else:
                interview = session.get(NicheInterview, resume)
                if interview is None:
                    typer.echo("Entrevue inconnue.", err=True)
                    raise typer.Exit(1)
            typer.echo(
                f"Entrevue {interview.id}. Pour la reprendre : cm niche new --resume {interview.id}"
            )

            while (question := current_question(interview)) is not None:
                typer.echo(f"\n{question}")
                answer = typer.prompt(">", prompt_suffix=" ")
                if answer.strip() == "/fin":
                    finish_interview(session, interview, claude)
                    break
                try:
                    answer_interview(session, interview, answer, claude)
                except InterviewError as exc:
                    typer.echo(str(exc), err=True)
        except (ClaudeError, anthropic.APIError) as exc:
            raise _claude_failure(exc) from exc

        niche = session.get(Niche, interview.niche_id)
        active = sum(1 for item in niche.topics if item["status"] == "new")
        typer.echo(f"\nNiche {niche.id} créée : {niche.name}. {active} sujet(s) en banque.")
        if not active:
            typer.echo(f"Banque de sujets vide : relance cm niche topics {niche.id}.", err=True)


@niche_cli.command("list")
def niche_list_command() -> None:
    """Liste les niches."""
    with get_session_factory()() as session:
        niches = session.scalars(select(Niche).order_by(Niche.id)).all()
        if not niches:
            typer.echo("Aucune niche.")
        for item in niches:
            typer.echo(f"#{item.id} {item.name} (version {item.version})")


@niche_cli.command("show")
def niche_show_command(niche_id: int) -> None:
    """Affiche la fiche avatar."""
    with get_session_factory()() as session:
        typer.echo(_load_niche(session, niche_id).avatar_markdown)


@niche_cli.command("topics")
def niche_topics_command(
    niche_id: int,
    refresh: Annotated[bool, typer.Option("--refresh", help="Complète la banque.")] = False,
    limit: Annotated[int, typer.Option("--limit", help="Nombre à afficher.")] = 50,
) -> None:
    """Affiche la banque de sujets, du plus intense au moins intense."""
    with get_session_factory()() as session:
        niche = _load_niche(session, niche_id)
        if refresh or not niche.topics:
            try:
                refresh_topics(session, niche, default_claude())
            except (ClaudeError, anthropic.APIError) as exc:
                raise _claude_failure(exc) from exc
        active = [item for item in niche.topics if item["status"] == "new"]
        for item in active[:limit]:
            hypothesis = " (hypothèse)" if item["grounding"] == "inferred" else ""
            typer.echo(f"{item['intensity']:>2}/10 [{item['id']}] {item['title']}{hypothesis}")


@niche_cli.command("voice")
def niche_voice_command(
    niche_id: int,
    force: Annotated[bool, typer.Option("--force", help="Recalcule même sans changement.")] = False,
) -> None:
    """Met à jour ma voix depuis mes propres vidéos."""
    with get_session_factory()() as session:
        niche = _load_niche(session, niche_id)
        try:
            voice = refresh_voice(session, niche, default_claude(), force)
        except (ClaudeError, anthropic.APIError) as exc:
            raise _claude_failure(exc) from exc
        if voice is None:
            typer.echo("Aucune de mes vidéos avec de la parole : utilise cm ingest --own.")
            return
        typer.echo(
            f"{len(voice.video_ids)} vidéo(s), {voice.words_per_second:.1f} mots/s, "
            f"{voice.avg_sentence_words:.1f} mots par phrase."
        )
        typer.echo(voice.style_summary)


def _report_scripts(result: GenerationResult) -> None:
    for script in result.scripts:
        content = load_content(script)
        typer.echo(
            f"Script {script.id} : {content.internal_title} "
            f"(hook {content.final_critique.hook_score}/10, "
            f"{content.estimated_duration_s:.0f} s, {len(content.to_verify)} à vérifier)"
        )
        typer.echo(f"  {script_path(script.id)}")
    for failure in result.failures:
        typer.echo(f"Script abandonné : {failure}", err=True)


@cli.command("adapt")
def adapt_command(
    video_id: int,
    niche: Annotated[int, typer.Option("--niche", help="Id de la niche.")],
    variants: Annotated[int, typer.Option("-n", help="Nombre de variantes.")] = 3,
) -> None:
    """Adapte une vidéo analysée à ta niche."""
    with get_session_factory()() as session:
        try:
            _report_scripts(adapt(session, video_id, niche, default_claude(), variants))
        except (LookupError, GenerationError, ClaudeError, anthropic.APIError) as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(1) from exc


@cli.command("ideas")
def ideas_command(
    niche: Annotated[int, typer.Option("--niche", help="Id de la niche.")],
    count: Annotated[int, typer.Option("-n", help="Nombre d'idées proposées.")] = 5,
    develop: Annotated[
        int, typer.Option("--develop", help="Nombre d'idées développées en scripts.")
    ] = 3,
    objectif: Annotated[str | None, typer.Option("--objectif", help="Objectif visé.")] = None,
) -> None:
    """Propose des idées et développe les meilleures en scripts."""
    with get_session_factory()() as session:
        try:
            _report_scripts(
                generate(session, niche, default_claude(), objectif, count, min(develop, count))
            )
        except (LookupError, GenerationError, ClaudeError, anthropic.APIError) as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(1) from exc


def _ask(label: str, parse: Callable[[str], Any]) -> Any:
    """Une question facultative : laisser vide veut dire « je ne sais pas »."""
    while True:
        raw = typer.prompt(label, default="", show_default=False)
        if not raw.strip():
            return None
        try:
            return parse(raw)
        except ValueError:
            typer.echo("Valeur illisible, recommence ou laisse vide.", err=True)


@cli.command("stats")
def stats_command(script_id: int) -> None:
    """Saisit les stats d'une vidéo publiée, puis met à jour ce que la machine sait."""
    with get_session_factory()() as session:
        script = session.get(Script, script_id)
        if script is None:
            typer.echo("Script inconnu.", err=True)
            raise typer.Exit(1)

        typer.echo(f"Script {script.id} : {load_content(script).internal_title}")
        typer.echo("Laisse vide ce que tu ne connais pas.")
        horizon = typer.prompt("Horizon en heures (24, 72 ou 168)", type=int, default=72)
        url = None
        if script.published_video_id is None:
            url = _ask("Lien de la vidéo", str.strip)

        try:
            data = PerformanceInput(
                horizon_h=horizon,
                video_url=url,
                views=_ask("Vues", parse_count),
                likes=_ask("Likes", parse_count),
                retention_3s=_ask("Rétention à 3 s (%)", parse_rate),
                retention_50=_ask("Rétention à 50 % (%)", parse_rate),
                completion_rate=_ask("Taux de complétion (%)", parse_rate),
                avg_watch_time_s=_ask("Temps de visionnage moyen (s)", parse_seconds),
                shares=_ask("Partages", parse_count),
                saves=_ask("Sauvegardes", parse_count),
                comments=_ask("Commentaires", parse_count),
                follows=_ask("Abonnés gagnés", parse_count),
                notes=_ask("Notes", str.strip),
            )
        except ValidationError as exc:
            typer.echo(f"Stats invalides : {exc}", err=True)
            raise typer.Exit(1) from exc

        result = record_performance(session, script, data, default_claude())
        row = result.performance
        if row.composite is None:
            typer.echo("Enregistré. Pas encore de score : il manque de l'historique à cet horizon.")
        else:
            typer.echo(f"Score : {row.composite:.2f}x ta baseline ({row.label}).")
        if result.post_mortem:
            typer.echo(result.post_mortem.content["summary"])
        if result.lessons_updated:
            typer.echo("Leçons mises à jour.")
        for warning in result.warnings:
            typer.echo(warning, err=True)


@cli.command("stats-import")
def stats_import_command(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Fichier CSV.")],
    horizon: Annotated[int, typer.Option("--horizon", help="24, 72 ou 168.")],
    preset: Annotated[str | None, typer.Option("--preset", help="tiktok, meta ou youtube.")] = None,
    mapping: Annotated[
        list[str] | None, typer.Option("--map", help="Colonne=champ, par exemple Vues=views.")
    ] = None,
) -> None:
    """Importe un export CSV, puis lance cm learn pour les post-mortems."""
    columns = dict(item.split("=", 1) for item in mapping or [])
    with get_session_factory()() as session:
        try:
            result = import_csv(
                session, path.read_text(encoding="utf-8-sig"), horizon, preset, columns
            )
        except ImportError_ as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(1) from exc
    typer.echo(f"{result.imported} ligne(s) importée(s).")
    for line in result.skipped:
        typer.echo(line, err=True)


@cli.command("learn")
def learn_command(
    force: Annotated[
        bool, typer.Option("--force", help="Réécrit les leçons sans attendre.")
    ] = False,
) -> None:
    """Rattrape les post-mortems manquants et met à jour les leçons si c'est le moment."""
    claude = default_claude()
    with get_session_factory()() as session:
        for script, row in scored_results(session):
            _, _, warnings = run_learning(session, script, row, claude)
            for warning in warnings:
                typer.echo(warning, err=True)
        for niche in session.scalars(select(Niche).where(Niche.profile.is_not(None))).all():
            if maybe_synthesize(session, niche, claude, force):
                typer.echo(f"Leçons réécrites pour {niche.name}.")
    typer.echo("Terminé.")


@cli.command("report")
def report_command(
    niche: Annotated[int | None, typer.Option("--niche", help="Id de la niche.")] = None,
) -> None:
    """Affiche le rapport de suivi."""
    with get_session_factory()() as session:
        typer.echo(json.dumps(build_report(session, niche), ensure_ascii=False, indent=2))


@cli.command("publish")
def publish_command(
    script_id: int,
    video: Annotated[int | None, typer.Option("--video", help="Id de la vidéo ingérée.")] = None,
    at: Annotated[
        datetime | None,
        typer.Option("--at", help="Date de publication, si ce n'est pas maintenant."),
    ] = None,
) -> None:
    """Note qu'un script est publié : c'est toi qui publies, ça lance les rappels de stats."""
    with get_session_factory()() as session:
        script = session.get(Script, script_id)
        if script is None:
            typer.echo("Script inconnu.", err=True)
            raise typer.Exit(1)
        published_at = at.astimezone(UTC) if at else None
        try:
            update_script(
                session,
                script,
                status=ScriptStatus.PUBLISHED,
                published_video_id=video,
                published_at=published_at,
            )
        except LookupError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(1) from exc
        typer.echo(f"Script {script_id} publié. Rappels de stats à 24 h, 72 h et 7 jours.")


def _show_step(step: Step) -> None:
    if step.kind == "text":
        typer.echo(step.detail)
    elif step.kind == "tool_start":
        typer.echo(f"  → {step.name} {step.detail}")
    elif step.is_error:
        typer.echo(f"  ✗ {step.name} : {step.detail[:300]}", err=True)


@cli.command("chat")
def chat_command(
    niche: Annotated[int, typer.Option("--niche", help="Id de la niche.")],
    resume: Annotated[int | None, typer.Option("--resume", help="Id de la session.")] = None,
) -> None:
    """Parle à l'agent. Tape /fin pour quitter."""
    claude = default_claude()
    with get_session_factory()() as session:
        try:
            agent_session = (
                session.get(AgentSession, resume) if resume else start_session(session, niche)
            )
        except AgentError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(1) from exc
        if agent_session is None or agent_session.niche_id != niche:
            typer.echo("Session inconnue pour cette niche.", err=True)
            raise typer.Exit(1)

        typer.echo(f"Session {agent_session.id}. Tape /fin pour quitter.")
        while True:
            message = typer.prompt("\n>", prompt_suffix=" ").strip()
            if message == "/fin":
                break
            try:
                result = chat(session, agent_session, message, claude, _show_step)
            except ClaudeError as exc:
                raise _claude_failure(exc) from exc
            typer.echo(f"\n{result.answer}")
            if result.unverified:
                typer.echo(f"Références introuvables : {', '.join(result.unverified)}", err=True)
            typer.echo(f"({result.tokens} tokens, {result.cost_usd:.2f} $)")


@cli.command("notifications")
def notifications_command(
    all_: Annotated[bool, typer.Option("--all", help="Aussi celles déjà lues.")] = False,
) -> None:
    """Affiche les notifications et les marque comme lues."""
    with get_session_factory()() as session:
        statement = select(Notification).order_by(Notification.id)
        if not all_:
            statement = statement.where(Notification.read_at.is_(None))
        rows = session.scalars(statement).all()
        if not rows:
            typer.echo("Aucune notification.")
        for row in rows:
            typer.echo(f"#{row.id} [{row.kind}] {row.title}\n{row.body}\n")
            row.read_at = row.read_at or utcnow()
        session.commit()


@cli.command("actions")
def actions_command(
    limit: Annotated[int, typer.Option("--limit", help="Nombre d'actions.")] = 20,
) -> None:
    """Affiche le budget du jour et le journal de l'agent."""
    with get_session_factory()() as session:
        budget = budget_status(session)
        typer.echo(
            f"Budget du jour : {budget['used']}/{budget['limit']} tokens "
            f"({budget['cost_usd']:.2f} $)"
        )
        rows = session.scalars(
            select(AgentAction).order_by(AgentAction.id.desc()).limit(limit)
        ).all()
        for row in reversed(rows):
            summary = (row.summary or "")[:100].replace("\n", " ")
            typer.echo(
                f"#{row.id} {row.created_at:%Y-%m-%d %H:%M} {row.source}/{row.action} "
                f"{row.status.value} {summary}"
            )


@cli.command("routine")
def routine_command(
    name: Annotated[str, typer.Argument(help="daily, weekly, reminders ou inbox.")],
    force: Annotated[bool, typer.Option("--force", help="Lance même si ce n'est pas dû.")] = False,
) -> None:
    """Lance une routine à la main, par exemple quand l'API ne tourne pas."""
    if name == "daily":
        result = run_daily(Services(default_claude(), default_deps()), force)
        typer.echo("Pas due." if result is None else json.dumps(result))
    elif name == "weekly":
        reports = run_weekly(Services(default_claude(), default_deps()), force)
        typer.echo(f"Rapports créés : {reports}" if reports else "Pas due.")
    elif name == "reminders":
        typer.echo(f"{run_reminders()} rappel(s) créé(s).")
    elif name == "inbox":
        videos = Inbox().scan(Services(default_claude(), default_deps()), wait_for_copy=False)
        typer.echo(f"Vidéos analysées : {videos}")
    else:
        typer.echo("Routine inconnue : daily, weekly, reminders ou inbox.", err=True)
        raise typer.Exit(2)


@cli.command("status")
def status_command(video_id: int) -> None:
    """Affiche le statut d'une vidéo."""
    with get_session_factory()() as session:
        video = session.get(Video, video_id)
        if video is None:
            typer.echo("Vidéo inconnue.", err=True)
            raise typer.Exit(1)
        video = canonical(session, video)
        job = latest_job(session, video.id)
        status = job.status.value if job else "aucun job"
        stage = f", étape {job.stage}" if job and job.stage else ""
        error = f"\n{job.error}" if job and job.error else ""
        typer.echo(f"Vidéo {video.id} : {status}{stage}{error}")


@cli.command("dev")
def dev_command(
    lan: Annotated[
        bool, typer.Option("--lan", help="Ouvre au réseau local, pour le téléphone.")
    ] = False,
    port: Annotated[int, typer.Option("--port", help="Port.")] = 8000,
) -> None:
    """Lance tout d'un coup : migrations, API, worker, routines et dashboard."""
    settings = get_settings()
    if lan and settings.dashboard_password is None:
        typer.echo("--lan exige DASHBOARD_PASSWORD dans le fichier .env.", err=True)
        raise typer.Exit(1)

    command.upgrade(Config("alembic.ini"), "head")
    with get_session_factory()() as session:
        pending = [item.name for item in _weekly_due(session)]
    if pending and settings.agent_scheduler_enabled and settings.agent_weekly_scripts > 0:
        typer.echo(
            f"Attention : le plan de la semaine n'a pas été fait. Il va partir dans 30 s "
            f"({settings.agent_weekly_scripts} scripts, environ 1,20 $ chacun). "
            "AGENT_WEEKLY_SCRIPTS=0 dans .env pour l'éviter.",
            err=True,
        )
    typer.echo(f"Dashboard : http://127.0.0.1:{port}")
    if lan:
        typer.echo(f"Téléphone : http://{_lan_address()}:{port}")
    uvicorn.run("app.main:app", host="0.0.0.0" if lan else "127.0.0.1", port=port)


def _weekly_due(session: Session) -> list[Niche]:
    niches = session.scalars(select(Niche).where(Niche.profile.is_not(None))).all()
    return [item for item in niches if weekly_due(session, item)]


def _lan_address() -> str:
    """L'adresse que le routeur voit. Aucun paquet n'est envoyé avec UDP sans données."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect(("10.255.255.255", 1))
        return probe.getsockname()[0]


@cli.command("worker")
def worker_command() -> None:
    """Traite la file en continu, sans lancer l'API."""
    worker = Worker(default_deps())
    worker.start()
    typer.echo("Worker démarré. Ctrl+C pour arrêter.")
    try:
        while worker.is_alive():
            worker.join(timeout=1)
    except KeyboardInterrupt:
        worker.stop()


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    cli()
