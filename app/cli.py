"""Commande `cm`."""

import logging
from pathlib import Path
from typing import Annotated

import anthropic
import typer

from app.analysis.service import AnalysisError, analyze, pending_video_ids, report_path
from app.db.models import IngestJob, JobStatus, Pattern, PatternType, Video
from app.db.session import get_session_factory
from app.ingest.jobs import Submission, Worker, canonical, drain, latest_job, run_job, submit
from app.ingest.pipeline import IngestDeps, IngestError
from app.llm.client import ClaudeClient, get_claude
from app.llm.errors import ClaudeError
from app.patterns.search import score_mean, to_detail, top_patterns

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}

cli = typer.Typer(help="ContentMachine", no_args_is_help=True)
patterns_cli = typer.Typer(help="Bibliothèque de patterns.", no_args_is_help=True)
cli.add_typer(patterns_cli, name="patterns")

Own = Annotated[bool, typer.Option("--own", help="C'est une de mes vidéos.")]
Niche = Annotated[str | None, typer.Option("--niche", help="Niche d'origine de la vidéo.")]
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
    niche: Niche = None,
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
    niche: Niche = None,
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
