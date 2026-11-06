"""Pages du dashboard. Rendu par le serveur, mises à jour partielles avec HTMX.

Tout ce qui appelle Claude passe par une tâche de fond : une page ne reste jamais bloquée.
"""

import shutil
import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.agent.budget import (
    TOKENS_ADAPT_BASE,
    TOKENS_ANALYSIS,
    TOKENS_SCRIPT,
    budget_status,
    check_budget,
    paid_work,
)
from app.agent.chat import Step, chat, check_citations, start_session
from app.agent.tools import VIDEO_EXTENSIONS, wait_for_job
from app.analysis.report import HOOK_TYPES, ROLES, clock, render_report
from app.analysis.schemas import AnalysisDocument
from app.analysis.service import analyze
from app.config import get_settings
from app.db.base import utcnow
from app.db.models import (
    AgentSession,
    Analysis,
    IngestJob,
    JobStatus,
    Lesson,
    Niche,
    Notification,
    Pattern,
    PatternType,
    Performance,
    PostMortem,
    Script,
    ScriptStatus,
    SourceType,
    Video,
    VideoPattern,
    WeeklyReport,
)
from app.db.session import get_db, get_session_factory
from app.generate.guards import spoken_texts, verification_notes
from app.generate.report import render_script
from app.generate.schemas import ScriptDraft
from app.generate.service import active_lessons, adapt, load_content, update_script
from app.ingest.jobs import canonical, is_url, latest_job, submit
from app.ingest.pipeline import IngestError
from app.learning.imports import parse_count, parse_rate, parse_seconds
from app.learning.report import build_report
from app.learning.schemas import PerformanceInput
from app.learning.service import run_learning, save_performance
from app.llm.client import ClaudeClient, get_claude
from app.web import views
from app.web.media import MediaError, playable_path
from app.web.security import COOKIE, session_token
from app.web.tasks import Task, runner

WEB_DIR = Path(__file__).resolve().parents[2] / "web"
PATTERN_LABELS = {
    "hook": "Hook",
    "angle": "Angle",
    "structure": "Structure",
    "retention_device": "Rétention",
    "ending": "Fin",
    "engagement_trigger": "Engagement",
    "visual_device": "Visuel",
    "pacing": "Rythme",
}
LABELS = {"winner": "Winner", "neutral": "Neutre", "loser": "Loser"}
STATUS_LABELS = {"draft": "Brouillon", "approved": "Approuvé", "published": "Publié"}
STAGES = {
    "acquire": "Récupération",
    "audio": "Transcription",
    "shots": "Découpage",
    "vision": "Lecture des images",
    "signals": "Signaux",
    "timeline": "Timeline",
}

Db = Annotated[Session, Depends(get_db)]
router = APIRouter(prefix="/ui", include_in_schema=False)


def _unread(request: Request) -> dict[str, Any]:
    with get_session_factory()() as session:
        count = session.scalar(
            select(func.count()).select_from(Notification).where(Notification.read_at.is_(None))
        )
    return {"unread_notifications": count, "path": request.url.path}


templates = Jinja2Templates(directory=WEB_DIR / "templates", context_processors=[_unread])
templates.env.filters["markdown"] = views.render_markdown
templates.env.filters["linked"] = views.linked_text
templates.env.filters["verify"] = views.highlight_verify
templates.env.filters["clock"] = clock
templates.env.globals.update(
    PATTERN_LABELS=PATTERN_LABELS,
    LABELS=LABELS,
    STATUS_LABELS=STATUS_LABELS,
    HOOK_TYPES=HOOK_TYPES,
    ROLES=ROLES,
    STAGES=STAGES,
)


def claude_client() -> ClaudeClient:
    return get_claude()


def render(request: Request, name: str, **context: Any) -> HTMLResponse:
    return templates.TemplateResponse(request, name, context)


def default_niche(session: Session) -> Niche | None:
    return session.scalar(select(Niche).where(Niche.profile.is_not(None)).order_by(Niche.id))


def _niche_or_404(session: Session, niche_id: int | None = None) -> Niche:
    niche = session.get(Niche, niche_id) if niche_id else default_niche(session)
    if niche is None or niche.profile is None:
        raise HTTPException(404, "Aucune niche : crée-la avec uv run cm niche new.")
    return niche


@router.get("/login")
def login_page(request: Request) -> HTMLResponse:
    return render(request, "login.html", error=None)


@router.post("/login", response_model=None)
def login(request: Request, password: Annotated[str, Form()]) -> Response:
    secret = get_settings().dashboard_password
    if secret is None or password != secret.get_secret_value():
        return render(request, "login.html", error="Mot de passe incorrect.")
    response = RedirectResponse("/ui/inbox", status_code=303)
    response.set_cookie(COOKIE, session_token(password), httponly=True, samesite="lax")
    return response


def _analyze_when_ingested(video_id: int, job_id: int | None) -> None:
    """Enchaîne l'analyse après l'ingestion, comme le dossier surveillé."""
    with get_session_factory()() as session:
        if job_id is not None:
            job = wait_for_job(session, job_id)
            if job.status is JobStatus.FAILED:
                raise IngestError(job.error or "Ingestion échouée.")
        exists = session.scalar(select(Analysis.id).where(Analysis.video_id == video_id))
        if exists is None:
            check_budget(session, TOKENS_ANALYSIS)
        with paid_work:
            analyze(session, video_id, claude_client())


def _queue_analysis(session: Session, source: str, kind: SourceType, own: bool) -> int:
    niche = default_niche(session)
    submission = submit(
        session,
        source,
        source_type=kind,
        is_own_content=own,
        niche_source=niche.name if niche else None,
    )
    video_id = canonical(session, session.get(Video, submission.video_id)).id
    job_id = None if submission.deduplicated else submission.job_id
    runner.start(
        f"Analyse de la vidéo {video_id}",
        lambda _: _analyze_when_ingested(video_id, job_id),
        key=f"video:{video_id}",
    )
    return video_id


def _queue_rows(session: Session) -> list[dict[str, Any]]:
    # Une saisie de stats crée une fiche sans fichier : elle n'a rien à faire dans la file.
    videos = session.scalars(
        select(Video)
        .where(
            Video.duplicate_of_id.is_(None),
            Video.ingested_at.is_not(None) | Video.id.in_(select(IngestJob.video_id)),
        )
        .order_by(Video.id.desc())
        .limit(30)
    ).all()
    rows = []
    for video in videos:
        job = latest_job(session, video.id)
        analyzed = session.scalar(select(Analysis.id).where(Analysis.video_id == video.id))
        active = runner.running_key(f"video:{video.id}")
        failure = runner.last_failure(f"video:{video.id}")
        if job is not None and job.status is JobStatus.FAILED:
            state, tone, detail = "Échec", "bad", f"{job.stage} : {job.error}"
        elif analyzed is not None:
            state, tone, detail = "Analysée", "good", ""
        elif failure is not None:
            state, tone, detail = "Analyse non faite", "bad", failure.error
        elif job is not None and job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
            state, tone = "En cours", "busy"
            detail = STAGES.get(job.stage or "", "En attente")
        elif active is not None:
            state, tone, detail = "Analyse en cours", "busy", "Quelques minutes"
        elif video.ingested_at is not None:
            state, tone, detail = "À analyser", "idle", ""
        else:
            state, tone, detail = "En attente", "idle", ""
        rows.append(
            {"video": video, "state": state, "tone": tone, "detail": detail, "done": bool(analyzed)}
        )
    return rows


@router.get("")
def home() -> RedirectResponse:
    return RedirectResponse("/ui/inbox", status_code=303)


@router.get("/inbox")
def inbox(request: Request, db: Db) -> HTMLResponse:
    return render(request, "inbox.html", rows=_queue_rows(db), budget=budget_status(db))


@router.get("/inbox/queue")
def inbox_queue(request: Request, db: Db) -> HTMLResponse:
    return render(request, "_queue.html", rows=_queue_rows(db))


@router.post("/inbox/links")
def inbox_links(
    request: Request,
    db: Db,
    links: Annotated[str, Form()],
    own: Annotated[bool, Form()] = False,
) -> HTMLResponse:
    urls = [item.strip() for item in links.split() if item.strip()]
    bad = [item for item in urls if not is_url(item)]
    for url in urls:
        if is_url(url):
            _queue_analysis(db, url, SourceType.URL, own)
    error = f"Lien ignoré : {bad[0]}" if bad else None
    return render(request, "_queue.html", rows=_queue_rows(db), error=error)


@router.post("/inbox/files")
def inbox_files(
    request: Request,
    db: Db,
    files: Annotated[list[UploadFile], File()],
    own: Annotated[bool, Form()] = False,
) -> HTMLResponse:
    folder = get_settings().data_dir.resolve() / "uploads"
    folder.mkdir(parents=True, exist_ok=True)
    error = None
    for upload in files:
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in VIDEO_EXTENSIONS:
            error = f"Fichier ignoré : {upload.filename}"
            continue
        path = folder / f"{uuid.uuid4().hex}{suffix}"  # Le nom du client n'est jamais utilisé.
        with path.open("wb") as output:
            shutil.copyfileobj(upload.file, output)
        _queue_analysis(db, str(path), SourceType.UPLOAD, own)
    return render(request, "_queue.html", rows=_queue_rows(db), error=error)


@router.post("/videos/{video_id}/analyze")
def analyze_video(request: Request, video_id: int, db: Db) -> HTMLResponse:
    if db.get(Video, video_id) is None:
        raise HTTPException(404, "Vidéo inconnue.")
    runner.start(
        f"Analyse de la vidéo {video_id}",
        lambda _: _analyze_when_ingested(video_id, None),
        key=f"video:{video_id}",
    )
    return render(request, "_queue.html", rows=_queue_rows(db))


@router.get("/media/{video_id}")
def media(video_id: int, db: Db) -> FileResponse:
    video = db.get(Video, video_id)
    if video is None:
        raise HTTPException(404, "Vidéo inconnue.")
    try:
        return FileResponse(playable_path(video), media_type="video/mp4")
    except MediaError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/videos/{video_id}")
def video_page(request: Request, video_id: int, db: Db) -> HTMLResponse:
    video = db.get(Video, video_id)
    if video is None:
        raise HTTPException(404, "Vidéo inconnue.")
    video = canonical(db, video)
    row = db.scalar(
        select(Analysis).where(Analysis.video_id == video.id).order_by(Analysis.id.desc()).limit(1)
    )
    context: dict[str, Any] = {"video": video, "analysis": None, "moments": [], "patterns": []}
    if row is not None:
        data = row.json_result["analysis"]
        context["analysis"] = data
        context["moments"] = views.moments(data, video.duration_s or 0)
        context["reasons"] = sorted(data["verdict"]["reasons"], key=lambda item: item["rank"])
        context["report"] = views.render_report_html(
            render_report(AnalysisDocument.model_validate(row.json_result))
        )
        context["patterns"] = db.execute(
            select(Pattern.id, Pattern.type, Pattern.name)
            .join(VideoPattern, VideoPattern.pattern_id == Pattern.id)
            .where(VideoPattern.video_id == video.id)
            .distinct()
            .order_by(Pattern.id)
        ).all()
    return render(request, "video.html", **context)


def _patterns(db: Session, kind: PatternType | None) -> list[dict[str, Any]]:
    statement = select(Pattern)
    if kind is not None:
        statement = statement.where(Pattern.type == kind.value)
    rows = [{"item": item, **views.confidence(item)} for item in db.scalars(statement)]
    rows.sort(key=lambda row: (-row["score"], -row["tests"], -len(row["item"].examples)))
    return rows


@router.get("/patterns")
def patterns_page(request: Request, db: Db, type: PatternType | None = None) -> HTMLResponse:
    name = "_patterns.html" if request.headers.get("hx-request") else "patterns.html"
    return render(request, name, rows=_patterns(db, type), active=type.value if type else None)


def _script_view(script: Script) -> dict[str, Any]:
    return {"script": script, "c": load_content(script)}


def _analyzed_videos(db: Session) -> list[Video]:
    return list(
        db.scalars(
            select(Video).where(Video.id.in_(select(Analysis.video_id))).order_by(Video.id.desc())
        )
    )


@router.get("/studio")
def studio(
    request: Request,
    db: Db,
    ids: str | None = None,
    video: int | None = None,
) -> HTMLResponse:
    if ids:
        wanted = [int(item) for item in ids.split(",") if item.isdigit()]
        scripts = list(db.scalars(select(Script).where(Script.id.in_(wanted)).order_by(Script.id)))
    else:
        scripts = list(db.scalars(select(Script).order_by(Script.id.desc()).limit(3)))
    return render(
        request,
        "studio.html",
        videos=_analyzed_videos(db),
        niches=db.scalars(select(Niche).where(Niche.profile.is_not(None))).all(),
        selected=video,
        cards=[_script_view(item) for item in scripts],
    )


@router.post("/studio/adapt")
def studio_adapt(
    request: Request,
    db: Db,
    video_id: Annotated[int, Form()],
    niche_id: Annotated[int, Form()],
    variants: Annotated[int, Form(ge=1, le=5)] = 3,
) -> HTMLResponse:
    def work(task: Task) -> None:
        with get_session_factory()() as session:
            check_budget(session, TOKENS_ADAPT_BASE + variants * TOKENS_SCRIPT)
            result = adapt(session, video_id, niche_id, claude_client(), variants)
            ids = ",".join(str(item.id) for item in result.scripts)
            task.redirect = f"/ui/studio?ids={ids}"

    task = runner.start(f"{variants} variante(s) de la vidéo {video_id}", work)
    return render(request, "_task.html", task=task)


def _refresh_checks(script: Script, changes: dict[str, Any]) -> dict[str, Any]:
    """Ce qui reste à vérifier suit ce que le créateur vient d'écrire."""
    merged = {**script.content, **changes}
    draft = ScriptDraft.model_validate({key: merged[key] for key in ScriptDraft.model_fields})
    return {**changes, "to_verify": verification_notes(spoken_texts(draft))}


def _script_or_404(db: Session, script_id: int) -> Script:
    script = db.get(Script, script_id)
    if script is None:
        raise HTTPException(404, "Script inconnu.")
    return script


@router.post("/scripts/{script_id}/beats/{index}")
def edit_beat(
    request: Request, script_id: int, index: int, db: Db, spoken: Annotated[str, Form()]
) -> HTMLResponse:
    script = _script_or_404(db, script_id)
    beats = [dict(item) for item in script.content["beats"]]
    if not 0 <= index < len(beats):
        raise HTTPException(404, "Beat inconnu.")
    if not spoken.strip():
        raise HTTPException(422, "Un beat ne peut pas être vide.")
    beats[index]["spoken"] = spoken.strip()
    update_script(db, script, changes=_refresh_checks(script, {"beats": beats}))
    return render(request, "_script_meta.html", **_script_view(script))


@router.post("/scripts/{script_id}/hook/{index}")
def choose_hook(request: Request, script_id: int, index: int, db: Db) -> HTMLResponse:
    """Le hook retenu passe en tête : c'est lui que le téléprompteur et la durée utilisent."""
    script = _script_or_404(db, script_id)
    hooks = list(script.content["hooks"])
    if not 0 <= index < len(hooks):
        raise HTTPException(404, "Hook inconnu.")
    hooks.insert(0, hooks.pop(index))
    update_script(db, script, changes=_refresh_checks(script, {"hooks": hooks}))
    return render(request, "_script_card.html", **_script_view(script))


@router.post("/scripts/{script_id}/approve")
def approve(request: Request, script_id: int, db: Db) -> HTMLResponse:
    script = _script_or_404(db, script_id)
    if script.status is ScriptStatus.DRAFT:
        update_script(db, script, status=ScriptStatus.APPROVED)
    return render(request, "_script_card.html", **_script_view(script))


@router.get("/scripts/{script_id}/teleprompter")
def teleprompter(request: Request, script_id: int, db: Db) -> HTMLResponse:
    return render(request, "teleprompter.html", **_script_view(_script_or_404(db, script_id)))


@router.get("/scripts/{script_id}/print")
def print_script(request: Request, script_id: int, db: Db) -> HTMLResponse:
    return render(request, "print.html", **_script_view(_script_or_404(db, script_id)))


@router.get("/scripts/{script_id}/markdown")
def script_markdown(script_id: int, db: Db) -> Response:
    content = load_content(_script_or_404(db, script_id))
    return Response(render_script(content), media_type="text/markdown; charset=utf-8")


def _latest_reading(db: Session, script: Script) -> Performance | None:
    return db.scalar(
        select(Performance)
        .where(Performance.script_id == script.id)
        .order_by(Performance.horizon_h.desc())
        .limit(1)
    )


@router.get("/results")
def results(request: Request, db: Db) -> HTMLResponse:
    published = db.scalars(
        select(Script)
        .where(Script.status == ScriptStatus.PUBLISHED)
        .order_by(Script.published_at.desc())
    ).all()
    ready = db.scalars(
        select(Script).where(Script.status == ScriptStatus.APPROVED).order_by(Script.id.desc())
    ).all()
    return render(
        request,
        "results.html",
        published=[
            {**_script_view(item), "reading": _latest_reading(db, item)} for item in published
        ],
        ready=[_script_view(item) for item in ready],
    )


@router.post("/scripts/{script_id}/publish")
def publish(script_id: int, db: Db) -> RedirectResponse:
    update_script(db, _script_or_404(db, script_id), status=ScriptStatus.PUBLISHED)
    return RedirectResponse(f"/ui/results/{script_id}", status_code=303)


@router.get("/results/{script_id}")
def result_detail(request: Request, script_id: int, db: Db, error: str | None = None):
    script = _script_or_404(db, script_id)
    readings = db.scalars(
        select(Performance)
        .where(Performance.script_id == script.id)
        .order_by(Performance.horizon_h)
    ).all()
    post_mortems = db.scalars(
        select(PostMortem).where(PostMortem.script_id == script.id).order_by(PostMortem.horizon_h)
    ).all()
    own = db.scalars(
        select(Video)
        .where(Video.is_own_content.is_(True), Video.ingested_at.is_not(None))
        .order_by(Video.id.desc())
        .limit(30)
    ).all()
    return render(
        request,
        "result_detail.html",
        **_script_view(script),
        readings=readings,
        post_mortems=post_mortems,
        own_videos=own,
        learning=runner.running_key(f"script:{script_id}"),
        error=error,
    )


def _parsed(form: dict[str, str], name: str, parse: Any) -> Any:
    raw = form.get(name, "").strip()
    return parse(raw) if raw else None


@router.post("/results/{script_id}")
async def save_stats(request: Request, script_id: int, db: Db) -> RedirectResponse:
    """Les stats sont enregistrées tout de suite, le post-mortem suit en tâche de fond."""
    script = _script_or_404(db, script_id)
    form = {key: str(value) for key, value in (await request.form()).items()}
    try:
        data = PerformanceInput(
            horizon_h=int(form.get("horizon_h", "72")),
            video_url=form.get("video_url", "").strip() or None,
            views=_parsed(form, "views", parse_count),
            likes=_parsed(form, "likes", parse_count),
            shares=_parsed(form, "shares", parse_count),
            saves=_parsed(form, "saves", parse_count),
            comments=_parsed(form, "comments", parse_count),
            follows=_parsed(form, "follows", parse_count),
            retention_3s=_parsed(form, "retention_3s", parse_rate),
            retention_50=_parsed(form, "retention_50", parse_rate),
            completion_rate=_parsed(form, "completion_rate", parse_rate),
            avg_watch_time_s=_parsed(form, "avg_watch_time_s", parse_seconds),
            notes=form.get("notes", "").strip() or None,
        )
    except (ValueError, ValidationError):
        return RedirectResponse(f"/ui/results/{script_id}?error=Valeur+illisible", status_code=303)

    linked = form.get("published_video_id", "")
    if linked.isdigit():
        update_script(db, script, published_video_id=int(linked))
    row_id = save_performance(db, script, data).id

    def learn(_: Task) -> None:
        with get_session_factory()() as session:
            row = session.get(Performance, row_id)
            run_learning(session, session.get(Script, script_id), row, claude_client())

    runner.start(f"Post-mortem du script {script_id}", learn, key=f"script:{script_id}")
    return RedirectResponse(f"/ui/results/{script_id}", status_code=303)


@router.get("/brain")
def brain(request: Request, db: Db) -> HTMLResponse:
    niche = _niche_or_404(db)
    weekly = db.scalar(
        select(WeeklyReport)
        .where(WeeklyReport.niche_id == niche.id)
        .order_by(WeeklyReport.week_start.desc())
        .limit(1)
    )
    report = build_report(db, niche.id)
    return render(
        request,
        "brain.html",
        report=report,
        lessons=active_lessons(db, niche.id),
        retired=db.scalars(
            select(Lesson)
            .where(Lesson.active.is_(False), Lesson.niche_id == niche.id)
            .order_by(Lesson.deactivated_at.desc())
            .limit(5)
        ).all(),
        chart=views.composite_chart(db, niche.id),
        trends=views.pattern_trends(db, niche.id),
        weekly=weekly,
        budget=budget_status(db),
    )


def _chat_messages(row: AgentSession | None) -> list[dict[str, Any]]:
    """Ce que l'utilisateur doit voir : ni la mémoire, ni les résultats d'outils bruts."""
    shown: list[dict[str, Any]] = []
    for message in row.messages if row else []:
        content = message["content"]
        if message["role"] == "user":
            if isinstance(content, str):
                shown.append({"role": "user", "text": content})
            elif content[0]["type"] == "text":
                shown.append({"role": "user", "text": content[-1]["text"]})
            continue
        tools = [block["name"] for block in content if block["type"] == "tool_use"]
        text = "\n".join(block["text"] for block in content if block["type"] == "text")
        if tools:
            shown.append({"role": "tools", "tools": tools})
        if text:
            shown.append({"role": "assistant", "text": text})
    return shown


@router.get("/chat")
def chat_page(request: Request, db: Db, session: int | None = None) -> HTMLResponse:
    niche = _niche_or_404(db)
    row = db.get(AgentSession, session) if session else None
    messages = _chat_messages(row)
    for message in messages:
        if message["role"] == "assistant":
            message["unverified"] = check_citations(db, message["text"])[1]
    sessions = db.scalars(
        select(AgentSession)
        .where(AgentSession.niche_id == niche.id)
        .order_by(AgentSession.id.desc())
        .limit(15)
    ).all()
    return render(
        request,
        "chat.html",
        session_id=row.id if row else None,
        messages=messages,
        sessions=sessions,
    )


@router.post("/chat")
def chat_send(
    request: Request,
    db: Db,
    message: Annotated[str, Form(min_length=1)],
    session_id: Annotated[int | None, Form()] = None,
) -> HTMLResponse:
    niche = _niche_or_404(db)
    row = db.get(AgentSession, session_id) if session_id else start_session(db, niche.id)
    if row is None:
        raise HTTPException(404, "Session inconnue.")
    target = row.id

    def work(task: Task) -> None:
        def show(step: Step) -> None:
            if step.kind == "tool_start":
                task.steps.append(f"{step.name}")
            elif step.kind == "text":
                task.steps.append(step.detail)

        with get_session_factory()() as session:
            agent_session = session.get(AgentSession, target)
            chat(session, agent_session, message, claude_client(), show)
        task.redirect = f"/ui/chat?session={target}"

    task = runner.start("L'agent travaille", work, key=f"chat:{target}")
    return render(request, "_task.html", task=task, sent=message)


@router.get("/tasks/{task_id}")
def task_status(request: Request, task_id: str) -> Response:
    task = runner.get(task_id)
    if task is None:
        return HTMLResponse('<p class="text-muted">Tâche introuvable : le serveur a redémarré.</p>')
    if task.status == "done" and task.redirect:
        return Response(headers={"HX-Redirect": task.redirect})
    return render(request, "_task.html", task=task)


@router.get("/notifications")
def notifications(request: Request, db: Db) -> HTMLResponse:
    rows = db.scalars(select(Notification).order_by(Notification.id.desc()).limit(50)).all()
    return render(request, "notifications.html", rows=rows)


@router.post("/notifications/{notification_id}/read")
def notification_read(notification_id: int, db: Db) -> RedirectResponse:
    row = db.get(Notification, notification_id)
    if row is None:
        raise HTTPException(404, "Notification inconnue.")
    row.read_at = row.read_at or utcnow()
    db.commit()
    return RedirectResponse("/ui/notifications", status_code=303)
