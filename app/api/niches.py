"""Routes des niches."""

from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.errors import claude_errors
from app.db.models import InterviewStatus, Niche, NicheInterview
from app.db.session import get_db
from app.llm.client import ClaudeClient, get_claude
from app.niche.interview import (
    InterviewError,
    answer_interview,
    current_question,
    finish_interview,
    start_interview,
)
from app.niche.profile import update_niche, versions
from app.niche.schemas import NicheProfile, VoiceProfile
from app.niche.topics import refresh_topics, set_topic_status
from app.niche.voice import refresh_voice

router = APIRouter(prefix="/niches", tags=["niches"])

Db = Annotated[Session, Depends(get_db)]
Claude = Annotated[ClaudeClient, Depends(get_claude)]


class InterviewStart(BaseModel):
    name: str | None = None


class InterviewAnswer(BaseModel):
    answer: str


class InterviewOut(BaseModel):
    id: int
    status: InterviewStatus
    question: str | None
    covered: list[str]
    niche_id: int | None


class NicheSummary(BaseModel):
    id: int
    name: str
    version: int
    topic_count: int


class NicheOut(NicheSummary):
    profile: NicheProfile | None
    avatar_markdown: str | None
    topics: list[dict[str, Any]]


class NichePatch(BaseModel):
    name: str | None = None
    profile: dict[str, Any] | None = None
    change_note: str = ""


class VersionOut(BaseModel):
    version: int
    name: str
    change_note: str
    created_at: datetime
    profile: dict[str, Any] | None
    topics: list[dict[str, Any]]


class TopicStatusPatch(BaseModel):
    status: Literal["new", "used", "retired"]


def _interview_out(interview: NicheInterview) -> InterviewOut:
    return InterviewOut(
        id=interview.id,
        status=interview.status,
        question=current_question(interview),
        covered=interview.covered,
        niche_id=interview.niche_id,
    )


def _summary_fields(niche: Niche) -> dict[str, Any]:
    active = sum(1 for item in niche.topics if item["status"] == "new")
    return {"id": niche.id, "name": niche.name, "version": niche.version, "topic_count": active}


def _niche_out(niche: Niche) -> NicheOut:
    return NicheOut(
        **_summary_fields(niche),
        profile=niche.profile,
        avatar_markdown=niche.avatar_markdown,
        topics=niche.topics,
    )


def _get_niche(db: Session, niche_id: int) -> Niche:
    niche = db.get(Niche, niche_id)
    if niche is None:
        raise HTTPException(404, "Niche inconnue.")
    return niche


def _with_profile(db: Session, niche_id: int) -> Niche:
    niche = _get_niche(db, niche_id)
    if niche.profile is None:
        raise HTTPException(409, "Cette niche n'a pas de profil.")
    return niche


def _get_interview(db: Session, interview_id: int) -> NicheInterview:
    interview = db.get(NicheInterview, interview_id)
    if interview is None:
        raise HTTPException(404, "Entrevue inconnue.")
    return interview


@router.post("/interviews", response_model=InterviewOut, status_code=status.HTTP_201_CREATED)
def create_interview(body: InterviewStart, db: Db, claude: Claude) -> InterviewOut:
    with claude_errors():
        return _interview_out(start_interview(db, claude, body.name))


@router.get("/interviews/{interview_id}", response_model=InterviewOut)
def get_interview(interview_id: int, db: Db) -> InterviewOut:
    return _interview_out(_get_interview(db, interview_id))


@router.post("/interviews/{interview_id}/answers", response_model=InterviewOut)
def post_answer(interview_id: int, body: InterviewAnswer, db: Db, claude: Claude) -> InterviewOut:
    interview = _get_interview(db, interview_id)
    with claude_errors():
        try:
            return _interview_out(answer_interview(db, interview, body.answer, claude))
        except InterviewError as exc:
            raise HTTPException(409, str(exc)) from exc


@router.post("/interviews/{interview_id}/finish", response_model=InterviewOut)
def post_finish(interview_id: int, db: Db, claude: Claude) -> InterviewOut:
    """Termine l'entrevue sans attendre que toutes les sections soient couvertes."""
    interview = _get_interview(db, interview_id)
    with claude_errors():
        try:
            return _interview_out(finish_interview(db, interview, claude))
        except InterviewError as exc:
            raise HTTPException(409, str(exc)) from exc


@router.get("", response_model=list[NicheSummary])
def list_niches(db: Db) -> list[NicheSummary]:
    niches = db.scalars(select(Niche).order_by(Niche.id))
    return [NicheSummary(**_summary_fields(niche)) for niche in niches]


@router.get("/{niche_id}", response_model=NicheOut)
def get_niche(niche_id: int, db: Db) -> NicheOut:
    return _niche_out(_get_niche(db, niche_id))


@router.patch("/{niche_id}", response_model=NicheOut)
def patch_niche(niche_id: int, body: NichePatch, db: Db) -> NicheOut:
    """Édition partielle : les objets sont fusionnés, une liste fournie remplace l'ancienne."""
    niche = _with_profile(db, niche_id)
    try:
        return _niche_out(update_niche(db, niche, body.profile, body.name, body.change_note))
    except ValidationError as exc:
        db.rollback()
        raise HTTPException(422, exc.errors(include_url=False, include_context=False)) from exc
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(409, "Ce nom de niche existe déjà.") from exc


@router.get("/{niche_id}/versions", response_model=list[VersionOut])
def list_versions(niche_id: int, db: Db) -> list[VersionOut]:
    _get_niche(db, niche_id)
    return [
        VersionOut(
            version=item.version,
            name=item.name,
            change_note=item.change_note,
            created_at=item.created_at,
            profile=item.profile,
            topics=item.topics,
        )
        for item in versions(db, niche_id)
    ]


@router.post("/{niche_id}/topics", response_model=NicheOut)
def post_topics(niche_id: int, db: Db, claude: Claude) -> NicheOut:
    """Complète la banque jusqu'à 50 sujets actifs."""
    niche = _with_profile(db, niche_id)
    with claude_errors():
        refresh_topics(db, niche, claude)
    return _niche_out(niche)


@router.patch("/{niche_id}/topics/{topic_id}")
def patch_topic(niche_id: int, topic_id: str, body: TopicStatusPatch, db: Db) -> dict[str, Any]:
    niche = _get_niche(db, niche_id)
    try:
        return set_topic_status(db, niche, topic_id, body.status)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/{niche_id}/voice", response_model=VoiceProfile | None)
def post_voice(niche_id: int, db: Db, claude: Claude, force: bool = False) -> VoiceProfile | None:
    """Recalcule la voix depuis mes vidéos. Renvoie null si je n'en ai aucune avec de la parole."""
    niche = _with_profile(db, niche_id)
    with claude_errors():
        return refresh_voice(db, niche, claude, force)
