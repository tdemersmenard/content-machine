"""Routes de l'agent. `POST /agent/chat` est synchrone, `GET /agent/actions` suit son avancement."""

from dataclasses import asdict
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.budget import budget_status
from app.agent.chat import AgentError, chat, start_session
from app.api.errors import claude_errors
from app.db.base import utcnow
from app.db.models import ActionStatus, AgentAction, AgentSession, Notification, WeeklyReport
from app.db.session import get_db
from app.llm.client import ClaudeClient, get_claude

router = APIRouter(prefix="/agent", tags=["agent"])

Db = Annotated[Session, Depends(get_db)]
Claude = Annotated[ClaudeClient, Depends(get_claude)]


class ChatRequest(BaseModel):
    niche_id: int
    message: str = Field(min_length=1)
    session_id: int | None = None


class StepOut(BaseModel):
    kind: str
    name: str
    detail: str
    is_error: bool


class ChatOut(BaseModel):
    session_id: int
    answer: str
    steps: list[StepOut]
    evidence: dict[str, list[int]]
    unverified: list[str]
    interrupted: str | None
    tokens: int
    cost_usd: float


class ActionOut(BaseModel):
    id: int
    session_id: int | None
    source: str
    action: str
    input: dict[str, Any]
    status: ActionStatus
    summary: str | None
    created_at: datetime
    finished_at: datetime | None


class NotificationOut(BaseModel):
    id: int
    niche_id: int | None
    kind: str
    title: str
    body: str
    script_id: int | None
    horizon_h: int | None
    read_at: datetime | None
    created_at: datetime


class WeeklyReportOut(BaseModel):
    id: int
    niche_id: int
    week_start: str
    learnings: dict[str, Any] | None
    markdown: str
    script_ids: list[int]
    plan_error: str | None
    created_at: datetime


def _session_for(db: Session, body: ChatRequest) -> AgentSession:
    if body.session_id is None:
        try:
            return start_session(db, body.niche_id)
        except AgentError as exc:
            raise HTTPException(404, str(exc)) from exc
    row = db.get(AgentSession, body.session_id)
    if row is None or row.niche_id != body.niche_id:
        raise HTTPException(404, "Session inconnue pour cette niche.")
    return row


@router.post("/chat", response_model=ChatOut)
def post_chat(body: ChatRequest, db: Db, claude: Claude) -> ChatOut:
    with claude_errors():
        result = chat(db, _session_for(db, body), body.message, claude)
    return ChatOut(**asdict(result))


@router.get("/sessions/{session_id}")
def get_session(session_id: int, db: Db) -> dict[str, Any]:
    row = db.get(AgentSession, session_id)
    if row is None:
        raise HTTPException(404, "Session inconnue.")
    return {"id": row.id, "niche_id": row.niche_id, "messages": row.messages}


@router.get("/actions", response_model=list[ActionOut])
def list_actions(
    db: Db,
    session_id: int | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[AgentAction]:
    statement = select(AgentAction).order_by(AgentAction.id.desc()).limit(limit)
    if session_id is not None:
        statement = statement.where(AgentAction.session_id == session_id)
    return list(db.scalars(statement))


@router.get("/budget")
def get_budget(db: Db) -> dict[str, Any]:
    return budget_status(db)


@router.get("/notifications", response_model=list[NotificationOut])
def list_notifications(
    db: Db,
    unread: bool = True,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[Notification]:
    statement = select(Notification).order_by(Notification.id.desc()).limit(limit)
    if unread:
        statement = statement.where(Notification.read_at.is_(None))
    return list(db.scalars(statement))


@router.post("/notifications/{notification_id}/read", response_model=NotificationOut)
def read_notification(notification_id: int, db: Db) -> Notification:
    row = db.get(Notification, notification_id)
    if row is None:
        raise HTTPException(404, "Notification inconnue.")
    row.read_at = row.read_at or utcnow()
    db.commit()
    return row


@router.get("/reports", response_model=list[WeeklyReportOut])
def list_reports(
    db: Db,
    niche_id: int | None = None,
    limit: Annotated[int, Query(ge=1, le=52)] = 10,
) -> list[WeeklyReport]:
    statement = select(WeeklyReport).order_by(WeeklyReport.week_start.desc()).limit(limit)
    if niche_id is not None:
        statement = statement.where(WeeklyReport.niche_id == niche_id)
    return list(db.scalars(statement))
