"""Routes de génération de scripts. Synchrones : compter plusieurs minutes."""

from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.errors import claude_errors
from app.db.models import Script, ScriptStatus
from app.db.session import get_db
from app.generate.report import render_script
from app.generate.schemas import ScriptContent
from app.generate.service import (
    GenerationError,
    GenerationResult,
    adapt,
    generate,
    load_content,
    update_script,
)
from app.llm.client import ClaudeClient, get_claude

router = APIRouter(prefix="/scripts", tags=["scripts"])

Db = Annotated[Session, Depends(get_db)]
Claude = Annotated[ClaudeClient, Depends(get_claude)]


class AdaptRequest(BaseModel):
    video_id: int
    niche_id: int
    n_variantes: int = Field(default=3, ge=1, le=5)


class GenerateRequest(BaseModel):
    niche_id: int
    objectif: str | None = None
    n: int = Field(default=5, ge=1, le=20)
    develop: int = Field(default=3, ge=1, le=10)


class ScriptPatch(BaseModel):
    status: ScriptStatus | None = None
    content: dict[str, Any] | None = None
    published_video_id: int | None = None


class ScriptOut(BaseModel):
    id: int
    niche_id: int
    source_video_id: int | None
    pattern_ids: list[int]
    prompt_version: str
    status: ScriptStatus
    published_video_id: int | None
    predicted_score: float | None
    created_at: datetime
    content: ScriptContent
    markdown: str


class GenerationOut(BaseModel):
    scripts: list[ScriptOut]
    failures: list[str]


def _out(script: Script) -> ScriptOut:
    content = load_content(script)
    return ScriptOut(
        id=script.id,
        niche_id=script.niche_id,
        source_video_id=script.source_video_id,
        pattern_ids=script.pattern_ids,
        prompt_version=script.prompt_version,
        status=script.status,
        published_video_id=script.published_video_id,
        predicted_score=script.predicted_score,
        created_at=script.created_at,
        content=content,
        markdown=render_script(content),
    )


def _generation_out(result: GenerationResult) -> GenerationOut:
    return GenerationOut(scripts=[_out(item) for item in result.scripts], failures=result.failures)


def _run(call) -> GenerationOut:
    with claude_errors():
        try:
            return _generation_out(call())
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc
        except GenerationError as exc:
            raise HTTPException(409, str(exc)) from exc


@router.post("/adapt", response_model=GenerationOut, status_code=201)
def post_adapt(body: AdaptRequest, db: Db, claude: Claude) -> GenerationOut:
    return _run(lambda: adapt(db, body.video_id, body.niche_id, claude, body.n_variantes))


@router.post("/generate", response_model=GenerationOut, status_code=201)
def post_generate(body: GenerateRequest, db: Db, claude: Claude) -> GenerationOut:
    return _run(lambda: generate(db, body.niche_id, claude, body.objectif, body.n, body.develop))


@router.get("", response_model=list[ScriptOut])
def list_scripts(
    db: Db,
    niche_id: int | None = None,
    status: ScriptStatus | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[ScriptOut]:
    statement = select(Script).order_by(Script.id.desc()).limit(limit)
    if niche_id is not None:
        statement = statement.where(Script.niche_id == niche_id)
    if status is not None:
        statement = statement.where(Script.status == status)
    return [_out(item) for item in db.scalars(statement)]


@router.patch("/{script_id}", response_model=ScriptOut)
def patch_script(script_id: int, body: ScriptPatch, db: Db) -> ScriptOut:
    script = db.get(Script, script_id)
    if script is None:
        raise HTTPException(404, "Script inconnu.")
    try:
        return _out(update_script(db, script, body.status, body.content, body.published_video_id))
    except ValidationError as exc:
        db.rollback()
        raise HTTPException(422, exc.errors(include_url=False, include_context=False)) from exc
    except LookupError as exc:
        db.rollback()
        raise HTTPException(404, str(exc)) from exc
