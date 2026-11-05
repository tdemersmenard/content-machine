"""Routes de l'apprentissage."""

from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.models import Niche, Script
from app.db.session import get_db
from app.learning.imports import ImportError_, ImportResult, import_csv
from app.learning.report import build_report
from app.learning.schemas import PerformanceInput
from app.learning.service import record_performance
from app.llm.client import ClaudeClient, get_claude

router = APIRouter(tags=["learning"])

FORM_PATH = Path(__file__).resolve().parents[2] / "web" / "performance.html"

Db = Annotated[Session, Depends(get_db)]
Claude = Annotated[ClaudeClient, Depends(get_claude)]


class PerformanceOut(BaseModel):
    id: int
    script_id: int
    video_id: int
    horizon_h: int
    composite: float | None
    label: str | None
    indexes: dict[str, float] | None
    baseline_size: int | None
    post_mortem: dict[str, Any] | None
    lessons_updated: bool
    warnings: list[str]


class ImportOut(BaseModel):
    imported: int
    skipped: list[str]


@router.post("/scripts/{script_id}/performance", response_model=PerformanceOut, status_code=201)
def post_performance(
    script_id: int, body: PerformanceInput, db: Db, claude: Claude
) -> PerformanceOut:
    """Les stats sont enregistrées même si le post-mortem échoue : voir `warnings`."""
    script = db.get(Script, script_id)
    if script is None:
        raise HTTPException(404, "Script inconnu.")
    result = record_performance(db, script, body, claude)
    row = result.performance
    return PerformanceOut(
        id=row.id,
        script_id=script_id,
        video_id=row.video_id,
        horizon_h=row.horizon_h,
        composite=row.composite,
        label=row.label,
        indexes=row.indexes,
        baseline_size=row.baseline_size,
        post_mortem=result.post_mortem.content if result.post_mortem else None,
        lessons_updated=result.lessons_updated,
        warnings=result.warnings,
    )


@router.post("/performance/import", response_model=ImportOut)
def post_import(
    db: Db,
    file: Annotated[UploadFile, File()],
    horizon_h: Annotated[int, Form()],
    preset: Annotated[str | None, Form()] = None,
) -> ImportResult:
    try:
        return import_csv(db, file.file.read().decode("utf-8-sig"), horizon_h, preset)
    except ImportError_ as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/learning/report")
def get_report(db: Db, niche_id: int | None = None) -> dict[str, Any]:
    if niche_id is not None and db.get(Niche, niche_id) is None:
        raise HTTPException(404, "Niche inconnue.")
    return build_report(db, niche_id)


@router.get("/forms/performance", include_in_schema=False)
def performance_form() -> FileResponse:
    return FileResponse(FORM_PATH)
