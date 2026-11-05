"""Routes d'analyse."""

from datetime import datetime
from typing import Annotated

import anthropic
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.analysis.report import render_report
from app.analysis.schemas import AnalysisDocument
from app.analysis.service import AnalysisError, analyze
from app.db.models import Analysis
from app.db.session import get_db
from app.llm.client import ClaudeClient, get_claude
from app.llm.errors import ClaudeCredentialsError, ClaudeError

router = APIRouter(tags=["analyses"])


class AnalysisResponse(BaseModel):
    id: int
    video_id: int
    prompt_version: str
    model: str
    created_at: datetime
    patterns_extracted: bool
    result: AnalysisDocument
    report_markdown: str


def _response(row: Analysis) -> AnalysisResponse:
    document = AnalysisDocument.model_validate(row.json_result)
    return AnalysisResponse(
        id=row.id,
        video_id=row.video_id,
        prompt_version=row.prompt_version,
        model=row.model,
        created_at=row.created_at,
        patterns_extracted=row.patterns_extracted_at is not None,
        result=document,
        report_markdown=render_report(document),
    )


@router.post(
    "/videos/{video_id}/analyze",
    response_model=AnalysisResponse,
    status_code=status.HTTP_201_CREATED,
)
def analyze_video(
    video_id: int,
    db: Annotated[Session, Depends(get_db)],
    claude: Annotated[ClaudeClient, Depends(get_claude)],
    force: bool = False,
) -> AnalysisResponse:
    """Synchrone : deux appels à Claude, compter quelques minutes."""
    try:
        return _response(analyze(db, video_id, claude, force=force))
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except AnalysisError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ClaudeCredentialsError as exc:
        raise HTTPException(503, str(exc)) from exc
    except (ClaudeError, anthropic.APIError) as exc:
        raise HTTPException(502, f"Échec de l'appel à Claude : {exc}") from exc


@router.get("/analyses/{analysis_id}", response_model=AnalysisResponse)
def get_analysis(analysis_id: int, db: Annotated[Session, Depends(get_db)]) -> AnalysisResponse:
    row = db.get(Analysis, analysis_id)
    if row is None:
        raise HTTPException(404, "Analyse inconnue.")
    return _response(row)
