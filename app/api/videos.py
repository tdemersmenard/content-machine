"""Routes d'ingestion."""

import shutil
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import JobStatus, SourceType, Video
from app.db.session import get_db
from app.ingest.jobs import canonical, is_url, latest_job, load_bundle, submit
from app.ingest.schemas import VideoBundle

router = APIRouter(prefix="/videos", tags=["videos"])


class SubmissionResponse(BaseModel):
    video_id: int
    job_id: int | None
    status: JobStatus
    deduplicated: bool


class VideoStatusResponse(BaseModel):
    id: int
    status: JobStatus | None
    stage: str | None
    error: str | None
    bundle: VideoBundle | None


def _save_upload(upload: UploadFile) -> Path:
    folder = get_settings().data_dir.resolve() / "uploads"
    folder.mkdir(parents=True, exist_ok=True)
    # Le nom vient du client : seule l'extension est conservée.
    path = folder / f"{uuid.uuid4().hex}{Path(upload.filename or '').suffix.lower()}"
    with path.open("wb") as output:
        shutil.copyfileobj(upload.file, output)
    return path


@router.post("", response_model=SubmissionResponse, status_code=status.HTTP_202_ACCEPTED)
def create_video(
    response: Response,
    db: Annotated[Session, Depends(get_db)],
    file: Annotated[UploadFile | None, File()] = None,
    url: Annotated[str | None, Form()] = None,
    is_own_content: Annotated[bool, Form()] = False,
    niche_source: Annotated[str | None, Form()] = None,
) -> SubmissionResponse:
    if (file is None) == (url is None):
        raise HTTPException(422, "Fournir soit `file`, soit `url`.")
    if url is not None and not is_url(url):
        raise HTTPException(422, "`url` doit commencer par http:// ou https://.")

    source = url if url is not None else str(_save_upload(file))
    submission = submit(
        db,
        source,
        source_type=SourceType.URL if url is not None else SourceType.UPLOAD,
        is_own_content=is_own_content,
        niche_source=niche_source,
    )

    if submission.deduplicated:
        response.status_code = status.HTTP_200_OK
        job_status = JobStatus.DONE
    else:
        job_status = latest_job(db, submission.video_id).status
    return SubmissionResponse(
        video_id=submission.video_id,
        job_id=submission.job_id,
        status=job_status,
        deduplicated=submission.deduplicated,
    )


@router.get("/{video_id}", response_model=VideoStatusResponse)
def get_video(video_id: int, db: Annotated[Session, Depends(get_db)]) -> VideoStatusResponse:
    video = db.get(Video, video_id)
    if video is None:
        raise HTTPException(404, "Vidéo inconnue.")

    video = canonical(db, video)
    job = latest_job(db, video.id)
    return VideoStatusResponse(
        id=video.id,
        status=job.status if job else None,
        stage=job.stage if job else None,
        error=job.error if job else None,
        bundle=load_bundle(db, video) if video.ingested_at else None,
    )
