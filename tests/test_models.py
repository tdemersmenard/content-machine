from datetime import UTC, datetime, timedelta, timezone

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, StatementError
from sqlalchemy.orm import Session

from app.db.models import (
    Lesson,
    LessonScope,
    Niche,
    Pattern,
    Script,
    ScriptStatus,
    Shot,
    Transcript,
    Video,
    VideoPattern,
)


def test_video_defaults(db: Session) -> None:
    video = Video(source_url="https://example.com/1", platform="tiktok")
    db.add(video)
    db.commit()

    assert video.is_own_content is False
    assert video.created_at.tzinfo is not None


def test_pattern_starts_with_neutral_prior(db: Session) -> None:
    pattern = Pattern(type="hook", name="Question choc")
    db.add(pattern)
    db.commit()

    assert (pattern.score_alpha, pattern.score_beta, pattern.times_used) == (1.0, 1.0, 0)
    assert pattern.examples == []


def test_datetimes_are_stored_in_utc(db: Session) -> None:
    paris = timezone(timedelta(hours=2))
    video = Video(posted_at=datetime(2026, 6, 1, 12, 0, tzinfo=paris))
    db.add(video)
    db.commit()
    db.expire_all()

    assert db.get(Video, video.id).posted_at == datetime(2026, 6, 1, 10, 0, tzinfo=UTC)


def test_naive_datetime_is_rejected(db: Session) -> None:
    db.add(Video(posted_at=datetime(2026, 6, 1, 12, 0)))

    with pytest.raises(StatementError, match="naïf"):
        db.commit()


def test_foreign_keys_are_enforced(db: Session) -> None:
    db.add(Shot(video_id=999, index=0, start_s=0, end_s=1))

    with pytest.raises(IntegrityError):
        db.commit()


def test_deleting_video_cascades_to_children(db: Session) -> None:
    video = Video()
    pattern = Pattern(type="hook", name="x")
    db.add_all([video, pattern])
    db.flush()
    db.add_all(
        [
            Transcript(video_id=video.id, full_text="bonjour"),
            Shot(video_id=video.id, index=0, start_s=0, end_s=1),
            VideoPattern(video_id=video.id, pattern_id=pattern.id, role="hook"),
        ]
    )
    db.commit()

    db.execute(text("DELETE FROM videos WHERE id = :id"), {"id": video.id})
    db.commit()

    for table in ("transcripts", "shots", "video_patterns"):
        assert db.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one() == 0


def test_script_status_defaults_to_draft_and_rejects_unknown_values(db: Session) -> None:
    niche = Niche(name="fitness")
    db.add(niche)
    db.flush()
    script = Script(niche_id=niche.id, prompt_version="generate/v1", content={"hook": "x"})
    db.add(script)
    db.commit()
    assert script.status is ScriptStatus.DRAFT

    with pytest.raises(IntegrityError):
        db.execute(text("UPDATE scripts SET status = 'archived'"))


def test_niche_lesson_requires_niche_id(db: Session) -> None:
    db.add(Lesson(scope=LessonScope.GLOBAL, statement="Accroche en moins de 2 s"))
    db.commit()

    db.add(Lesson(scope=LessonScope.NICHE, statement="Sans niche"))
    with pytest.raises(IntegrityError):
        db.commit()
