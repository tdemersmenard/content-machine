"""Métadonnées et ratios d'une vidéo, partagés par l'analyse et l'extraction de patterns."""

from typing import Any

from app.analysis.schemas import PerformanceMetrics
from app.db.models import Video


def _ratio(numerator: int | None, denominator: int | None) -> float | None:
    if numerator is None or not denominator:
        return None
    return numerator / denominator


def compute_metrics(video: Video) -> PerformanceMetrics:
    return PerformanceMetrics(
        views_per_follower=_ratio(video.views, video.creator_followers),
        share_rate=_ratio(video.shares, video.views),
        save_rate=_ratio(video.saves, video.views),
        comment_rate=_ratio(video.comments, video.views),
        like_rate=_ratio(video.likes, video.views),
    )


def build_metadata(video: Video, metrics: PerformanceMetrics) -> dict[str, Any]:
    return {
        "platform": video.platform,
        "creator": video.creator,
        "creator_followers": video.creator_followers,
        "posted_at": video.posted_at.isoformat() if video.posted_at else None,
        "description": video.description,
        "hashtags": video.hashtags,
        "sound": video.sound,
        "views": video.views,
        "likes": video.likes,
        "comments": video.comments,
        "shares": video.shares,
        "saves": video.saves,
        **metrics.model_dump(),
    }
