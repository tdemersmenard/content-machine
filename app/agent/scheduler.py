"""Planification dans le process de l'API.

Un seul process doit la porter : avec plusieurs workers uvicorn, chacun lancerait ses routines.
La base évite de repayer un rapport, mais pas de lancer deux analyses en parallèle.
"""

import logging
from collections.abc import Callable
from datetime import timedelta
from functools import wraps

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger

from app.agent.budget import local_now, local_zone
from app.agent.routines import Inbox, run_daily, run_reminders, run_weekly
from app.config import get_settings

logger = logging.getLogger(__name__)

CATCH_UP_DELAY = timedelta(seconds=30)


def _guarded(job: Callable[[], object]) -> Callable[[], None]:
    @wraps(job)
    def run() -> None:
        try:
            job()
        except Exception:
            logger.exception("Routine en échec : %s", job.__name__)

    return run


def _catch_up() -> None:
    """Ce qui aurait dû tourner pendant que l'ordinateur était éteint."""
    run_reminders()
    run_daily()
    run_weekly()


def build_scheduler() -> BackgroundScheduler:
    settings = get_settings()
    zone = local_zone()
    scheduler = BackgroundScheduler(
        timezone=zone,
        job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": 3600},
    )
    inbox = Inbox()

    scheduler.add_job(
        _guarded(run_daily), CronTrigger(hour=settings.agent_daily_hour, timezone=zone), id="daily"
    )
    scheduler.add_job(
        _guarded(run_weekly),
        CronTrigger(day_of_week="mon", hour=settings.agent_weekly_hour, timezone=zone),
        id="weekly",
    )
    scheduler.add_job(_guarded(run_reminders), IntervalTrigger(hours=1), id="reminders")
    scheduler.add_job(
        _guarded(inbox.scan),
        IntervalTrigger(seconds=settings.agent_inbox_interval_s),
        id="inbox",
    )
    scheduler.add_job(
        _guarded(_catch_up),
        DateTrigger(run_date=local_now() + CATCH_UP_DELAY, timezone=zone),
        id="catch_up",
    )
    return scheduler
