"""Tâches longues du dashboard : analyse, scripts, chat.

En mémoire et dans le process de l'API. Un redémarrage perd l'affichage de l'avancement, pas le
travail : ce qui est terminé est en base, et chaque étape payante sait reprendre.
"""

import logging
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

MAX_KEPT_TASKS = 200


@dataclass
class Task:
    id: str
    label: str
    key: str = ""  # Par exemple « video:12 », pour afficher ce qui est en cours.
    status: str = "running"  # running, done ou failed.
    steps: list[str] = field(default_factory=list)
    result: Any = None
    error: str | None = None
    redirect: str | None = None  # Où aller une fois la tâche finie.


class TaskRunner:
    def __init__(self) -> None:
        self._pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="web-task")
        self._tasks: dict[str, Task] = {}
        self._lock = threading.Lock()
        self.inline = False  # Les tests exécutent les tâches sur place.

    def start(self, label: str, work: Callable[[Task], Any], key: str = "") -> Task:
        task = Task(id=uuid.uuid4().hex, label=label, key=key)
        with self._lock:
            self._tasks[task.id] = task
            for old in list(self._tasks)[:-MAX_KEPT_TASKS]:
                del self._tasks[old]
        if self.inline:
            self._run(task, work)
        else:
            self._pool.submit(self._run, task, work)
        return task

    @staticmethod
    def _run(task: Task, work: Callable[[Task], Any]) -> None:
        try:
            task.result = work(task)
            task.status = "done"
        except Exception as exc:
            logger.exception("Tâche en échec : %s", task.label)
            task.error = str(exc)
            task.status = "failed"

    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def running_key(self, key: str) -> Task | None:
        return next(
            (
                item
                for item in reversed(list(self._tasks.values()))
                if item.key == key and item.status == "running"
            ),
            None,
        )

    def last_failure(self, key: str) -> Task | None:
        latest = next(
            (item for item in reversed(list(self._tasks.values())) if item.key == key), None
        )
        return latest if latest is not None and latest.status == "failed" else None


runner = TaskRunner()
