"""A one-at-a-time job queue with a replayable event stream.

A remix pegs a core and holds the whole source spectrogram in RAM, so running
two at once on a laptop is worse than running them in order: the queue has a
single worker and everything else waits, with its place in the line reported.

Every job keeps its events in a list rather than pushing them at whoever is
listening. A browser that reloads mid-remix, or connects a moment after
starting the job, replays from event zero and sees the phases it missed --
which is also what makes the stream testable without racing it.
"""

from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"

MAX_JOBS = 64


@dataclass
class Job:
    """One unit of queued work and everything said about it."""

    id: str
    label: str = ""
    state: str = QUEUED
    created: float = field(default_factory=time.time)
    started: float = 0.0
    finished: float = 0.0
    result: Any = None
    error: str = ""
    events: list[dict] = field(default_factory=list)
    _cv: threading.Condition = field(default_factory=threading.Condition, repr=False)

    def emit(self, type: str, **data: Any) -> dict:
        """Append an event and wake every follower."""
        event = {"type": type, "at": round(time.time() - (self.started or self.created), 3),
                 **data}
        with self._cv:
            self.events.append(event)
            self._cv.notify_all()
        return event

    @property
    def is_finished(self) -> bool:
        return self.state in (DONE, FAILED)

    def follow(self, start: int = 0, timeout: float = 300.0) -> Iterator[dict]:
        """Yield events from ``start``, blocking until the job finishes.

        Yields ``None`` when nothing has happened for a second so the caller
        can send an SSE keep-alive and notice a browser that walked away.
        """
        i = start
        deadline = time.time() + timeout
        while True:
            with self._cv:
                while i >= len(self.events) and not self.is_finished:
                    if time.time() > deadline:
                        return
                    if not self._cv.wait(1.0):
                        break               # nothing happened; let the caller ping
                pending = self.events[i:]
                i += len(pending)
                finished = self.is_finished
            if pending:
                yield from pending
                continue
            if finished:
                return
            yield None                      # keep-alive tick

    def to_dict(self) -> dict:
        return {
            "id": self.id, "label": self.label, "state": self.state,
            "created": self.created, "error": self.error,
            "elapsed": round((self.finished or time.time()) - (self.started or self.created), 2),
        }


class JobQueue:
    """A single background worker draining a FIFO of callables."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._pending: list[tuple[Job, Callable[[Job], Any]]] = []
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="fourfloor-worker",
                                        daemon=True)
        self._thread.start()

    # -- submission -------------------------------------------------------

    def submit(self, job_id: str, fn: Callable[[Job], Any], label: str = "") -> Job:
        """Queue ``fn``; it is called with its :class:`Job` and returns a result."""
        job = Job(id=job_id, label=label)
        with self._wake:
            self._jobs[job_id] = job
            self._order.append(job_id)
            self._pending.append((job, fn))
            place = len(self._pending) - 1
            self._prune()
            self._wake.notify()
        job.emit("queued", position=place)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def depth(self) -> int:
        with self._lock:
            return len(self._pending)

    def close(self, timeout: float = 5.0) -> None:
        with self._wake:
            self._stop = True
            self._wake.notify_all()
        self._thread.join(timeout)

    # -- worker -----------------------------------------------------------

    def _prune(self) -> None:
        while len(self._order) > MAX_JOBS:
            old = self._order.pop(0)
            job = self._jobs.get(old)
            if job is not None and job.is_finished:
                self._jobs.pop(old, None)
            else:                                   # still live: keep it, try later
                self._order.append(old)
                return

    def _run(self) -> None:
        while True:
            with self._wake:
                while not self._pending and not self._stop:
                    self._wake.wait(0.5)
                if self._stop and not self._pending:
                    return
                job, fn = self._pending.pop(0)
                waiting = list(self._pending)
            for i, (other, _) in enumerate(waiting):
                other.emit("queued", position=i + 1)

            job.state = RUNNING
            job.started = time.time()
            job.emit("start")
            try:
                result = fn(job)
            except Exception as exc:                # noqa: BLE001 - reported to the browser
                job.error = str(exc) or exc.__class__.__name__
                job.state = FAILED
                job.finished = time.time()
                job.emit("error", message=job.error,
                         detail=traceback.format_exc(limit=3).strip().splitlines()[-1])
            else:
                job.result = result
                job.state = DONE
                job.finished = time.time()
                job.emit("done", result=result,
                         elapsed=round(job.finished - job.started, 2))


class PhaseTimer:
    """Turn ``remix``'s ``progress(name, detail)`` calls into job events.

    The pipeline reports the phase it is *entering*, so the elapsed time of a
    phase is only known when the next one starts -- or when :meth:`close` runs
    at the end. Both emit a ``phase_done``.
    """

    def __init__(self, job: Job) -> None:
        self.job = job
        self.current: str | None = None
        self._t0 = time.time()

    def __call__(self, name: str, detail: str = "") -> None:
        self.close()
        self.current = name
        self._t0 = time.time()
        self.job.emit("phase", name=name, detail=detail)

    def close(self) -> None:
        if self.current is not None:
            self.job.emit("phase_done", name=self.current,
                          elapsed=round(time.time() - self._t0, 2))
            self.current = None
