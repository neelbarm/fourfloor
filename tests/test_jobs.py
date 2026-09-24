"""The job queue's end-of-job handoff, which a follower must never miss."""

from __future__ import annotations

import threading
import time

from fourfloor import jobs, store


def test_a_follower_never_stops_without_the_last_event(monkeypatch) -> None:
    """A finished state used to land before its ``error`` event did.

    A follower whose one-second wait ran out in that gap saw a finished job
    with nothing to read and stopped; the page got ``end`` and never heard the
    remix had failed. ``format_exc`` is slowed here to hold the gap open.
    """
    real = jobs.traceback.format_exc

    def slow_format_exc(*a, **kw):
        time.sleep(1.6)
        return real(*a, **kw)

    monkeypatch.setattr(jobs.traceback, "format_exc", slow_format_exc)
    q = jobs.JobQueue()
    try:
        started = threading.Event()

        def fn(job):
            started.set()
            raise ValueError("the render fell over")

        job = q.submit(store.new_id(), fn)
        started.wait(5)
        seen = [e for e in job.follow(0, timeout=10) if e]
        assert seen[-1]["type"] == "error", [e["type"] for e in seen]
        assert "fell over" in seen[-1]["message"]
    finally:
        q.close()


def test_finish_sets_the_state_and_the_event_together() -> None:
    job = jobs.Job(id="x")
    job.started = time.time()
    event = job.finish(jobs.DONE, "done", result={"ok": 1})
    assert job.state == jobs.DONE and job.finished > 0
    assert job.events[-1] is event and event["result"] == {"ok": 1}
    assert [e["type"] for e in job.follow(0, timeout=1) if e] == ["done"]
