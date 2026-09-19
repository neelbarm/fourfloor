"""Listening notes, written down next to the render.

fourfloor can measure a remix -- tempo, key, peak, bar count -- but it cannot
hear one. One person on this project can, and his time is the scarce resource,
so everything he says while listening is captured where an engineer will find
it: ``~/.fourfloor/remixes/<id>/feedback.json``, beside the mp3 it is about.

The file is **append-only**. A marker, a rating and an A/B vote are all events
with a timestamp; nothing is ever edited in place, so re-rating a remix after a
fix leaves the first opinion visible and the pair reads as a before/after. The
current rating is simply the last one.

Every marker is also copied into the remix's ``*.session.json`` under a
``feedback`` array, so the DJ handoff file carries the notes with it -- a
session file is the thing that gets moved to another machine, and a note about
bar 41 is worth nothing if it stays behind.

``python -m fourfloor.feedback`` prints the digest: per remix, the markers
grouped by category, each with its bar number and the plan slot it lands in.
That is the block an engineer agent reads before touching the arranger.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

SCHEMA = 1

#: The file, inside the remix's own folder.
FEEDBACK_FILE = "feedback.json"

#: The quick chips, in the order the app shows them. A category is a closed
#: set so the digest can group by it; the free-text note carries everything
#: else. ``good`` is here on purpose -- "this bit is right, don't touch it" is
#: as useful to an engineer as a complaint.
CATEGORIES: dict[str, str] = {
    "off-beat": "Off-beat",
    "vocal-buried": "Vocal buried",
    "drums-fake": "Drums fake",
    "clash": "Clash / wrong notes",
    "transition": "Rough transition",
    "too-loud": "Too loud",
    "too-quiet": "Too quiet",
    "boring": "Boring / repetitive",
    "good": "Good",
}

#: Which way a vote went.
PREFERENCES = ("a", "b")

MAX_NOTE = 400
MAX_LABEL = 120
MAX_MARKERS = 500

_LOCK = threading.Lock()


class FeedbackError(ValueError):
    """The browser sent something this module will not store."""


# ---------------------------------------------------------------------------
# reading and writing
# ---------------------------------------------------------------------------

def _write_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf8")
    tmp.replace(path)
    return path


def blank(rid: str = "") -> dict:
    return {"schema": SCHEMA, "remix": rid, "markers": [], "ratings": [], "votes": []}


def read(remix_dir: str | Path, rid: str = "") -> dict:
    """The feedback for one remix; an empty document if nothing was said yet."""
    path = Path(remix_dir) / FEEDBACK_FILE
    try:
        data = json.loads(path.read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError):
        return blank(rid)
    if not isinstance(data, dict):
        return blank(rid)
    out = blank(rid or str(data.get("remix") or ""))
    for key in ("markers", "ratings", "votes"):
        rows = data.get(key)
        if isinstance(rows, list):
            out[key] = [r for r in rows if isinstance(r, dict)]
    return out


def latest_rating(data: dict) -> dict:
    """The rating that stands: stars and verdict from the most recent events.

    They are separate fields on separate events -- a verdict typed after a star
    was picked must not erase the star -- so each is taken from the last event
    that carried one.
    """
    stars, verdict, at = None, "", 0.0
    for row in data.get("ratings", []):
        if row.get("stars"):
            stars = int(row["stars"])
            at = max(at, float(row.get("at") or 0.0))
        if row.get("verdict"):
            verdict = str(row["verdict"])
            at = max(at, float(row.get("at") or 0.0))
    return {"stars": stars, "verdict": verdict, "at": at}


# ---------------------------------------------------------------------------
# cleaning what the browser sent
# ---------------------------------------------------------------------------

def clean_text(value, limit: int = MAX_NOTE) -> str:
    """Free text reduced to one safe, short line."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", str(value or ""))
    return re.sub(r"\s+", " ", text).strip()[:limit]


def clean_category(value) -> str:
    code = str(value or "").strip().lower()
    if code not in CATEGORIES:
        raise FeedbackError(
            f"{code or 'that'} is not a feedback category; pick one of "
            + ", ".join(CATEGORIES))
    return code


def clean_time(value, duration: float = 0.0) -> float:
    try:
        t = float(value)
    except (TypeError, ValueError):
        raise FeedbackError("a marker needs a time in seconds") from None
    if t != t or t in (float("inf"), float("-inf")) or t < 0:
        raise FeedbackError("a marker needs a time in seconds")
    if duration > 0:
        t = min(t, duration)
    return round(t, 3)


# ---------------------------------------------------------------------------
# where a marker lands: bar, and the slot the arranger built there
# ---------------------------------------------------------------------------

def bar_duration(session: dict) -> float:
    """Seconds per bar, from the session file's own numbers."""
    beat = float(session.get("beat_duration_sec") or 0.0)
    if beat > 0:
        return beat * float(session.get("beats_per_bar") or 4)
    bpm = float(session.get("bpm") or 0.0)
    if bpm > 0:
        return 60.0 / bpm * float(session.get("beats_per_bar") or 4)
    return 0.0


def bar_of(session: dict, seconds: float) -> int:
    """The 1-based bar number a moment falls in. 0 when the grid is unknown."""
    dur = bar_duration(session)
    if dur <= 0:
        return 0
    return int(seconds // dur) + 1


def slot_of(plan: dict, bar: int) -> dict | None:
    """The arrangement slot covering a 1-based bar number."""
    if bar <= 0:
        return None
    index = bar - 1
    for slot in plan.get("slots", []):
        start = int(slot.get("start_bar", 0))
        if start <= index < start + int(slot.get("bars", 0)):
            return slot
    return None


def slot_label(slot: dict | None) -> str:
    if not slot:
        return ""
    start = int(slot.get("start_bar", 0))
    bars = int(slot.get("bars", 0))
    return f"{slot.get('kind', 'slot')} bars {start + 1}-{start + bars}"


def _read_side_file(remix_dir: Path, name: str) -> dict:
    try:
        data = json.loads((remix_dir / name).read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def session_of(remix_dir: str | Path) -> dict:
    return _read_side_file(Path(remix_dir), "remix.session.json")


def plan_of(remix_dir: str | Path) -> dict:
    return _read_side_file(Path(remix_dir), "remix.plan.json")


# ---------------------------------------------------------------------------
# appending
# ---------------------------------------------------------------------------

def _stamp(row: dict) -> dict:
    row["at"] = round(time.time(), 3)
    return row


def add_marker(remix_dir: str | Path, time_sec, category, note: str = "",
               rid: str = "") -> dict:
    """Drop a marker. Returns the marker as it was stored.

    The bar and the slot are worked out here from the session and plan files
    rather than trusted from the browser: the page is a view of this data, not
    the source of it, and the digest is only useful if the bar numbers are the
    ones the arranger used.
    """
    remix_dir = Path(remix_dir)
    session = session_of(remix_dir)
    plan = plan_of(remix_dir)
    code = clean_category(category)
    t = clean_time(time_sec, float(session.get("duration") or 0.0))
    bar = bar_of(session, t)
    slot = slot_of(plan, bar)
    marker = _stamp({
        "time": t,
        "bar": bar,
        "category": code,
        "note": clean_text(note),
        "slot": (slot or {}).get("kind", ""),
        "slot_bars": slot_label(slot),
    })

    with _LOCK:
        data = read(remix_dir, rid)
        if len(data["markers"]) >= MAX_MARKERS:
            raise FeedbackError(
                f"this remix already has {MAX_MARKERS} markers; that is a rewrite, "
                "not a note")
        data["markers"].append(marker)
        _write_json(remix_dir / FEEDBACK_FILE, data)
        _mirror_into_session(remix_dir, marker)
    return marker


def _mirror_into_session(remix_dir: Path, marker: dict) -> None:
    """Copy a marker into ``*.session.json`` so the DJ handoff carries it.

    Additive only: the session schema is a contract with MixPilot and every
    other reader, so this appends to a ``feedback`` array and touches nothing
    else. A reader that does not know the field ignores it.
    """
    path = remix_dir / "remix.session.json"
    session = _read_side_file(remix_dir, "remix.session.json")
    if not session:
        return
    rows = session.get("feedback")
    if not isinstance(rows, list):
        rows = []
    rows.append({k: marker[k] for k in ("time", "bar", "category", "note", "at")})
    session["feedback"] = rows
    _write_json(path, session)


def add_rating(remix_dir: str | Path, stars=None, verdict=None,
               rid: str = "") -> dict:
    """Append a star rating, a one-line verdict, or both."""
    row: dict = {}
    if stars is not None and stars != "":
        try:
            n = int(stars)
        except (TypeError, ValueError):
            raise FeedbackError("a rating is a whole number of stars") from None
        if not 1 <= n <= 5:
            raise FeedbackError("a rating runs from 1 to 5 stars")
        row["stars"] = n
    if verdict is not None:
        text = clean_text(verdict)
        if text:
            row["verdict"] = text
    if not row:
        raise FeedbackError("a rating needs stars or a verdict")
    _stamp(row)
    with _LOCK:
        data = read(remix_dir, rid)
        data["ratings"].append(row)
        _write_json(Path(remix_dir) / FEEDBACK_FILE, data)
    return row


def add_vote(remix_dir: str | Path, other: str, prefer: str, reason: str = "",
             label: str = "", other_label: str = "", rid: str = "") -> dict:
    """Append an A/B vote. ``prefer`` is ``a`` (this remix) or ``b`` (``other``).

    ``other`` is stored as opaque text -- it is a remix id, or ``source``, or
    ``reference`` for the real remix off disk -- and is never used to build a
    path. The digest prints it; resolving it is the reader's business.
    """
    side = str(prefer or "").strip().lower()
    if side not in PREFERENCES:
        raise FeedbackError("a vote is for 'a' or 'b'")
    row = _stamp({
        "other": clean_text(other, MAX_LABEL),
        "prefer": side,
        "winner": "this" if side == "a" else "other",
        "reason": clean_text(reason),
        "label": clean_text(label, MAX_LABEL),
        "other_label": clean_text(other_label, MAX_LABEL),
    })
    with _LOCK:
        data = read(remix_dir, rid)
        data["votes"].append(row)
        _write_json(Path(remix_dir) / FEEDBACK_FILE, data)
    return row


def apply(remix_dir: str | Path, payload: dict, rid: str = "") -> dict:
    """Take one POST body -- marker, rating and/or vote -- and store it.

    Returns the whole feedback document, so the page can re-render from the
    server's copy rather than from what it hoped it had sent.
    """
    if not isinstance(payload, dict):
        raise FeedbackError("body must be a JSON object")
    did = False
    marker = payload.get("marker")
    if isinstance(marker, dict):
        add_marker(remix_dir, marker.get("time"), marker.get("category"),
                   marker.get("note", ""), rid)
        did = True
    if payload.get("stars") is not None or payload.get("verdict") is not None:
        add_rating(remix_dir, payload.get("stars"), payload.get("verdict"), rid)
        did = True
    vote = payload.get("vote")
    if isinstance(vote, dict):
        add_vote(remix_dir, vote.get("other", ""), vote.get("prefer", ""),
                 vote.get("reason", ""), vote.get("label", ""),
                 vote.get("other_label", ""), rid)
        did = True
    if not did:
        raise FeedbackError("nothing to save: send a marker, a rating or a vote")
    return read(remix_dir, rid)


# ---------------------------------------------------------------------------
# the digest
# ---------------------------------------------------------------------------

def fmt_time(seconds: float) -> str:
    m = int(seconds // 60)
    return f"{m}:{seconds - m * 60:05.2f}"


def collect(lib) -> list[dict]:
    """Every remix that has feedback, newest remix first.

    ``lib`` is a :class:`fourfloor.store.Library`.
    """
    out = []
    for meta in lib.list_remixes():
        rid = meta["id"]
        d = lib.remix_dir(rid)
        data = read(d, rid)
        if not (data["markers"] or data["ratings"] or data["votes"]):
            continue
        out.append({"meta": meta, "feedback": data,
                    "rating": latest_rating(data),
                    "plan": plan_of(d), "session": session_of(d)})
    return out


def digest(lib, rid: str = "") -> str:
    """One text block an engineer agent can read and act on.

    Grouped by category rather than by time on purpose: a fix is per category
    ("the hats drag"), and the bar numbers under it are the evidence.
    """
    rows = collect(lib)
    if rid:
        rows = [r for r in rows if r["meta"]["id"] == rid]
    if not rows:
        return ("No listening notes yet.\n"
                "Play a remix in the app, press M where it sounds wrong, and "
                "the note lands here.")

    lines = [f"fourfloor listening notes — {len(rows)} "
             f"remix{'' if len(rows) == 1 else 'es'} with feedback", ""]
    for row in rows:
        meta, data, plan = row["meta"], row["feedback"], row["plan"]
        rating = row["rating"]
        head = (f"{meta.get('title', 'Untitled')}  ·  {meta['id']}  ·  "
                f"{float(meta.get('bpm', 0)):.0f} BPM  ·  "
                f"{meta.get('camelot', '?')} {meta.get('key', '')}  ·  "
                f"{meta.get('form', '?')} form")
        lines.append(head)
        lines.append("─" * min(len(head), 78))
        if rating["stars"] or rating["verdict"]:
            stars = ("★" * rating["stars"] + "☆" * (5 - rating["stars"])
                     if rating["stars"] else "unrated")
            verdict = f'  “{rating["verdict"]}”' if rating["verdict"] else ""
            lines.append(f"  rating  {stars}{verdict}")

        said: set[str] = set()                    # each slot explains itself once
        by_cat: dict[str, list[dict]] = {}
        for m in data["markers"]:
            by_cat.setdefault(m.get("category", "?"), []).append(m)
        if not by_cat:
            lines.append("  (no markers — rating only)")
        for code in CATEGORIES:
            marks = by_cat.pop(code, [])
            if not marks:
                continue
            lines.append(f"  {CATEGORIES[code]}  ({len(marks)})")
            for m in sorted(marks, key=lambda x: x.get("time", 0)):
                slot = slot_of(plan, int(m.get("bar", 0)))
                where = slot_label(slot) or m.get("slot_bars") or "outside the plan"
                note = f'  — {m["note"]}' if m.get("note") else ""
                lines.append(f"      bar {int(m.get('bar', 0)):>3}  "
                             f"{fmt_time(float(m.get('time', 0))):>7}  "
                             f"[{where}]{note}")
                if slot and slot.get("note") and where not in said:
                    said.add(where)
                    lines.append(f"{'':>25}the arranger said: {slot['note']}")
        for code, marks in by_cat.items():            # anything unknown, kept
            lines.append(f"  {code}  ({len(marks)})")

        for v in data["votes"]:
            won = "this one" if v.get("prefer") == "a" else (
                v.get("other_label") or v.get("other") or "the other one")
            reason = f'  — {v["reason"]}' if v.get("reason") else ""
            lines.append(f"  A/B vs {v.get('other_label') or v.get('other')}: "
                         f"preferred {won}{reason}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def as_json(lib) -> dict:
    """The same thing as data, for ``GET /api/feedback``."""
    rows = collect(lib)
    return {
        "categories": [{"code": c, "label": lab} for c, lab in CATEGORIES.items()],
        "remixes": [
            {
                "id": r["meta"]["id"],
                "title": r["meta"].get("title", ""),
                "bpm": r["meta"].get("bpm"),
                "camelot": r["meta"].get("camelot"),
                "form": r["meta"].get("form"),
                "created": r["meta"].get("created"),
                "rating": r["rating"],
                "feedback": r["feedback"],
            }
            for r in rows
        ],
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    from .store import Library

    p = argparse.ArgumentParser(
        prog="python -m fourfloor.feedback",
        description="Print every listening note taken in the app, grouped so an "
                    "engineer can act on it.")
    p.add_argument("--json", action="store_true", help="print the raw records")
    p.add_argument("--home", default=None,
                   help="library folder (default ~/.fourfloor)")
    p.add_argument("--remix", default="", help="just this remix id")
    args = p.parse_args(argv)

    lib = Library(args.home)
    if args.json:
        data = as_json(lib)
        if args.remix:
            data["remixes"] = [r for r in data["remixes"] if r["id"] == args.remix]
        print(json.dumps(data, indent=2))
    else:
        print(digest(lib, args.remix), end="")
    return 0


if __name__ == "__main__":                              # pragma: no cover
    raise SystemExit(main())
