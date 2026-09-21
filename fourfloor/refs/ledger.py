"""What has been done to every link, so a long night survives a closed laptop.

A DJ pastes forty links. Each one is a metadata probe, a download, a search,
two more downloads, two Demucs runs and a kit build; somewhere around link
twelve the wifi drops, or the battery does. The ledger is the answer: one JSON
file in the reference folder, rewritten after every step of every link, keyed by
URL. Re-running the same command skips whatever already finished and picks up
where it stopped.

It lives with the audio, not in the repository, because it is a list of the
records somebody likes -- their titles, their links, the lot. Only the aggregate
numbers ever go near the project.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA = 1

#: A link's state.
#:
#: ``done``         -- an original was found, verified and filed.
#: ``needs_review`` -- something scored between the thresholds; a person decides.
#: ``standalone``   -- no original was found; the remix was kept on its own.
#: ``failed``       -- the link, the download or the search fell over.
#: ``pending``      -- started and not finished, which is what a crash leaves.
STATUSES = ("done", "needs_review", "standalone", "failed", "pending")

#: Statuses a re-run leaves alone.
SETTLED = ("done", "standalone", "needs_review")


@dataclass
class Entry:
    """One link, and everything the pipeline learned about it."""

    url: str = ""
    slug: str = ""
    status: str = "pending"
    title: str = ""
    uploader: str = ""
    site: str = ""
    artist: str = ""
    track: str = ""
    remixer: str = ""
    kind: str = ""
    remix_path: str = ""
    original_path: str = ""
    original_url: str = ""
    original_title: str = ""
    score: float = 0.0
    verdict: str = ""
    method: str = ""
    tempo_ratio: float = 0.0
    semitones: int = 0
    bpm_original: float = 0.0
    bpm_remix: float = 0.0
    lattice: str = ""
    treatment: str = ""
    kit: str = ""
    candidates: list = field(default_factory=list)
    error: str = ""
    added: str = ""
    updated: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Entry":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


class Ledger:
    """Every link seen, on disk, written after every change."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.entries: list[Entry] = []
        self._load()

    # -- reading ----------------------------------------------------------
    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf8"))
        except (OSError, json.JSONDecodeError):
            # a half-written ledger is not worth dying over; keep the broken one
            # beside the new one so nothing a person did is silently thrown away
            backup = self.path.with_suffix(".json.broken")
            try:
                self.path.replace(backup)
            except OSError:
                pass
            return
        rows = data.get("links") if isinstance(data, dict) else data
        self.entries = [Entry.from_dict(r) for r in (rows or []) if isinstance(r, dict)]

    def by_url(self, url: str) -> Entry | None:
        key = (url or "").strip()
        return next((e for e in self.entries if e.url == key), None)

    def by_slug(self, slug: str) -> Entry | None:
        return next((e for e in self.entries if e.slug == slug), None)

    def of_status(self, *statuses: str) -> list[Entry]:
        return [e for e in self.entries if e.status in statuses]

    def slugs(self) -> set[str]:
        return {e.slug for e in self.entries if e.slug}

    # -- writing ----------------------------------------------------------
    def put(self, entry: Entry) -> Entry:
        """Insert or replace by URL, stamp it, and write the file."""
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        entry.updated = now
        existing = self.by_url(entry.url)
        if existing is None:
            entry.added = entry.added or now
            self.entries.append(entry)
        else:
            entry.added = existing.added or now
            self.entries[self.entries.index(existing)] = entry
        self.save()
        return entry

    def save(self) -> Path:
        """Write the ledger atomically: a crash cannot leave half a file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema": SCHEMA,
                   "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "links": [e.to_dict() for e in self.entries]}
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), prefix=".ledger-",
                                   suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf8") as fh:
                json.dump(payload, fh, indent=2)
                fh.write("\n")
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return self.path

    def free_slug(self, slug: str, taken: set[str] | None = None) -> str:
        """``slug``, or ``slug-2``/``slug-3`` when a different link had it."""
        used = set(self.slugs()) | set(taken or ())
        if slug not in used:
            return slug
        for n in range(2, 100):
            candidate = f"{slug}-{n}"
            if candidate not in used:
                return candidate
        return f"{slug}-{int(time.time()) % 10000}"
