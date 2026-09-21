"""Find the record a remix was made from, and put the likeliest one first.

yt-dlp will search YouTube for us -- ``ytsearch5:<query>`` returns five results
as JSON without downloading anything -- so the work here is not fetching, it is
*choosing*. A search for "Don Toliver E85 official audio" comes back with the
official upload, three other people's remixes of it, a sped-up edit and an hour
loop, and picking the wrong one wastes a download and a Demucs run.

Three things decide the order:

* **Title similarity.** Token F1 between what the remix title claimed and what
  the candidate is called, after both have been through the same de-junking the
  parser uses. Artist and track tokens are weighted the same, because half the
  titles DJs trade have them the wrong way round.
* **Duration.** A song is two to six minutes. A 40-second snippet and a
  one-hour loop are not the record, whatever they are called.
* **The words that give away an edit.** ``remix``, ``bootleg``, ``sped up``,
  ``slowed``, ``nightcore``, ``8d``, ``cover``, ``live``, ``karaoke``,
  ``instrumental``, ``mashup`` -- each one is a subtraction, and the remixer's
  own name appearing in a candidate title is the biggest subtraction of all,
  because that candidate *is* the remix we already have.

The score is a number in 0..1 with its reasons attached, so ``refs review`` can
show a person why a candidate was preferred.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from .. import fetch as fetch_mod
from . import titles as T

#: How many results to ask for per query.
PER_QUERY = 5

#: A song is this long. Outside the band the score falls away smoothly rather
#: than off a cliff, because a 1:55 interlude is still plausibly the record.
MIN_SECONDS = 120.0
MAX_SECONDS = 360.0

#: Words in a candidate title that say it is not the original recording, and
#: what each one costs.
PENALTIES: dict[str, float] = {
    "remix": 0.55, "rmx": 0.45, "bootleg": 0.55, "edit": 0.35, "flip": 0.45,
    "rework": 0.4, "refix": 0.4, "mashup": 0.55, "mash-up": 0.55,
    "sped up": 0.6, "spedup": 0.6, "speed up": 0.5, "slowed": 0.6,
    "reverb": 0.5, "nightcore": 0.7, "8d": 0.6, "9d": 0.6, "16d": 0.6,
    "bass boosted": 0.5, "cover": 0.5, "live": 0.4, "concert": 0.4,
    "karaoke": 0.7, "instrumental": 0.5, "acapella": 0.5, "acappella": 0.5,
    "tiktok": 0.35, "loop": 0.4, "1 hour": 0.8, "one hour": 0.8,
    "10 hours": 0.9, "extended mix": 0.3, "vip": 0.35, "reaction": 0.8,
    "tutorial": 0.8, "type beat": 0.9, "full album": 0.7, "mix 20": 0.5,
    "dj set": 0.8, "mashup mix": 0.6, "chopped": 0.5, "screwed": 0.5,
    "clean version": 0.2, "trailer": 0.6, "teaser": 0.6,
}

#: Words that say this *is* the record, and what each one is worth.
BONUSES: dict[str, float] = {
    "official audio": 0.16, "official video": 0.08, "official music video": 0.08,
    "audio": 0.05, "lyrics": 0.06, "lyric video": 0.06, "original": 0.05,
    "full song": 0.03, "hq": 0.01,
}

#: A YouTube "Art Track" -- the auto-generated ``Artist - Topic`` channel -- is
#: the label's own upload, which is exactly what we want.
TOPIC = re.compile(r"\s-\s*topic\s*$", re.I)

_WORD = re.compile(r"[a-z0-9']+")
_QUERY_BAD = re.compile(r"[\r\n\t\x00]")


class SearchError(RuntimeError):
    """The search itself failed -- no network, no yt-dlp, a refused query."""


@dataclass
class Candidate:
    """One possible original, with the reasons it scored what it did."""

    url: str = ""
    title: str = ""
    uploader: str = ""
    duration: float = 0.0
    query: str = ""
    score: float = 0.0
    similarity: float = 0.0
    penalties: list[str] = field(default_factory=list)
    bonuses: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "url": self.url, "title": self.title, "uploader": self.uploader,
            "duration": round(float(self.duration or 0.0), 1),
            "query": self.query, "score": round(self.score, 4),
            "similarity": round(self.similarity, 4),
            "penalties": list(self.penalties), "bonuses": list(self.bonuses),
        }


# ---------------------------------------------------------------------------
# comparing titles
# ---------------------------------------------------------------------------

def tokens(text: str) -> list[str]:
    """Comparable words: de-junked, accent-folded, lowercase."""
    clean = T.normalize(str(text or ""))
    for pattern in T.NOISE:
        clean = re.sub(pattern, " ", clean, flags=re.I)
    clean = TOPIC.sub(" ", clean)
    return _WORD.findall(T.strip_accents(clean).lower())


def title_similarity(wanted: str, found: str) -> float:
    """Token F1 between two titles, in 0..1.

    F1 rather than a plain overlap because both directions matter: a candidate
    missing half the track's words is wrong, and so is one that adds ten words
    of its own. Stop-words are left in -- "the" is load-bearing in *The Sweet
    Escape* -- and repeated words are counted once.
    """
    a, b = set(tokens(wanted)), set(tokens(found))
    if not a or not b:
        return 0.0
    shared = len(a & b)
    if not shared:
        return 0.0
    precision, recall = shared / len(b), shared / len(a)
    return float(2 * precision * recall / (precision + recall))


def duration_fit(seconds: float) -> float:
    """1.0 for a plausible song length, falling away outside 2-6 minutes."""
    d = float(seconds or 0.0)
    if d <= 0:
        return 0.6                                   # unknown: neither reward nor punish
    if MIN_SECONDS <= d <= MAX_SECONDS:
        return 1.0
    if d < MIN_SECONDS:
        return max(0.05, d / MIN_SECONDS) ** 1.5
    return max(0.05, MAX_SECONDS / d) ** 1.2


def score_candidate(parsed: T.Parsed, cand: Candidate) -> Candidate:
    """Fill in ``score``, ``similarity`` and the reasons, and return ``cand``."""
    wanted = " ".join(x for x in (parsed.artist, parsed.track) if x)
    haystack = f"{cand.title} {cand.uploader}".lower()
    cand.similarity = title_similarity(wanted, cand.title)

    penalty, names = 0.0, []
    for word, cost in PENALTIES.items():
        if re.search(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])", haystack):
            penalty += cost
            names.append(word)
    if parsed.remixer and len(parsed.remixer) > 2:
        if re.search(re.escape(parsed.remixer.lower()), haystack):
            penalty += 0.8                            # this *is* the remix we have
            names.append(f"credited to {parsed.remixer}")

    bonus, got = 0.0, []
    for word, worth in BONUSES.items():
        if word in haystack:
            bonus += worth
            got.append(word)
    if TOPIC.search(cand.uploader or ""):
        bonus += 0.18
        got.append("topic channel")
    # The artist's own channel is evidence, but only a little of it: everything
    # Don Toliver ever released is on Don Toliver's channel, and only one of
    # them is the record this remix was made from. It broke the ranking once by
    # putting a different song by the right artist above the right song.
    if parsed.artist and title_similarity(parsed.artist, cand.uploader) >= 0.75:
        bonus += 0.10
        got.append("artist's own channel")

    base = cand.similarity * duration_fit(cand.duration)
    cand.score = float(max(0.0, min(1.0, base + bonus - penalty)))
    cand.penalties, cand.bonuses = names, got
    return cand


def rank(parsed: T.Parsed, candidates: list[Candidate]) -> list[Candidate]:
    """Score every candidate and return them best first, duplicates removed."""
    seen: set[str] = set()
    out: list[Candidate] = []
    for cand in candidates:
        key = (cand.url or cand.title).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(score_candidate(parsed, cand))
    return sorted(out, key=lambda c: (-c.score, -c.similarity, c.duration))


# ---------------------------------------------------------------------------
# asking yt-dlp
# ---------------------------------------------------------------------------

def _entries(query: str, n: int, timeout: float) -> list[dict]:
    """Raw yt-dlp search results for one query."""
    q = _QUERY_BAD.sub(" ", str(query or "")).strip()[:200]
    if not q:
        return []
    cmd = [fetch_mod.ytdlp(), "--dump-single-json", "--flat-playlist",
           "--no-warnings", "--no-color", "--ignore-config",
           "--socket-timeout", "20", f"ytsearch{max(1, int(n))}:{q}"]
    code, log = fetch_mod._run(cmd, timeout)
    if code != 0:
        raise SearchError(fetch_mod._friendly(log, f"a search for {q!r}"))
    for line in log.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                return fetch_mod.entries_of(json.loads(line))
            except ValueError:
                continue
    return []


def _as_candidate(entry: dict, query: str) -> Candidate | None:
    url = str(entry.get("url") or entry.get("webpage_url") or "").strip()
    if url and not url.startswith("http"):
        url = f"https://www.youtube.com/watch?v={entry.get('id')}"
    if not url:
        return None
    return Candidate(
        url=url,
        title=str(entry.get("title") or "").strip(),
        uploader=str(entry.get("uploader") or entry.get("channel")
                     or entry.get("uploader_id") or "").strip(),
        duration=float(entry.get("duration") or 0.0),
        query=query,
    )


def find_original(parsed: T.Parsed, per_query: int = PER_QUERY,
                  queries: list[str] | None = None, pause: float = 1.0,
                  timeout: float = fetch_mod.PROBE_TIMEOUT,
                  on_note=None) -> list[Candidate]:
    """Search for the original of ``parsed`` and return candidates, best first.

    Queries are run in order and the search stops early once something scores
    convincingly, so the common case costs one search rather than four. A short
    pause between queries keeps the request rate polite.
    """
    note = on_note or (lambda *_a: None)
    qs = queries if queries is not None else T.search_queries(parsed)
    found: list[Candidate] = []
    for i, q in enumerate(qs):
        if i:
            time.sleep(max(0.0, pause))
        note(f"searching: {q}")
        try:
            entries = _entries(q, per_query, timeout)
        except fetch_mod.FetchError as exc:
            raise SearchError(str(exc)) from None
        for e in entries:
            cand = _as_candidate(e, q)
            if cand is not None:
                found.append(cand)
        best = rank(parsed, found)
        if best and best[0].score >= 0.80:            # no need to keep looking
            return best
    return rank(parsed, found)
