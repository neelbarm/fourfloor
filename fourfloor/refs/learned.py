"""What the pairs teach, and the two decisions that read it.

A folder of finished house records tells you what one sounds like. A folder of
*pairs* tells you what a remixer **did** -- and the decision this project keeps
getting asked about is the vocal one: play the voice as it was sung, or cut it
into pieces that each start on a syllable and land on a beat.

:func:`treatment_of` measures that on one pair, from the separated vocals of
both halves:

* the **original's lattice fit** -- does the voice sit on a straight sixteenth
  grid or a triplet one (:func:`~fourfloor.analysis.alignment.vocal_fit`);
* what the **remixer did** -- continuous or chopped, from the ratio of the two
  median phrase lengths and from how long a remix phrase is in beats;
* how much of the singing **survived**, as the ratio of the two duty cycles.

:func:`table` tabulates those against the tempo the remixer chose, across every
pair, and that table is written into ``~/.fourfloor/style.json``. Two call sites
read it when it is there:

* :func:`~fourfloor.house.vocal.choose_vocal`, which is what ``--vocal auto``
  runs, takes its straight-lock threshold from the pairs instead of from the two
  records it was originally calibrated on;
* :func:`~fourfloor.analysis.suggest_house_tempo` biases its tie-break toward
  the tempo the references actually sit at instead of a flat 124.

Both are conservative by construction: no file, no change, and a table built
from fewer than :data:`MIN_PAIRS` pairs is not trusted with either.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

#: How many pairs it takes before the table is allowed to move a default. One
#: remix is an anecdote.
MIN_PAIRS = 3

#: A remix phrase shorter than this many beats is a chop however long the
#: original's phrases were: at 128 BPM two beats is under a second, which is a
#: word and a half.
CHOP_BEATS = 2.0

#: …and a remixer who kept phrases this much shorter than the original's was
#: cutting, not playing.
CHOP_RATIO = 0.65

#: How much better the triplet lattice has to fit before an original's vocal is
#: called a triplet flow. The same question ``house/vocal.py`` asks; the same
#: number, kept here so the table and the engine cannot disagree.
TRIPLET_EDGE = 0.06


def home(path: str | os.PathLike | None = None) -> Path:
    """``~/.fourfloor``, or ``FOURFLOOR_HOME`` when it is set.

    Reads the same environment variable the kit store reads, so a test that
    isolates one isolates both.
    """
    return Path(path or os.environ.get("FOURFLOOR_HOME") or (Path.home() / ".fourfloor"))


def style_path(home_dir: str | os.PathLike | None = None) -> Path:
    """Where the learned style profile lives."""
    return home(home_dir) / "style.json"


_CACHE: dict[str, tuple[float, dict]] = {}


def load(home_dir: str | os.PathLike | None = None) -> dict | None:
    """The learned style profile, or ``None`` when nothing has been learned.

    Cached by modification time: this is read on every remix, and re-reading a
    small JSON file is cheap but re-reading it inside a loop is silly.
    """
    path = style_path(home_dir)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    hit = _CACHE.get(str(path))
    if hit is not None and hit[0] == mtime:
        return hit[1]
    try:
        data = json.loads(path.read_text(encoding="utf8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    _CACHE[str(path)] = (mtime, data)
    return data


def forget() -> None:
    """Drop the cache. For tests, and for a profile rewritten in-process."""
    _CACHE.clear()


# ---------------------------------------------------------------------------
# what the engine asks
# ---------------------------------------------------------------------------

def learned_bpm(home_dir: str | os.PathLike | None = None) -> float | None:
    """The tempo the references sit at, if enough of them were measured."""
    data = load(home_dir)
    if not data:
        return None
    n = int(data.get("n_refs") or 0)
    bpm = float(data.get("bpm") or 0.0)
    if n < MIN_PAIRS or not (100.0 <= bpm <= 150.0):
        return None
    return bpm


def vocal_thresholds(home_dir: str | os.PathLike | None = None) -> dict | None:
    """Learned ``straight_lock`` / ``triplet_edge`` for ``--vocal auto``.

    The rule the pairs imply is simple and it is the one a producer would state:
    the lowest straight-lattice fit among the originals that were nonetheless
    played *continuously* is the highest a voice can sit at and still be worth
    playing straight. The threshold is set just under it, and never outside a
    sane band, so one odd pair cannot turn the decision inside out.
    """
    data = load(home_dir)
    if not data:
        return None
    vocal = data.get("vocal") if isinstance(data.get("vocal"), dict) else None
    if not vocal:
        return None
    if int(vocal.get("n_pairs") or 0) < MIN_PAIRS:
        return None
    lock = vocal.get("straight_lock")
    if lock is None:
        return None
    lock = float(lock)
    if not (0.5 <= lock <= 0.9):
        return None
    return {"straight_lock": lock,
            "triplet_edge": float(vocal.get("triplet_edge") or TRIPLET_EDGE),
            "n_pairs": int(vocal.get("n_pairs") or 0)}


# ---------------------------------------------------------------------------
# what a pair says
# ---------------------------------------------------------------------------

#: The tolerance :func:`~fourfloor.analysis.alignment.vocal_fit` counts a hit
#: within, in seconds. Repeated here because the chance correction below needs
#: it, and it is the one number both readings depend on.
FIT_TOL = 0.030


def chance_level(bpm: float, division: int, tol: float = FIT_TOL) -> float:
    """What share of a *random* vocal lands on this lattice by luck alone.

    A triplet lattice puts six points in a beat and a straight one puts four,
    so at a fixed tolerance the triplet grid simply has more places to land: at
    146 BPM it covers 87% of the timeline and the straight grid covers 58%.
    Compared raw, almost every singer on earth reads as a triplet flow. The
    honest comparison is each reading's *excess over chance*.
    """
    step = 60.0 / max(bpm, 1e-6) / max(division, 1)
    return float(min(1.0, 2.0 * tol / step))


def lattice_of(fit: dict, bpm: float | None = None) -> str:
    """``"straight"`` or ``"triplet"`` for one vocal-fit measurement.

    With ``bpm``, both readings are corrected for how much of the timeline
    their lattice covers; without it they are compared as measured.
    """
    if not fit:
        return "unknown"
    straight = float(fit.get("straight") or 0.0)
    triplet = float(fit.get("triplet") or 0.0)
    if bpm:
        straight -= chance_level(bpm, 4)
        triplet -= chance_level(bpm, 6)
    return "triplet" if triplet - straight >= TRIPLET_EDGE else "straight"


def treatment_of(original, remix, tempo_ratio: float = 1.0) -> dict:
    """What the remixer did with the voice, from both fingerprints.

    ``original`` and ``remix`` are :class:`~fourfloor.refs.verify.Fingerprint`
    objects, so every number here was measured on the same excerpts the match
    was proved on -- which is the honest scope of the claim: this is what the
    remixer did *in the stretch that matched*, not across the whole record.
    """
    o_phr = float(getattr(original, "phrase_median", 0.0) or 0.0)
    r_phr = float(getattr(remix, "phrase_median", 0.0) or 0.0)
    o_duty = float(getattr(original, "duty", 0.0) or 0.0)
    r_duty = float(getattr(remix, "duty", 0.0) or 0.0)
    bpm = float(getattr(remix, "bpm", 0.0) or 0.0)
    ratio = (r_phr / o_phr) if o_phr > 0 else 0.0
    beats = (r_phr / (60.0 / bpm)) if bpm > 0 else 0.0
    chopped = bool(r_phr > 0 and (beats <= CHOP_BEATS or
                                  (ratio and ratio <= CHOP_RATIO)))
    fit = dict(getattr(original, "lattice", {}) or {})
    o_bpm = float(getattr(original, "bpm", 0.0) or 0.0)
    return {
        "original_lattice": lattice_of(fit, o_bpm),
        "bpm_original": round(o_bpm, 2),
        "straight": round(float(fit.get("straight") or 0.0), 4),
        "triplet": round(float(fit.get("triplet") or 0.0), 4),
        "treatment": "chopped" if chopped else "continuous",
        "phrase_original": round(o_phr, 3),
        "phrase_remix": round(r_phr, 3),
        "phrase_ratio": round(ratio, 3),
        "phrase_beats": round(beats, 2),
        "duty_original": round(o_duty, 3),
        "duty_remix": round(r_duty, 3),
        # how much of the singing survived, as a ratio of duty cycles over the
        # excerpts that matched -- 1.0 means the remix sings as much of the time
        # as the record does
        "vocal_kept": round(min(1.5, r_duty / o_duty), 3) if o_duty > 0 else 0.0,
        "tempo_ratio": round(float(tempo_ratio), 4),
    }


def table(rows: list[dict]) -> dict:
    """Tabulate lattice against treatment and tempo across every pair.

    ``rows`` are the ``vocal`` blocks of the per-pair sidecars. The result is
    the thing ``refs list`` prints and the thing the engine reads: the counts
    in each of the four cells, the median tempo the remixers chose in each, and
    the two thresholds :func:`vocal_thresholds` hands back.
    """
    import numpy as np

    cells: dict[str, dict] = {}
    for row in rows:
        key = f"{row.get('original_lattice', 'unknown')}/{row.get('treatment', '?')}"
        cell = cells.setdefault(key, {"n": 0, "tempo_ratio": [], "kept": [],
                                      "straight": [], "bpm": []})
        cell["n"] += 1
        for name, field in (("tempo_ratio", "tempo_ratio"), ("kept", "vocal_kept"),
                            ("straight", "straight"), ("bpm", "bpm_remix")):
            value = row.get(field)
            if value is not None:
                cell[name].append(float(value))

    out: dict = {"n_pairs": len(rows), "cells": {}}
    for key, cell in sorted(cells.items()):
        out["cells"][key] = {
            "n": cell["n"],
            "tempo_ratio": round(float(np.median(cell["tempo_ratio"])), 4)
            if cell["tempo_ratio"] else None,
            "vocal_kept": round(float(np.median(cell["kept"])), 3)
            if cell["kept"] else None,
            "straight_fit": round(float(np.median(cell["straight"])), 3)
            if cell["straight"] else None,
            "bpm": round(float(np.median(cell["bpm"])), 2) if cell["bpm"] else None,
        }

    played = [float(r.get("straight") or 0.0) for r in rows
              if r.get("treatment") == "continuous"]
    chopped = [float(r.get("straight") or 0.0) for r in rows
               if r.get("treatment") == "chopped"]
    if played:
        # just under the loosest voice anybody still played straight through,
        # and never below the highest voice anybody chopped
        lock = min(played) - 0.01
        if chopped:
            lock = max(lock, min(max(chopped) + 0.01, min(played) - 0.005))
        out["straight_lock"] = round(float(min(0.9, max(0.5, lock))), 3)
    out["triplet_edge"] = TRIPLET_EDGE
    return out
