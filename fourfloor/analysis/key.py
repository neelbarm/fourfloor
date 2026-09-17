"""Key detection and per-bar chord estimation.

Key uses the Krumhansl-Schmuckler algorithm: correlate the track's average
chromagram against the 24 rotations of the major and minor key profiles measured
in Krumhansl & Kessler, "Tracing the dynamic changes in perceived tonal
organization", Psychological Review 89(4), 1982. Chords use the simpler binary
triad templates of Fujishima's chroma matching (ICMC 1999).

Camelot codes are the wheel used by DJ software: 8B = C major, 8A = A minor,
+1 = up a perfect fifth, A = minor, B = major.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl & Kessler (1982) probe-tone profiles.
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Camelot: index = pitch class, value = wheel number. C major = 8B, A minor = 8A.
_MAJOR_CAMELOT = {0: 8, 1: 3, 2: 10, 3: 5, 4: 12, 5: 7, 6: 2, 7: 9, 8: 4, 9: 11, 10: 6, 11: 1}
_MINOR_CAMELOT = {9: 8, 10: 3, 11: 10, 0: 5, 1: 12, 2: 7, 3: 2, 4: 9, 5: 4, 6: 11, 7: 6, 8: 1}


@dataclass
class KeyEstimate:
    """Detected key with a Camelot code and a confidence in [0, 1]."""

    tonic: int          # pitch class 0-11
    is_minor: bool
    confidence: float
    tuning: float = 0.0

    @property
    def name(self) -> str:
        return f"{PITCH_NAMES[self.tonic]}{'m' if self.is_minor else ''}"

    @property
    def camelot(self) -> str:
        table = _MINOR_CAMELOT if self.is_minor else _MAJOR_CAMELOT
        return f"{table[self.tonic]}{'A' if self.is_minor else 'B'}"

    def to_dict(self) -> dict:
        return {
            "key": self.name,
            "camelot": self.camelot,
            "tonic_pc": self.tonic,
            "mode": "minor" if self.is_minor else "major",
            "confidence": round(float(self.confidence), 3),
            "tuning_semitones": round(float(self.tuning), 3),
        }


def camelot_to_key(code: str) -> tuple[int, bool]:
    """Parse a Camelot code like ``8A`` into (pitch class, is_minor)."""
    code = code.strip().upper()
    if len(code) < 2 or code[-1] not in "AB" or not code[:-1].isdigit():
        raise ValueError(f"not a Camelot code: {code!r}")
    num, letter = int(code[:-1]), code[-1]
    if not 1 <= num <= 12:
        raise ValueError(f"Camelot number out of range: {code!r}")
    table = _MINOR_CAMELOT if letter == "A" else _MAJOR_CAMELOT
    for pc, n in table.items():
        if n == num:
            return pc, letter == "A"
    raise ValueError(code)


def parse_key(text: str) -> tuple[int, bool]:
    """Parse ``Am``, ``F#`` , ``Bbm``, ``8A`` … into (pitch class, is_minor)."""
    t = text.strip()
    if t and t[0].isdigit():
        return camelot_to_key(t)
    t = t.replace("maj", "").replace("MAJ", "")
    minor = t.lower().endswith("m") or t.lower().endswith("min")
    core = t[:-3] if t.lower().endswith("min") else (t[:-1] if minor else t)
    core = core.strip().capitalize().replace("Bb", "A#").replace("Db", "C#") \
        .replace("Eb", "D#").replace("Gb", "F#").replace("Ab", "G#")
    if len(core) == 2 and core[1] == "b":
        core = PITCH_NAMES[(PITCH_NAMES.index(core[0]) - 1) % 12]
    if core not in PITCH_NAMES:
        raise ValueError(f"unrecognised key: {text!r}")
    return PITCH_NAMES.index(core), minor


def camelot_neighbours(code: str) -> list[str]:
    """Harmonically compatible Camelot codes: same, ±1 on the wheel, relative."""
    num, letter = int(code[:-1]), code[-1]
    other = "A" if letter == "B" else "B"
    return [
        code,
        f"{(num % 12) + 1}{letter}",
        f"{((num - 2) % 12) + 1}{letter}",
        f"{num}{other}",
    ]


def detect_key(chroma: np.ndarray, tuning: float = 0.0) -> KeyEstimate:
    """Krumhansl-Schmuckler key detection over an averaged chromagram.

    Confidence is the normalised gap between the winning correlation and the
    best correlation of any key *outside the winner's Camelot neighbourhood*.
    Relative-major/minor confusion is harmless for a DJ -- those keys mix -- so
    scoring it as uncertainty would understate how usable the answer is; landing
    in the wrong neighbourhood is the failure that matters.
    """
    if chroma.size == 0:
        return KeyEstimate(0, False, 0.0, tuning)
    avg = chroma.mean(axis=1)
    if avg.sum() <= 0:
        return KeyEstimate(0, False, 0.0, tuning)
    avg = avg / avg.sum()

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        a, b = a - a.mean(), b - b.mean()
        d = np.sqrt(np.sum(a * a) * np.sum(b * b))
        return float(np.sum(a * b) / d) if d > 0 else 0.0

    scores: list[tuple[float, int, bool]] = []
    for pc in range(12):
        scores.append((corr(avg, np.roll(KS_MAJOR, pc)), pc, False))
        scores.append((corr(avg, np.roll(KS_MINOR, pc)), pc, True))
    scores.sort(key=lambda t: -t[0])
    best, pc, minor = scores[0]
    family = set(camelot_neighbours(KeyEstimate(pc, minor, 0.0).camelot))
    outside = [s for s, p, m in scores[1:]
               if KeyEstimate(p, m, 0.0).camelot not in family]
    runner = max(outside) if outside else 0.0
    margin = (best - runner) / max(abs(best), 1e-6)
    conf = float(np.clip(margin * 2.5, 0.0, 1.0)) * float(np.clip(best * 1.4, 0.0, 1.0))
    return KeyEstimate(tonic=pc, is_minor=minor, confidence=conf, tuning=tuning)


_TRIADS = [(0, 4, 7), (0, 3, 7)]  # major, minor


def _triad_templates() -> tuple[np.ndarray, list[tuple[int, bool]]]:
    """24 binary triad templates plus their (root, is_minor) labels."""
    tmpl, labels = [], []
    for minor, iv in enumerate(_TRIADS):
        for root in range(12):
            v = np.zeros(12)
            for s in iv:
                v[(root + s) % 12] = 1.0
            tmpl.append(v / v.sum())
            labels.append((root, bool(minor)))
    return np.asarray(tmpl), labels


def chords_per_bar(chroma: np.ndarray, fps: float, downbeat_times: np.ndarray,
                   bar_dur: float) -> list[dict]:
    """Estimate one triad per bar by matching the bar's mean chroma to templates.

    The returned roots drive the bass line, so ambiguous bars (low margin) fall
    back to the previous bar's root, which avoids a bass line that jumps around
    on percussion-only bars.
    """
    templates, labels = _triad_templates()
    out: list[dict] = []
    prev: tuple[int, bool] | None = None
    for t0 in downbeat_times:
        a = int(round(t0 * fps))
        b = int(round((t0 + bar_dur) * fps))
        a, b = max(0, a), min(chroma.shape[1], b)
        if b - a < 2:
            root, minor, score = (prev if prev else (0, True)) + (0.0,)
        else:
            v = chroma[:, a:b].mean(axis=1)
            s = v.sum()
            if s <= 0:
                root, minor, score = (prev if prev else (0, True)) + (0.0,)
            else:
                sims = templates @ (v / s)
                k = int(np.argmax(sims))
                srt = np.sort(sims)[::-1]
                margin = float((srt[0] - srt[1]) / max(srt[0], 1e-9))
                root, minor = labels[k]
                score = margin
                if margin < 0.04 and prev is not None:
                    root, minor = prev
        prev = (root, minor)
        out.append({
            "time": round(float(t0), 4),
            "root_pc": int(root),
            "quality": "min" if minor else "maj",
            "name": f"{PITCH_NAMES[root]}{'m' if minor else ''}",
            "confidence": round(float(score), 3),
        })
    return out


def semitone_shift(src: KeyEstimate, target_pc: int, target_minor: bool,
                   max_shift: int = 6) -> int:
    """Smallest signed semitone shift from ``src`` tonic to the target tonic.

    Mode is not changed by pitch shifting, so a major→minor request is honoured
    on the tonic only and reported as such by the caller.
    """
    raw = (target_pc - src.tonic) % 12
    shift = raw if raw <= max_shift else raw - 12
    return int(shift)


def nearest_compatible(src: KeyEstimate, other: KeyEstimate,
                       max_shift: int = 3) -> tuple[int, str]:
    """Pick the shift within ±``max_shift`` semitones landing on a key that is
    Camelot-compatible with ``other``.

    Returns ``(shift, target_camelot)``; shift 0 if we are already compatible.
    """
    wanted = set(camelot_neighbours(other.camelot))
    if src.camelot in wanted:
        return 0, src.camelot
    best: tuple[int, str] | None = None
    for shift in sorted(range(-max_shift, max_shift + 1), key=abs):
        cand = KeyEstimate((src.tonic + shift) % 12, src.is_minor, src.confidence)
        if cand.camelot in wanted and (best is None or abs(shift) < abs(best[0])):
            best = (shift, cand.camelot)
    return best if best else (0, src.camelot)
