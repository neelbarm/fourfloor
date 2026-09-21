"""Scoring: the six sub-scores, the weights, and the verdict.

The weights and thresholds in ``WEIGHTS`` and ``BANDS`` were fitted so the
ranking reproduces Neel's verdicts on the calibration set: real
references first, then ``body.classic`` ("better, some of it is off
beat"), then ``body.fourfloor`` ("off beat, everything overlapping") a
long way back. The fitted table is in the README.

See :mod:`fourfloor.critic` for what each sub-score means.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import features as F
from .features import Measured, clamp01

__all__ = ["critique", "Critique", "SubScore", "Measured", "WEIGHTS", "BANDS"]

#: How much each sub-score contributes to the total. Groove dominates
#: because every render Neel rejected, he rejected for timing first.
WEIGHTS: dict[str, float] = {
    "groove": 0.30,
    "similarity": 0.22,
    "clarity": 0.18,
    "clicks": 0.10,
    "vocal": 0.10,
    "loudness": 0.10,
}

#: (low, high) anchors: a measurement at ``low`` scores 0 on that term, at
#: ``high`` it scores 1. A pair may descend, meaning "less is better".
#: Every number here was read off the calibration set -- six reference
#: remixes, two non-house originals, and the renders Neel rated -- and
#: placed so the two rated renders fall on opposite sides of each term
#: that should separate them. The README has the resulting table.
BANDS: dict[str, tuple[float, float]] = {
    # groove
    "on_grid": (0.55, 0.85),        # windowed half-beat adherence
    "on_grid_beat": (0.30, 0.65),   # windowed beat adherence
    "pulse": (0.40, 0.90),          # beat autocorrelation contrast
    "bar": (0.35, 0.85),
    "split": (0.15, 0.65),          # rival tempo peak, less is better
    # clarity
    "density": (3.3, 6.5),          # onsets per beat, less is better
    "mod_depth": (30.0, 120.0),     # 2-8 Hz envelope peak over median
    "flatness": (0.035, 0.085),     # less is better
    # clicks
    "click_rate": (3.0, 35.0),      # weighted events per minute, less is better
    # vocal
    "vocal_lo": (-20.0, -8.0),
    "vocal_hi": (2.0, 9.0),         # above this the vocal band is too hot
    "vocal_mod": (0.50, 1.80),
    # loudness
    "rms_lo": (-20.0, -12.0),
    "rms_hi": (-6.0, -2.0),
    "crest_lo": (6.0, 10.0),
    "crest_hi": (16.0, 22.0),
}

#: ``total <= GATE[0] + GATE[1] * groove``.
#:
#: A render whose groove has collapsed cannot be rescued by the other
#: five: "off beat" was the first thing Neel said about the render he
#: threw out, and a well-mastered arrhythmic track is still unplayable.
#: Deliberately set so it does *not* bind anywhere on the calibration set
#: -- it is a guard against a pathological case, not a tuning knob. It
#: starts to bite around a groove score of 25 with everything else high.
GROOVE_GATE = (30.0, 0.90)


@dataclass
class SubScore:
    key: str
    label: str
    score: float
    weight: float
    detail: str
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"score": round(self.score, 1), "weight": self.weight,
                "detail": self.detail, "stats": self.stats}


@dataclass
class Critique:
    path: str
    score: float
    verdict: str
    subs: list[SubScore]
    measured: Measured
    backend: str
    refs: str | None = None
    elapsed: float = 0.0
    gated: bool = False
    notes: list[str] = field(default_factory=list)
    gemini: dict | None = None

    def sub(self, key: str) -> SubScore | None:
        return next((s for s in self.subs if s.key == key), None)

    def to_dict(self) -> dict:
        m = asdict(self.measured)
        out = {
            "file": self.path,
            "score": round(self.score, 1),
            "verdict": self.verdict,
            "embedding": self.backend,
            "refs": self.refs,
            "gated": self.gated,
            "elapsed_sec": round(self.elapsed, 2),
            "sub_scores": {s.key: s.to_dict() for s in self.subs},
            "measured": {k: (round(v, 4) if isinstance(v, float) else v)
                         for k, v in m.items() if k != "extra"},
            "notes": self.notes,
        }
        if self.gemini is not None:
            out["gemini"] = self.gemini
        return out


# ---------------------------------------------------------------------------
# session file
# ---------------------------------------------------------------------------

def session_for(path: Path) -> dict | None:
    """The ``*.session.json`` beside a render, if fourfloor wrote one."""
    cand = path.with_suffix(".session.json")
    if not cand.exists():
        cand = path.with_name(path.stem + ".session.json")
    if not cand.exists():
        return None
    try:
        return json.loads(cand.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _cue_times(session: dict | None, kind: str | None = None) -> list[float]:
    if not session:
        return []
    return [float(c["time"]) for c in session.get("cues", [])
            if "time" in c and (kind is None or c.get("kind") == kind)]


# ---------------------------------------------------------------------------
# sub-scores
# ---------------------------------------------------------------------------

def _band(x: float, key: str) -> float:
    lo, hi = BANDS[key]
    return F._lerp01(x, lo, hi)


def score_groove(m: Measured) -> SubScore:
    """Is there one pulse, is it sharp, and do the onsets land on it.

    Grid adherence carries most of the weight and the split-peak term is
    its own term rather than a multiplier, because of what the
    calibration set showed: ``body.fourfloor`` has the *sharpest* beat
    autocorrelation peak of every render measured. Its synthesised kick
    is perfectly quantised. Everything layered over that kick is not.
    Scoring peak sharpness alone would have called the track Neel threw
    out the tightest one in the set; grid adherence (0.59 against the
    references' 0.79) and the split peak (0.61 against their 0.00-0.41)
    are the two numbers that actually notice.

    Half-beat autocorrelation contrast is deliberately *not* here,
    although a house record is supposed to have a peak there. On the
    calibration set it pointed the wrong way: ``body.fourfloor`` scores
    0.77 on it and a clean four-to-the-floor render scores 0.06, because
    a kick with nothing syncopated over it has little half-beat energy
    by construction. Two grid terms replace it -- adherence to the
    half-beat grid and to the beat grid -- which ask the question
    directly instead of inferring it from the spectrum of the envelope.
    """
    grid = _band(m.on_grid, "on_grid")
    grid_beat = _band(m.on_grid_beat, "on_grid_beat")
    pulse = _band(m.beat_contrast, "pulse")
    bar = _band(m.bar_contrast, "bar")
    split = 1.0 - _band(m.split_peak, "split")
    score = 100.0 * (0.42 * grid + 0.20 * grid_beat + 0.13 * pulse
                     + 0.08 * bar + 0.17 * split)
    if m.on_grid >= 0.80 and m.beat_contrast > 0.18:
        detail = f"{m.on_grid * 100:.0f}% of onsets land on the half-beat grid; pulse is sharp"
    elif m.on_grid >= 0.60:
        detail = f"{m.on_grid * 100:.0f}% on grid -- some of it drifts off the beat"
    else:
        detail = f"only {m.on_grid * 100:.0f}% on grid; the pulse is smeared"
    if split < 0.55:
        detail += ", and the beat peak is split -- two layers at different tempi"
    return SubScore("groove", "groove", score, WEIGHTS["groove"], detail, {
        "bpm": round(m.bpm, 2), "on_grid": round(m.on_grid, 3),
        "on_grid_beat": round(m.on_grid_beat, 3),
        "beat_contrast": round(m.beat_contrast, 3),
        "half_contrast": round(m.half_contrast, 3),
        "bar_contrast": round(m.bar_contrast, 3),
        "split_peak": round(m.split_peak, 3),
    })


def score_clarity(m: Measured) -> SubScore:
    """Is the arrangement one thing, or two things on top of each other."""
    density = 1.0 - _band(m.onsets_per_beat, "density")
    mod = _band(m.mod_depth, "mod_depth")
    flat = 1.0 - _band(m.flatness, "flatness")
    score = 100.0 * (0.45 * density + 0.30 * mod + 0.25 * flat)
    if m.onsets_per_beat > BANDS["density"][0] and m.mod_depth < BANDS["mod_depth"][1] * 0.45:
        detail = (f"{m.onsets_per_beat:.1f} onsets per beat with a weak 2-8 Hz pulse "
                  f"-- sounds like two parts playing over each other")
    elif m.mod_depth < BANDS["mod_depth"][1] * 0.45:
        detail = "the 2-8 Hz modulation is shallow; the groove does not breathe"
    else:
        detail = f"{m.onsets_per_beat:.1f} onsets per beat, modulation depth {m.mod_depth:.1f}x"
    return SubScore("clarity", "clarity", score, WEIGHTS["clarity"], detail, {
        "onsets_per_beat": round(m.onsets_per_beat, 2),
        "mod_depth": round(m.mod_depth, 2),
        "flatness": round(m.flatness, 4),
    })


def score_clicks(m: Measured) -> SubScore:
    score = 100.0 * (1.0 - _band(m.click_rate, "click_rate"))
    n = m.extra.get("n_clicks", 0)
    if m.click_rate < 2.0:
        detail = "no audible splices"
    else:
        detail = f"{n} discontinuit{'y' if n == 1 else 'ies'} ({m.click_rate:.0f}/min weighted at cues)"
    return SubScore("clicks", "clicks", score, WEIGHTS["clicks"], detail, {
        "click_rate_per_min": round(m.click_rate, 2),
        "n_events": n, "worst_ratio": round(m.click_worst, 1),
    })


def score_vocal(m: Measured) -> SubScore:
    buried = _band(m.vocal_ratio_db, "vocal_lo")
    hot = 1.0 - _band(m.vocal_ratio_db, "vocal_hi")
    level = min(buried, hot)
    mod = _band(m.vocal_mod, "vocal_mod")
    score = 100.0 * (0.6 * level + 0.4 * mod)
    if buried < 0.3:
        detail = f"vocal band sits {m.vocal_ratio_db:.0f} dB under the backing -- buried"
    elif hot < 0.3:
        detail = f"vocal band is {m.vocal_ratio_db:.0f} dB relative -- too hot, it fights the drums"
    elif mod < 0.3:
        detail = "little syllabic movement; the vocal reads as a pad, not words"
    else:
        detail = f"vocal band {m.vocal_ratio_db:.0f} dB relative, syllabic energy {m.vocal_mod:.1f}x"
    return SubScore("vocal", "vocal", score, WEIGHTS["vocal"], f"{detail} ({m.vocal_source})", {
        "ratio_db": round(m.vocal_ratio_db, 2), "syllabic": round(m.vocal_mod, 3),
        "source": m.vocal_source,
    })


def score_loudness(m: Measured) -> SubScore:
    quiet = _band(m.rms_db, "rms_lo")
    loud = 1.0 - _band(m.rms_db, "rms_hi")
    squashed = _band(m.crest_db, "crest_lo")
    floppy = 1.0 - _band(m.crest_db, "crest_hi")
    clip = 1.0 - clamp01(m.clip_fraction / 0.002)
    score = 100.0 * (0.45 * min(quiet, loud) + 0.30 * min(squashed, floppy) + 0.25 * clip)
    if m.clip_fraction > 0.0005:
        detail = f"{m.clip_fraction * 100:.2f}% of samples are clipped"
    elif quiet < 0.4:
        detail = f"quiet master at {m.rms_db:.1f} dBFS RMS"
    elif loud < 0.4:
        detail = f"hot master at {m.rms_db:.1f} dBFS RMS"
    else:
        detail = f"{m.rms_db:.1f} dBFS RMS, {m.crest_db:.1f} dB crest"
    return SubScore("loudness", "loudness", score, WEIGHTS["loudness"], detail, {
        "rms_db": round(m.rms_db, 2), "peak_db": round(m.peak_db, 2),
        "crest_db": round(m.crest_db, 2), "clip_fraction": round(m.clip_fraction, 6),
    })


def score_similarity(raw: float, anchors: tuple[float, float], backend: str,
                     per_window: list[float], n_refs: int,
                     weight_factor: float = 1.0) -> SubScore:
    lo, hi = anchors
    # ``hi`` is the mean leave-one-out cosine the references reach against
    # each other and maps to 90, not 100 -- a render that scored 100 would
    # *be* one of the references. ``lo`` is the best any non-house original
    # managed and maps to 0. Nothing is clamped at the top, so a render that
    # out-houses the average reference is allowed to say so.
    score = min(100.0, max(0.0, 90.0 * (raw - lo) / max(hi - lo, 1e-6)))
    if score >= 75:
        detail = f"drops sit in house territory (cos {raw:.3f} to {n_refs} reference windows)"
    elif score >= 45:
        detail = f"drops are house-adjacent but thin (cos {raw:.3f})"
    else:
        detail = f"drops do not sound like the references (cos {raw:.3f})"
    return SubScore("similarity", "similarity", score,
                    WEIGHTS["similarity"] * weight_factor,
                    f"{detail} [{backend}]", {
                        "cosine": round(raw, 4), "backend": backend,
                        "per_window": [round(v, 4) for v in per_window],
                        "ref_windows": n_refs,
                    })


# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------

_HEADLINE = [
    (82, "This one is gig-ready."),
    (68, "Playable, with one thing to fix."),
    (52, "Half there -- it reads as house but it would not survive a floor."),
    (35, "Not playable yet."),
    (0, "This is not a house remix yet."),
]

_BLAME = {
    "groove": "the timing is the problem",
    "clarity": "the arrangement is piling parts on top of each other",
    "similarity": "it does not sound like the reference records",
    "clicks": "there are audible edits",
    "vocal": "the vocal balance is wrong",
    "loudness": "the master level is wrong",
}


def verdict_for(score: float, subs: list[SubScore], gated: bool) -> str:
    head = next(text for cut, text in _HEADLINE if score >= cut)
    worst = sorted(subs, key=lambda s: s.score)[:2]
    bad = [s for s in worst if s.score < 55]
    if not bad:
        best = max(subs, key=lambda s: s.score)
        return f"{head} Strongest on {best.label}; nothing is obviously broken."
    blame = _BLAME.get(bad[0].key, bad[0].label)
    tail = f" Mainly {blame}: {bad[0].detail}."
    if len(bad) > 1:
        tail += f" Also {_BLAME.get(bad[1].key, bad[1].label)}."
    if gated:
        tail += " The score is capped by the groove: nothing else matters until it is on the beat."
    return head + tail


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def measure(path: Path, session: dict | None = None,
            demucs: bool = False) -> tuple[Measured, np.ndarray, int]:
    """Every local number for one file. Returns ``(measured, mono, sr)``."""
    from ..audio import decode

    clip = decode(path)
    mono_full = F.downmix(clip.samples)
    m = Measured(duration=clip.duration)
    F.measure_loudness(clip.samples, m)
    F.measure_clicks(mono_full, clip.sr, m, _cue_times(session))
    mono = F.resample_to(mono_full, clip.sr, F.ANALYSIS_SR)
    sp = F.spectral(mono)
    bpm_hint = float(session["bpm"]) if session and session.get("bpm") else None
    F.measure_groove(sp, bpm_hint, m)
    F.measure_mush(sp, m)
    vocals = _demucs_vocals(path) if demucs else None
    F.measure_vocal(sp, m, vocals, clip.sr)
    return m, mono_full, clip.sr


def _demucs_vocals(path: Path) -> np.ndarray | None:
    """Separate a vocal stem with Demucs. Minutes on CPU, so opt in only.

    This shells out to ``demucs --two-stems=vocals`` rather than going
    through ``fourfloor.stems``: that module's ``Stems`` deliberately folds
    vocals in with the other harmonic content, which is right for building
    a remix and useless for judging one.
    """
    import shutil
    import subprocess
    import sys
    import tempfile

    tmp = tempfile.mkdtemp(prefix="ff-critic-")
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "demucs", "--two-stems=vocals",
             "-o", tmp, "-n", "htdemucs", str(path)],
            capture_output=True, text=True, check=False,
        )
        if proc.returncode != 0:
            return None
        found = list(Path(tmp).rglob("vocals.wav"))
        if not found:
            return None
        from ..audio import decode

        return F.downmix(decode(found[0]).samples)
    except (OSError, RuntimeError):
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def critique(path: str | Path, refs: str | Path | None = None, embed: str = "auto",
             demucs: bool = False, windows: int = 4, on_step=None) -> Critique:
    """Score one render. ``refs`` is a folder of real house remixes."""
    started = time.time()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    session = session_for(path)
    notes: list[str] = []

    if on_step:
        on_step("measuring")
    m, mono, sr = measure(path, session, demucs=demucs)

    subs = [score_groove(m), score_clarity(m), score_clicks(m),
            score_vocal(m), score_loudness(m)]

    if refs:
        from . import embed as E

        backend = E.load_backend(embed)
        if on_step:
            on_step(f"embedding ({backend.name})")
        bank = E.reference_bank(Path(refs), backend, limit=windows,
                                on_file=lambda n: on_step and on_step(f"reference {n}"))
        drops = _cue_times(session, "drop")
        wins = [w for _, w in E.drop_windows(
            F.resample_to(mono, sr, backend.sr), backend.sr, drops, windows)]
        vecs = backend.embed(wins)
        raw, per_window = E.cosine_to_bank(vecs, bank)
        subs.insert(0, score_similarity(raw, backend.anchors, backend.name,
                                        per_window, len(bank),
                                        getattr(backend, "weight_factor", 1.0)))
        backend_name = backend.name
        if getattr(backend, "caveat", ""):
            notes.append(backend.caveat)
    else:
        backend_name = "none"
        notes.append("no --refs given: the learned-similarity score was skipped "
                     "and the other five were reweighted")

    total_w = sum(s.weight for s in subs)
    score = sum(s.score * s.weight for s in subs) / max(total_w, 1e-9)
    groove = next(s.score for s in subs if s.key == "groove")
    cap = GROOVE_GATE[0] + GROOVE_GATE[1] * groove
    gated = score > cap
    score = min(score, cap)

    return Critique(path=str(path), score=score,
                    verdict=verdict_for(score, subs, gated), subs=subs, measured=m,
                    backend=backend_name, refs=str(refs) if refs else None,
                    elapsed=time.time() - started, gated=gated, notes=notes)
