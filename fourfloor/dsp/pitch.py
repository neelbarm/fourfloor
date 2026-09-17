"""Pitch shifting by resample-after-stretch, and the tempo targeting rule."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy import signal as sps

from .phasevocoder import time_stretch

MAX_COMFORTABLE_RATIO = 1.25
MIN_COMFORTABLE_RATIO = 0.8


def resample_ratio(x: np.ndarray, ratio: float) -> np.ndarray:
    """Resample by ``ratio`` (output length ≈ len(x) * ratio) with a polyphase FIR.

    ``ratio`` is approximated as a rational with denominator ≤ 2000, which keeps
    the pitch error under 0.1 cent while keeping the filter cheap.
    """
    if abs(ratio - 1.0) < 1e-9:
        return x
    frac = Fraction(ratio).limit_denominator(2000)
    up, down = frac.numerator, frac.denominator
    if x.ndim == 2:
        return np.stack([sps.resample_poly(x[:, c], up, down) for c in range(x.shape[1])],
                        axis=1).astype(np.float32)
    return sps.resample_poly(x, up, down).astype(np.float32)


def pitch_shift(x: np.ndarray, semitones: float) -> np.ndarray:
    """Shift pitch by ``semitones`` while preserving duration.

    Stretch by ``2**(n/12)`` then resample back down by the same factor: the
    resampling moves every partial by the interval and undoes the length change.
    """
    if abs(semitones) < 1e-6:
        return x
    r = 2.0 ** (semitones / 12.0)
    stretched = time_stretch(x, 1.0 / r)          # longer by r
    out = resample_ratio(stretched, 1.0 / r)      # shorter by r, pitch up by r
    n = len(x)
    if len(out) < n:
        pad = n - len(out)
        out = np.concatenate([out, np.zeros((pad,) + out.shape[1:], dtype=out.dtype)])
    return out[:n]


def f0_autocorr(x: np.ndarray, sr: int, fmin: float = 35.0, fmax: float = 180.0,
                clarity: float = 0.35) -> float:
    """Fundamental of a short monophonic segment, or 0.0 if it is not pitched.

    Normalised autocorrelation with parabolic interpolation on the peak. A bass
    line is about the most favourable signal there is for this -- one note at a
    time, strong fundamental, few partials -- which is exactly why it is worth
    asking the recording what note is playing instead of asking a chromagram
    what chord it thinks the whole mix implies.

    ``clarity`` is how periodic the segment has to be before an answer is
    returned at all. A bass drop, a sub-less passage or a gap comes back as 0.0
    and the caller should play nothing rather than guess.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    n = len(x)
    if n < 4 * int(sr / max(fmin, 1e-6)):
        return 0.0
    x = x - x.mean()
    energy = float(np.dot(x, x))
    if energy <= 1e-9:
        return 0.0
    spec = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(spec * np.conj(spec))[:n]
    lo = max(1, int(sr / fmax))
    hi = min(n - 2, int(sr / fmin))
    if hi <= lo + 1:
        return 0.0
    window = ac[lo:hi + 1] / max(ac[0], 1e-12)
    k = int(np.argmax(window))
    if window[k] < clarity:
        return 0.0
    lag = lo + k
    a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]
    den = a - 2 * b + c
    if abs(den) > 1e-12:
        lag = lag + float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    return float(sr / max(lag, 1e-9))


@dataclass(frozen=True)
class TempoPlan:
    """How a source tempo is mapped onto the target grid."""

    source_bpm: float
    target_bpm: float
    beat_multiple: float     # target beats occupied by one source beat (0.5, 1 or 2)
    ratio: float             # playback speed factor applied to the source
    warning: str | None = None

    @property
    def interpretation(self) -> str:
        return {0.5: "double-time", 1.0: "straight", 2.0: "half-time"}.get(
            self.beat_multiple, f"x{self.beat_multiple}")

    def to_dict(self) -> dict:
        return {
            "source_bpm": round(self.source_bpm, 2),
            "target_bpm": round(self.target_bpm, 2),
            "beat_multiple": self.beat_multiple,
            "stretch_ratio": round(self.ratio, 4),
            "interpretation": self.interpretation,
            "warning": self.warning,
        }


def plan_tempo(source_bpm: float, target_bpm: float) -> TempoPlan:
    """Choose the metrical interpretation that needs the least time-stretching.

    Candidates are the straight mapping ``target/source``, the half-time mapping
    ``target/(2*source)`` and the double-time mapping ``2*target/source``; we
    take the smallest ``|log ratio|``. Laying a half-time vocal over a
    four-on-the-floor kick is a standard house move, and it beats stretching a
    vocal by 1.55x, which sounds like a chipmunk in a wind tunnel.
    """
    src = max(source_bpm, 1e-6)
    candidates = [
        (2.0, target_bpm / (2.0 * src)),    # source beat spans two target beats
        (1.0, target_bpm / src),
        (0.5, (2.0 * target_bpm) / src),
    ]
    multiple, ratio = min(candidates, key=lambda c: abs(math.log(max(c[1], 1e-9))))
    warning = None
    if ratio > MAX_COMFORTABLE_RATIO or ratio < MIN_COMFORTABLE_RATIO:
        warning = (f"stretch ratio {ratio:.3f} is outside the comfortable "
                   f"{MIN_COMFORTABLE_RATIO}-{MAX_COMFORTABLE_RATIO} window; "
                   "expect audible time-stretch artefacts")
    return TempoPlan(source_bpm=source_bpm, target_bpm=target_bpm,
                     beat_multiple=multiple, ratio=ratio, warning=warning)
