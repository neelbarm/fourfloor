"""Sidechain ducking, saturation, limiting and the master chain."""

from __future__ import annotations

import numpy as np

from . import filters as FL


def sidechain_envelope(n: int, sr: int, trigger_times: np.ndarray, depth: float = 0.65,
                       attack: float = 0.004, release: float = 0.22,
                       shape: float = 2.0) -> np.ndarray:
    """Kick-triggered ducking envelope in [1-depth, 1].

    Each trigger drops the gain to ``1 - depth`` over ``attack`` seconds, then
    recovers over ``release`` seconds along a power curve. ``shape`` > 1 gives
    the slow-then-fast recovery that reads as "pumping" rather than as a gate.
    """
    env = np.ones(n, dtype=np.float32)
    a = max(1, int(attack * sr))
    r = max(1, int(release * sr))
    duck = np.concatenate([
        np.linspace(1.0, 1.0 - depth, a, dtype=np.float32),
        (1.0 - depth) + depth * (np.linspace(0.0, 1.0, r, dtype=np.float32) ** shape),
    ])
    for t in trigger_times:
        s = int(round(t * sr))
        if s >= n:
            continue
        s = max(0, s)
        e = min(n, s + len(duck))
        env[s:e] = np.minimum(env[s:e], duck[: e - s])
    return env


def apply_sidechain(x: np.ndarray, env: np.ndarray) -> np.ndarray:
    """Multiply a buffer by a ducking envelope (broadcast over channels)."""
    e = env[: len(x)]
    if len(e) < len(x):
        e = np.concatenate([e, np.ones(len(x) - len(e), dtype=np.float32)])
    return (x * (e[:, None] if x.ndim == 2 else e)).astype(np.float32)


def saturate(x: np.ndarray, drive: float = 1.4) -> np.ndarray:
    """Odd-harmonic soft clip. Normalised so unity input stays near unity out."""
    if drive <= 1.0:
        return x
    return (np.tanh(drive * x) / np.tanh(drive)).astype(np.float32)


def soft_limit(x: np.ndarray, ceiling: float = 0.891, sr: int = 44100,
               release: float = 0.05) -> np.ndarray:
    """Smooth gain-reduction limiter (no lookahead).

    A per-sample required-gain curve is computed from the rectified peak, then
    smoothed with a one-pole release so gain changes are gradual; the attack is
    instantaneous, which for a signal that is already close to the ceiling
    produces a fraction of a dB of transient overshoot at most, removed by the
    final trim in ``master``.
    """
    mag = np.max(np.abs(x), axis=1) if x.ndim == 2 else np.abs(x)
    need = np.minimum(1.0, ceiling / np.maximum(mag, 1e-9))
    alpha = float(np.exp(-1.0 / max(release * sr, 1.0)))
    gain = np.empty_like(need)
    g = 1.0
    for i, v in enumerate(need):
        g = v if v < g else alpha * g + (1.0 - alpha) * v   # fast down, slow up
        gain[i] = g
    return (x * (gain[:, None] if x.ndim == 2 else gain)).astype(np.float32)


def normalize_peak(x: np.ndarray, target_db: float = -1.0) -> np.ndarray:
    """Scale so the absolute peak sits at ``target_db`` dBFS."""
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak <= 1e-9:
        return x
    return (x * (10.0 ** (target_db / 20.0) / peak)).astype(np.float32)


#: Median band balance (fraction of total power) measured across a reference
#: set of commercial house remixes, used as the matched-EQ target. Bands are
#: 40-80, 80-250, 250-2k, 2k-8k and 8k-20k Hz.
HOUSE_BALANCE = ((40.0, 80.0, 0.40), (80.0, 250.0, 0.23), (250.0, 2000.0, 0.21),
                 (2000.0, 8000.0, 0.066), (8000.0, 20000.0, 0.017))


def band_balance(x: np.ndarray, sr: int) -> list[float]:
    """Fraction of total power in each ``HOUSE_BALANCE`` band."""
    mono = x.mean(axis=1) if x.ndim == 2 else x
    n = 1 << 15
    if len(mono) < n:
        n = 1 << int(np.floor(np.log2(max(len(mono), 16))))
    starts = range(0, max(len(mono) - n, 1), max(n, 1))
    acc = np.zeros(len(HOUSE_BALANCE))
    frames = 0
    win = np.hanning(n)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    for a in starts:
        seg = mono[a:a + n]
        if len(seg) < n:
            break
        power = np.abs(np.fft.rfft(seg * win)) ** 2
        total = power.sum()
        if total <= 0:
            continue
        for i, (lo, hi, _) in enumerate(HOUSE_BALANCE):
            acc[i] += power[(freqs >= lo) & (freqs < hi)].sum() / total
        frames += 1
    return list(acc / max(frames, 1))


def match_balance(x: np.ndarray, sr: int, max_db: float = 5.5,
                  strength: float = 0.62) -> np.ndarray:
    """Nudge a mix toward the reference house band balance with gentle shelves.

    A crude matched EQ: measure how far each band sits from the reference in dB,
    scale it by ``strength`` (matching fully would flatten the track's own
    character), clamp to ``max_db`` and apply one shelf or bell per band. This is what keeps a
    synthesised kit from burying the source under sub.
    """
    have = band_balance(x, sr)
    y = x
    shapes = ("lowshelf", "peak", "peak", "peak", "highshelf")
    for (lo, hi, want), got, shape in zip(HOUSE_BALANCE, have, shapes):
        if got <= 1e-9:
            continue
        delta = float(np.clip(10.0 * np.log10(want / got) * strength, -max_db, max_db))
        if abs(delta) < 0.25:
            continue
        centre = float(np.sqrt(lo * hi))
        if shape == "lowshelf":
            y = FL.apply(y, "lowshelf", sr, hi, q=0.707, gain_db=delta)
        elif shape == "highshelf":
            y = FL.apply(y, "highshelf", sr, lo, q=0.707, gain_db=delta)
        else:
            y = FL.apply(y, "peak", sr, centre, q=0.75, gain_db=delta)
    return y.astype(np.float32)


def master(x: np.ndarray, sr: int, peak_db: float = -1.0, drive: float = 1.35,
           rms_target_db: float | None = -8.5, match: bool = True) -> np.ndarray:
    """Master bus: corrective EQ, matched EQ, saturation, limiter, peak trim.

    The static EQ is a 25 Hz rumble cut, a small kick-seating shelf, a mud cut at
    350 Hz and an air shelf; ``match_balance`` then pulls the spectrum toward the
    reference house balance. Gain is staged into the limiter so the drops land
    near ``rms_target_db`` before the final peak normalisation.
    """
    y = FL.apply(x, "highpass", sr, 28.0, q=0.707, order=2)
    y = FL.apply(y, "lowshelf", sr, 80.0, q=0.707, gain_db=0.5)
    y = FL.apply(y, "peak", sr, 350.0, q=1.0, gain_db=-1.4)     # clear the mud
    y = FL.apply(y, "highshelf", sr, 11000.0, q=0.707, gain_db=1.6)
    if match:
        y = match_balance(y, sr)

    if rms_target_db is not None:
        rms = float(np.sqrt(np.mean(np.square(y)))) if len(y) else 0.0
        if rms > 1e-7:
            # aim a little under target: saturation and limiting add ~1 dB back
            want = 10.0 ** ((rms_target_db - 1.0) / 20.0)
            y = y * float(np.clip(want / rms, 0.25, 4.0))

    y = saturate(y, drive)
    y = soft_limit(y, ceiling=10.0 ** (peak_db / 20.0) * 0.96, sr=sr)
    return normalize_peak(y, peak_db)
