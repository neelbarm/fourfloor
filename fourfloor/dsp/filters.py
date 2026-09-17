"""Biquad filters, including blockwise sweeps with interpolated cutoff.

Coefficients follow Robert Bristow-Johnson's Audio EQ Cookbook. Sweeps are
rendered by splitting the signal into short blocks, recomputing the biquad per
block from an interpolated cutoff, and carrying the filter state across block
boundaries with ``lfilter``'s ``zi`` so there is no click at the seams.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

SWEEP_BLOCK = 256


def biquad(kind: str, sr: int, freq: float, q: float = 0.707,
           gain_db: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Audio EQ Cookbook biquad coefficients ``(b, a)``, normalised by a0."""
    freq = float(np.clip(freq, 10.0, sr * 0.49))
    w0 = 2.0 * np.pi * freq / sr
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2.0 * q)
    A = 10.0 ** (gain_db / 40.0)

    if kind == "lowpass":
        b = np.array([(1 - cw) / 2, 1 - cw, (1 - cw) / 2])
        a = np.array([1 + alpha, -2 * cw, 1 - alpha])
    elif kind == "highpass":
        b = np.array([(1 + cw) / 2, -(1 + cw), (1 + cw) / 2])
        a = np.array([1 + alpha, -2 * cw, 1 - alpha])
    elif kind == "bandpass":
        b = np.array([alpha, 0.0, -alpha])
        a = np.array([1 + alpha, -2 * cw, 1 - alpha])
    elif kind == "peak":
        b = np.array([1 + alpha * A, -2 * cw, 1 - alpha * A])
        a = np.array([1 + alpha / A, -2 * cw, 1 - alpha / A])
    elif kind == "lowshelf":
        sq = 2.0 * np.sqrt(A) * alpha
        b = A * np.array([(A + 1) - (A - 1) * cw + sq, 2 * ((A - 1) - (A + 1) * cw),
                          (A + 1) - (A - 1) * cw - sq])
        a = np.array([(A + 1) + (A - 1) * cw + sq, -2 * ((A - 1) + (A + 1) * cw),
                      (A + 1) + (A - 1) * cw - sq])
    elif kind == "highshelf":
        sq = 2.0 * np.sqrt(A) * alpha
        b = A * np.array([(A + 1) + (A - 1) * cw + sq, -2 * ((A - 1) + (A + 1) * cw),
                          (A + 1) + (A - 1) * cw - sq])
        a = np.array([(A + 1) - (A - 1) * cw + sq, 2 * ((A - 1) - (A + 1) * cw),
                      (A + 1) - (A - 1) * cw - sq])
    else:
        raise ValueError(f"unknown filter kind: {kind}")
    return b / a[0], a / a[0]


def apply(x: np.ndarray, kind: str, sr: int, freq: float, q: float = 0.707,
          gain_db: float = 0.0, order: int = 1) -> np.ndarray:
    """Apply a static biquad ``order`` times (12 dB/oct per pass)."""
    b, a = biquad(kind, sr, freq, q, gain_db)
    y = x
    for _ in range(max(1, order)):
        y = sps.lfilter(b, a, y, axis=0)
    return y.astype(np.float32)


def sweep(x: np.ndarray, kind: str, sr: int, cutoffs: np.ndarray, q: float = 0.707,
          order: int = 1, block: int = SWEEP_BLOCK) -> np.ndarray:
    """Time-varying filter: ``cutoffs`` is a per-sample (or coarser) cutoff curve.

    The curve is resampled to one value per block and the biquad is rebuilt each
    block; filter state carries over so the transition is continuous.
    """
    n = len(x)
    if n == 0:
        return x
    cutoffs = np.asarray(cutoffs, dtype=float)
    if cutoffs.ndim == 0:
        cutoffs = np.full(n, float(cutoffs))
    if len(cutoffs) != n:
        cutoffs = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(cutoffs)), cutoffs)

    chans = 1 if x.ndim == 1 else x.shape[1]
    out = np.zeros_like(x, dtype=np.float32)
    states = [[np.zeros(2) if chans == 1 else np.zeros((2, chans))
               for _ in range(max(1, order))]]
    zi = states[0]
    for start in range(0, n, block):
        end = min(n, start + block)
        f = float(np.mean(cutoffs[start:end]))
        b, a = biquad(kind, sr, f, q)
        seg = x[start:end]
        for k in range(max(1, order)):
            seg, zi[k] = sps.lfilter(b, a, seg, axis=0, zi=zi[k])
        out[start:end] = seg
    return out


def exp_curve(n: int, start: float, end: float) -> np.ndarray:
    """Exponential (musically linear) interpolation between two frequencies."""
    return np.exp(np.linspace(np.log(max(start, 1.0)), np.log(max(end, 1.0)), max(n, 1)))
