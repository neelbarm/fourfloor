"""Synthesised reverb: a Schroeder network and an FFT-convolution plate IR."""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

from . import filters as FL

# Schroeder's original comb delays, scaled from 25 kHz to the working rate.
_COMB_MS = (29.7, 37.1, 41.1, 43.7)
_ALLPASS_MS = (5.0, 1.7)


def synth_ir(sr: int, seconds: float = 1.8, decay: float = 4.5, damping: float = 6500.0,
             seed: int = 7) -> np.ndarray:
    """Exponentially decaying, frequency-damped noise burst: a plate-ish IR.

    Velvet-noise-style sparse early reflections are added in the first 60 ms so
    the tail has some pre-delay structure instead of starting as a wall.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    ir = rng.standard_normal((n, 2)) * np.exp(-decay * t)[:, None]
    early = np.zeros((n, 2))
    for _ in range(14):
        pos = int(rng.uniform(0.004, 0.06) * sr)
        if pos < n:
            early[pos, rng.integers(0, 2)] += rng.uniform(-0.6, 0.6)
    ir = ir * 0.5 + early
    ir = FL.apply(ir, "lowpass", sr, damping, order=2)
    ir = FL.apply(ir, "highpass", sr, 180.0)
    ir[:8] *= np.linspace(0.0, 1.0, 8)[:, None]
    peak = float(np.max(np.abs(ir)))
    return (ir / peak * 0.5).astype(np.float32) if peak > 0 else ir.astype(np.float32)


def convolve(x: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """FFT convolution of a stereo buffer with a stereo IR, truncated to len(x)."""
    n = len(x)
    xs = x if x.ndim == 2 else np.stack([x, x], axis=1)
    out = np.zeros((n, 2), dtype=np.float32)
    for c in range(2):
        y = sps.fftconvolve(xs[:, c], ir[:, min(c, ir.shape[1] - 1)])[:n]
        out[: len(y), c] = y
    return out


def schroeder(x: np.ndarray, sr: int, room: float = 0.82, damping: float = 0.35,
              mix: float = 0.3) -> np.ndarray:
    """Schroeder reverberator: 4 parallel damped combs into 2 series allpasses.

    M. R. Schroeder, "Natural sounding artificial reverberation", JAES 10(3),
    1962. Cheap, and for a vocal throw in a house track it is indistinguishable
    from anything fancier once the kick is on top of it.

    Each stage is expressed as a sparse IIR so scipy runs the recursion in C: a
    feedback comb is ``a = [1, 0..0, -g]`` and an allpass is
    ``b = [-g, 0..0, 1], a = [1, 0..0, -g]``. Damping is applied once to the comb
    sum rather than inside each feedback path, which is a close approximation
    and orders of magnitude faster than a per-sample Python loop.
    """
    xs = x if x.ndim == 2 else np.stack([x, x], axis=1)
    n = len(xs)
    wet = np.zeros_like(xs, dtype=np.float32)
    for ch in range(2):
        sig = xs[:, ch].astype(np.float64)
        acc = np.zeros(n)
        for ms in _COMB_MS:
            # decorrelate the channels by detuning the delay lines slightly
            d = max(1, int(ms * 0.001 * sr * (1.0 + 0.017 * ch)))
            a = np.zeros(d + 1)
            a[0], a[d] = 1.0, -float(np.clip(room, 0.0, 0.97))
            acc += sps.lfilter([1.0], a, sig)
        acc /= len(_COMB_MS)
        for ms in _ALLPASS_MS:
            d = max(1, int(ms * 0.001 * sr * (1.0 + 0.017 * ch)))
            g = 0.7
            b = np.zeros(d + 1)
            b[0], b[d] = -g, 1.0
            a = np.zeros(d + 1)
            a[0], a[d] = 1.0, -g
            acc = sps.lfilter(b, a, acc)
        wet[:, ch] = acc
    cutoff = float(np.clip(12000.0 * (1.0 - damping), 800.0, 16000.0))
    wet = FL.apply(wet, "lowpass", sr, cutoff, order=2)
    return ((1.0 - mix) * xs + mix * wet).astype(np.float32)
