"""Bass and chord-stab synthesis driven by the source's detected chords."""

from __future__ import annotations

import numpy as np

from ..dsp import filters as FL


def note_hz(pitch_class: int, octave: int = 2) -> float:
    """Frequency of a pitch class in a given octave (C2 = 65.41 Hz)."""
    midi = 12 * (octave + 1) + pitch_class
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def _saw(freq: float, n: int, sr: int, detune: float = 0.0) -> np.ndarray:
    """Band-limited-ish sawtooth by additive synthesis up to Nyquist."""
    t = np.arange(n) / sr
    f = freq * (1.0 + detune)
    out = np.zeros(n, dtype=np.float32)
    h = 1
    while f * h < sr * 0.45 and h <= 24:
        out += (np.sin(2.0 * np.pi * f * h * t) / h).astype(np.float32)
        h += 1
    return out * (2.0 / np.pi)


def bass_note(sr: int, freq: float, seconds: float, cutoff: float = 1100.0,
              sub_mix: float = 0.28, drive: float = 1.8) -> np.ndarray:
    """One rolling-bass note: detuned saws through an envelope-swept low-pass.

    The filter opens on the attack and closes over the note, which is what gives
    a house bass its forward "plucked" motion; a sine an octave down supplies
    the sub that the saw's fundamental alone cannot.
    """
    n = max(8, int(seconds * sr))
    t = np.arange(n) / sr
    saw = 0.6 * _saw(freq, n, sr) + 0.4 * _saw(freq, n, sr, detune=0.006)
    sub = np.sin(2.0 * np.pi * freq * 0.5 * t).astype(np.float32) * sub_mix

    amp = np.exp(-t / max(seconds * 0.55, 0.04)).astype(np.float32)
    a = max(2, int(0.006 * sr))
    amp[:a] *= np.linspace(0.0, 1.0, a)
    amp[-min(n, 64):] *= np.linspace(1.0, 0.0, min(n, 64))

    fenv = cutoff * (0.35 + 0.65 * np.exp(-t / max(seconds * 0.4, 0.03)))
    sig = FL.sweep(saw, "lowpass", sr, np.maximum(fenv, freq * 2.2), q=1.1, order=2)
    out = (sig + sub) * amp
    out = np.tanh(drive * out) / np.tanh(drive)
    out = FL.apply(out, "highpass", sr, 38.0, q=0.707, order=2)
    return (out * 0.55).astype(np.float32)


def stab(sr: int, root_pc: int, minor: bool, seconds: float, octave: int = 4,
         cutoff: float = 2600.0) -> np.ndarray:
    """Short filtered triad stab for offbeat chords in the drops."""
    n = max(8, int(seconds * sr))
    t = np.arange(n) / sr
    intervals = (0, 3, 7, 10) if minor else (0, 4, 7, 11)
    sig = np.zeros(n, dtype=np.float32)
    for k, iv in enumerate(intervals):
        f = note_hz((root_pc + iv) % 12, octave + (1 if (root_pc + iv) >= 12 else 0))
        sig += _saw(f, n, sr, detune=0.004 * (k - 1.5)) * (0.9 ** k)
    amp = np.exp(-t / max(seconds * 0.3, 0.02)).astype(np.float32)
    a = max(2, int(0.004 * sr))
    amp[:a] *= np.linspace(0.0, 1.0, a)
    sig = FL.apply(sig, "lowpass", sr, cutoff, q=0.9, order=2)
    sig = FL.apply(sig, "highpass", sr, 220.0)
    out = sig * amp
    peak = float(np.max(np.abs(out)))
    return (out / peak * 0.3).astype(np.float32) if peak > 0 else out


def pick_bass_octave(root_pc: int) -> int:
    """Keep the bass fundamental inside 41-82 Hz, where a club system lives."""
    return 1 if note_hz(root_pc, 1) >= 41.0 else 2
