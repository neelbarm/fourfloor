"""Bass and chord-stab synthesis driven by the source's detected chords."""

from __future__ import annotations

import numpy as np

from ..dsp import filters as FL


#: How much of a bass part's sub-band attack energy may land on the beat before
#: it counts as a second kick drum. Measured on five records warped to 128 BPM:
#: three house remixes and a lo-fi track put 5-25% of it there, because a bass
#: that belongs over four-on-the-floor dodges the kick. Don Toliver's *Body*
#: puts 54% of it there.
COLLISION_LIMIT = 0.35

#: A part has to be playing a rhythm before where it lands can be a problem.
#: A bass holding one note a bar starts that note on the downbeat, so 100% of
#: its attacks are "on the beat", and there is nothing wrong with it.
RHYTHMIC_PER_BAR = 4.0

#: ...unless it is also relentless. A part this busy and this far off the
#: eighth-note grid is a pattern rather than a bass line, whatever it does on
#: the beat.
BUSY_PER_BAR = 10.0
BUSY_OFF_EIGHTH = 0.55


def choose_bass(bass, sr: int, bpm: float) -> tuple[str, dict, str]:
    """Play the record's bass, or replace it with a sub that follows its pitch.

    Returns ``(mode, measurement, why)``. The measurement is
    :func:`analysis.alignment.bass_collision`; ``why`` is a sentence for the
    remix report, because a decision the engine makes on its own about how the
    low end of somebody's track is going to sound should say so out loud.
    """
    from ..analysis.alignment import bass_collision

    m = bass_collision(bass, sr, bpm)
    if m["rms_db"] < -45.0:
        return "sub", m, ("the separated bass is almost silent, so the sub is "
                          "synthesised from what pitch there is")
    if m["per_bar"] >= RHYTHMIC_PER_BAR and m["on_beat"] >= COLLISION_LIMIT:
        return "sub", m, (
            f"{m['on_beat']:.0%} of the source bass hits exactly where the kick "
            "goes -- that is an 808 doubling the kick, not a bass line, so the "
            "low end is re-synthesised on the house grid at the pitches it plays")
    if m["per_bar"] >= BUSY_PER_BAR and m["off_eighth"] >= BUSY_OFF_EIGHTH:
        return "sub", m, (
            f"the source bass plays {m['per_bar']:.0f} notes a bar with "
            f"{m['off_eighth']:.0%} of them off the eighth-note grid; that is a "
            "pattern rather than a bass line, so it is re-synthesised")
    if m["per_bar"] < RHYTHMIC_PER_BAR:
        return "source", m, (
            f"the source bass plays {m['per_bar']:.1f} notes a bar -- it is "
            "holding notes, not playing a pattern -- so it is kept as it is")
    return "source", m, (
        f"the source bass puts only {m['on_beat']:.0%} of itself on the beat, so "
        "it dodges the kick and is kept as it is")


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
