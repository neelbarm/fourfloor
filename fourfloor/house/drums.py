"""Synthesised house drum kit and 16-step patterns.

Every voice is built from scratch in numpy: no samples, no external kit. The
16-step velocity-pattern representation is ported from the sibling `groovebox`
project's `rhythm.ts`, where a pattern is a list of ``(step, velocity)`` pairs
over one bar of 16th notes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..dsp import filters as FL

Pattern = dict[str, list[tuple[int, float]]]


def _env(n: int, attack: int, decay: float, sr: int, curve: float = 1.0) -> np.ndarray:
    """Attack-decay envelope: linear attack, exponential decay."""
    t = np.arange(n) / sr
    e = np.exp(-t / max(decay, 1e-4)) ** curve
    a = min(max(attack, 1), n)
    e[:a] *= np.linspace(0.0, 1.0, a)
    return e.astype(np.float32)


def kick(sr: int, length: float = 0.55, f_start: float = 190.0, f_end: float = 50.0,
         pitch_decay: float = 0.028, amp_decay: float = 0.135, click: float = 0.3,
         drive: float = 2.6) -> np.ndarray:
    """Classic synthesised house kick.

    A sine whose frequency falls exponentially from ``f_start`` to ``f_end``
    (the pitch envelope is what makes it read as a drum rather than a bass
    note), an exponential amplitude envelope, a short broadband click for the
    beater attack, and tanh drive so it has harmonics that survive on a laptop
    speaker as well as sub that carries on a system.
    """
    n = int(length * sr)
    t = np.arange(n) / sr
    freq = f_end + (f_start - f_end) * np.exp(-t / pitch_decay)
    phase = 2.0 * np.pi * np.cumsum(freq) / sr
    body = np.sin(phase) * _env(n, 4, amp_decay, sr)
    sub = np.sin(2.0 * np.pi * np.cumsum(np.full(n, f_end * 0.92)) / sr) \
        * _env(n, 8, amp_decay * 1.15, sr) * 0.30

    rng = np.random.default_rng(11)
    nclick = int(0.006 * sr)
    tick = rng.standard_normal(nclick).astype(np.float32) * _env(nclick, 1, 0.0016, sr)
    tick = FL.apply(tick, "highpass", sr, 1200.0, order=2) * click

    out = np.zeros(n, dtype=np.float32)
    out[:n] = body + sub
    out[:nclick] += tick
    out = np.tanh(drive * out) / np.tanh(drive)
    out = FL.apply(out, "lowpass", sr, 7000.0)
    out[-64:] *= np.linspace(1.0, 0.0, 64)
    return (out * 0.95).astype(np.float32)


def clap(sr: int, spread: float = 0.011, bursts: int = 4, decay: float = 0.19,
         seed: int = 3) -> np.ndarray:
    """Hand clap: a short burst train (the "many hands" flam) into a noise tail.

    Bandpassed 900-3400 Hz, which is where a clap's formant sits, with a small
    synthesised room tail so it sits behind the kick instead of on top of it.
    """
    rng = np.random.default_rng(seed)
    n = int(0.42 * sr)
    out = np.zeros(n, dtype=np.float32)
    for i in range(bursts):
        pos = int(i * spread * sr * (1.0 + 0.25 * rng.random()))
        ln = int(0.012 * sr)
        if pos + ln >= n:
            break
        burst = rng.standard_normal(ln).astype(np.float32) * _env(ln, 1, 0.0035, sr)
        out[pos:pos + ln] += burst * (1.0 - 0.18 * i)
    tail_start = int(bursts * spread * sr)
    tail = rng.standard_normal(n - tail_start).astype(np.float32) \
        * _env(n - tail_start, 2, decay, sr) * 0.42
    out[tail_start:] += tail
    out = FL.apply(out, "bandpass", sr, 1500.0, q=0.62, order=2)
    out = FL.apply(out, "peak", sr, 2400.0, q=1.2, gain_db=4.0)
    peak = float(np.max(np.abs(out)))
    return (out / peak * 0.9).astype(np.float32) if peak > 0 else out


def hat(sr: int, open_: bool = False, seed: int = 5) -> np.ndarray:
    """Hi-hat from highpassed noise with a metallic ring.

    Six detuned square partials at inharmonic ratios (the 808/909 trick) are
    mixed under the noise so the hat has a pitch centre instead of reading as a
    burst of static.
    """
    rng = np.random.default_rng(seed)
    decay = 0.34 if open_ else 0.032
    n = int((0.7 if open_ else 0.13) * sr)
    t = np.arange(n) / sr
    noise = rng.standard_normal(n).astype(np.float32)
    ring = np.zeros(n, dtype=np.float32)
    for r in (1.0, 1.4471, 1.6170, 1.9265, 2.5028, 2.6637):
        ring += np.sign(np.sin(2.0 * np.pi * 320.0 * r * t)).astype(np.float32)
    sig = 0.72 * noise + 0.18 * ring / 6.0
    sig *= _env(n, 1, decay, sr)
    sig = FL.apply(sig, "highpass", sr, 7200.0 if not open_ else 6200.0, order=3)
    sig = FL.apply(sig, "peak", sr, 9500.0, q=0.8, gain_db=3.0)
    peak = float(np.max(np.abs(sig)))
    return (sig / peak * (0.46 if open_ else 0.52)).astype(np.float32) if peak > 0 else sig


def shaker(sr: int, seed: int = 9) -> np.ndarray:
    """Shaker/ride texture: brighter, softer-attacked noise for the top end."""
    rng = np.random.default_rng(seed)
    n = int(0.16 * sr)
    sig = rng.standard_normal(n).astype(np.float32) * _env(n, int(0.004 * sr), 0.04, sr)
    sig = FL.apply(sig, "highpass", sr, 5000.0, order=2)
    sig = FL.apply(sig, "lowpass", sr, 13000.0)
    peak = float(np.max(np.abs(sig)))
    return (sig / peak * 0.3).astype(np.float32) if peak > 0 else sig


def tom(sr: int, freq: float = 150.0, seed: int = 13) -> np.ndarray:
    """Pitched tom for phrase-end fills."""
    n = int(0.3 * sr)
    t = np.arange(n) / sr
    f = freq * (1.0 + 0.6 * np.exp(-t / 0.05))
    sig = np.sin(2.0 * np.pi * np.cumsum(f) / sr).astype(np.float32) * _env(n, 3, 0.11, sr)
    rng = np.random.default_rng(seed)
    nn = int(0.004 * sr)
    sig[:nn] += rng.standard_normal(nn).astype(np.float32) * 0.3 * _env(nn, 1, 0.002, sr)
    return (np.tanh(1.8 * sig) * 0.6).astype(np.float32)


def riser_noise(sr: int, seconds: float, f0: float = 300.0, f1: float = 12000.0,
                seed: int = 17) -> np.ndarray:
    """Filtered-noise riser with a rising bandpass and a rising sine on top."""
    n = int(seconds * sr)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n).astype(np.float32)
    cut = FL.exp_curve(n, f0, f1)
    swept = FL.sweep(noise, "bandpass", sr, cut, q=0.8, order=2)
    t = np.arange(n) / sr
    tone_f = FL.exp_curve(n, 220.0, 1760.0)
    tone = np.sin(2.0 * np.pi * np.cumsum(tone_f) / sr).astype(np.float32) * 0.22
    ramp = (np.linspace(0.0, 1.0, n) ** 2.1).astype(np.float32)
    out = (swept * 1.3 + tone) * ramp
    peak = float(np.max(np.abs(out)))
    return (out / peak * 0.5).astype(np.float32) if peak > 0 else out


def impact(sr: int, seed: int = 23) -> np.ndarray:
    """Downbeat impact: a sub drop plus a reversed-feel noise swell."""
    n = int(1.6 * sr)
    t = np.arange(n) / sr
    f = 120.0 * np.exp(-t / 0.22) + 34.0
    sub = np.sin(2.0 * np.pi * np.cumsum(f) / sr).astype(np.float32) * np.exp(-t / 0.5)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n).astype(np.float32) * np.exp(-t / 0.28)
    noise = FL.apply(noise, "lowpass", sr, 4000.0, order=2)
    out = 0.85 * sub + 0.4 * noise
    peak = float(np.max(np.abs(out)))
    return (out / peak * 0.72).astype(np.float32) if peak > 0 else out


# ---------------------------------------------------------------------------
# patterns
# ---------------------------------------------------------------------------

def _every(n: int, vel: float, offset: int = 0, jitter: float = 0.0) -> list[tuple[int, float]]:
    return [(s, vel * (1.0 - jitter * ((s // n) % 2))) for s in range(offset, 16, n)]


PATTERNS: dict[str, Pattern] = {
    # Four on the floor, offbeat open hats, clap on 2 and 4.
    "drop": {
        "kick": _every(4, 1.0),
        "clap": [(4, 0.92), (12, 0.92)],
        "hat": [(s, 0.30 if s % 4 == 0 else (0.46 if s % 2 else 0.24)) for s in range(16)],
        "ohat": _every(4, 0.55, 2),
        "shaker": _every(2, 0.3, 1),
        "sub": _every(4, 0.9),
    },
    "drop_var": {
        "kick": _every(4, 1.0),
        "clap": [(4, 0.92), (12, 0.92), (15, 0.34)],
        "hat": [(s, 0.32 if s % 4 == 0 else (0.5 if s % 2 else 0.26)) for s in range(16)],
        "ohat": _every(4, 0.6, 2),
        "shaker": _every(1, 0.18),
        "sub": _every(4, 0.9),
    },
    "intro": {
        "kick": _every(4, 0.86),
        "clap": [],
        "hat": _every(4, 0.3, 2),
        "ohat": [],
        "shaker": _every(4, 0.2, 1),
        "sub": _every(4, 0.6),
    },
    "intro_full": {
        "kick": _every(4, 0.94),
        "clap": [(12, 0.6)],
        "hat": _every(2, 0.36, 1),
        "ohat": _every(4, 0.42, 2),
        "shaker": _every(2, 0.22, 1),
        "sub": _every(4, 0.75),
    },
    "build": {
        "kick": _every(4, 0.96),
        "clap": [(4, 0.7), (12, 0.7)],
        "hat": [(s, 0.4) for s in range(16)],
        "ohat": _every(4, 0.5, 2),
        "shaker": _every(1, 0.24),
        "sub": _every(4, 0.7),
    },
    "breakdown": {
        "kick": [],
        "clap": [(12, 0.42)],
        "hat": _every(4, 0.2, 2),
        "ohat": [],
        "shaker": _every(2, 0.16, 1),
        "sub": [],
    },
    "outro": {
        "kick": _every(4, 0.9),
        "clap": [(4, 0.5)],
        "hat": _every(2, 0.24, 1),
        "ohat": _every(8, 0.32, 2),
        "shaker": [],
        "sub": _every(4, 0.7),
    },
}


@dataclass
class Kit:
    """Pre-rendered one-shots, rendered once and reused across the whole track."""

    sr: int
    swing: float = 0.08
    samples: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.samples:
            self.samples = {
                "kick": kick(self.sr),
                "clap": clap(self.sr),
                "hat": hat(self.sr, False),
                "ohat": hat(self.sr, True),
                "shaker": shaker(self.sr),
                "tom": tom(self.sr),
            }

    def step_time(self, bar_start: float, step: int, step_dur: float) -> float:
        """Time of a 16th step, with swing applied to the odd 16ths.

        Swing delays every other 16th by ``swing`` of a step, the standard
        MPC-style shuffle; 0 is straight, 0.66 would be full triplet feel.
        """
        off = self.swing * step_dur if step % 2 == 1 else 0.0
        return bar_start + step * step_dur + off
