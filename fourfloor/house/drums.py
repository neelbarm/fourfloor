"""Synthesised house drum kit and 16-step patterns.

Every voice is built from scratch in numpy: no samples, no external kit. The
16-step velocity-pattern representation is ported from the sibling `groovebox`
project's `rhythm.ts`, where a pattern is a list of ``(step, velocity)`` pairs
over one bar of 16th notes.

The voice parameters are not taste: they are fitted to measurements of the
drum stems of six commercial house remixes, isolated with Demucs and analysed
hit by hit. :data:`REFERENCE` records those measured ranges, :func:`kit_report`
measures the synthesised kit the same way, and the two are compared in the
tests. The short version of what the references say, and what this module had
to change to meet it:

* a house kick is a **short** kick -- its 40-100 Hz tail is 20 dB down within
  57-130 ms. The old kick rang for 310 ms, which is what made every render
  sound like mud rather than a record.
* the fundamental settles at 42-46 Hz, not 55+.
* closed hats are centred around 7.2-9.0 kHz. The old hat sat at 13.4 kHz,
  which reads as static rather than as a hi-hat.
* claps have an audible room: 30-60 ms of tail, not a dry 8 ms tick.
* the hat line is weighted, not flat: downbeats are the loudest slot, the "a"
  16ths carry the push and the "e" 16ths are the quiet ones.
* they are **straight**. Every reference quantises its 16ths to within 3 ms,
  so :class:`Kit` defaults to no swing at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sps

from ..dsp import filters as FL

Pattern = dict[str, list[tuple[int, float]]]


#: Measured ranges from the reference drum stems, as ``(low, high)``.
#:
#: Sources: six commercial house remixes, Demucs ``htdemucs`` drum stems,
#: measured twice over -- once per isolated hit, once over whole 16-bar windows
#: -- and reconciled. The voice entries are checked against a single synthesised
#: one-shot; the ``band_*`` entries against a rendered drum stem, so they stay
#: comparable with a commercial one. Levels are in dB, times in ms, frequencies
#: in Hz. The ranges are wider than the corpus medians on purpose: they are a
#: regression fence, not a target. The medians the kit is actually tuned to are
#: kick 43 Hz / 64 ms, clap 7.1 kHz / 46 ms / -4.5 dB, hat 7.6 kHz / -12.8 dB.
REFERENCE: dict[str, tuple[float, float]] = {
    "kick_f0_hz": (41.0, 53.0),
    "kick_sub_t20_ms": (45.0, 135.0),
    "kick_click_db": (-39.0, -18.0),
    "clap_centroid_hz": (4000.0, 9500.0),
    "clap_t20_ms": (28.0, 110.0),
    "clap_vs_kick_db": (-12.0, 3.0),
    "hat_centroid_hz": (7000.0, 9200.0),
    "hat_t20_ms": (28.0, 90.0),
    "hat_vs_kick_db": (-16.0, -7.0),
    "ohat_t20_ms": (60.0, 130.0),
    "swing": (-0.06, 0.06),
    "onsets_per_bar": (5.5, 19.0),
    # band balance of a whole drum stem, dB relative to its broadband energy
    "band_sub_db": (-7.0, -1.0),
    "band_low_db": (-10.5, -3.5),
    "band_lowmid_db": (-15.5, -6.5),
    "band_mid_db": (-20.0, -8.5),
    "band_himid_db": (-24.5, -10.5),
    "band_pres_db": (-19.0, -11.0),
    # the kit sits at the bright end of the corpus on purpose: the hat level is
    # pinned to the references' -12.8 dB peak ratio against the kick, and 16ths
    # at that level put more 6-16 kHz energy into the stem than a corpus whose
    # hat lines are only two thirds occupied.
    "band_air_db": (-18.0, -9.5),
}

#: Analysis bands for :func:`band_balance`, matching the reference analysis.
BANDS: tuple[tuple[str, float, float], ...] = (
    ("sub", 20.0, 60.0),
    ("low", 60.0, 120.0),
    ("lowmid", 120.0, 250.0),
    ("mid", 250.0, 800.0),
    ("himid", 800.0, 2500.0),
    ("pres", 2500.0, 6000.0),
    ("air", 6000.0, 16000.0),
)


def _env(n: int, attack: int, decay: float, sr: int, curve: float = 1.0) -> np.ndarray:
    """Attack-decay envelope: linear attack, exponential decay."""
    t = np.arange(n) / sr
    e = np.exp(-t / max(decay, 1e-4)) ** curve
    a = min(max(attack, 1), n)
    e[:a] *= np.linspace(0.0, 1.0, a)
    return e.astype(np.float32)


def _norm(x: np.ndarray, peak: float) -> np.ndarray:
    """Scale to an exact peak, and fade the tail so a truncated voice cannot click."""
    p = float(np.max(np.abs(x)))
    out = (x / p * peak) if p > 0 else x
    tail = min(64, len(out))
    if tail > 1:
        out[-tail:] *= np.linspace(1.0, 0.0, tail)
    return np.asarray(out, dtype=np.float32)


def _room(n: int, sr: int, decay: float, density: float = 0.022,
          seed: int = 101) -> np.ndarray:
    """A sparse exponentially-decaying impulse train: a cheap synthetic room.

    Convolving a dry hit with this is what turns a burst of noise into a clap
    recorded in a space. Real claps in records always have one; the measured
    30-60 ms of clap tail in the references is mostly room, not the hands.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    taps = (rng.random(n) < density).astype(np.float32)
    taps[0] = 1.0
    ir = taps * rng.standard_normal(n).astype(np.float32) * np.exp(-t / max(decay, 1e-4))
    return ir.astype(np.float32)


# ---------------------------------------------------------------------------
# voices
# ---------------------------------------------------------------------------

def kick(sr: int, length: float = 0.46, f_start: float = 152.0, f_end: float = 43.0,
         pitch_decay: float = 0.019, amp_decay: float = 0.022, click: float = 0.3,
         drive: float = 2.0, sub_decay: float | None = 0.028, sub_level: float = 0.68,
         punch: float = 0.55, seed: int = 11) -> np.ndarray:
    """Synthesised house kick, fitted to the reference stems.

    The pitch envelope runs in two stages -- a very fast drop out of
    ``f_start`` for the beater snap, then a slower settle onto ``f_end`` -- so
    the attack reads as a drum and the tail reads as a note. ``amp_decay``
    is deliberately short: the references put the 40-100 Hz tail 20 dB down in
    57-130 ms, and the single biggest fault of the previous kick was a 310 ms
    ring that smeared the whole low end.

    ``sub_decay`` governs the sine tail that
    carries on a system after the body has gone; ``punch`` bends the body
    envelope so the first few milliseconds fall faster than the tail, which is
    what a compressed acoustic kick does. The transient is three layers -- a
    noise tick, a damped beater tone and a bandpassed snap -- because a single
    highpassed noise burst is audibly a single highpassed noise burst.
    """
    n = max(int(length * sr), 64)
    t = np.arange(n) / sr
    pd = max(pitch_decay, 1e-4)
    # two-stage pitch envelope: snap out of f_start, then settle onto f_end
    freq = (f_end
            + (f_start - f_end) * 0.62 * np.exp(-t / (pd * 0.34))
            + (f_start - f_end) * 0.38 * np.exp(-t / (pd * 2.6)))
    phase = 2.0 * np.pi * np.cumsum(freq) / sr
    body = np.sin(phase) * _env(n, 3, amp_decay, sr)
    # the punch stage: an extra fast-decaying copy of the body, so the first
    # ~15 ms is hotter than the tail without lengthening the tail.
    body = body * (1.0 - punch) + body * punch * 2.0 * _env(n, 3, amp_decay * 0.34, sr)

    sd = amp_decay * 1.45 if sub_decay is None else sub_decay
    # exactly f_end, not a detuned copy: a sub a fraction of a hertz away from
    # the body beats against it, and a 0.6 Hz beat over a 460 ms kick is a
    # wobble in the tail that reads as a tuning problem rather than as weight.
    sub = (np.sin(2.0 * np.pi * np.cumsum(np.full(n, f_end)) / sr)
           * _env(n, max(4, int(0.004 * sr)), sd, sr) * sub_level)

    rng = np.random.default_rng(seed)
    n_tick = min(n, int(0.008 * sr))
    tt = np.arange(n_tick) / sr
    tick = rng.standard_normal(n_tick).astype(np.float32) * _env(n_tick, 1, 0.0013, sr)
    tick = FL.apply(tick, "highpass", sr, 1400.0, order=2)
    beater = (np.sin(2.0 * np.pi * 2150.0 * tt).astype(np.float32)
              * _env(n_tick, 1, 0.0022, sr) * 0.55)
    snap = FL.apply(rng.standard_normal(n_tick).astype(np.float32)
                    * _env(n_tick, 1, 0.0007, sr), "bandpass", sr, 4200.0, q=0.9, order=2)
    transient = (tick + beater + snap * 0.8) * click

    out = np.zeros(n, dtype=np.float32)
    out += body.astype(np.float32) + sub.astype(np.float32)
    out[:n_tick] += transient
    # asymmetric drive: a little second harmonic before the tanh, which is what
    # gives a kick weight on a speaker that cannot reproduce 45 Hz at all.
    out = out + 0.07 * out * np.abs(out)
    out = np.tanh(drive * out) / np.tanh(drive)
    out = FL.apply(out, "highpass", sr, 27.0, order=2)
    out = FL.apply(out, "lowpass", sr, 7200.0)
    # 0.92, not 1.0: the engine jitters velocity by up to +8%, and a kick that
    # peaks at unity here would clip its own stem on the loud hits.
    return _norm(out, 0.92)


def clap(sr: int, spread: float = 0.0105, bursts: int = 4, decay: float = 0.020,
         seed: int = 3, room: float = 0.62, bright: float = 1.0) -> np.ndarray:
    """Hand clap: a burst train (the "many hands" flam) into a real room tail.

    Three fast pre-hits and a louder main hit, bandpassed around the 1-4 kHz
    clap formant, then convolved with a sparse decaying impulse train so the
    tail is a room rather than a fade on white noise. ``room`` sets how much of
    that convolved signal is mixed back in; it is what takes the measured decay
    from 8 ms (the old dry clap) into the 30-60 ms the references show.
    """
    rng = np.random.default_rng(seed)
    n = int(0.42 * sr)
    dry = np.zeros(n, dtype=np.float32)
    nb = max(1, bursts)
    for i in range(nb):
        pos = int(i * spread * sr * (1.0 + 0.22 * rng.random()))
        ln = int((0.010 if i < nb - 1 else 0.020) * sr)
        if pos + ln >= n:
            break
        burst = rng.standard_normal(ln).astype(np.float32) \
            * _env(ln, 1, 0.0030 if i < nb - 1 else 0.0075, sr)
        # the last hit is the one the ear hears as "the" clap
        dry[pos:pos + ln] += burst * (0.62 - 0.10 * i if i < nb - 1 else 1.0)

    tail_start = int(nb * spread * sr)
    tail = rng.standard_normal(n - tail_start).astype(np.float32) \
        * _env(n - tail_start, 2, max(decay, 1e-4), sr) * 0.30
    dry[tail_start:] += tail

    ir = _room(int(0.16 * sr), sr, max(decay, 1e-4) * 1.5, 0.026, seed + 71)
    wet = sps.fftconvolve(dry, ir)[:n].astype(np.float32)
    wpk = float(np.max(np.abs(wet)))
    if wpk > 0:
        wet /= wpk
    out = dry + wet * room

    out = FL.apply(out, "bandpass", sr, 2600.0, q=0.45, order=2)
    out = FL.apply(out, "peak", sr, 2300.0, q=1.1, gain_db=4.0)
    out = FL.apply(out, "peak", sr, 6200.0, q=0.6, gain_db=11.0 * bright)
    out = FL.apply(out, "highshelf", sr, 6400.0, q=0.707, gain_db=9.0 * bright)
    out = FL.apply(out, "highpass", sr, 380.0, order=2)
    # the references put the clap 4.5 dB under the kick peak, not level with it
    return _norm(out, 0.60)


def snare(sr: int, decay: float = 0.045, tone: float = 0.38, seed: int = 7,
          bright: float = 0.55) -> np.ndarray:
    """Layering snare for the 2 and 4, and for build rolls.

    Two detuned shell modes under a bandpassed noise body plus a short wire
    buzz. Sits under the clap in a drop; carries the roll on its own in a
    build.
    """
    rng = np.random.default_rng(seed)
    n = int(0.28 * sr)
    t = np.arange(n) / sr
    shell = (np.sin(2.0 * np.pi * 183.0 * t) + 0.7 * np.sin(2.0 * np.pi * 247.0 * t))
    shell = (shell * _env(n, 2, decay * 0.45, sr)).astype(np.float32) * tone
    body = rng.standard_normal(n).astype(np.float32) * _env(n, 1, decay, sr)
    body = FL.apply(body, "bandpass", sr, 1700.0, q=0.5, order=2)
    buzz = rng.standard_normal(n).astype(np.float32) * _env(n, 1, decay * 1.5, sr) * 0.42
    buzz = FL.apply(buzz, "highpass", sr, 4200.0, order=2)
    out = shell + body + buzz * bright
    out = FL.apply(out, "peak", sr, 3400.0, q=0.8, gain_db=3.5)
    out = FL.apply(out, "highpass", sr, 150.0, order=2)
    out = np.tanh(1.6 * out) / np.tanh(1.6)
    return _norm(out, 0.78)


def hat(sr: int, open_: bool = False, seed: int = 5, tone: float = 0.34,
        decay: float | None = None, low: float | None = None,
        high: float = 9800.0, room: float = 0.30) -> np.ndarray:
    """Hi-hat: metallic partials plus noise through a band-pass.

    Six detuned square partials at the classic inharmonic 808/909 ratios sit
    under the noise so the hat has a pitch centre instead of reading as a burst
    of static. The band is the important part: the references put the closed
    hat's spectral centroid at 7.2-9.0 kHz, so the noise is band-passed rather
    than merely high-passed. A short room tail keeps it from sounding like a
    sample-and-hold click.
    """
    rng = np.random.default_rng(seed)
    dec = (0.042 if open_ else 0.022) if decay is None else decay
    n = int((0.42 if open_ else 0.16) * sr)
    t = np.arange(n) / sr
    noise = rng.standard_normal(n).astype(np.float32)
    ring = np.zeros(n, dtype=np.float32)
    for r in (1.0, 1.4471, 1.6170, 1.9265, 2.5028, 2.6637):
        ring += np.sign(np.sin(2.0 * np.pi * 620.0 * r * t)).astype(np.float32)
    sig = (1.0 - tone) * noise + tone * ring / 6.0
    sig = (sig * _env(n, 1, dec, sr)).astype(np.float32)

    if room > 0:
        ir = _room(int((0.14 if open_ else 0.07) * sr), sr, dec * 1.6, 0.03, seed + 41)
        wet = sps.fftconvolve(sig, ir)[:n].astype(np.float32)
        wpk = float(np.max(np.abs(wet)))
        if wpk > 0:
            sig = sig + wet / wpk * room

    lo = (4000.0 if open_ else 4200.0) if low is None else low
    sig = FL.apply(sig, "highpass", sr, lo, order=2)
    sig = FL.apply(sig, "lowpass", sr, high, order=2)
    sig = FL.apply(sig, "peak", sr, 8200.0, q=0.7, gain_db=1.5)
    return _norm(sig, 0.46 if open_ else 0.52)


def shaker(sr: int, seed: int = 9) -> np.ndarray:
    """Shaker/ride texture: brighter, softer-attacked noise for the top end."""
    rng = np.random.default_rng(seed)
    n = int(0.16 * sr)
    sig = rng.standard_normal(n).astype(np.float32) * _env(n, int(0.005 * sr), 0.030, sr)
    sig = FL.apply(sig, "highpass", sr, 6200.0, order=2)
    sig = FL.apply(sig, "lowpass", sr, 12500.0, order=2)
    return _norm(sig, 0.24)


def rim(sr: int, freq: float = 1720.0, seed: int = 19) -> np.ndarray:
    """Rimshot/click for fills and for the sparse perc in an intro."""
    n = int(0.09 * sr)
    t = np.arange(n) / sr
    sig = (np.sin(2.0 * np.pi * freq * t) * _env(n, 1, 0.0075, sr)
           + 0.6 * np.sin(2.0 * np.pi * freq * 1.54 * t) * _env(n, 1, 0.0042, sr))
    rng = np.random.default_rng(seed)
    nn = int(0.003 * sr)
    sig[:nn] += rng.standard_normal(nn) * 0.45 * _env(nn, 1, 0.0012, sr)
    sig = FL.apply(sig.astype(np.float32), "highpass", sr, 700.0, order=2)
    return _norm(sig, 0.55)


def perc(sr: int, freq: float = 420.0, seed: int = 29) -> np.ndarray:
    """Conga-ish pitched percussion: the sparse colour in intros and fills."""
    n = int(0.20 * sr)
    t = np.arange(n) / sr
    f = freq * (1.0 + 0.35 * np.exp(-t / 0.018))
    sig = np.sin(2.0 * np.pi * np.cumsum(f) / sr).astype(np.float32) * _env(n, 2, 0.055, sr)
    rng = np.random.default_rng(seed)
    skin = rng.standard_normal(n).astype(np.float32) * _env(n, 1, 0.010, sr) * 0.35
    skin = FL.apply(skin, "bandpass", sr, 2400.0, q=0.7, order=2)
    out = np.tanh(1.7 * (sig + skin))
    return _norm(out, 0.6)


def tom(sr: int, freq: float = 150.0, seed: int = 13) -> np.ndarray:
    """Pitched tom for phrase-end fills."""
    n = int(0.3 * sr)
    t = np.arange(n) / sr
    f = freq * (1.0 + 0.62 * np.exp(-t / 0.042))
    sig = np.sin(2.0 * np.pi * np.cumsum(f) / sr).astype(np.float32) * _env(n, 3, 0.070, sr)
    rng = np.random.default_rng(seed)
    nn = int(0.005 * sr)
    skin = rng.standard_normal(nn).astype(np.float32) * 0.35 * _env(nn, 1, 0.0018, sr)
    sig[:nn] += FL.apply(skin, "highpass", sr, 900.0, order=2)
    sig = FL.apply(sig, "highpass", sr, 60.0, order=2)
    return _norm(np.tanh(1.8 * sig), 0.72)


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


#: Closed-hat weight per 16th slot, from the references' measured slot
#: occupancy (downbeats 0.81, "e" 0.41, offbeat 8ths 0.67, "a" 0.65). The "a"
#: carries the push, the "e" is the quiet one -- the reverse of the accent
#: pattern this module used to have, and the difference between a hat line that
#: grooves and one that ticks.
HAT_SLOT_WEIGHT: tuple[float, float, float, float] = (1.0, 0.54, 0.61, 0.82)


def _hats(level: float = 0.56, weights: tuple[float, float, float, float] | None = None,
          thin: float = 0.0) -> list[tuple[int, float]]:
    """16th closed hats weighted by the references' per-slot occupancy.

    ``thin`` drops the quietest slots first -- the references thin a hat line by
    clearing the "a" and "e" 16ths, never by pulling the downbeats.
    """
    w = HAT_SLOT_WEIGHT if weights is None else weights
    hits = [(s, round(level * w[s % 4], 3)) for s in range(16)]
    if thin > 0:
        keep = max(1, int(round(len(hits) * (1.0 - thin))))
        hits = sorted(sorted(hits, key=lambda h: -h[1])[:keep])
    return hits


PATTERNS: dict[str, Pattern] = {
    # Four on the floor, offbeat open hats, clap on 2 and 4.
    "drop": {
        "kick": _every(4, 1.0),
        "clap": [(4, 0.92), (12, 0.92)],
        "snare": [(4, 0.30), (12, 0.30)],
        "hat": _hats(),
        "ohat": _every(4, 0.52, 2),
        "shaker": _every(2, 0.26, 1),
        "perc": [(7, 0.30), (14, 0.24)],
        "sub": _every(4, 0.9),
    },
    "drop_var": {
        "kick": _every(4, 1.0),
        "clap": [(4, 0.92), (12, 0.92), (15, 0.32)],
        "snare": [(4, 0.34), (12, 0.34), (11, 0.20)],
        "hat": _hats(0.60),
        "ohat": _every(4, 0.56, 2),
        "shaker": _every(1, 0.17),
        "rim": [(3, 0.28), (10, 0.24)],
        "perc": [(7, 0.32), (13, 0.26)],
        "sub": _every(4, 0.9),
    },
    # DJ intro: kick and hats so the track can be beatmatched, nothing else.
    "intro": {
        "kick": _every(4, 0.86),
        "clap": [],
        "hat": _every(4, 0.30, 2),
        "ohat": [],
        "shaker": _every(4, 0.20, 1),
        "rim": [(6, 0.22)],
        "sub": _every(4, 0.6),
    },
    # Hats and percussion only: the top-of-track variant with no kick at all.
    "intro_perc": {
        "kick": [],
        "clap": [],
        "hat": _hats(0.34, thin=0.25),
        "ohat": [],
        "shaker": _every(2, 0.22, 1),
        "rim": [(6, 0.26), (14, 0.22)],
        "perc": [(3, 0.30), (11, 0.26)],
        "sub": [],
    },
    "intro_full": {
        "kick": _every(4, 0.94),
        "clap": [(12, 0.6)],
        "hat": _hats(0.42, thin=0.25),
        "ohat": _every(4, 0.40, 2),
        "shaker": _every(2, 0.22, 1),
        "perc": [(7, 0.26)],
        "sub": _every(4, 0.75),
    },
    # Build: denser hats, a snare figure that thickens into the last beat, and
    # rising velocities. The engine renders every bar of a slot identically, so
    # the "roll" is the shape inside the bar rather than across the bars.
    "build": {
        "kick": _every(4, 0.96),
        "clap": [(4, 0.7), (12, 0.7)],
        "snare": [(8, 0.34), (10, 0.40), (12, 0.46), (13, 0.54),
                  (14, 0.64), (15, 0.76)],
        "hat": [(s, 0.30 + 0.018 * s) for s in range(16)],
        "ohat": _every(4, 0.48, 2),
        "shaker": _every(1, 0.22),
        "sub": _every(4, 0.7),
    },
    # Breakdown: no kick. Shaker and a little perc keep the pulse alive.
    "breakdown": {
        "kick": [],
        "clap": [(12, 0.42)],
        "hat": [],
        "ohat": [],
        "shaker": _every(2, 0.18, 1),
        "perc": [(6, 0.22)],
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
    swing: float = 0.0
    samples: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.samples:
            self.samples = {
                "kick": kick(self.sr),
                "clap": clap(self.sr),
                "snare": snare(self.sr),
                "hat": hat(self.sr, False),
                "ohat": hat(self.sr, True),
                "shaker": shaker(self.sr),
                "rim": rim(self.sr),
                "perc": perc(self.sr),
                "tom": tom(self.sr),
            }

    def step_time(self, bar_start: float, step: int, step_dur: float) -> float:
        """Time of a 16th step, with swing applied to the odd 16ths.

        Swing delays every other 16th by ``swing`` of a step, the standard
        MPC-style shuffle; 0 is straight, 0.66 would be full triplet feel.

        The default is 0, and that is a finding rather than a fallback: all six
        references are straight 16ths, with mean offbeat offsets of +1.4 and
        +2.7 ms against a 115 ms sixteenth. (An earlier pass read -50 to -70 ms
        of shuffle out of them; that was a grid-phase artifact -- the same
        offset showed up on the downbeats, which by definition cannot swing.)
        Anything above about 0.06 is an effect, not house feel.
        """
        off = self.swing * step_dur if step % 2 == 1 else 0.0
        return bar_start + step * step_dur + off


# ---------------------------------------------------------------------------
# rendering and measurement
# ---------------------------------------------------------------------------

def render_pattern(sr: int, pattern: str = "drop", bars: int = 16, bpm: float = 128.0,
                   swing: float = 0.0, seed: int = 0, fill_every: int = 8,
                   per_voice: bool = False,
                   kit: Kit | None = None) -> np.ndarray | dict[str, np.ndarray]:
    """Render a pattern over ``bars`` bars, exactly as ``engine.render_drums`` does.

    With ``per_voice`` the voices come back as a dict of mono buffers instead of
    one summed buffer, so a mixer can balance them independently. The engine
    sums them itself today; the split is here for whatever wants it.
    """
    k = kit if kit is not None else Kit(sr=sr, swing=swing)
    bar_dur = 4.0 * 60.0 / bpm
    step_dur = bar_dur / 16.0
    n = int(round(bars * bar_dur * sr)) + sr
    pat = PATTERNS.get(pattern, PATTERNS["drop"])
    rng = np.random.default_rng(seed)
    voices = sorted(set(pat) | {"tom"}) if per_voice else []
    out: dict[str, np.ndarray] = {v: np.zeros(n, dtype=np.float32) for v in voices}
    summed = np.zeros(n, dtype=np.float32)

    def place(voice: str, sample: np.ndarray, at: int, gain: float) -> None:
        if at < 0 or at >= n:
            return
        m = min(len(sample), n - at)
        buf = out[voice] if per_voice else summed
        buf[at:at + m] += sample[:m] * gain

    for b in range(bars):
        bar_t = b * bar_dur
        last = fill_every > 0 and (b % fill_every) == fill_every - 1
        for voice, steps in pat.items():
            if voice == "sub":
                continue
            sample = k.samples.get(voice)
            if sample is None:
                continue
            for step, vel in steps:
                if last and voice in ("hat", "shaker") and step >= 12:
                    continue
                t = k.step_time(bar_t, step, step_dur)
                place(voice, sample, int(round(t * sr)),
                      vel * (0.92 + 0.16 * rng.random()))
        if last:
            for i, (step, f) in enumerate(zip((12, 13, 14, 15),
                                              (220.0, 180.0, 150.0, 120.0))):
                place("tom", tom(sr, f, seed=13 + i),
                      int(round((bar_t + step * step_dur) * sr)), 0.5 + 0.12 * i)
    return out if per_voice else summed


def _hp(x: np.ndarray, sr: int, f: float, order: int = 4) -> np.ndarray:
    sos = sps.butter(order, max(f, 1.0) / (sr * 0.5), btype="high", output="sos")
    return sps.sosfiltfilt(sos, np.asarray(x, dtype=np.float64))


def _bp(x: np.ndarray, sr: int, lo: float, hi: float, order: int = 4) -> np.ndarray:
    ny = sr * 0.5
    sos = sps.butter(order, [max(lo, 1.0) / ny, min(hi, ny * 0.99) / ny],
                     btype="band", output="sos")
    return sps.sosfiltfilt(sos, np.asarray(x, dtype=np.float64))


def _amp_env(x: np.ndarray, sr: int, smooth_ms: float = 3.0) -> np.ndarray:
    a = np.abs(sps.hilbert(np.asarray(x, dtype=np.float64)))
    w = max(1, int(smooth_ms * 1e-3 * sr))
    k = np.hanning(w * 2 + 1)
    return np.convolve(a, k / k.sum(), mode="same")


def _t20_ms(env: np.ndarray, sr: int, cap_ms: float = 500.0) -> float:
    """Time from the envelope peak to the point it stays 20 dB down, in ms.

    Measured against the running maximum of the remaining envelope rather than
    its instantaneous value: two detuned layers beat against each other, and a
    plain first-crossing test reads the first beat null as the end of the tail,
    which makes the number jump around by 5x for a 1 Hz change in the sub.
    """
    if len(env) == 0:
        return float("nan")
    i0 = int(np.argmax(env))
    pk = float(env[i0])
    if pk <= 0:
        return float("nan")
    seg = np.asarray(env[i0:], dtype=np.float64)
    remaining_max = np.maximum.accumulate(seg[::-1])[::-1]
    below = np.nonzero(remaining_max <= pk * 0.1)[0]
    if len(below) == 0:
        return float(min(cap_ms, len(seg) / sr * 1000.0))
    return float(min(cap_ms, below[0] / sr * 1000.0))


def _centroid_hz(x: np.ndarray, sr: int, lo: float = 300.0, hi: float = 20000.0) -> float:
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 64 or not np.any(x):
        return float("nan")
    nfft = max(4096, 1 << int(np.ceil(np.log2(len(x)))))
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=nfft))
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    m = (f >= lo) & (f <= hi)
    s = spec[m]
    return float((f[m] * s).sum() / s.sum()) if s.sum() > 0 else float("nan")


def _peak_freq(x: np.ndarray, sr: int, lo: float, hi: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    nfft = max(1 << 16, 1 << int(np.ceil(np.log2(max(len(x), 2)))))
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=nfft))
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    m = (f >= lo) & (f <= hi)
    return float(f[m][int(np.argmax(spec[m]))]) if m.any() else float("nan")


def band_balance(x: np.ndarray, sr: int) -> dict[str, float]:
    """Per-band energy in dB relative to the buffer's broadband energy.

    A shape, not a level, so a synthesised drum stem and a commercial one can be
    compared without matching their loudness first.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    nper = min(len(x), 8192)
    f, p = sps.welch(x, sr, nperseg=max(nper, 256), noverlap=max(nper, 256) // 2)
    total = float(np.trapezoid(p, f))
    out: dict[str, float] = {}
    for name, lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        e = float(np.trapezoid(p[m], f[m])) if m.sum() > 1 else 1e-20
        out[name] = round(10 * np.log10(max(e, 1e-20) / max(total, 1e-20)), 2)
    return out


def _band_env(x: np.ndarray, sr: int, lo: float, hi: float,
              smooth_ms: float = 3.0) -> np.ndarray:
    """Band envelope of a one-shot, with the filter's edge transient kept out of it.

    ``sosfiltfilt`` runs the filter both ways, so the abrupt end of a one-shot
    buffer throws a transient back into the last few milliseconds -- enough to
    climb back over a -20 dB threshold and make a decay measurement read as the
    whole buffer. Padding with silence moves that artifact past the hit, and the
    envelope is trimmed back to the real signal.
    """
    x = np.asarray(x, dtype=np.float64)
    pad = int(0.25 * sr)
    padded = np.concatenate([x, np.zeros(pad)])
    return _amp_env(_bp(padded, sr, lo, hi), sr, smooth_ms)[:len(x)]


def measure_voice(x: np.ndarray, sr: int, kind: str = "perc") -> dict[str, float]:
    """Measure one synthesised one-shot the way the reference stems were measured."""
    x = np.asarray(x, dtype=np.float64)
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if kind == "kick":
        low = _bp(np.concatenate([x, np.zeros(int(0.25 * sr))]), sr, 40.0, 100.0)[:len(x)]
        first = x[:int(0.010 * sr)]
        e_all = float(np.sum(first ** 2)) + 1e-20
        e_hi = float(np.sum(_hp(first, sr, 2000.0, order=3) ** 2)) + 1e-20
        return {
            # after the pitch envelope has settled: the fundamental the tail
            # sits on, not the average of the drop into it
            "f0_hz": round(_peak_freq(low[int(0.030 * sr):int(0.220 * sr)],
                                      sr, 25.0, 140.0), 2),
            "sub_t20_ms": round(_t20_ms(_band_env(x, sr, 40.0, 100.0, 4.0), sr), 1),
            "click_db": round(float(10 * np.log10(e_hi / e_all)), 2),
            "peak": round(peak, 4),
        }
    band = {"clap": (900.0, 4500.0), "low": (80.0, 1200.0)}.get(kind, (2000.0, 16000.0))
    env = _band_env(x, sr, band[0], band[1], 2.0)
    return {
        "centroid_hz": round(_centroid_hz(x, sr), 1),
        "t20_ms": round(_t20_ms(env, sr, 450.0), 1),
        "peak": round(peak, 4),
    }


def _swing_of(x: np.ndarray, sr: int, bpm: float, t0: float = 0.0) -> float:
    """Measured swing of the odd 16ths, in units of a 16th step.

    Calibrated against renders of known swing: 0.00 reads +0.008, 0.04 reads
    +0.046, 0.08 reads +0.088.
    """
    step = 60.0 / bpm / 4.0
    env = _amp_env(_hp(x, sr, 3000.0), sr, 2.0)
    n_steps = int((len(x) / sr - t0) / step) - 1
    ref = np.median([env[int((t0 + s * step) * sr):
                         int((t0 + (s + 0.5) * step) * sr)].max()
                     for s in range(0, max(n_steps, 1), 4)] or [1.0])
    offs = []
    for s in range(1, n_steps, 2):
        a = max(int((t0 + (s - 0.15) * step) * sr), 0)
        b = int((t0 + (s + 0.55) * step) * sr)
        w = env[a:b]
        if len(w) < 32 or w.max() < 0.25 * ref:
            continue
        d = np.maximum(np.diff(w, prepend=w[0]), 0.0)
        if d.max() <= 0:
            continue
        offs.append((-0.15 * step + int(np.argmax(d)) / sr) / step)
    return round(float(np.median(offs)), 4) if len(offs) >= 4 else 0.0


def _onsets_per_bar(x: np.ndarray, sr: int, bpm: float) -> float:
    env = _amp_env(_bp(x, sr, 100.0, 16000.0), sr, 2.0)
    d = np.maximum(np.diff(env, prepend=env[0]), 0.0)
    if d.max() <= 0:
        return 0.0
    pk, _ = sps.find_peaks(d / d.max(), height=0.15, distance=int(0.035 * sr))
    bars = max(len(x) / sr / (4.0 * 60.0 / bpm), 1e-6)
    return round(len(pk) / bars, 2)


def kit_report(sr: int = 44100, pattern: str = "drop", bars: int = 16,
               bpm: float = 128.0, swing: float = 0.0, seed: int = 0) -> dict:
    """Measure the synthesised kit, in the units the reference stems were measured in.

    Returns the per-voice one-shot numbers, the numbers for a rendered
    ``bars``-bar stem of ``pattern``, and :data:`REFERENCE` alongside, so a
    caller (or a test) can check every value against the range measured from
    real records without repeating the analysis code.
    """
    k = Kit(sr=sr, swing=swing)
    voices = {
        "kick": measure_voice(k.samples["kick"], sr, "kick"),
        "clap": measure_voice(k.samples["clap"], sr, "clap"),
        "snare": measure_voice(k.samples["snare"], sr, "clap"),
        "hat": measure_voice(k.samples["hat"], sr, "hat"),
        "ohat": measure_voice(k.samples["ohat"], sr, "hat"),
        "shaker": measure_voice(k.samples["shaker"], sr, "hat"),
        "rim": measure_voice(k.samples["rim"], sr, "clap"),
        "perc": measure_voice(k.samples["perc"], sr, "low"),
        "tom": measure_voice(k.samples["tom"], sr, "low"),
    }
    # the reference levels were measured on hits inside a stem, so compare the
    # voices the way the pattern actually plays them, velocity included
    pat = PATTERNS.get(pattern, PATTERNS["drop"])

    def hit_peak(voice: str) -> float:
        vels = [v for _, v in pat.get(voice, [])]
        return voices[voice]["peak"] * (float(np.median(vels)) if vels else 1.0)

    kpk = max(hit_peak("kick"), 1e-9)
    stem = render_pattern(sr, pattern, bars, bpm, swing, seed, kit=k)
    assert isinstance(stem, np.ndarray)
    rms = float(np.sqrt(np.mean(stem ** 2))) if len(stem) else 0.0
    return {
        "voices": voices,
        "kick_f0_hz": voices["kick"]["f0_hz"],
        "kick_sub_t20_ms": voices["kick"]["sub_t20_ms"],
        "kick_click_db": voices["kick"]["click_db"],
        "clap_centroid_hz": voices["clap"]["centroid_hz"],
        "clap_t20_ms": voices["clap"]["t20_ms"],
        "clap_vs_kick_db": round(20 * np.log10(max(hit_peak("clap"), 1e-9) / kpk), 2),
        "hat_centroid_hz": voices["hat"]["centroid_hz"],
        "hat_t20_ms": voices["hat"]["t20_ms"],
        "hat_vs_kick_db": round(20 * np.log10(max(hit_peak("hat"), 1e-9) / kpk), 2),
        "ohat_t20_ms": voices["ohat"]["t20_ms"],
        "swing": _swing_of(stem, sr, bpm),
        "onsets_per_bar": _onsets_per_bar(stem, sr, bpm),
        "band_balance_db": band_balance(stem, sr),
        "stem_peak": round(float(np.max(np.abs(stem))) if len(stem) else 0.0, 4),
        "stem_rms_db": round(20 * np.log10(max(rms, 1e-9)), 2),
        "pattern": pattern,
        "reference": REFERENCE,
    }
