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


#: Where a lead vocal actually lives. Intelligibility is decided here: below it
#: the kick and bass own the spectrum, above it only air and sibilance.
VOCAL_BAND = (300.0, 4000.0)

#: The narrower band that carries consonants and "presence". Boosting a vocal
#: here and cutting the kit by the same amount buys clarity without loudness.
PRESENCE_BAND = (2000.0, 5000.0)


def band_limit(x: np.ndarray, sr: int, lo: float, hi: float,
               order: int = 2) -> np.ndarray:
    """Restrict a buffer to ``[lo, hi]`` with 24 dB/oct skirts, for measurement."""
    y = FL.apply(x, "highpass", sr, lo, q=0.707, order=order)
    return FL.apply(y, "lowpass", sr, hi, q=0.707, order=order)


def rms(x: np.ndarray) -> float:
    """Plain RMS of a buffer (0.0 when empty)."""
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


def rms_db(x: np.ndarray) -> float:
    """RMS in dBFS, floored at -120."""
    return 20.0 * np.log10(max(rms(x), 1e-6))


def band_rms(x: np.ndarray, sr: int, band: tuple[float, float] = VOCAL_BAND,
             n: int = 8192) -> float:
    """RMS of the part of ``x`` inside ``band``, by averaged periodograms.

    Parseval on Hann-windowed frames, corrected for the window's power loss.
    Measuring in the frequency domain rather than filtering and taking an RMS
    is both exact about the band edges and cheap enough to run on every slot of
    every render.
    """
    mono = x.mean(axis=1) if x.ndim == 2 else x
    if len(mono) < 64:
        return 0.0
    n = min(n, 1 << int(np.floor(np.log2(len(mono)))))
    win = np.hanning(n)
    wpow = float(np.mean(win ** 2))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    sel = (freqs >= band[0]) & (freqs < band[1])
    edge = (freqs == 0) | (freqs == freqs[-1])
    acc, frames = 0.0, 0
    for a in range(0, len(mono) - n + 1, n):
        spec = np.abs(np.fft.rfft(mono[a:a + n] * win)) ** 2
        power = 2.0 * spec[sel & ~edge].sum() + spec[sel & edge].sum()
        acc += power / (n * n) / max(wpow, 1e-12)
        frames += 1
    return float(np.sqrt(acc / max(frames, 1)))


def band_rms_db(x: np.ndarray, sr: int, band: tuple[float, float] = VOCAL_BAND) -> float:
    """RMS in dBFS of the part of ``x`` inside ``band``."""
    return 20.0 * np.log10(max(band_rms(x, sr, band), 1e-6))


#: Control-rate hop for envelope followers: 256 samples is 5.8 ms at 44.1 kHz,
#: far finer than any gain move we want to hear and 256x cheaper than running
#: the one-pole per sample.
CTRL_HOP = 256


def follow(level: np.ndarray, rate: float, attack: float = 0.02,
           release: float = 0.15) -> np.ndarray:
    """One-pole attack/release smoothing of a control signal at ``rate`` Hz."""
    a_att = float(np.exp(-1.0 / max(attack * rate, 1.0)))
    a_rel = float(np.exp(-1.0 / max(release * rate, 1.0)))
    out = np.empty(len(level), dtype=np.float32)
    g = float(level[0]) if len(level) else 0.0
    for i, v in enumerate(level):
        a = a_att if v > g else a_rel
        g = a * g + (1.0 - a) * float(v)
        out[i] = g
    return out


def band_follower(x: np.ndarray, sr: int, band: tuple[float, float] = VOCAL_BAND,
                  attack: float = 0.02, release: float = 0.15,
                  hop: int = CTRL_HOP) -> np.ndarray:
    """Smoothed per-sample RMS envelope of ``x`` inside ``band``.

    The RMS is taken per ``hop``, smoothed at control rate and interpolated back
    to one value per sample, which is what a gain curve needs to be free of
    zipper noise while costing a fraction of a per-sample follower.
    """
    n = len(x)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    mono = x.mean(axis=1) if x.ndim == 2 else x
    mono = band_limit(mono, sr, band[0], band[1])
    frames = max(1, n // hop)
    block = mono[: frames * hop].reshape(frames, hop)
    level = np.sqrt(np.mean(np.square(block), axis=1))
    smooth = follow(level, sr / hop, attack=attack, release=release)
    centres = np.arange(frames) * hop + hop * 0.5
    return np.interp(np.arange(n), centres, smooth).astype(np.float32)


#: Where a split-band sidechain divides. Below it the kick owns the spectrum
#: and the source can be ducked as hard as the mix wants; above it the vocal
#: lives and ducking it only makes the kick louder than the singer.
SIDECHAIN_CROSSOVER_HZ = 200.0

#: How much of the ducking survives above the crossover.
SIDECHAIN_HIGH_SCALE = 0.3


def split_sidechain(x: np.ndarray, sr: int, env: np.ndarray,
                    crossover: float = SIDECHAIN_CROSSOVER_HZ,
                    high_scale: float = SIDECHAIN_HIGH_SCALE) -> np.ndarray:
    """Duck ``x`` hard below ``crossover`` and gently above it.

    A kick and a vocal do not compete for the same frequencies, but a full-band
    sidechain makes them compete anyway: every kick drags the whole vocal down
    with the sub, four times a bar, which is most of what "the vocals get
    drowned out by the drums" actually is. Splitting at 200 Hz and letting only
    ``high_scale`` of the ducking through above it means the kick still owns the
    bottom while the voice stays where it was.

    The split is a Linkwitz-Riley pair -- two cascaded Butterworth sections each
    way -- not ``low`` and ``x - low``. Subtraction reconstructs perfectly but
    separates nothing: the low-pass at 60 Hz is very nearly unity in *magnitude*
    and 50 degrees late in phase, so ``x - low`` keeps 84% of a 60 Hz tone and
    the "low band" ducking barely reaches the kick's own register. An LR4 pair
    sums to an all-pass, so the magnitude is still flat, and each band actually
    contains its own half of the spectrum.
    """
    e = env[: len(x)]
    if len(e) < len(x):
        e = np.concatenate([e, np.ones(len(x) - len(e), dtype=np.float32)])
    low = FL.apply(x, "lowpass", sr, crossover, q=0.707, order=2)
    high = FL.apply(x, "highpass", sr, crossover, q=0.707, order=2)
    e_hi = 1.0 - (1.0 - e) * float(np.clip(high_scale, 0.0, 1.0))
    if x.ndim == 2:
        e, e_hi = e[:, None], e_hi[:, None]
    return (low * e + high * e_hi).astype(np.float32)


def moving_bell(x: np.ndarray, sr: int, centre: float, gain: np.ndarray | float,
                q: float = 0.8) -> np.ndarray:
    """A bell whose gain moves per sample while the filter stands still.

    A dry signal plus a scaled constant-0 dB-peak band-pass is a bell: unity far
    from ``centre``, ``1 + gain`` at it. It is not quite the Audio EQ Cookbook's
    peaking filter -- that one narrows its denominator as the gain rises, so its
    Q stays constant where this one's does not -- but it has the property that
    matters here. The coefficients never change, so a gain that follows an
    envelope cannot zipper and cannot make the filter ring as it moves; only a
    multiply moves.

    ``gain`` is the bell's peak gain minus one: 0 is flat, +0.41 is +3 dB at
    ``centre``, -0.29 is -3 dB.
    """
    g = np.asarray(gain, dtype=np.float32)
    if g.ndim == 1:
        g = g[: len(x)]
        if len(g) < len(x):
            g = np.concatenate([g, np.zeros(len(x) - len(g), dtype=np.float32)])
        if x.ndim == 2:
            g = g[:, None]
    bp = FL.apply(x, "bandpass", sr, centre, q=q, order=1)
    return (x + bp * g).astype(np.float32)


def db_to_bell(gain_db: np.ndarray | float) -> np.ndarray:
    """dB of bell gain as the ``A - 1`` factor ``moving_bell`` wants."""
    return (10.0 ** (np.asarray(gain_db, dtype=np.float32) / 20.0) - 1.0).astype(np.float32)


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


#: Integrated RMS the master aims for. The reference corpus sits at -10.8 dBFS
#: with a 12 dB crest factor; fourfloor used to aim at -8.5, which at a -1 dBFS
#: peak means a crest of 7.5 dB -- two and a half dB of limiting that the
#: references do not do, spent squashing exactly the transients a vocal needs
#: to stay in front of a kit.
RMS_TARGET_DB = -10.5


def master(x: np.ndarray, sr: int, peak_db: float = -1.0, drive: float = 1.28,
           rms_target_db: float | None = RMS_TARGET_DB, match: bool = True) -> np.ndarray:
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

    ceiling = 10.0 ** (peak_db / 20.0) * 0.96

    def stage(sig: np.ndarray) -> np.ndarray:
        return normalize_peak(soft_limit(saturate(sig, drive), ceiling=ceiling, sr=sr),
                              peak_db)

    if rms_target_db is None:
        return stage(y)

    got = rms(y)
    if got > 1e-7:
        y = y * float(np.clip(10.0 ** ((rms_target_db - 1.0) / 20.0) / got, 0.25, 4.0))
    out = stage(y)

    # The peak normalisation at the end of the chain, not the staging at the
    # front, is what decides the final RMS: whatever crest survives the limiter
    # sets it. Aiming at a target and then not measuring whether the target was
    # hit meant the loudness of a render depended on how peaky its kit happened
    # to be. One bounded corrective pass pushes the mix further into the limiter
    # until the target is met, so the number in the constant is the number that
    # comes out whatever the drums and bass are doing.
    shortfall = rms_target_db - rms_db(out)
    if shortfall > 0.3:
        makeup = 10.0 ** (min(shortfall, 4.0) / 20.0)
        out = stage(out * makeup)
    return out
