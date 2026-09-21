"""Signal features the critic scores a render on.

Everything here is numpy/scipy only and runs on the mono mixdown. One STFT
pass produces the mel band matrix, the onset envelope and the per-frame
spectral flatness; the detectors then read those three arrays. The sample
level work (clicks) runs on the full rate buffer because a splice is a
one sample event and survives no decimation.

Frame geometry: 22.05 kHz, 1024-point FFT, 128-sample hop. The hop is
5.8 ms, chosen so that "within +/- 20 ms of the grid" is a question the
frame rate can actually answer, and onset peaks are parabolically
interpolated on top of that for roughly millisecond placement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

ANALYSIS_SR = 22050
N_FFT = 1024
HOP = 128
FPS = ANALYSIS_SR / HOP  # 172.27 frames per second
N_MELS = 48
_BLOCK = 8192  # STFT frames per chunk, keeps peak memory flat on long tracks


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------

def clamp01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _lerp01(x: float, lo: float, hi: float) -> float:
    """Map ``x`` from the range [lo, hi] onto [0, 1], clamped. Handles hi < lo."""
    if hi == lo:
        return 0.0
    return clamp01((x - lo) / (hi - lo))


def downmix(samples: np.ndarray) -> np.ndarray:
    """Mono float64 view of a (n, 2) or (n,) buffer."""
    if samples.ndim == 2:
        return samples.mean(axis=1).astype(np.float64)
    return samples.astype(np.float64)


def resample_to(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Linear resample. Good enough: every consumer downstream is an envelope."""
    if sr_in == sr_out:
        return x
    n_out = int(round(len(x) * sr_out / float(sr_in)))
    if n_out <= 1:
        return x[:1].copy()
    src = np.linspace(0.0, len(x) - 1.0, n_out)
    return np.interp(src, np.arange(len(x), dtype=np.float64), x)


def _mel_filters(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """A (n_bins, n_mels) triangular filterbank, area-normalised."""
    def to_mel(f):
        return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)

    def from_mel(m):
        return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)

    f_min, f_max = 30.0, min(sr / 2.0 - 1.0, 11000.0)
    edges = from_mel(np.linspace(to_mel(f_min), to_mel(f_max), n_mels + 2))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    bank = np.zeros((len(freqs), n_mels), dtype=np.float64)
    for m in range(n_mels):
        lo, mid, hi = edges[m], edges[m + 1], edges[m + 2]
        left = (freqs - lo) / max(mid - lo, 1e-9)
        right = (hi - freqs) / max(hi - mid, 1e-9)
        tri = np.maximum(0.0, np.minimum(left, right))
        area = tri.sum()
        if area > 0:
            bank[:, m] = tri / area
    return bank


def _frame(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Overlapping frames as a strided view (no copy)."""
    if len(x) < n_fft:
        x = np.concatenate([x, np.zeros(n_fft - len(x))])
    n_frames = 1 + (len(x) - n_fft) // hop
    stride = x.strides[0]
    return np.lib.stride_tricks.as_strided(
        x, shape=(n_frames, n_fft), strides=(stride * hop, stride), writeable=False
    )


@dataclass
class Spectral:
    """One STFT pass, reduced to the three arrays every detector needs."""

    mel: np.ndarray        # (n_frames, N_MELS) magnitude
    onset: np.ndarray      # (n_frames,) spectral flux, >= 0
    flatness: np.ndarray   # (n_frames,) Wiener entropy of the power spectrum
    fps: float
    freqs: np.ndarray


def spectral(x: np.ndarray, sr: int = ANALYSIS_SR) -> Spectral:
    """Mel magnitudes, half-wave rectified log flux and spectral flatness."""
    window = np.hanning(N_FFT)
    bank = _mel_filters(sr, N_FFT, N_MELS)
    frames = _frame(x, N_FFT, HOP)
    mel_parts: list[np.ndarray] = []
    flat_parts: list[np.ndarray] = []
    for start in range(0, len(frames), _BLOCK):
        block = frames[start: start + _BLOCK] * window
        mag = np.abs(np.fft.rfft(block, axis=1))
        power = np.square(mag) + 1e-12
        # Wiener entropy: geometric over arithmetic mean, in [0, 1]
        log_mean = np.exp(np.mean(np.log(power), axis=1))
        flat_parts.append(log_mean / np.mean(power, axis=1))
        mel_parts.append(mag @ bank)
    mel = np.concatenate(mel_parts) if mel_parts else np.zeros((0, N_MELS))
    flat = np.concatenate(flat_parts) if flat_parts else np.zeros(0)
    log_mel = np.log1p(200.0 * mel)
    flux = np.maximum(0.0, np.diff(log_mel, axis=0, prepend=log_mel[:1])).sum(axis=1)
    if flux.size:
        flux = flux - median_filter(flux, size=int(FPS * 0.5) | 1, mode="nearest")
        flux = np.maximum(0.0, flux)
    return Spectral(mel=mel, onset=flux, flatness=flat, fps=sr / HOP,
                    freqs=np.fft.rfftfreq(N_FFT, 1.0 / sr))


# ---------------------------------------------------------------------------
# onsets
# ---------------------------------------------------------------------------

def pick_onsets(env: np.ndarray, fps: float,
                min_gap: float = 0.028) -> tuple[np.ndarray, np.ndarray]:
    """Peak-pick an onset envelope.

    Returns ``(times, strengths)``. A peak has to clear an adaptive
    threshold (local median plus a fraction of the local spread) and beat
    every neighbour inside ``min_gap``; its time is refined by fitting a
    parabola through the three envelope samples around it, which is what
    buys sub-frame placement.
    """
    if env.size < 3 or not np.any(env > 0):
        return np.zeros(0), np.zeros(0)
    win = int(fps * 0.4) | 1
    floor = median_filter(env, size=win, mode="nearest")
    spread = median_filter(np.abs(env - floor), size=win, mode="nearest")
    thresh = floor + 1.2 * spread + 0.02 * float(env.max())
    cand = np.flatnonzero((env[1:-1] > env[:-2]) & (env[1:-1] >= env[2:])
                          & (env[1:-1] > thresh[1:-1])) + 1
    if cand.size == 0:
        return np.zeros(0), np.zeros(0)
    # greedy strongest-first suppression inside min_gap
    gap = max(1, int(round(min_gap * fps)))
    order = cand[np.argsort(-env[cand])]
    keep: list[int] = []
    taken = np.zeros(env.size, dtype=bool)
    for i in order:
        if taken[max(0, i - gap): i + gap + 1].any():
            continue
        taken[i] = True
        keep.append(int(i))
    idx = np.array(sorted(keep))
    a, b, c = env[idx - 1], env[idx], env[idx + 1]
    denom = a - 2.0 * b + c
    shift = np.where(np.abs(denom) > 1e-12, 0.5 * (a - c) / np.where(denom == 0, 1, denom), 0.0)
    shift = np.clip(shift, -0.5, 0.5)
    return (idx + shift) / fps, env[idx]


def autocorr(env: np.ndarray) -> np.ndarray:
    """Unbiased normalised autocorrelation of an envelope, via FFT."""
    x = env - env.mean() if env.size else env
    if x.size == 0 or not np.any(x):
        return np.zeros(max(env.size, 1))
    n = 1 << int(np.ceil(np.log2(2 * x.size)))
    spec = np.fft.rfft(x, n)
    ac = np.fft.irfft(spec * np.conj(spec), n)[: x.size]
    counts = np.arange(x.size, 0, -1, dtype=np.float64)
    ac = ac / counts
    return ac / max(ac[0], 1e-12)


def _peak_contrast(ac: np.ndarray, lag: float, tol: float = 0.035,
                   halo: float = 0.30) -> tuple[float, float]:
    """How much an autocorrelation peak near ``lag`` stands above its region.

    Returns ``(contrast, offset)`` where offset is the peak's fractional
    distance from the requested lag. A sharp house pulse gives a high
    contrast at a near-zero offset; a smeared or dragged one gives a low
    contrast, and a split one puts the peak off to the side.
    """
    n = len(ac)
    lo, hi = int(lag * (1 - tol)), int(np.ceil(lag * (1 + tol)))
    lo, hi = max(1, lo), min(n - 1, hi)
    if hi <= lo:
        return 0.0, 0.0
    seg = ac[lo: hi + 1]
    k = int(np.argmax(seg)) + lo
    peak = float(ac[k])
    h_lo, h_hi = max(1, int(lag * (1 - halo))), min(n - 1, int(lag * (1 + halo)))
    if h_hi <= h_lo:
        return 0.0, 0.0
    halo_vals = np.concatenate([ac[h_lo: lo], ac[hi + 1: h_hi + 1]])
    base = float(np.median(halo_vals)) if halo_vals.size else 0.0
    return peak - base, (k - lag) / max(lag, 1e-9)


GRID_WINDOW = 8.0
"""Seconds of audio each grid phase is fitted over.

Not a free parameter. Folding a whole track modulo one beat period is
useless: a 0.1 % tempo error accumulates to a quarter of a second over
four minutes, which is ten times the tolerance being tested, so every
track -- good, bad, or not house at all -- comes out at chance. Eight
seconds is about four bars: long enough that a phase fit means something,
short enough that drift inside it stays under the 20 ms window. Measured
on the calibration set, this is the window at which the references, the
"some of it is off beat" render and the "off beat" render separate.
"""


def windowed_grid(times: np.ndarray, strengths: np.ndarray, step: float,
                  tol: float = 0.020, window: float = GRID_WINDOW) -> float:
    """Grid adherence measured locally and averaged.

    Each window gets its own phase, so this answers "are the onsets
    quantised to *a* grid right here", which is what off-beat sounds
    like, rather than "has the track held one absolute phase for four
    minutes", which no record does.
    """
    if times.size < 4 or step <= 0:
        return 0.0
    fractions, weights = [], []
    for start in np.arange(0.0, float(times[-1]) + window, window):
        sel = (times >= start) & (times < start + window)
        if sel.sum() < 4:
            continue
        frac, _ = grid_fraction(times[sel] - start, strengths[sel], step, tol)
        fractions.append(frac)
        weights.append(float(strengths[sel].sum()))
    if not fractions or sum(weights) <= 0:
        return 0.0
    return float(np.average(fractions, weights=weights))


def grid_fraction(times: np.ndarray, strengths: np.ndarray, step: float,
                  tol: float = 0.020) -> tuple[float, float]:
    """Strength-weighted fraction of onsets within ``tol`` of the best grid.

    The grid has period ``step`` and a free phase. Rather than sweeping the
    phase, every onset is folded into one period and the best phase is the
    circular window of width ``2 * tol`` holding the most weight -- exact,
    and linear in the number of onsets after the sort.
    """
    if times.size == 0 or step <= 0:
        return 0.0, 0.0
    w = strengths.astype(np.float64)
    total = w.sum()
    if total <= 0:
        return 0.0, 0.0
    r = np.mod(times, step)
    order = np.argsort(r)
    r, w = r[order], w[order]
    # duplicate one wrap so the window can straddle the seam
    rr = np.concatenate([r, r + step])
    ww = np.concatenate([w, w])
    cum = np.concatenate([[0.0], np.cumsum(ww)])
    width = 2.0 * tol
    ends = np.searchsorted(rr, rr + width, side="right")
    sums = cum[ends] - cum[np.arange(len(rr))]
    best = int(np.argmax(sums[: len(r)]))
    return float(sums[best] / total), float((rr[best] + tol) % step)


# ---------------------------------------------------------------------------
# detectors
# ---------------------------------------------------------------------------

@dataclass
class Measured:
    """Raw numbers from one track, before any scoring opinion is applied."""

    duration: float = 0.0
    bpm: float = 0.0
    beat_period: float = 0.0
    beat_contrast: float = 0.0
    half_contrast: float = 0.0
    bar_contrast: float = 0.0
    beat_offset: float = 0.0
    split_peak: float = 0.0
    on_grid: float = 0.0
    on_grid_beat: float = 0.0
    onsets_per_beat: float = 0.0
    flatness: float = 0.0
    mod_depth: float = 0.0
    click_rate: float = 0.0
    click_worst: float = 0.0
    rms_db: float = -120.0
    peak_db: float = -120.0
    crest_db: float = 0.0
    clip_fraction: float = 0.0
    vocal_ratio_db: float = -120.0
    vocal_mod: float = 0.0
    vocal_source: str = "band-proxy"
    extra: dict = field(default_factory=dict)


def tempo(env: np.ndarray, fps: float, bpm_hint: float | None = None) -> tuple[float, np.ndarray]:
    """Beat period in seconds plus the autocorrelation it was read from.

    ``bpm_hint`` (from a session file) is trusted only as a neighbourhood:
    the peak is still located inside +/- 6 % of it, so a render that drifted
    off its own stated tempo is measured where it actually landed.
    """
    ac = autocorr(env)
    n = len(ac)
    if bpm_hint and bpm_hint > 0:
        want = 60.0 / bpm_hint
        lo, hi = int(want * 0.94 * fps), int(np.ceil(want * 1.06 * fps))
        lo, hi = max(2, lo), min(n - 1, hi)
        if hi > lo:
            return (int(np.argmax(ac[lo: hi + 1])) + lo) / fps, ac
    lo, hi = int(0.30 * fps), min(n - 1, int(0.92 * fps))
    if hi <= lo:
        return 0.5, ac
    seg = ac[lo: hi + 1].copy()
    # mild preference for the middle of the house range, so a strong
    # half-time or double-time lag does not win by a hair
    lags = np.arange(lo, hi + 1) / fps
    seg *= np.exp(-0.5 * (np.log(lags / 0.48) / 0.55) ** 2)
    return (int(np.argmax(seg)) + lo) / fps, ac


def measure_groove(sp: Spectral, bpm_hint: float | None, m: Measured) -> None:
    """Beat period, autocorrelation peak shape and grid adherence."""
    period, ac = tempo(sp.onset, sp.fps, bpm_hint)
    m.beat_period = period
    m.bpm = 60.0 / period if period > 0 else 0.0
    lag = period * sp.fps
    m.beat_contrast, m.beat_offset = _peak_contrast(ac, lag)
    m.half_contrast, _ = _peak_contrast(ac, lag / 2.0)
    m.bar_contrast, _ = _peak_contrast(ac, lag * 4.0, tol=0.02, halo=0.18)
    # split peak: a competing peak 5-14 % away from the beat lag. An
    # off-beat mix (two layers at slightly different tempi, or a stretch
    # that slipped) shows two shoulders instead of one spike.
    n = len(ac)
    lo, hi = int(lag * 1.05), min(n - 1, int(lag * 1.30))
    lo2, hi2 = max(2, int(lag * 0.70)), int(lag * 0.95)
    side = []
    if hi > lo:
        side.append(float(ac[lo: hi + 1].max()))
    if hi2 > lo2:
        side.append(float(ac[lo2: hi2 + 1].max()))
    peak = float(ac[max(2, int(lag * 0.965)): min(n - 1, int(lag * 1.035)) + 1].max()) if n > 3 else 0.0
    rival = max(side) if side else 0.0
    m.split_peak = clamp01(rival / max(peak, 1e-6)) if peak > 0 else 1.0

    times, strengths = pick_onsets(sp.onset, sp.fps)
    m.extra["n_onsets"] = int(times.size)
    m.on_grid = windowed_grid(times, strengths, period / 2.0)
    m.on_grid_beat = windowed_grid(times, strengths, period)
    if m.duration > 0 and period > 0:
        m.onsets_per_beat = times.size / (m.duration / period)


def measure_mush(sp: Spectral, m: Measured) -> None:
    """Spectral flatness and the depth of the 2-8 Hz pulse in the envelope.

    A house record is strongly amplitude-modulated at the beat and its
    subdivisions, so the envelope spectrum has a spike in 2-8 Hz. Two source
    spans playing over each other fill the gaps in, which both flattens that
    modulation and raises the broadband flatness.
    """
    m.flatness = float(np.mean(sp.flatness)) if sp.flatness.size else 0.0
    env = sp.onset
    if env.size < int(sp.fps * 4):
        return
    # Welch-style average of the envelope spectrum over 8 s windows
    win_n = int(sp.fps * 8)
    step = win_n // 2
    win = np.hanning(win_n)
    acc = np.zeros(win_n // 2 + 1)
    count = 0
    for start in range(0, max(1, env.size - win_n + 1), step):
        seg = env[start: start + win_n]
        if len(seg) < win_n:
            break
        seg = (seg - seg.mean()) * win
        acc += np.abs(np.fft.rfft(seg)) ** 2
        count += 1
    if count == 0:
        return
    acc /= count
    freqs = np.fft.rfftfreq(win_n, 1.0 / sp.fps)
    band = (freqs >= 2.0) & (freqs <= 8.0)
    ref = (freqs >= 0.4) & (freqs <= 20.0)
    if not band.any() or not ref.any():
        return
    m.mod_depth = float(acc[band].max() / max(np.median(acc[ref]), 1e-12))


#: (short floor ms, long floor ms, short ratio, long ratio, absolute jump)
CLICK = (1.5, 50.0, 12.0, 35.0, 0.08)


def measure_clicks(x: np.ndarray, sr: int, m: Measured,
                   cues: list[float] | None = None) -> None:
    """Splices: sample jumps that are both locally isolated and globally large.

    One test is not enough. Against a long derivative floor a kick attack
    looks exactly like a splice -- both are a sudden jump in a quiet
    neighbourhood -- and on the calibration set a long floor alone fired
    forty times a minute on professionally mastered references. Against a
    short floor alone, mp3 quantisation noise fires hundreds of times a
    minute. A real discontinuity is the one event that clears both: one
    sample wide (so it towers over its immediate neighbours) *and* far
    above the derivative the passage has been running at.

    Events within 200 ms of a section cue count triple. That is where a bad
    edit actually lands, and where a listener notices it.
    """
    if len(x) < 8:
        return
    short_ms, long_ms, short_t, long_t, abs_t = CLICK
    d = np.abs(np.diff(x))
    short = median_filter(d, size=max(3, int(sr * short_ms / 1000.0) | 1), mode="nearest")
    long_ = median_filter(d, size=max(3, int(sr * long_ms / 1000.0) | 1), mode="nearest")
    # uniform_filter1d can return a tiny negative for an all-zero stretch
    local_rms = np.sqrt(np.maximum(
        0.0, uniform_filter1d(np.square(x), size=max(3, int(sr * 0.02)))[:-1]))
    ratio = d / (long_ + 1e-6)
    hits = np.flatnonzero((d / (short + 1e-6) > short_t) & (ratio > long_t)
                          & (d > abs_t) & (d > 0.15 * local_rms))
    if hits.size == 0:
        m.extra["n_clicks"] = 0
        return
    # merge hits inside 5 ms into one event
    gap = max(1, int(sr * 0.005))
    starts = hits[np.concatenate([[True], np.diff(hits) > gap])]
    weights = np.ones(len(starts))
    if cues:
        t = starts / float(sr)
        cue_arr = np.asarray(cues, dtype=np.float64)
        near = np.min(np.abs(t[:, None] - cue_arr[None, :]), axis=1) < 0.20
        weights[near] = 3.0
    minutes = max(len(x) / float(sr) / 60.0, 1e-6)
    m.click_rate = float(weights.sum() / minutes)
    m.click_worst = float(ratio[hits].max())
    m.extra["n_clicks"] = int(len(starts))


def measure_loudness(samples: np.ndarray, m: Measured) -> None:
    """RMS, peak, crest and the fraction of samples pinned at full scale."""
    x = samples if samples.ndim == 1 else samples.reshape(-1)
    if not x.size:
        return
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    peak = float(np.max(np.abs(x)))
    m.rms_db = 20.0 * np.log10(max(rms, 1e-6))
    m.peak_db = 20.0 * np.log10(max(peak, 1e-6))
    m.crest_db = m.peak_db - m.rms_db
    m.clip_fraction = float(np.mean(np.abs(x) >= 0.9985))


def measure_vocal(sp: Spectral, m: Measured, vocal_mono: np.ndarray | None = None,
                  sr: int = ANALYSIS_SR) -> None:
    """Syllabic-rate energy in the vocal band, and its level against the rest.

    With a real Demucs vocal stem this reads the stem. Without one it falls
    back to a band proxy: the 300-3400 Hz mel bands stand in for the voice
    and the bands outside it for the backing. The proxy cannot tell a vocal
    from a lead synth, which is exactly why it is reported as a proxy.
    """
    if vocal_mono is not None and vocal_mono.size:
        vsp = spectral(resample_to(vocal_mono, sr, ANALYSIS_SR))
        voice = vsp.mel.sum(axis=1)
        m.vocal_source = "demucs"
    else:
        lo = np.searchsorted(np.linspace(30, 11000, N_MELS), 300.0)
        hi = np.searchsorted(np.linspace(30, 11000, N_MELS), 3400.0)
        voice = sp.mel[:, lo:hi].sum(axis=1)
        m.vocal_source = "band-proxy"
    rest = sp.mel.sum(axis=1) - (voice if m.vocal_source == "band-proxy" else 0.0)
    v_rms = float(np.sqrt(np.mean(np.square(voice)))) if voice.size else 0.0
    r_rms = float(np.sqrt(np.mean(np.square(rest)))) if rest.size else 0.0
    m.vocal_ratio_db = 20.0 * np.log10(max(v_rms, 1e-9) / max(r_rms, 1e-9))
    if voice.size < int(sp.fps * 2):
        return
    env = voice - uniform_filter1d(voice, size=int(sp.fps * 1.0) | 1)
    win_n = min(len(env), int(sp.fps * 4))
    spec = np.abs(np.fft.rfft((env[:win_n] - env[:win_n].mean()) * np.hanning(win_n))) ** 2
    freqs = np.fft.rfftfreq(win_n, 1.0 / sp.fps)
    syll = (freqs >= 4.0) & (freqs <= 8.0)
    ref = (freqs >= 0.5) & (freqs <= 20.0)
    if syll.any() and ref.any():
        m.vocal_mod = float(spec[syll].mean() / max(spec[ref].mean(), 1e-12))
