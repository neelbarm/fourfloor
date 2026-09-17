"""Tempo, beat and downbeat estimation.

Pipeline:

1. Onset strength envelope (``features.onset_strength``).
2. Tempo from the autocorrelation of that envelope, weighted by a log-normal
   prior over 60-200 BPM, with an explicit octave-disambiguation pass.
3. Beat times by dynamic programming over the envelope, balancing onset
   strength against deviation from the estimated period -- Ellis, "Beat
   Tracking by Dynamic Programming", J. New Music Research 36(1), 2007.
4. Downbeats by scoring each of the four possible bar phases with low-band and
   onset energy on beat 1, plus a windowed consistency check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sps

from . import features as F

MIN_BPM = 60.0
MAX_BPM = 200.0
PRIOR_CENTER_BPM = 124.0
PRIOR_WIDTH_OCTAVES = 0.95


@dataclass
class BeatGrid:
    """Beat positions with tempo and bar phase."""

    bpm: float
    beats: np.ndarray            # beat times, seconds
    downbeat_index: int          # index into `beats` of the first beat 1
    beats_per_bar: int = 4
    tempo_candidates: list[tuple[float, float]] = field(default_factory=list)
    downbeat_confidence: float = 0.0

    @property
    def period(self) -> float:
        """Mean inter-beat interval in seconds."""
        return 60.0 / self.bpm

    @property
    def first_downbeat(self) -> float:
        """Time of the first beat 1 in seconds."""
        if not len(self.beats):
            return 0.0
        return float(self.beats[min(self.downbeat_index, len(self.beats) - 1)])

    @property
    def downbeats(self) -> np.ndarray:
        """Times of every beat 1."""
        return self.beats[self.downbeat_index::self.beats_per_bar]

    def to_dict(self) -> dict:
        return {
            "bpm": round(float(self.bpm), 2),
            "beat_count": int(len(self.beats)),
            "first_beat_sec": round(float(self.beats[0]), 4) if len(self.beats) else 0.0,
            "first_downbeat_sec": round(self.first_downbeat, 4),
            "beats_per_bar": self.beats_per_bar,
            "downbeat_confidence": round(float(self.downbeat_confidence), 3),
            "tempo_candidates": [[round(b, 2), round(s, 3)] for b, s in self.tempo_candidates],
        }


def _tempo_prior(bpms: np.ndarray, center: float = PRIOR_CENTER_BPM,
                 width: float = PRIOR_WIDTH_OCTAVES) -> np.ndarray:
    """Log-normal perceptual tempo prior (Ellis 2007 eq. 5)."""
    return np.exp(-0.5 * (np.log2(np.maximum(bpms, 1e-6) / center) / width) ** 2)


def estimate_tempo(env: np.ndarray, fps: float) -> tuple[float, list[tuple[float, float]]]:
    """Return (bpm, ranked candidates) from the onset envelope.

    The envelope's unbiased autocorrelation is evaluated over lags corresponding
    to 60-200 BPM, weighted by the log-normal prior. Because autocorrelation is
    inherently ambiguous by factors of 2 and 3, each peak is rescored with a
    comb filter that sums envelope energy at multiples of the period; the winner
    is then compared against its half, double and 2/3 relatives under the prior.
    """
    x = env - env.mean()
    n = len(x)
    if n < 16:
        return PRIOR_CENTER_BPM, []
    nfft = int(2 ** np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(x, nfft)
    ac = np.fft.irfft(spec * np.conj(spec), nfft)[:n]
    ac /= np.maximum(np.arange(n, 0, -1), 1)      # unbiased
    ac = np.maximum(ac, 0.0)
    if ac[0] > 0:
        ac /= ac[0]

    lag_min = max(2, int(np.floor(60.0 * fps / MAX_BPM)))
    lag_max = min(n - 1, int(np.ceil(60.0 * fps / MIN_BPM)))
    if lag_max <= lag_min:
        return PRIOR_CENTER_BPM, []
    lags = np.arange(lag_min, lag_max + 1)
    bpms = 60.0 * fps / lags
    score = ac[lags] * _tempo_prior(bpms)

    peaks, _ = sps.find_peaks(score)
    if not len(peaks):
        peaks = np.array([int(np.argmax(score))])
    order = peaks[np.argsort(score[peaks])[::-1]][:8]

    def comb(period_lag: float) -> float:
        """Sum envelope energy at integer multiples of a candidate period."""
        total, count = 0.0, 0
        for k in range(1, 5):
            lag = int(round(period_lag * k))
            if lag_min <= lag < n:
                total += float(ac[lag])
                count += 1
        return total / max(count, 1)

    cands: list[tuple[float, float]] = []
    for p in order:
        lag = float(lags[p])
        bpm = 60.0 * fps / lag
        s = float(score[p]) * (0.5 + 0.5 * comb(lag) / max(ac[lag_min:lag_max].max(), 1e-9))
        cands.append((bpm, s))
    cands.sort(key=lambda t: -t[1])
    best_bpm, best_score = cands[0]

    # octave disambiguation: prefer the relative with the best prior-weighted comb score
    rels = [1.0, 0.5, 2.0, 2.0 / 3.0, 1.5]
    best = (best_bpm, -np.inf)
    for r in rels:
        bpm = best_bpm * r
        if not (MIN_BPM <= bpm <= MAX_BPM):
            continue
        lag = 60.0 * fps / bpm
        s = comb(lag) * float(_tempo_prior(np.array([bpm]))[0])
        if s > best[1]:
            best = (bpm, s)
    chosen = best[0]

    # refine with parabolic interpolation on the raw autocorrelation
    lag = 60.0 * fps / chosen
    li = int(round(lag))
    if 1 <= li < n - 1:
        a, b, c = ac[li - 1], ac[li], ac[li + 1]
        denom = a - 2 * b + c
        if abs(denom) > 1e-12:
            li = li + float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))
    refined = 60.0 * fps / max(li, 1e-6)
    if MIN_BPM <= refined <= MAX_BPM:
        chosen = refined
    return float(chosen), [(float(b), float(s)) for b, s in cands[:5]]


def track_beats(env: np.ndarray, fps: float, bpm: float, tightness: float = 300.0) -> np.ndarray:
    """Beat times (seconds) by dynamic programming (Ellis 2007, §3).

    Maximises ``sum_i env[t_i] + alpha * sum_i F(t_i - t_{i-1}, period)`` where
    ``F`` is a squared-log deviation penalty, then backtraces from the best
    ending beat. ``tightness`` is Ellis's alpha: larger keeps the grid more
    rigidly periodic, which is what we want since we re-grid onto a fixed tempo.
    """
    period = 60.0 * fps / bpm
    n = len(env)
    if n < 4 or period < 2:
        return np.array([])
    # local score: envelope smoothed by a period/32 Gaussian to tolerate jitter
    sigma = max(1.0, period / 32.0)
    k = int(np.ceil(4 * sigma))
    g = np.exp(-0.5 * (np.arange(-k, k + 1) / sigma) ** 2)
    local = np.convolve(env, g / g.sum(), mode="same")
    if local.max() > 0:
        local = local / local.max()

    back_max = int(round(-2.0 * period))
    back_min = int(round(-0.5 * period))
    if back_min <= back_max:
        return np.array([])
    offsets = np.arange(back_max, back_min + 1)          # negative offsets
    txcost = -tightness * (np.log(-offsets / period) ** 2)

    cumscore = np.zeros(n)
    backlink = np.full(n, -1, dtype=int)
    for t in range(n):
        idx = t + offsets
        valid = idx >= 0
        if not valid.any():
            cumscore[t] = local[t]
            continue
        cand = cumscore[idx[valid]] + txcost[valid]
        j = int(np.argmax(cand))
        cumscore[t] = local[t] + cand[j]
        backlink[t] = int(idx[valid][j])

    # end at the last strong local maximum of the cumulative score
    tail = cumscore[int(n * 0.0):]
    thresh = np.median(tail) + 0.5 * (tail.max() - np.median(tail))
    ends = np.where(cumscore >= thresh)[0]
    end = int(ends[-1]) if len(ends) else int(np.argmax(cumscore))

    beats = [end]
    while backlink[beats[-1]] >= 0:
        beats.append(backlink[beats[-1]])
    beats.reverse()
    idx = np.asarray(beats, dtype=int)

    # The recursion always starts and ends somewhere, so the first and last
    # beats can land on frames with no onset support at all -- padding either
    # end of the true grid. Ellis trims these; leaving them in biases every
    # downstream tempo estimate, because they stretch the span without adding a
    # real beat period.
    if len(idx) > 4:
        support = local[idx]
        floor = 0.12 * float(np.median(support))
        head = 0
        while head < len(idx) - 2 and support[head] < floor:
            head += 1
        tail = len(idx)
        while tail > head + 2 and support[tail - 1] < floor:
            tail -= 1
        idx = idx[head:tail]
    return idx.astype(float) / fps


def find_downbeats(beats: np.ndarray, env: np.ndarray, low: np.ndarray, fps: float,
                   beats_per_bar: int = 4) -> tuple[int, float]:
    """Pick the bar phase whose beat 1 carries the most low-band + onset energy.

    Returns ``(phase, confidence)``. Confidence is the margin between the best
    and second-best phase, multiplied by the fraction of 8-bar windows that vote
    for the winning phase (the consistency check).
    """
    if len(beats) < beats_per_bar * 2:
        return 0, 0.0
    idx = np.clip((beats * fps).astype(int), 0, min(len(env), len(low)) - 1)
    ln = low / max(low.max(), 1e-9)
    en = env / max(env.max(), 1e-9)
    strength = 0.65 * ln[idx] + 0.35 * en[idx]

    scores = np.array([strength[p::beats_per_bar].mean() for p in range(beats_per_bar)])
    phase = int(np.argmax(scores))
    srt = np.sort(scores)[::-1]
    margin = float((srt[0] - srt[1]) / max(srt[0], 1e-9))

    # consistency: re-vote inside 8-bar windows
    win = beats_per_bar * 8
    votes = []
    for s in range(0, len(strength) - win + 1, win):
        seg = strength[s:s + win]
        votes.append(int(np.argmax([seg[p::beats_per_bar].mean() for p in range(beats_per_bar)])))
    agree = float(np.mean([v == phase for v in votes])) if votes else 1.0
    return phase, float(np.clip(margin * 4.0, 0.0, 1.0) * agree)


def analyze_beats(x: np.ndarray, sr: int) -> BeatGrid:
    """Full beat analysis of a mono signal."""
    fps = F.frame_rate(sr)
    env = F.onset_strength(x, sr)
    bpm, cands = estimate_tempo(env, fps)
    beats = track_beats(env, fps, bpm)
    if len(beats) >= 8:
        # Re-estimate BPM from the realised grid. The DP can only place beats on
        # feature frames (11.6 ms), so any single interval -- and therefore the
        # median -- is quantised to that grid: at 124 BPM the true period is 41.7
        # frames, the median snaps to 42, and the tempo reads 123.0. Dividing the
        # *total span* by the number of periods it contains averages the
        # quantisation away over the whole track, giving a sub-frame estimate,
        # and is insensitive to a single inserted or dropped beat in the middle.
        iois = np.diff(beats)
        coarse = float(np.median(iois))
        span = float(beats[-1] - beats[0])
        n_periods = int(round(span / coarse)) if coarse > 0 else 0
        if n_periods >= 4:
            refined = 60.0 / (span / n_periods)
            if MIN_BPM <= refined <= MAX_BPM:
                bpm = refined
    low = F.band_energy(x, sr, 20.0, 160.0)
    phase, conf = find_downbeats(beats, env, low, fps)
    return BeatGrid(bpm=bpm, beats=beats, downbeat_index=phase,
                    tempo_candidates=cands, downbeat_confidence=conf)
