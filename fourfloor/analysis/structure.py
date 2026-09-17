"""Structural segmentation: boundaries, section labels, per-section loudness.

Boundaries come from Foote's checkerboard novelty on a self-similarity matrix
built from stacked chroma + MFCC features (J. Foote, "Automatic audio
segmentation using a measure of audio novelty", ICME 2000). Labels come from
agglomerative grouping of the resulting segments by feature distance, then a
naming pass: the most-repeated high-energy group becomes the hook.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sps

from . import features as F


@dataclass
class Section:
    """One structural span of the source track."""

    start: float
    end: float
    label: str
    group: int
    energy: float          # 0-1, normalised RMS
    rms_db: float
    repeats: int = 1

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "label": self.label,
            "group": self.group,
            "energy": round(self.energy, 3),
            "rms_db": round(self.rms_db, 2),
            "repeats": self.repeats,
        }


def _checkerboard(size: int) -> np.ndarray:
    """Gaussian-tapered checkerboard kernel of half-width ``size`` (Foote 2000)."""
    n = 2 * size
    g = np.outer(*(2 * [np.exp(-0.5 * (np.linspace(-2, 2, n)) ** 2)]))
    sign = np.ones((n, n))
    sign[:size, size:] = -1.0
    sign[size:, :size] = -1.0
    return g * sign


def _stack(feat: np.ndarray, lag: int = 4, step: int = 2) -> np.ndarray:
    """Time-delay embedding: stack ``lag`` frames spaced ``step`` apart."""
    cols = [np.roll(feat, -i * step, axis=1) for i in range(lag)]
    return np.vstack(cols)[:, : feat.shape[1] - (lag - 1) * step or None]


def novelty_curve(chroma: np.ndarray, timbre: np.ndarray, kernel: int = 32) -> np.ndarray:
    """Foote novelty from a cosine self-similarity matrix of stacked features."""
    def norm_rows(a: np.ndarray) -> np.ndarray:
        a = a - a.mean(axis=1, keepdims=True)
        s = a.std(axis=1, keepdims=True)
        return a / np.maximum(s, 1e-9)

    n = min(chroma.shape[1], timbre.shape[1])
    feat = np.vstack([norm_rows(chroma[:, :n]), 0.5 * norm_rows(timbre[:, :n])])
    feat = _stack(feat)
    norms = np.linalg.norm(feat, axis=0, keepdims=True)
    unit = feat / np.maximum(norms, 1e-9)
    ssm = unit.T @ unit
    m = ssm.shape[0]
    k = min(kernel, max(4, m // 8))
    kern = _checkerboard(k)
    nov = np.zeros(m)
    for i in range(k, m - k):
        nov[i] = float(np.sum(ssm[i - k:i + k, i - k:i + k] * kern))
    nov = np.maximum(nov, 0.0)
    if nov.max() > 0:
        nov /= nov.max()
    return nov


def _snap(times: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Snap each time to the nearest grid point (downbeats), keeping order."""
    if not len(grid):
        return times
    snapped = np.array([grid[int(np.argmin(np.abs(grid - t)))] for t in times])
    return np.unique(snapped)


def segment(x: np.ndarray, sr: int, downbeats: np.ndarray, bar_dur: float,
            chroma: np.ndarray | None = None,
            min_bars: int = 4) -> list[Section]:
    """Segment a track into labelled sections aligned to downbeats.

    ``min_bars`` is the shortest section we allow; house arrangements work in
    4- and 8-bar phrases so anything shorter is merged into its neighbour. It
    scales with the length of the track: a full song is described better by ten
    8-bar-minimum sections than by twenty 4-bar ones, while a short loop would
    collapse to a single block under the same rule.
    """
    fps = F.frame_rate(sr)
    total_bars = max(1, int((len(x) / sr) / max(bar_dur, 1e-6)))
    min_bars = min(min_bars * 2, max(min_bars, total_bars // 8))
    chroma = F.chromagram(x, sr) if chroma is None else chroma
    timbre = F.mfcc(x, sr)
    nov = novelty_curve(chroma, timbre)

    min_dist = max(2, int(round(min_bars * bar_dur * fps)))
    height = float(np.percentile(nov[nov > 0], 55)) if np.any(nov > 0) else 0.0
    peaks, _ = sps.find_peaks(nov, distance=min_dist, height=height)
    bounds = np.concatenate([[0.0], peaks / fps, [len(x) / sr]])
    if len(downbeats):
        inner = _snap(bounds[1:-1], downbeats) if len(bounds) > 2 else np.array([])
        bounds = np.unique(np.concatenate([[0.0], inner, [len(x) / sr]]))

    # merge anything shorter than min_bars
    merged = [float(bounds[0])]
    for b in bounds[1:]:
        if b - merged[-1] >= min_bars * bar_dur * 0.9 or b == bounds[-1]:
            merged.append(float(b))
    if len(merged) > 2 and merged[-1] - merged[-2] < min_bars * bar_dur * 0.9:
        merged.pop(-2)
    bounds = np.asarray(merged)
    if len(bounds) < 2:
        bounds = np.array([0.0, len(x) / sr])

    # per-segment descriptors
    descs, energies, rmsdbs = [], [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        fa, fb = int(a * fps), max(int(b * fps), int(a * fps) + 1)
        ca = chroma[:, fa:min(fb, chroma.shape[1])]
        ta = timbre[:, fa:min(fb, timbre.shape[1])]
        cv = ca.mean(axis=1) if ca.size else np.zeros(12)
        tv = ta.mean(axis=1) if ta.size else np.zeros(timbre.shape[0])
        descs.append(np.concatenate([cv / max(cv.sum(), 1e-9), tv / max(np.abs(tv).max(), 1e-9)]))
        seg = x[int(a * sr):int(b * sr)]
        r = float(np.sqrt(np.mean(np.square(seg)))) if len(seg) else 1e-6
        energies.append(r)
        rmsdbs.append(20.0 * np.log10(max(r, 1e-6)))
    D = np.asarray(descs)
    emax = max(max(energies), 1e-9)

    # group similar segments (single-link at a distance threshold)
    n = len(D)
    groups = list(range(n))
    if n > 1:
        dist = np.linalg.norm(D[:, None, :] - D[None, :, :], axis=2)
        iu = np.triu_indices(n, 1)
        thr = float(np.percentile(dist[iu], 30)) if len(iu[0]) else 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if dist[i, j] <= thr:
                    tgt, src = groups[i], groups[j]
                    if tgt != src:
                        groups = [tgt if g == src else g for g in groups]
    counts = {g: groups.count(g) for g in set(groups)}

    sections: list[Section] = []
    for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        sections.append(Section(
            start=float(a), end=float(b), label="", group=groups[i],
            energy=float(energies[i] / emax), rms_db=float(rmsdbs[i]),
            repeats=counts[groups[i]],
        ))
    _label(sections)
    return sections


def _label(sections: list[Section]) -> None:
    """Name sections: hook = most repeated high-energy group, then verse/etc."""
    if not sections:
        return
    by_group: dict[int, list[Section]] = {}
    for s in sections:
        by_group.setdefault(s.group, []).append(s)

    def group_score(items: list[Section]) -> float:
        return float(np.mean([s.energy for s in items])) * (1.0 + 0.35 * (len(items) - 1))

    ranked = sorted(by_group.items(), key=lambda kv: -group_score(kv[1]))
    hook_group = ranked[0][0]
    verse_group = ranked[1][0] if len(ranked) > 1 else hook_group

    for s in sections:
        if s.group == hook_group:
            s.label = "hook"
        elif s.group == verse_group:
            s.label = "verse"
        elif s.energy < 0.45:
            s.label = "breakdown"
        else:
            s.label = "section"
    # the first and last low-energy spans read as intro/outro
    if sections[0].energy < 0.7 and sections[0].label != "hook":
        sections[0].label = "intro"
    if len(sections) > 1 and sections[-1].energy < 0.7 and sections[-1].label != "hook":
        sections[-1].label = "outro"


def energy_per_bar(x: np.ndarray, sr: int, downbeats: np.ndarray, bar_dur: float) -> list[float]:
    """Normalised RMS energy for each bar, for the session file's energy curve."""
    vals = []
    for t in downbeats:
        a, b = int(t * sr), int((t + bar_dur) * sr)
        seg = x[max(0, a):min(len(x), b)]
        vals.append(float(np.sqrt(np.mean(np.square(seg)))) if len(seg) else 0.0)
    m = max(vals) if vals else 1.0
    return [round(v / max(m, 1e-9), 4) for v in vals]


# --------------------------------------------------------------------------
# Vocal phrasing
#
# Foote novelty tells us *where the music changes*; it says nothing about
# whether a singer happens to be halfway through the word "again" at that
# moment. Cutting there is the single most audible mistake an automatic edit
# can make, so the arranger needs a second, independent view of the source: a
# vocal-activity envelope and the gaps in it.
# --------------------------------------------------------------------------

#: Band that carries sung fundamentals and the first formants. Below this is
#: kick and bass; above it is mostly cymbals and sibilance, which are exactly
#: the things that make a purely broadband envelope look "busy" during a gap.
VOCAL_BAND = (180.0, 4200.0)

#: A gap shorter than this is a breath or a consonant stop, not a phrase end.
#: 300 ms is about one 16th note at 124 BPM: long enough that a cut inside it
#: cannot clip a syllable, short enough that most songs have several per verse.
GAP_MIN = 0.30

#: How far under the "someone is singing" level a frame has to sit to count as
#: silence. Measured against the 90th percentile of the envelope so it tracks
#: the mix rather than an absolute dBFS number.
GAP_DROP_DB = 13.0

#: Envelope hop. 43 Hz resolves a 300 ms gap to within a frame and costs a
#: quarter of what the 86 Hz feature rate would.
ENV_HOP = 1024

#: Below this dynamic range (p90 minus p10, dB) the "vocal" band is a pad, a
#: loop or a click track rather than a voice, and its gaps mean nothing.
MIN_CONTRAST_DB = 9.0

#: Fewer gaps than this over a whole track means we found noise-floor dips, not
#: phrase ends -- an instrumental, a loop, or a master squashed flat.
MIN_GAPS = 3

#: Shortest upper-quartile voiced run we will believe is a voice. Without it a
#: metronome looks like the most articulate singer alive: enormous contrast,
#: dozens of evenly spaced "phrase gaps", every one of them meaningless.
#:
#: Measured upper quartiles: a click track and a 16th hat pattern both sit at
#: 0.070 s and never exceed it at any percentile, while a vocal runs 0.60 s
#: (Demucs stem) to 0.72 s (centre-extracted mix). 0.35 s sits between them
#: with roughly a factor of two of margin on each side.
MIN_VOICED_RUN = 0.35


@dataclass(frozen=True)
class PhraseGap:
    """A stretch of source time with no vocal in it, wide enough to cut in."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def mid(self) -> float:
        return 0.5 * (self.start + self.end)

    def contains(self, t: float, pad: float = 0.0) -> bool:
        return (self.start - pad) <= t <= (self.end + pad)

    def to_dict(self) -> dict:
        return {"start": round(self.start, 3), "end": round(self.end, 3),
                "duration": round(self.duration, 3)}


@dataclass
class VocalMap:
    """Where the voice is, and where it is not, in one source track.

    ``env`` is a 0-1 activity curve at ``fps`` frames per second; ``gaps`` are
    the spans where it stays under ``threshold`` for at least ``GAP_MIN``.
    ``source`` records which signal it was measured from so a plan can say why
    it trusted (or ignored) the result.
    """

    env: np.ndarray
    fps: float
    threshold: float
    gaps: list[PhraseGap]
    source: str
    duration: float
    contrast_db: float
    voiced_run: float = 0.0     # 75th-pct length of a run above `threshold`

    @property
    def usable(self) -> bool:
        """True when the envelope has enough contrast to mean anything.

        An instrumental loop, a click track or a wall-of-sound master gives a
        near-flat curve; treating its noise floor as "phrase gaps" would snap
        cuts to arbitrary places with false confidence. When this is False the
        arranger falls back to plain downbeat alignment and says so.
        """
        return (len(self.gaps) >= MIN_GAPS
                and self.contrast_db >= MIN_CONTRAST_DB
                and self.voiced_run >= MIN_VOICED_RUN
                and sum(g.duration for g in self.gaps) >= 0.01 * max(self.duration, 1e-6))

    def _frame(self, t: float) -> int:
        return int(np.clip(round(t * self.fps), 0, max(len(self.env) - 1, 0)))

    def activity(self, t: float) -> float:
        """Vocal activity at one instant, 0-1."""
        if not len(self.env):
            return 0.0
        return float(self.env[self._frame(t)])

    def peak_activity(self, a: float, b: float) -> float:
        """Loudest vocal moment in a span -- the "is this mid-word?" measure.

        A cut sitting inside a syllable has energy on *both* sides of it, so the
        peak over a short window straddling the cut is high even when the
        instant itself happens to fall in a glottal dip.
        """
        if not len(self.env):
            return 0.0
        i, j = self._frame(min(a, b)), self._frame(max(a, b))
        return float(self.env[i:max(j + 1, i + 1)].max())

    def mean_activity(self, a: float, b: float) -> float:
        """Average vocal presence over a span -- how much singing it contains."""
        if not len(self.env) or b <= a:
            return 0.0
        i, j = self._frame(a), self._frame(b)
        return float(self.env[i:max(j + 1, i + 1)].mean())

    def in_gap(self, t: float, pad: float = 0.0) -> bool:
        return any(g.contains(t, pad) for g in self.gaps)

    def gap_at(self, t: float, pad: float = 0.0) -> PhraseGap | None:
        for g in self.gaps:
            if g.contains(t, pad):
                return g
        return None

    def phrase_starts(self) -> list[float]:
        """Times a vocal phrase begins: the end of every gap."""
        return [g.end for g in self.gaps]

    def phrase_ends(self) -> list[float]:
        """Times a vocal phrase finishes: the start of every gap."""
        return [g.start for g in self.gaps]

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "usable": self.usable,
            "threshold": round(self.threshold, 4),
            "contrast_db": round(self.contrast_db, 2),
            "voiced_run": round(self.voiced_run, 3),
            "gaps": len(self.gaps),
            "gap_seconds": round(sum(g.duration for g in self.gaps), 2),
        }


def _centre_magnitude(x: np.ndarray, hop: int) -> np.ndarray:
    """Magnitude spectrogram of what is panned dead centre.

    ``|mid| - |side|`` per bin. In a commercial stereo master the lead vocal is
    centred and almost everything else -- pads, guitars, reverb tails, stereo
    synths -- is spread, so this removes a large part of the *harmonic* backing
    that band-limiting alone cannot touch. Measured against a Demucs vocal stem
    on the reference pair it roughly doubles the number of true phrase gaps
    found, for the cost of one extra STFT.
    """
    left, right = x[:, 0].astype(np.float64), x[:, 1].astype(np.float64)
    mid = np.abs(F.stft(left + right, n_fft=2048, hop=hop))
    side = np.abs(F.stft(left - right, n_fft=2048, hop=hop))
    return np.maximum(mid - side, 0.0)


def vocal_envelope(x: np.ndarray, sr: int, hop: int = ENV_HOP,
                   band: tuple[float, float] = VOCAL_BAND) -> tuple[np.ndarray, float]:
    """A 0-1 curve of how much *voice-like* energy the signal has, over time.

    Two cheap discriminators, multiplied:

    * band energy in ``band``, which throws away the kick, the sub and most of
      the cymbals before anything else is measured;
    * tonality, ``1 - flatness``, where flatness is the geometric over the
      arithmetic mean of the band spectrum (Wiener entropy). A sung note is a
      handful of loud harmonics over a quiet floor and scores near 1; a snare,
      a hat or vinyl noise is broadband and scores near 0.

    Running this on a Demucs ``vocals`` stem instead of the full mix makes the
    first term nearly exact; the second still earns its keep by suppressing the
    bleed and separation artefacts that stem carries in its quiet moments.
    """
    x = np.asarray(x, dtype=np.float32)
    fps = sr / float(hop)
    if len(x) < hop * 2:
        return np.zeros(1, dtype=np.float32), fps

    if x.ndim > 1 and x.shape[1] == 2:
        spec = _centre_magnitude(x, hop)
    else:
        mono = x.mean(axis=1) if x.ndim > 1 else x
        spec = np.abs(F.stft(mono, n_fft=2048, hop=hop))
    freqs = np.fft.rfftfreq(2048, 1.0 / sr)
    lo = int(np.searchsorted(freqs, band[0]))
    hi = int(np.searchsorted(freqs, band[1]))
    sub = spec[lo:hi] ** 2
    if not sub.size:
        return np.zeros(spec.shape[1], dtype=np.float32), fps

    energy = np.sqrt(sub.mean(axis=0))
    log_mean = np.exp(np.log(sub + 1e-12).mean(axis=0))
    flatness = log_mean / np.maximum(sub.mean(axis=0), 1e-12)
    env = energy * (1.0 - np.clip(flatness, 0.0, 1.0))

    # ~45 ms smoothing: long enough to ride the glottal pulses inside one vowel
    # (a pitch period is 4-10 ms), short enough that it smears a phrase edge by
    # less than CUT_GUARD. If the smear were wider than the guard, a cut placed
    # exactly on a phrase start would read as mid-word, which is the one case
    # the snapper most wants to say yes to.
    w = max(1, int(round(0.045 * fps)))
    if w > 1:
        env = np.convolve(env, np.ones(w) / w, mode="same")
    peak = float(np.percentile(env, 99.0))
    env = env / max(peak, 1e-9)
    return np.clip(env, 0.0, 1.5).astype(np.float32), fps


def find_gaps(env: np.ndarray, fps: float, threshold: float,
              min_gap: float = GAP_MIN) -> list[PhraseGap]:
    """Runs of ``env`` below ``threshold`` lasting at least ``min_gap``."""
    if not len(env):
        return []
    quiet = env < threshold
    gaps: list[PhraseGap] = []
    start = None
    for i, q in enumerate(quiet):
        if q and start is None:
            start = i
        elif not q and start is not None:
            a, b = start / fps, i / fps
            if b - a >= min_gap:
                gaps.append(PhraseGap(a, b))
            start = None
    if start is not None:
        a, b = start / fps, len(quiet) / fps
        if b - a >= min_gap:
            gaps.append(PhraseGap(a, b))
    return gaps


def voiced_run(env: np.ndarray, fps: float, threshold: float,
               percentile: float = 75.0) -> float:
    """Upper-quartile length in seconds of a run above ``threshold``.

    The *median* is the wrong statistic here, and wrong in the direction that
    matters: a cleanly separated vocal stem falls to silence between syllables,
    so its median run is one syllable (0.30 s measured) while a bleedier
    centre-extracted mix of the same song reads 0.44 s. Judging by the median
    would reject the better input. The upper quartile asks a different question
    -- does this signal ever sustain the way a voice does? -- which a percussive
    track fails at every percentile, its runs being one decay envelope long.
    """
    if not len(env):
        return 0.0
    loud = env >= threshold
    runs, n = [], 0
    for v in loud:
        if v:
            n += 1
        elif n:
            runs.append(n / fps)
            n = 0
    if n:
        runs.append(n / fps)
    return float(np.percentile(runs, percentile)) if runs else 0.0


def vocal_map(x: np.ndarray, sr: int, vocals: np.ndarray | None = None,
              min_gap: float = GAP_MIN, drop_db: float = GAP_DROP_DB) -> VocalMap:
    """Build the :class:`VocalMap` the arranger cuts against.

    ``vocals`` is an isolated vocal (or harmonic) stem when one is available --
    ``fourfloor.stems.separate`` gives you one either way. When it is ``None``
    the envelope is measured from the full mix, which is noisier but still
    finds phrase boundaries in anything with a foreground voice.

    The threshold is relative: 13 dB under the 90th percentile of the envelope.
    An absolute level would find no gaps at all in a loud master and nothing
    but gaps in a quiet one.
    """
    signal = vocals if vocals is not None else x
    stereo = np.asarray(signal).ndim > 1 and np.asarray(signal).shape[-1] == 2
    source = ("stem" if vocals is not None else "mix") + ("+centre" if stereo else "")
    env, fps = vocal_envelope(signal, sr)
    duration = len(np.asarray(x)) / float(sr)

    if len(env) < 2:
        return VocalMap(env, fps, 0.0, [], source, duration, 0.0, 0.0)
    p90 = float(np.percentile(env, 90.0))
    p10 = float(np.percentile(env, 10.0))
    # a synthetic source can be digitally silent between events, which would
    # send the ratio to infinity; cap the usable range at 80 dB
    contrast = 20.0 * np.log10(max(p90, 1e-9) / max(p10, 1e-4 * p90, 1e-9))
    threshold = p90 * (10.0 ** (-drop_db / 20.0))
    gaps = find_gaps(env, fps, threshold, min_gap)
    return VocalMap(env, fps, threshold, gaps, source, duration, float(contrast),
                    voiced_run(env, fps, threshold))


#: How close to a cut a syllable has to be before the cut counts as mid-word.
#: 80 ms is under half a sung syllable, so a splice this close to vocal energy
#: audibly chops a word; further away and the ear hears a phrase boundary.
CUT_GUARD = 0.08

#: How far either side of a structural boundary we may move a cut, in bars.
#: Two bars is enough to reach the nearest phrase edge in almost every song and
#: short enough that the cut still lands where the segmentation meant it to.
SNAP_WINDOW_BARS = 2.0


@dataclass(frozen=True)
class CutPoint:
    """One chosen edit point, with the reasoning that produced it."""

    time: float
    reason: str
    mid_phrase: bool
    moved_bars: float
    cost: float = 0.0          # comparable across candidates; not serialised

    def to_dict(self) -> dict:
        return {"time": round(self.time, 3), "reason": self.reason,
                "mid_phrase": self.mid_phrase, "moved_bars": round(self.moved_bars, 2)}


def _bar_grid(t: float, downbeats: np.ndarray, bar_dur: float,
              window_bars: float) -> list[float]:
    """Downbeat candidates within ``window_bars`` of ``t``.

    Falls back to a synthetic grid hung off ``t`` when the source has no usable
    downbeat track, so the search window behaves the same either way.
    """
    span = window_bars * bar_dur
    if len(downbeats):
        near = [float(d) for d in downbeats if abs(d - t) <= span + 1e-6]
        if near:
            # always offer the nearest downbeat even if the window missed it
            nearest = float(downbeats[int(np.argmin(np.abs(downbeats - t)))])
            if nearest not in near:
                near.append(nearest)
            return sorted(near)
        return [float(downbeats[int(np.argmin(np.abs(downbeats - t)))])]
    k = int(window_bars)
    return [t + i * bar_dur for i in range(-k, k + 1)]


def _splice_energy(vmap: VocalMap, t: float, role: str, guard: float) -> float:
    """How much of a word a cut at ``t`` would tear in half.

    A splice is only "mid-word" when the voice is sounding *continuously
    across* it -- energy on both sides within ``guard``. Testing one side alone
    gets both ends wrong in the same way: a clean entry at a phrase start is
    loud immediately after it, and a clean exit at a phrase end is loud
    immediately before it, so a one-sided test rejects exactly the cuts a human
    editor would make. Taking the smaller of the two sides passes both and
    still catches a cut dropped into the middle of a sustained note.

    ``role`` does not change this measurement -- it changes which *bonuses*
    :func:`snap_cut` applies, since an entry wants a phrase start under it and
    an exit wants a phrase end.
    """
    lead = vmap.peak_activity(t - guard, t - guard * 0.1)
    tail = vmap.peak_activity(t + guard * 0.1, t + guard)
    return min(lead, tail)


def snap_cut(t: float, downbeats: np.ndarray, vmap: VocalMap, bar_dur: float,
             role: str = "entry", window_bars: float = SNAP_WINDOW_BARS,
             guard: float = CUT_GUARD) -> CutPoint:
    """Move a cut to the best nearby downbeat that is not inside a word.

    Every candidate is a downbeat, so the result is on the grid by
    construction. Among those, the cost prefers, in order: no vocal energy
    within ``guard`` of the splice; sitting inside a phrase gap; being a phrase
    *start* (for an entry) or a phrase *end* (for an exit); and staying close
    to where the segmentation put the boundary.

    When the source has no usable vocal contrast the cost collapses to the
    distance term and this is just "snap to the nearest downbeat", which is
    what the arranger did before -- the difference is that the plan now says so.
    """
    cands = _bar_grid(t, downbeats, bar_dur, window_bars)
    if not cands:
        return CutPoint(t, "no grid; left where the segmentation put it", False, 0.0, 0.0)

    if not vmap.usable:
        best = min(cands, key=lambda c: abs(c - t))
        return CutPoint(best, f"nearest downbeat ({vmap.source} has no vocal "
                              f"contrast to phrase against)", False,
                        (best - t) / bar_dur, 0.0)

    beat = bar_dur / 4.0
    scored: list[tuple[float, float, str, bool]] = []
    for c in cands:
        peak = _splice_energy(vmap, c, role, guard)
        mid = peak > vmap.threshold
        cost = 1.0 * min(peak / max(vmap.threshold, 1e-9), 4.0)
        cost += 0.12 * abs(c - t) / bar_dur

        why: list[str] = []
        gap = vmap.gap_at(c)
        if gap is not None:
            cost -= 0.35
            why.append(f"in a {gap.duration:.2f}s phrase gap")
        if role == "entry":
            near = [g for g in vmap.gaps if -0.25 * beat <= c - g.end <= 3.0 * beat]
            if near:
                cost -= 0.30
                why.append(f"phrase starts {abs(c - near[-1].end):.2f}s away")
        else:
            near = [g for g in vmap.gaps if -3.0 * beat <= g.start - c <= 0.25 * beat]
            if near:
                cost -= 0.30
                why.append(f"phrase ends {abs(near[0].start - c):.2f}s away")
        if not why:
            why.append(f"vocal at {peak / max(vmap.threshold, 1e-9):.1f}x the gap level")
        scored.append((cost, c, ", ".join(why), mid))

    cost, best, why, mid = min(scored, key=lambda s: (s[0], abs(s[1] - t)))
    moved = (best - t) / bar_dur
    move_txt = "on the boundary" if abs(moved) < 1e-6 else f"{moved:+.0f} bar(s)"
    return CutPoint(best, f"downbeat {move_txt}: {why}", mid, moved, cost)


def section_hook_score(sec: Section, vmap: VocalMap, max_repeats: int,
                       bar_dur: float) -> float:
    """How much a section behaves like the hook of the song, 0-1.

    Loudness alone picks the densest bar of the master, which in a modern mix
    is often a bridge or an ad-lib pile-up rather than the part anyone would
    sing back. Three signals, weighted:

    * **vocal presence** -- a house drop needs a voice over it, and a section
      that is 70% singing beats one that is 20% singing at the same RMS;
    * **repetition** -- the hook is, definitionally, the thing that comes back;
    * **energy** -- still matters, just no longer on its own.

    A short section is discounted: you cannot build a 32-bar drop out of four
    bars of source without hearing the loop.
    """
    presence = vmap.mean_activity(sec.start, sec.end) if vmap.usable else 0.5
    repeat = (sec.repeats - 1) / max(max_repeats - 1, 1)
    bars = sec.duration / max(bar_dur, 1e-6)
    length = float(np.clip(bars / 8.0, 0.0, 1.0))
    score = 0.34 * float(np.clip(presence * 1.6, 0.0, 1.0)) \
        + 0.26 * float(np.clip(repeat, 0.0, 1.0)) \
        + 0.28 * float(np.clip(sec.energy, 0.0, 1.0)) \
        + 0.12 * length
    if sec.label == "hook":
        score += 0.10          # the segmenter's own opinion, as a tie-break
    return float(score)
